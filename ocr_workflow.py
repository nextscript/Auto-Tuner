"""Document OCR workflow for llama.cpp multimodal/OCR models.

llama.cpp accepts raster media, not PDF or Office documents. This module keeps
that boundary explicit: Office files are converted to PDF with LibreOffice,
PDF pages are rendered with PyMuPDF, raster inputs are normalized with Pillow,
and each page is sent to a running llama-server through
``/v1/chat/completions``.

The module is UI-independent. Both the Qt launcher and terminal launcher use the
same presets, conversion code, request schema, output layout, cancellation, and
manifest format.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
)
PDF_EXTENSIONS = frozenset({".pdf"})
OFFICE_EXTENSIONS = frozenset(
    {
        ".doc",
        ".docx",
        ".odt",
        ".rtf",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
        ".ods",
    }
)
SUPPORTED_INPUT_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS | OFFICE_EXTENSIONS
MAX_IMAGE_PIXELS = 100_000_000
MAX_REQUEST_IMAGE_BYTES = 64 * 1024 * 1024
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class OcrModelPreset:
    """Model-family-specific OCR defaults verified against llama.cpp docs/tests."""

    family: str
    label: str
    prompt: str
    max_tokens: int = 4096
    min_llama_build: int = 0
    strip_grounding: bool = False
    notes: str = ""


_PRESETS = {
    "unlimited-ocr": OcrModelPreset(
        family="unlimited-ocr",
        label="Unlimited OCR",
        prompt="document parsing.",
        max_tokens=4096,
        min_llama_build=10287,
        strip_grounding=True,
        notes=(
            "Uses b10287+'s corrected 32-tile preprocessing and b10285+'s "
            "multi-row DeepSeek-OCR batching. PDF and Office pages are rendered "
            "before inference."
        ),
    ),
    "deepseek-ocr": OcrModelPreset(
        family="deepseek-ocr",
        label="DeepSeek OCR",
        prompt="Free OCR.",
        max_tokens=4096,
        strip_grounding=True,
    ),
    "paddleocr-vl": OcrModelPreset(
        family="paddleocr-vl",
        label="PaddleOCR-VL",
        prompt="OCR:",
        max_tokens=4096,
    ),
    "glm-ocr": OcrModelPreset(
        family="glm-ocr",
        label="GLM-OCR",
        prompt="Text Recognition:",
        max_tokens=4096,
    ),
    "dots-ocr": OcrModelPreset(
        family="dots-ocr",
        label="Dots.OCR",
        prompt="Extract the text content from this image.",
        max_tokens=4096,
    ),
    "hunyuan-ocr": OcrModelPreset(
        family="hunyuan-ocr", label="HunyuanOCR", prompt="OCR", max_tokens=4096
    ),
    "lighton-ocr": OcrModelPreset(
        family="lighton-ocr", label="LightOnOCR", prompt="", max_tokens=4096
    ),
    "qianfan-ocr": OcrModelPreset(
        family="qianfan-ocr",
        label="Qianfan-OCR",
        prompt="Parse this document to Markdown.",
        max_tokens=4096,
    ),
    "docling": OcrModelPreset(
        family="docling",
        label="Docling vision",
        prompt="Convert this document page to clean Markdown.",
        max_tokens=4096,
    ),
    "generic-ocr": OcrModelPreset(
        family="generic-ocr",
        label="OCR / document vision",
        prompt=(
            "Extract all text from this document image as clean Markdown. "
            "Preserve headings, lists, tables, and reading order."
        ),
        max_tokens=4096,
    ),
}


def _model_search_text(model: object) -> Tuple[str, str]:
    metadata = getattr(model, "metadata", {}) or {}
    name = str(getattr(model, "name", "") or "")
    general_name = str(metadata.get("general.name", "") or "")
    arch = str(
        metadata.get("general.architecture", getattr(model, "architecture", "")) or ""
    )
    return f"{name} {general_name}".lower(), arch.lower().strip()


def ocr_model_family(model: object) -> Optional[str]:
    """Return a curated OCR family id, or ``None`` for a non-OCR model."""
    text, arch = _model_search_text(model)
    # DeepSeek-OCR v1/v2 and Unlimited-OCR share deepseek2-ocr architecture.
    # Only the checkpoint name distinguishes their incompatible prompts.
    if "unlimited" in text and "ocr" in text:
        return "unlimited-ocr"
    if ("deepseek" in text and "ocr" in text) or ("deepseek" in arch and "ocr" in arch):
        return "deepseek-ocr"
    if "paddleocr" in text or "paddleocr" in arch:
        return "paddleocr-vl"
    if "glm" in text and "ocr" in text or "glm-ocr" in arch:
        return "glm-ocr"
    if "dots.ocr" in text or "dots-ocr" in text or "dotsocr" in arch:
        return "dots-ocr"
    if "hunyuan" in text and "ocr" in text or "hunyuan-ocr" in arch:
        return "hunyuan-ocr"
    if "lightonocr" in text or "lighton-ocr" in text:
        return "lighton-ocr"
    if "qianfan" in text and "ocr" in text:
        return "qianfan-ocr"
    if "docling" in text or "docling" in arch:
        return "docling"
    if "ocr" in text or "ocr" in arch:
        return "generic-ocr"
    return None


def is_ocr_model(model: object) -> bool:
    return ocr_model_family(model) is not None


def ocr_model_preset(model: object) -> OcrModelPreset:
    family = ocr_model_family(model) or "generic-ocr"
    return _PRESETS[family]


def ocr_projector_warning(model: object) -> str:
    """Return an actionable warning for stale Unlimited-OCR projectors."""
    if ocr_model_family(model) != "unlimited-ocr":
        return ""
    projector = getattr(model, "mmproj", None)
    if projector is None:
        return "Unlimited-OCR requires its matching mmproj projector."
    try:
        from scanner import read_gguf_metadata

        metadata = read_gguf_metadata(Path(projector))
        max_tiles = int(metadata.get("clip.vision.preproc_max_tiles", 0) or 0)
    except (OSError, TypeError, ValueError):
        max_tiles = 0
    if max_tiles >= 32:
        return ""
    return (
        f"{Path(projector).name} does not declare "
        "clip.vision.preproc_max_tiles=32. llama.cpp will fall back to the "
        "older 9-tile DeepSeek-OCR limit, reducing quality on tall/dense pages. "
        "Use a projector converted/re-uploaded with llama.cpp b10287 or newer "
        "before relying on reference-quality Unlimited-OCR results."
    )


@dataclass
class OcrJobOptions:
    inputs: List[Path]
    output_dir: Path
    prompt: str
    max_tokens: int = 4096
    dpi: int = 220
    page_range: str = ""
    output_format: str = "markdown"  # markdown | text
    keep_rendered: bool = False
    strip_grounding: bool = False
    continue_on_error: bool = True
    request_timeout_seconds: int = 1200
    stop_server_when_done: bool = True

    def normalized(self) -> "OcrJobOptions":
        fmt = str(self.output_format or "markdown").strip().lower()
        if fmt not in {"markdown", "text"}:
            fmt = "markdown"
        return OcrJobOptions(
            inputs=[Path(p).expanduser() for p in self.inputs],
            output_dir=Path(self.output_dir).expanduser(),
            prompt=str(self.prompt),
            max_tokens=max(1, min(32768, int(self.max_tokens))),
            dpi=max(72, min(600, int(self.dpi))),
            page_range=str(self.page_range or "").strip(),
            output_format=fmt,
            keep_rendered=bool(self.keep_rendered),
            strip_grounding=bool(self.strip_grounding),
            continue_on_error=bool(self.continue_on_error),
            request_timeout_seconds=max(
                10, min(7200, int(self.request_timeout_seconds))
            ),
            stop_server_when_done=bool(self.stop_server_when_done),
        )


@dataclass
class OcrPageResult:
    source: str
    page: int
    output_file: str
    characters: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""


@dataclass
class OcrJobResult:
    job_dir: Path
    combined_output: Path
    manifest_path: Path
    total_pages: int
    completed_pages: int
    failed_pages: int
    cancelled: bool
    page_results: List[OcrPageResult] = field(default_factory=list)


@dataclass(frozen=True)
class _DocumentSource:
    original: Path
    media: Path
    kind: str  # image | pdf
    pages: Tuple[int, ...]  # zero-based indices


ProgressCallback = Callable[[str, int, int, str], None]


class OcrWorkflowError(RuntimeError):
    pass


class OcrCancelled(OcrWorkflowError):
    pass


def parse_page_range(spec: str, page_count: int) -> Tuple[int, ...]:
    """Parse ``1-3,5`` into sorted zero-based page indices."""
    if page_count < 0:
        raise ValueError("page_count must not be negative")
    raw = str(spec or "").strip()
    if not raw:
        return tuple(range(page_count))
    selected: set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            left, right = chunk.split("-", 1)
            if not left.strip().isdigit() or not right.strip().isdigit():
                raise ValueError(f"invalid page range: {chunk!r}")
            start, end = int(left), int(right)
        else:
            if not chunk.isdigit():
                raise ValueError(f"invalid page number: {chunk!r}")
            start = end = int(chunk)
        if start < 1 or end < start:
            raise ValueError(f"invalid page range: {chunk!r}")
        if end > page_count:
            raise ValueError(f"page {end} exceeds document page count {page_count}")
        selected.update(range(start - 1, end))
    if not selected:
        raise ValueError("page range selects no pages")
    return tuple(sorted(selected))


def discover_input_files(
    inputs: Iterable[Path], cancel_event: Optional[threading.Event] = None
) -> List[Path]:
    """Expand files/directories, filter supported formats, and de-duplicate."""
    found: List[Path] = []
    seen: set[str] = set()
    for raw in inputs:
        if cancel_event is not None and cancel_event.is_set():
            raise OcrCancelled("OCR input discovery cancelled")
        path = Path(raw).expanduser()
        candidates: Iterable[Path]
        if path.is_dir():
            discovered: List[Path] = []
            for root, dirs, files in os.walk(path):
                if cancel_event is not None and cancel_event.is_set():
                    raise OcrCancelled("OCR input discovery cancelled")
                dirs.sort(key=str.lower)
                files.sort(key=str.lower)
                discovered.extend(Path(root) / name for name in files)
            candidates = discovered
        elif path.is_file():
            candidates = (path,)
        else:
            raise OcrWorkflowError(f"Input does not exist: {path}")
        for candidate in candidates:
            if cancel_event is not None and cancel_event.is_set():
                raise OcrCancelled("OCR input discovery cancelled")
            if candidate.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
                continue
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as exc:
                raise OcrWorkflowError(f"Input cannot be resolved: {candidate}: {exc}")
            key = os.path.normcase(str(resolved))
            if key not in seen:
                seen.add(key)
                found.append(resolved)
    if not found:
        supported = ", ".join(sorted(SUPPORTED_INPUT_EXTENSIONS))
        raise OcrWorkflowError(f"No supported input files found ({supported})")
    return found


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._")
    return stem[:100] or "document"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def _redact_command(command: Sequence[str]) -> List[str]:
    sensitive = {
        "--api-key",
        "--hf-token",
        "-hft",
        "--mcp-servers-json",
        "--connect-code",
    }
    redacted: List[str] = []
    hide_next = False
    for token in command:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        flag = token.split("=", 1)[0]
        if flag in sensitive:
            if "=" in token:
                redacted.append(flag + "=<redacted>")
            else:
                redacted.append(token)
                hide_next = True
        else:
            redacted.append(token)
    return redacted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_grounding_markup(text: str) -> str:
    """Remove OCR coordinate markup while retaining recognized text."""
    value = str(text or "")
    value = re.sub(
        r"<\|det\|>.*?<\|/det\|>", "", value, flags=re.IGNORECASE | re.DOTALL
    )
    value = re.sub(r"<\|/?ref\|>", "", value, flags=re.IGNORECASE)
    # llama-server may already consume the special-token delimiters and expose
    # DeepSeek-OCR grounding as ``text [x1, y1, x2, y2]...``. Treat that as
    # the same coordinate markup when clean output is requested.
    value = re.sub(
        r"(?im)^(?:header|title|text|table|figure|image|formula|caption|list)\s+"
        r"\[[^\]\r\n]+\]\s*",
        "",
        value,
    )
    value = re.sub(
        r"<\|(?:endoftext|end_of_sentence|eos)\|>", "", value, flags=re.IGNORECASE
    )
    value = value.strip()
    # Some chat-completions paths prepend one generic refusal/quality warning
    # before a long, valid parse. Remove it only when substantial OCR follows;
    # keep the same sentence when it is the model's sole result.
    lines = value.splitlines()
    if len(lines) >= 4 and len("\n".join(lines[1:]).strip()) >= 100:
        first = lines[0].strip().lower()
        if re.search(
            r"(?:too blurry|cannot recognize|can't recognize|unable to recognize|"
            r"no (?:readable )?text|no text content)",
            first,
        ):
            value = "\n".join(lines[1:]).lstrip()
    return value


def _candidate_soffice_paths() -> Iterable[Path]:
    if os.name == "nt":
        for env_name in ("PROGRAMFILES", "PROGRAMFILES(X86)"):
            root = os.environ.get(env_name)
            if root:
                yield Path(root) / "LibreOffice" / "program" / "soffice.com"
                yield Path(root) / "LibreOffice" / "program" / "soffice.exe"
    elif sys_platform() == "darwin":
        yield Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")
        yield Path.home() / "Applications/LibreOffice.app/Contents/MacOS/soffice"


def sys_platform() -> str:
    import sys

    return sys.platform


def find_libreoffice() -> Optional[str]:
    for name in ("soffice.com", "soffice", "libreoffice"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    for path in _candidate_soffice_paths():
        if path.is_file():
            return str(path)
    return None


def _process_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        # LibreOffice's soffice.com wrapper can terminate with 0xC0000409 when
        # placed in CREATE_NEW_PROCESS_GROUP from a windowed Python process.
        # Keep the normal Windows process group; taskkill /T handles cancellation.
        return {}
    return {"start_new_session": True}


def _close_quietly(connection: http.client.HTTPConnection) -> None:
    try:
        connection.close()
    except Exception:
        pass


def _abort_connection(connection: http.client.HTTPConnection) -> None:
    """Close the OS socket under an in-flight request from another thread."""
    sock = getattr(connection, "sock", None)
    if sock is None:
        return
    # Linux/macOS wake a blocked recv on shutdown(); Windows only does so
    # when the OS handle is closed. Do both, in that order.
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        handle = sock.detach()
    except OSError:
        return
    try:
        socket.close(handle)
    except OSError:
        pass


def _terminate_process(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        # Always target the complete LibreOffice process tree. soffice.com is
        # a wrapper around soffice.bin; terminating only the wrapper (or
        # returning after it exits) can leave conversion running invisibly.
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass
        return
    if proc.poll() is not None:
        return
    try:
        os.kill(-proc.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.kill(-proc.pid, getattr(signal, "SIGKILL", 9))
    except (OSError, ProcessLookupError):
        pass


def client_base_url(host: str, port: int) -> str:
    clean = str(host or "127.0.0.1").strip()
    if clean in {"0.0.0.0", "::", "[::]", ""}:
        clean = "127.0.0.1"
    if ":" in clean and not clean.startswith("["):
        clean = f"[{clean}]"
    return f"http://{clean}:{int(port)}"


def tcp_port_in_use(host: str, port: int, timeout_seconds: float = 0.25) -> bool:
    """Return whether a TCP listener already owns the requested local port."""
    base = urllib.parse.urlsplit(client_base_url(host, port))
    target_host = base.hostname or "127.0.0.1"
    try:
        with socket.create_connection((target_host, int(port)), timeout_seconds):
            return True
    except OSError:
        return False


def server_model_ids(
    base_url: str, timeout_seconds: float = 2.0
) -> Optional[List[str]]:
    """Read served model ids; ``None`` means the probe was unavailable."""
    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + "/v1/models", timeout=timeout_seconds
        ) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None
    data = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(data, list):
        return None
    return [
        str(item.get("id"))
        for item in data
        if isinstance(item, dict) and item.get("id") is not None
    ]


def wait_for_server(
    base_url: str,
    timeout_seconds: float = 300,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[ProgressCallback] = None,
    process_is_running: Optional[Callable[[], bool]] = None,
    expected_model_alias: str = "",
) -> None:
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    last_message = 0.0
    while time.monotonic() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            raise OcrCancelled("OCR job cancelled while waiting for llama-server")
        if process_is_running is not None and not process_is_running():
            raise OcrWorkflowError("llama-server exited before the OCR model was ready")
        try:
            with urllib.request.urlopen(
                base_url.rstrip("/") + "/health", timeout=1.0
            ) as response:
                if response.status == 200:
                    if expected_model_alias:
                        ids = server_model_ids(base_url)
                        if ids is None:
                            time.sleep(0.1)
                            continue
                        if expected_model_alias not in ids:
                            shown = ", ".join(ids) if ids else "none"
                            raise OcrWorkflowError(
                                "The responding endpoint is not the OCR server that "
                                f"AutoTuner started (expected model alias "
                                f"{expected_model_alias!r}, received: {shown})."
                            )
                    return
        except OcrWorkflowError:
            raise
        except (OSError, urllib.error.URLError):
            pass
        now = time.monotonic()
        if progress is not None and now - last_message >= 2.0:
            progress("server", 0, 0, "Waiting for llama-server to load the OCR model")
            last_message = now
        time.sleep(0.25)
    raise OcrWorkflowError(
        f"llama-server did not become ready within {int(timeout_seconds)} seconds"
    )


class OcrJobRunner:
    """Run one cancellable OCR job against an already-started llama-server."""

    def __init__(
        self,
        base_url: str,
        model_alias: str,
        options: OcrJobOptions,
        *,
        model_name: str = "",
        llama_build: Optional[int] = None,
        server_command: Optional[Sequence[str]] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_alias = model_alias
        self.model_name = model_name or model_alias
        self.llama_build = llama_build
        self.server_command = _redact_command(server_command or [])
        self.options = options.normalized()
        self.progress = progress
        self.cancel_event = threading.Event()
        self._state_lock = threading.Lock()
        self._active_response: Optional[Any] = None
        self._active_connection: Optional[http.client.HTTPConnection] = None
        self._active_process: Optional[subprocess.Popen] = None
        self._source_hashes: dict[str, str] = {}
        self._source_stats: dict[str, Tuple[int, int]] = {}
        self._prepared_job_dir: Optional[Path] = None
        self._prepared_sources: Optional[List[_DocumentSource]] = None

    def configure_server(
        self,
        base_url: str,
        model_alias: str,
        *,
        model_name: Optional[str] = None,
        llama_build: Optional[int] = None,
        server_command: Optional[Sequence[str]] = None,
    ) -> None:
        """Attach runtime details after document preparation is complete."""
        self.base_url = str(base_url).rstrip("/")
        self.model_alias = str(model_alias)
        if model_name is not None:
            self.model_name = str(model_name)
        self.llama_build = llama_build
        if server_command is not None:
            self.server_command = _redact_command(server_command)

    def cancel(self) -> None:
        self.cancel_event.set()
        with self._state_lock:
            connection = self._active_connection
            process = self._active_process
        if connection is not None:
            # ``response.close()`` / ``connection.close()`` would take the
            # buffer lock the worker holds while blocked in ``recv`` and
            # make the caller (usually the GUI thread) wait for the whole
            # page inference. Closing the OS socket handle wakes that
            # ``recv`` immediately on Windows and POSIX; the worker's
            # ``finally`` then closes the Python objects and maps the
            # resulting OSError to ``OcrCancelled``.
            _abort_connection(connection)
            # The Python-level close still runs (for connections without a
            # live socket, and to release the SocketIO), but never on the
            # caller's thread.
            threading.Thread(
                target=_close_quietly, args=(connection,), daemon=True
            ).start()
        if process is not None:
            # Cancellation can be initiated by the GUI thread. Never make that
            # thread wait for LibreOffice's shutdown/taskkill grace periods.
            threading.Thread(
                target=_terminate_process, args=(process,), daemon=True
            ).start()

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise OcrCancelled("OCR job cancelled")

    def _emit(self, stage: str, current: int, total: int, message: str) -> None:
        if self.progress is not None:
            self.progress(stage, current, total, message)

    def _new_job_dir(self) -> Path:
        root = self.options.output_dir
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = root / f"AutoTuner-OCR-{stamp}"
        candidate = base
        suffix = 2
        while candidate.exists():
            candidate = root / f"{base.name}-{suffix}"
            suffix += 1
        candidate.mkdir(parents=False)
        return candidate

    def _convert_office(self, source: Path, converted_dir: Path) -> Path:
        executable = find_libreoffice()
        if executable is None:
            raise OcrWorkflowError(
                "LibreOffice is required for Word/Office input but was not found. "
                "Install LibreOffice or convert the document to PDF first."
            )
        converted_dir.mkdir(parents=True, exist_ok=True)
        # Keep LibreOffice's private user profile in the system temp root.
        # Its internal directory tree is deep; placing it below the OCR job
        # can exceed legacy MAX_PATH and make soffice.com abort with 0xC0000409.
        profile_dir = Path(tempfile.mkdtemp(prefix="autotuner-ocr-lo-"))
        before = set(converted_dir.glob("*.pdf"))
        command = [
            executable,
            f"-env:UserInstallation={profile_dir.resolve().as_uri()}",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(converted_dir),
            str(source),
        ]
        self._emit("convert", 0, 0, f"Converting {source.name} to PDF")
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            **_process_group_kwargs(),
        )
        with self._state_lock:
            self._active_process = proc
        deadline = time.monotonic() + 300
        output = ""
        try:
            while True:
                self._check_cancelled()
                try:
                    output, _ = proc.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        _terminate_process(proc)
                        raise OcrWorkflowError(
                            f"LibreOffice conversion timed out: {source.name}"
                        )
        except (KeyboardInterrupt, OcrCancelled):
            # ``cancel()`` may have run between Popen and the registration
            # above; terminating here is idempotent and keeps the headless
            # soffice from outliving the cancelled job.
            _terminate_process(proc)
            raise
        finally:
            with self._state_lock:
                self._active_process = None
            shutil.rmtree(profile_dir, ignore_errors=True)
        if proc.returncode != 0:
            raise OcrWorkflowError(
                f"LibreOffice could not convert {source.name} (exit {proc.returncode}): "
                f"{output.strip()[-1000:]}"
            )
        expected = converted_dir / f"{source.stem}.pdf"
        if expected.is_file():
            return expected
        created = [p for p in converted_dir.glob("*.pdf") if p not in before]
        if len(created) == 1:
            return created[0]
        raise OcrWorkflowError(
            f"LibreOffice reported success but produced no unambiguous PDF for {source.name}"
        )

    def _image_frame_count(self, path: Path) -> int:
        try:
            from PIL import Image
        except ImportError as exc:
            raise OcrWorkflowError(
                "Pillow is required for image input. Install requirements.txt again."
            ) from exc
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                opened = Image.open(path)
            with opened as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise OcrWorkflowError(
                        f"Image {path.name} exceeds the {MAX_IMAGE_PIXELS:,}-pixel limit"
                    )
                return max(1, int(getattr(image, "n_frames", 1) or 1))
        except Exception as exc:
            raise OcrWorkflowError(f"Cannot inspect image {path.name}: {exc}") from exc

    def _pdf_page_count(self, path: Path) -> int:
        try:
            import pymupdf
        except ImportError as exc:
            raise OcrWorkflowError(
                "PyMuPDF is required for PDF input. Install requirements.txt again."
            ) from exc
        try:
            with pymupdf.open(path) as document:
                return int(document.page_count)
        except Exception as exc:
            raise OcrWorkflowError(f"Cannot open PDF {path.name}: {exc}") from exc

    def _prepare_sources(self, job_dir: Path) -> List[_DocumentSource]:
        files = discover_input_files(self.options.inputs, self.cancel_event)
        converted_dir = job_dir / "converted"
        prepared: List[_DocumentSource] = []
        for index, original in enumerate(files, 1):
            self._check_cancelled()
            self._emit("prepare", index, len(files), f"Preparing {original.name}")
            # Capture provenance before any converter reads the source, then
            # verify conversion did not race with an external edit.
            self._source_sha256(original)
            suffix = original.suffix.lower()
            media = original
            if suffix in OFFICE_EXTENSIONS:
                source_convert_dir = (
                    converted_dir / f"{index:04d}-{_safe_stem(original)}"
                )
                media = self._convert_office(original, source_convert_dir)
                self._assert_source_unchanged(original)
                suffix = ".pdf"
            if suffix in PDF_EXTENSIONS:
                count = self._pdf_page_count(media)
                pages = parse_page_range(self.options.page_range, count)
                prepared.append(_DocumentSource(original, media, "pdf", pages))
            elif suffix in IMAGE_EXTENSIONS:
                frame_count = self._image_frame_count(media)
                pages = (
                    parse_page_range(self.options.page_range, frame_count)
                    if frame_count > 1
                    else (0,)
                )
                prepared.append(_DocumentSource(original, media, "image", pages))
        if not prepared:
            raise OcrWorkflowError("No OCR inputs remained after preparation")
        return prepared

    def _render_pdf_page(self, pdf: Path, page_index: int, target: Path) -> None:
        try:
            import pymupdf
        except ImportError as exc:
            raise OcrWorkflowError("PyMuPDF is unavailable") from exc
        self._check_cancelled()
        scale = self.options.dpi / 72.0
        try:
            with pymupdf.open(pdf) as document:
                page = document.load_page(page_index)
                rect = page.rect
                pixels = int(rect.width * scale) * int(rect.height * scale)
                if pixels <= 0 or pixels > MAX_IMAGE_PIXELS:
                    raise OcrWorkflowError(
                        f"Rendered page would contain {pixels:,} pixels; "
                        f"limit is {MAX_IMAGE_PIXELS:,}. Lower the DPI."
                    )
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(scale, scale),
                    alpha=False,
                    colorspace=pymupdf.csRGB,
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                pixmap.save(target)
        except OcrWorkflowError:
            raise
        except Exception as exc:
            raise OcrWorkflowError(
                f"Could not render {pdf.name} page {page_index + 1}: {exc}"
            ) from exc

    def _normalize_image(
        self, source: Path, target: Path, frame_index: int = 0
    ) -> None:
        try:
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise OcrWorkflowError(
                "Pillow is required for image input. Install requirements.txt again."
            ) from exc
        self._check_cancelled()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                opened = Image.open(source)
            with opened as image:
                if frame_index < 0 or frame_index >= int(
                    getattr(image, "n_frames", 1) or 1
                ):
                    raise OcrWorkflowError(
                        f"Image {source.name} has no frame {frame_index + 1}"
                    )
                image.seek(frame_index)
                # Pillow opens lazily. Reject hostile dimensions before EXIF
                # transforms or image.load() allocate the decoded pixel buffer.
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise OcrWorkflowError(
                        f"Image {source.name} exceeds the {MAX_IMAGE_PIXELS:,}-pixel limit"
                    )
                image = ImageOps.exif_transpose(image)
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise OcrWorkflowError(
                        f"Image {source.name} exceeds the {MAX_IMAGE_PIXELS:,}-pixel limit"
                    )
                image.load()
                if image.mode == "P" and "transparency" in image.info:
                    image = image.convert("RGBA")
                if image.mode not in {"RGB", "L"}:
                    background = Image.new("RGB", image.size, "white")
                    if "A" in image.getbands():
                        background.paste(image, mask=image.getchannel("A"))
                        image = background
                    else:
                        image = image.convert("RGB")
                target.parent.mkdir(parents=True, exist_ok=True)
                image.save(target, format="PNG", optimize=True)
        except OcrWorkflowError:
            raise
        except Exception as exc:
            raise OcrWorkflowError(f"Cannot decode image {source.name}: {exc}") from exc

    def _request_page(self, image: Path) -> str:
        size = image.stat().st_size
        if size > MAX_REQUEST_IMAGE_BYTES:
            raise OcrWorkflowError(
                f"Rendered image is {size / (1024 * 1024):.1f} MiB; "
                f"request limit is {MAX_REQUEST_IMAGE_BYTES / (1024 * 1024):.0f} MiB"
            )
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        payload = {
            "model": self.model_alias,
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": self.options.max_tokens,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    # DeepSeek/Unlimited-OCR's canonical prompt is
                    # ``<image>document parsing.``. Preserve that ordering:
                    # llama-server replaces this media item with the model's
                    # image marker before appending the task text.
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                        {"type": "text", "text": self.options.prompt},
                    ],
                }
            ],
        }
        raw_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise OcrWorkflowError(f"Invalid llama-server URL: {self.base_url!r}")
        connection_type = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(
            parsed.hostname,
            parsed.port,
            timeout=self.options.request_timeout_seconds,
        )
        endpoint = (parsed.path.rstrip("/") + "/v1/chat/completions") or (
            "/v1/chat/completions"
        )
        response: Optional[http.client.HTTPResponse] = None
        try:
            self._check_cancelled()
            with self._state_lock:
                self._active_connection = connection
            self._check_cancelled()
            connection.request(
                "POST",
                endpoint,
                body=raw_payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            response = connection.getresponse()
            with self._state_lock:
                self._active_response = response
            raw = response.read()
            if response.status < 200 or response.status >= 300:
                detail = raw.decode("utf-8", errors="replace")[-2000:]
                raise OcrWorkflowError(
                    f"llama-server returned HTTP {response.status}: {detail}"
                )
        except OcrWorkflowError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            if self.cancel_event.is_set():
                raise OcrCancelled("OCR request cancelled") from exc
            raise OcrWorkflowError(f"llama-server OCR request failed: {exc}") from exc
        finally:
            with self._state_lock:
                self._active_response = None
                self._active_connection = None
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
            connection.close()
        self._check_cancelled()
        try:
            body = json.loads(raw.decode("utf-8", errors="replace"))
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise OcrWorkflowError(
                "llama-server returned an unexpected chat-completions response"
            ) from exc
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        text = str(content or "").strip()
        if self.options.strip_grounding:
            text = strip_grounding_markup(text)
        if not text:
            raise OcrWorkflowError("OCR model returned empty output")
        return text

    def _source_sha256(self, path: Path) -> str:
        key = os.path.normcase(str(path))
        if key not in self._source_hashes:
            before = path.stat()
            digest = _sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise OcrWorkflowError(f"Input changed while hashing: {path}")
            self._source_hashes[key] = digest
            self._source_stats[key] = (after.st_size, after.st_mtime_ns)
        return self._source_hashes[key]

    def _assert_source_unchanged(self, path: Path) -> None:
        key = os.path.normcase(str(path))
        expected_hash = self._source_sha256(path)
        expected_stat = self._source_stats.get(key)
        stat = path.stat()
        current_stat = (stat.st_size, stat.st_mtime_ns)
        if expected_stat == current_stat:
            return
        current_hash = _sha256(path)
        if current_hash != expected_hash:
            raise OcrWorkflowError(
                f"Input changed after OCR preparation: {path}. "
                "Restart the job so the manifest and processed bytes agree."
            )
        self._source_stats[key] = current_stat

    def _write_manifest(
        self,
        path: Path,
        *,
        status: str,
        sources: Sequence[_DocumentSource],
        page_results: Sequence[OcrPageResult],
        error: str = "",
    ) -> None:
        payload = {
            "manifest_version": MANIFEST_VERSION,
            "status": status,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": self.model_name,
            "model_alias": self.model_alias,
            "llama_build": self.llama_build,
            "base_url": self.base_url,
            "server_command": self.server_command,
            "options": {
                **asdict(self.options),
                "inputs": [str(p) for p in self.options.inputs],
                "output_dir": str(self.options.output_dir),
            },
            "sources": [
                {
                    "path": str(source.original),
                    "sha256": self._source_sha256(source.original),
                    "kind": source.kind,
                    "selected_pages": [page + 1 for page in source.pages],
                }
                for source in sources
            ],
            "pages": [asdict(result) for result in page_results],
            "error": error,
        }
        _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def prepare(self) -> Tuple[Path, int]:
        """Convert Office inputs and enumerate pages before model loading.

        Running LibreOffice before llama-server claims RAM avoids conversion
        failures on memory-constrained desktops. The prepared PDFs remain in
        the private job directory until :meth:`run` finishes.
        """
        if self._prepared_job_dir is not None and self._prepared_sources is not None:
            return self._prepared_job_dir, sum(
                len(source.pages) for source in self._prepared_sources
            )
        job_dir = self._new_job_dir()
        self._prepared_job_dir = job_dir
        manifest = job_dir / "manifest.json"
        try:
            sources = self._prepare_sources(job_dir)
            total = sum(len(source.pages) for source in sources)
            if total <= 0:
                raise OcrWorkflowError("Selected page range contains no pages")
            self._write_manifest(
                manifest, status="prepared", sources=sources, page_results=[]
            )
            self._prepared_job_dir = job_dir
            self._prepared_sources = sources
            return job_dir, total
        except (Exception, KeyboardInterrupt) as exc:
            cancelled = isinstance(exc, (OcrCancelled, KeyboardInterrupt))
            fallback = {
                "manifest_version": MANIFEST_VERSION,
                "status": "cancelled" if cancelled else "failed",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "model": self.model_name,
                "error": str(exc),
            }
            _atomic_write(
                manifest, json.dumps(fallback, ensure_ascii=False, indent=2) + "\n"
            )
            if not self.options.keep_rendered:
                shutil.rmtree(job_dir / "converted", ignore_errors=True)
            if isinstance(exc, KeyboardInterrupt):
                self.cancel_event.set()
                raise OcrCancelled("OCR preparation cancelled by user") from exc
            raise

    def finalize_prepared(self, status: str, error: str) -> None:
        """Finalize a prepared job when server startup never reaches inference."""
        job_dir = self._prepared_job_dir
        sources = self._prepared_sources
        if job_dir is None or sources is None:
            return
        normalized_status = status if status in {"failed", "cancelled"} else "failed"
        self._write_manifest(
            job_dir / "manifest.json",
            status=normalized_status,
            sources=sources,
            page_results=[],
            error=str(error),
        )
        if not self.options.keep_rendered:
            shutil.rmtree(job_dir / "rendered", ignore_errors=True)
            shutil.rmtree(job_dir / "converted", ignore_errors=True)

    def run(self) -> OcrJobResult:
        options = self.options
        if self._prepared_job_dir is None or self._prepared_sources is None:
            try:
                self.prepare()
            except OcrCancelled:
                assert self._prepared_job_dir is not None
                job_dir = self._prepared_job_dir
                extension = ".md" if options.output_format == "markdown" else ".txt"
                return OcrJobResult(
                    job_dir=job_dir,
                    combined_output=job_dir / ("OCR-result" + extension),
                    manifest_path=job_dir / "manifest.json",
                    total_pages=0,
                    completed_pages=0,
                    failed_pages=0,
                    cancelled=True,
                    page_results=[],
                )
        assert self._prepared_job_dir is not None
        assert self._prepared_sources is not None
        job_dir = self._prepared_job_dir
        rendered_dir = job_dir / "rendered"
        pages_dir = job_dir / "pages"
        manifest = job_dir / "manifest.json"
        extension = ".md" if options.output_format == "markdown" else ".txt"
        combined_output = job_dir / ("OCR-result" + extension)
        sources: List[_DocumentSource] = list(self._prepared_sources)
        page_results: List[OcrPageResult] = []
        grouped: dict[str, List[Tuple[int, str]]] = {}
        status = "failed"
        fatal_error = ""
        try:
            total = sum(len(source.pages) for source in sources)
            self._write_manifest(
                manifest, status="running", sources=sources, page_results=[]
            )
            current = 0
            used_names: dict[str, int] = {}
            for source in sources:
                key = str(source.original)
                grouped.setdefault(key, [])
                stem = _safe_stem(source.original)
                used_names[stem] = used_names.get(stem, 0) + 1
                if used_names[stem] > 1:
                    stem = f"{stem}-{used_names[stem]}"
                for page_index in source.pages:
                    self._check_cancelled()
                    current += 1
                    page_number = page_index + 1
                    image_path = rendered_dir / f"{stem}-page-{page_number:04d}.png"
                    output_path = (
                        pages_dir / f"{stem}-page-{page_number:04d}{extension}"
                    )
                    started = time.monotonic()
                    try:
                        self._assert_source_unchanged(source.original)
                        self._emit(
                            "render",
                            current,
                            total,
                            f"Rendering {source.original.name}, page {page_number}",
                        )
                        if source.kind == "pdf":
                            self._render_pdf_page(source.media, page_index, image_path)
                        else:
                            self._normalize_image(source.media, image_path, page_index)
                        self._emit(
                            "ocr",
                            current,
                            total,
                            f"OCR {source.original.name}, page {page_number}",
                        )
                        text = self._request_page(image_path)
                        elapsed = time.monotonic() - started
                        _atomic_write(output_path, text.rstrip() + "\n")
                        grouped[key].append((page_number, text))
                        result = OcrPageResult(
                            source=key,
                            page=page_number,
                            output_file=str(output_path.relative_to(job_dir)),
                            characters=len(text),
                            elapsed_seconds=round(elapsed, 3),
                        )
                    except OcrCancelled:
                        raise
                    except Exception as exc:
                        elapsed = time.monotonic() - started
                        result = OcrPageResult(
                            source=key,
                            page=page_number,
                            output_file="",
                            elapsed_seconds=round(elapsed, 3),
                            error=str(exc),
                        )
                        if not options.continue_on_error:
                            page_results.append(result)
                            raise
                    page_results.append(result)
                    self._write_manifest(
                        manifest,
                        status="running",
                        sources=sources,
                        page_results=page_results,
                    )

            # A source can be edited while long HTTP inference is running.
            # Reject that race before declaring the immutable job complete.
            for source in sources:
                self._assert_source_unchanged(source.original)

            combined_parts: List[str] = []
            for source in sources:
                key = str(source.original)
                entries = grouped.get(key, [])
                if not entries:
                    continue
                if options.output_format == "markdown":
                    combined_parts.append(f"# {source.original.name}\n")
                    for page_number, text in entries:
                        combined_parts.append(
                            f"\n## Page {page_number}\n\n{text.strip()}\n"
                        )
                else:
                    combined_parts.append(f"===== {source.original.name} =====\n")
                    for page_number, text in entries:
                        combined_parts.append(
                            f"\n----- Page {page_number} -----\n{text.strip()}\n"
                        )
            _atomic_write(combined_output, "\n".join(combined_parts).rstrip() + "\n")
            status = (
                "completed_with_errors"
                if any(r.error for r in page_results)
                else "completed"
            )
        except OcrCancelled as exc:
            status = "cancelled"
            fatal_error = str(exc)
        except KeyboardInterrupt:
            self.cancel_event.set()
            status = "cancelled"
            fatal_error = "OCR job cancelled by user"
        except Exception as exc:
            status = "failed"
            fatal_error = str(exc)
        finally:
            if sources:
                self._write_manifest(
                    manifest,
                    status=status,
                    sources=sources,
                    page_results=page_results,
                    error=fatal_error,
                )
            else:
                fallback = {
                    "manifest_version": MANIFEST_VERSION,
                    "status": status,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "model": self.model_name,
                    "error": fatal_error,
                }
                _atomic_write(
                    manifest,
                    json.dumps(fallback, ensure_ascii=False, indent=2) + "\n",
                )
            if not options.keep_rendered:
                shutil.rmtree(rendered_dir, ignore_errors=True)
                shutil.rmtree(job_dir / "converted", ignore_errors=True)

        failed = sum(1 for result in page_results if result.error)
        completed = sum(1 for result in page_results if not result.error)
        total_pages = sum(len(source.pages) for source in sources)
        result = OcrJobResult(
            job_dir=job_dir,
            combined_output=combined_output,
            manifest_path=manifest,
            total_pages=total_pages,
            completed_pages=completed,
            failed_pages=failed,
            cancelled=status == "cancelled",
            page_results=page_results,
        )
        if status == "failed":
            raise OcrWorkflowError(fatal_error or "OCR job failed")
        return result
