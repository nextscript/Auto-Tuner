"""AutoTuner for llama.cpp — interactive launcher.

Workflow:
  1. Detect the system (CPU, RAM, GPUs across AMD / NVIDIA / Intel / Apple).
  2. Scan a folder of GGUF files, pair each main model with its mmproj.
  3. Show a numbered terminal menu so the user can pick a model.
  4. Match the model against per-family YAML profiles in settings/.
  5. Compute an optimal llama-server config that fits in free RAM/VRAM.
  6. Start llama-server with proper Ctrl+C handling.

Usage:
  python auto_tuner.py
  python auto_tuner.py --models-path D:/models --port 1234
  python auto_tuner.py --model Devstral --dry-run
  python auto_tuner.py --gui          # Qt log-viewer alongside the server

Environment variables:
  AUTOTUNER_MODELS    default models path
  LLAMA_SERVER        path to the llama-server binary
  LLAMA_CPP_DIR       llama.cpp checkout (build/bin/[Release/]llama-server is auto-found)
"""

from __future__ import annotations

import argparse
import builtins
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from autotuner_version import VERSION

from hardware import detect_system, format_system, SystemInfo
from launcher import launch
from scanner import scan_models, group_entries, read_gguf_metadata, ModelEntry
from settings_loader import load_profiles, match_profile, ModelProfile
from tuner import (
    build_command,
    build_diffusion_command,
    build_diffusion_server_command,
    check_draft_model_build,
    check_model_build,
    check_profile_build,
    compute_config,
    effective_load_mode,
    gemma_draft_needs_ik_fork,
    prepare_command_for_binary,
    probe_binary_build_number,
    TunedConfig,
)
from performance_target import (
    resolve_performance_target,
    list_target_names,
    describe_targets,
)
import app_settings
from ocr_workflow import (
    OcrCancelled,
    OcrJobOptions,
    OcrJobRunner,
    OcrWorkflowError,
    client_base_url,
    is_ocr_model,
    ocr_model_preset,
    ocr_projector_warning,
    tcp_port_in_use,
    wait_for_server,
)

# ---------------------------------------------------------------------------
# Terminal presentation helpers

_DEBUG_MODE = os.environ.get("AUTOTUNER_DEBUG", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_DEBUG_CATEGORIES: set[str] = set()
_DEBUG_SINK: Optional[Callable[[str], None]] = None
_BAR = "─" * 64


def _ascii_safe(value: object) -> str:
    """Normalize AutoTuner-owned output for legacy consoles and log files."""
    replacements = str.maketrans(
        {
            "—": "-",
            "–": "-",
            "…": "...",
            "→": "->",
            "←": "<-",
            "·": "/",
            "⚠": "WARNING:",
            "👁": "[V]",
            "⚡": "[D]",
            "🧠": "[T]",
            "🛠": "[U]",
            "═": "=",
            "─": "-",
            "╌": "-",
        }
    )
    return (
        str(value)
        .translate(replacements)
        .encode("ascii", "backslashreplace")
        .decode("ascii")
    )


def _tui_print(*values: object, **kwargs: object) -> None:
    """ASCII-normalize this module's output without changing child process streams."""
    if _RENDERER.plain:
        separator = str(kwargs.pop("sep", " "))
        ending = str(kwargs.pop("end", "\n"))
        text = separator.join(_ascii_safe(value) for value in values)
        width = max(20, _RENDERER.width)
        lines = []
        for raw_line in text.splitlines() or [""]:
            lines.extend(
                raw_line[i : i + width] for i in range(0, len(raw_line), width)
            )
        builtins.print("\n".join(lines), end=ending, **kwargs)
        return
    builtins.print(*values, **kwargs)


print = _tui_print  # type: ignore[assignment]


def _windows_vt_enabled() -> bool:
    """Allow Windows ANSI only when the active console accepts VT processing."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not handle or not ctypes.windll.kernel32.GetConsoleMode(
            handle, ctypes.byref(mode)
        ):
            return False
        enable_vt = 0x0004
        return bool(
            mode.value & enable_vt
            or ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | enable_vt)
        )
    except (AttributeError, OSError):
        return False


class _TerminalRenderer:
    """Small dependency-free renderer with safe ANSI and ASCII fallbacks."""

    def __init__(self, *, plain: bool = False, no_color: bool = False) -> None:
        encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
        utf_capable = encoding.startswith("utf") or encoding == "cp65001"
        self.plain = plain or not sys.stdout.isatty() or not utf_capable
        self.width = max(20, shutil.get_terminal_size(fallback=(80, 24)).columns)
        self.rule_width = min(self.width, 110)
        self.color = (
            not self.plain
            and not no_color
            and sys.stdout.isatty()
            and os.environ.get("TERM", "").lower() != "dumb"
            and "NO_COLOR" not in os.environ
            and _windows_vt_enabled()
        )
        self.line = ("-" if self.plain else "─") * self.rule_width

    def style(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def _lines(self, text: str, indent: int = 2) -> List[str]:
        usable = max(8, self.width - indent)
        return [text[i : i + usable] for i in range(0, len(text), usable)] or [""]

    def heading(self, text: str) -> None:
        print(self.line)
        for line in self._lines(text):
            print(self.style(f"  {line}", "1;36"))
        print(self.line)

    def section(self, text: str) -> None:
        print()
        for line in self._lines(text):
            print(self.style(f"  {line}", "1;35"))
        print(
            self.style(
                "  " + ("-" if self.plain else "╌") * min(self.rule_width - 2, 48), "2"
            )
        )

    def truncate(self, text: str, width: int) -> str:
        if len(text) <= width:
            return text
        suffix = "..." if self.plain else "…"
        return text[: max(1, width - len(suffix))] + suffix


_RENDERER = _TerminalRenderer(plain=True)


def _configure_renderer(args: argparse.Namespace) -> None:
    global _RENDERER, _BAR
    _RENDERER = _TerminalRenderer(
        plain=bool(getattr(args, "plain", False)),
        no_color=bool(getattr(args, "no_color", False)),
    )
    _BAR = _RENDERER.line


def _effective_mode(args: argparse.Namespace) -> str:
    """Resolve CLI mode first, then the shared persisted GUI preference."""
    return args.mode or app_settings.get_mode() or "chat"


def _effective_prompt_cache_mib(args: argparse.Namespace) -> int:
    """Resolve the CLI cache limit while retaining the shared saved default."""
    if args.no_prompt_cache:
        return 0
    return max(
        -1,
        int(
            args.cache_ram_mib
            if args.cache_ram_mib is not None
            else app_settings.get_prompt_cache_ram_mib()
        ),
    )


def set_debug_mode(enabled: bool, sink: Optional[Callable[[str], None]] = None) -> None:
    """Enable internal diagnostics without changing llama-server verbosity."""
    global _DEBUG_MODE, _DEBUG_SINK
    _DEBUG_MODE = bool(enabled)
    if sink is not None:
        _DEBUG_SINK = sink


def set_debug_sink(sink: Optional[Callable[[str], None]]) -> None:
    global _DEBUG_SINK
    _DEBUG_SINK = sink


def _emit_debug(prefix: str, *args, **kwargs) -> None:
    message = " ".join(str(arg) for arg in args)
    print(prefix, *args, **kwargs)
    if _DEBUG_SINK is not None:
        try:
            _DEBUG_SINK(f"{prefix} {message}".rstrip())
        except Exception:
            pass


def _debug_print(*args, **kwargs) -> None:
    if _DEBUG_MODE:
        _emit_debug("[DEBUG]", *args, **kwargs)


def enable_debug_category(category: str) -> None:
    _DEBUG_CATEGORIES.add(category)


def debug_cat(category: str, *args, **kwargs) -> None:
    if _DEBUG_MODE or category in _DEBUG_CATEGORIES:
        _emit_debug(f"[DEBUG:{category.upper()}]", *args, **kwargs)


def _print_banner() -> None:
    print()
    _RENDERER.heading(f"AutoTuner v{VERSION}  |  llama.cpp launcher")
    print("  Hardware-aware model selection, memory planning, and server launch.")


def _print_system(info: SystemInfo) -> None:
    print(format_system(info))


def _capability_markers(entry: ModelEntry, *, plain: Optional[bool] = None) -> str:
    """Return GUI-parity capability markers, including embedded MTP drafts."""
    use_plain = _RENDERER.plain if plain is None else plain
    symbols = (
        ("[V]", "[D]", "[T]", "[U]", "[O]")
        if use_plain
        else ("👁", "⚡", "🧠", "🛠", "📄")
    )
    markers = []
    if entry.has_vision:
        markers.append(symbols[0])
    if entry.has_speculative_draft:
        markers.append(symbols[1])
    if entry.supports_thinking:
        markers.append(symbols[2])
    if entry.supports_tool_use:
        markers.append(symbols[3])
    if is_ocr_model(entry):
        markers.append(symbols[4])
    return " ".join(markers) or "-"


def _print_menu(groups: dict) -> List[ModelEntry]:
    """Render a responsive grouped model table and return its display order."""
    flat: List[ModelEntry] = []
    _RENDERER.section("Available models")
    legend = (
        "[V] vision / [D] draft/MTP / [T] thinking / [U] tools / [O] OCR"
        if _RENDERER.plain
        else "👁 vision / ⚡ draft/MTP / 🧠 thinking / 🛠 tools / 📄 OCR"
    )
    for line in _RENDERER._lines(legend):
        print("  " + line)
    compact = _RENDERER.width < 60
    name_width = max(12, _RENDERER.width - 34)
    idx = 1
    for group_name in sorted(groups):
        entries = sorted(groups[group_name], key=lambda entry: entry.name.lower())
        if not entries:
            continue
        print(_RENDERER.style(f"\n  [{group_name}]", "1"))
        for entry in entries:
            context = (
                f"{entry.native_context // 1024}k" if entry.native_context else "-"
            )
            markers = _capability_markers(entry)
            if compact:
                print(
                    f"  {idx:>3}. {_RENDERER.truncate(entry.name, max(8, _RENDERER.width - 7))}"
                )
                print(f"       {entry.size_gb:.1f}G | ctx {context} | {markers}")
            else:
                name = _RENDERER.truncate(entry.name, name_width)
                print(
                    f"  {idx:>3}  {name:<{name_width}}  {entry.size_gb:>5.1f}G  "
                    f"{context:>5}  {markers}"
                )
            flat.append(entry)
            idx += 1
    print()
    return flat


def _print_config(
    model: ModelEntry, profile: ModelProfile, cfg: TunedConfig, system: SystemInfo
) -> None:
    print(_BAR)
    print(f"Model:    {model.name}")
    print(
        f"Profile:  {profile.display_name}"
        + (f"  ({profile.source_file})" if profile.source_file else "")
    )
    if profile.notes:
        print(f"Notes:    {profile.notes}")
    if model.mmproj is not None:
        print(f"Vision:   {model.mmproj.name}")
    print(_BAR)

    n_cpu_moe = getattr(cfg, "n_cpu_moe", None)
    if n_cpu_moe is not None and n_cpu_moe > 0:
        placement = (
            f"MoE hybrid: {n_cpu_moe} expert layers on CPU, "
            "remaining resident weights on GPU"
        )
    elif cfg.full_offload:
        placement = f"GPU full offload (ngl=all of {model.n_layers or '?'})"
    elif cfg.ngl > 0:
        placement = f"hybrid: {cfg.ngl} layers on GPU, rest on CPU"
    else:
        placement = "CPU only"
    print(f"  Placement       : {placement}")
    print(f"  Perf target     : {cfg.performance_target}")
    print(f"  Context         : {cfg.ctx:,} tokens")
    print(f"  KV cache quant  : K={cfg.cache_k}  V={cfg.cache_v}")
    print(f"  Threads         : {cfg.threads} (batch: {cfg.batch_threads})")
    print(f"  Batch / ubatch  : {cfg.batch} / {cfg.ubatch}")
    print(f"  Flash attention : {'on' if cfg.flash_attn else 'off'}")
    load_mode = effective_load_mode(cfg)
    if load_mode is not None:
        print(f"  Load mode       : {load_mode}")
    if cfg.numa:
        print(f"  NUMA            : {cfg.numa}")
    if cfg.no_context_shift:
        print("  no-context-shift: on (better performance for large context)")
    if cfg.tensor_split:
        print(f"  Tensor split    : {cfg.tensor_split}")
    if cfg.main_gpu is not None:
        print(f"  Main GPU        : {cfg.main_gpu}")

    s = cfg.sampling
    print(
        f"  Sampling        : temp={s.get('temperature')} "
        f"top_k={s.get('top_k')} top_p={s.get('top_p')} "
        f"min_p={s.get('min_p')} rep={s.get('repeat_penalty')}"
    )

    print()
    print("  Memory estimate (with current options):")
    mapped_model_ram_gb = float(getattr(cfg, "mapped_model_ram_gb", 0.0) or 0.0)
    mapped_model_resident_gb = float(
        getattr(cfg, "mapped_model_resident_gb", mapped_model_ram_gb) or 0.0
    )
    runtime_ram_overhead_gb = float(getattr(cfg, "runtime_ram_overhead_gb", 0.0) or 0.0)
    gpu_label = "accelerator" if cfg.unified_memory else "GPU"
    cpu_label = "host" if cfg.unified_memory else "CPU"
    print(f"    Model {gpu_label:<11}: ~ {cfg.estimated_model_vram_gb:5.1f} GB")
    print(f"    Model {cpu_label:<11}: ~ {cfg.estimated_model_ram_gb:5.1f} GB")
    if mapped_model_ram_gb:
        print(
            f"    Lazy mapped file : ~ {mapped_model_ram_gb:5.1f} GB "
            f"(active budget {mapped_model_resident_gb:.1f} GB)"
        )
    if cfg.vision_vram_gb:
        print(f"    Vision GPU       : ~ {cfg.vision_vram_gb:5.1f} GB")
    if cfg.vision_ram_gb:
        print(f"    Vision RAM       : ~ {cfg.vision_ram_gb:5.1f} GB")
    if cfg.runtime_vram_overhead_gb:
        print(f"    Runtime GPU      : ~ {cfg.runtime_vram_overhead_gb:5.1f} GB")
    if runtime_ram_overhead_gb:
        print(f"    Runtime RAM      : ~ {runtime_ram_overhead_gb:5.1f} GB")
    if cfg.batch_vram_overhead_gb:
        print(f"    Batch workspace  : ~ {cfg.batch_vram_overhead_gb:5.1f} GB")
    compute_vram_gb = float(getattr(cfg, "compute_vram_gb", 0.0) or 0.0)
    compute_ram_gb = float(getattr(cfg, "compute_ram_gb", 0.0) or 0.0)
    if compute_vram_gb:
        print(f"    Compute GPU      : ~ {compute_vram_gb:5.1f} GB")
    if compute_ram_gb:
        print(f"    Compute RAM      : ~ {compute_ram_gb:5.1f} GB")
    print(f"    KV cache         : ~ {cfg.estimated_kv_gb:5.1f} GB")
    recurrent_total = cfg.recurrent_state_vram_gb + cfg.recurrent_state_ram_gb
    if recurrent_total:
        print(f"    Recurrent state  : ~ {recurrent_total:5.1f} GB")
    if cfg.prompt_cache_ram_gb:
        print(f"    Prompt RAM       : ~ {cfg.prompt_cache_ram_gb:5.1f} GB")
    if cfg.unified_memory:
        total_unified = (
            cfg.estimated_model_vram_gb
            + cfg.estimated_model_ram_gb
            + mapped_model_resident_gb
            + cfg.vision_vram_gb
            + cfg.vision_ram_gb
            + cfg.draft_vram_gb
            + cfg.estimated_kv_gb
            + cfg.recurrent_state_vram_gb
            + cfg.recurrent_state_ram_gb
            + cfg.runtime_vram_overhead_gb
            + float(getattr(cfg, "compute_vram_gb", 0.0) or 0.0)
            + runtime_ram_overhead_gb
            + float(getattr(cfg, "compute_ram_gb", 0.0) or 0.0)
            + cfg.batch_vram_overhead_gb
            + cfg.prompt_cache_ram_gb
        )
        print(
            f"    Unified total    : ~ {total_unified:5.1f} GB   "
            f"({min(system.free_ram_gb, system.free_vram_gb):.1f} GB "
            "accelerator-addressable before launch)"
        )
    else:
        print(
            f"    Capacity         : {system.free_vram_gb:.1f} GB VRAM / "
            f"{system.free_ram_gb:.1f} GB RAM free before launch"
        )
    warning = getattr(cfg, "warning", None)
    if warning:
        print(f"    Warning          : {warning}")
    print(_BAR)


# ---------------------------------------------------------------------------
# Selection


def _confirm(prompt: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    try:
        raw = input(f"{prompt} {suffix} ").strip().lower()
    except EOFError:
        return default_yes
    if not raw:
        return default_yes
    return raw in ("y", "yes", "j", "ja")


def _ask_interactive_features(
    model: ModelEntry,
    draft_model: Optional[ModelEntry],
    settings_path: Path,
    force_ngram: bool = False,
    disable_ngram: bool = False,
    non_interactive: bool = False,
    draft_available: Optional[bool] = None,
    thinking_available: Optional[bool] = None,
    draft_default: bool = True,
) -> tuple[bool, bool, bool, bool, Optional[ModelEntry]]:
    """Interaktive Fragen-Kette nach Modellauswahl.

    Returns:
        (use_vision, use_draft, use_thinking, use_ngram, effective_draft)
    """
    # Embedded MTP is a draft capability even without an external model file.
    draft_available = (
        draft_model is not None or model.has_embedded_mtp
        if draft_available is None
        else draft_available
    )
    thinking_available = (
        model.supports_thinking if thinking_available is None else thinking_available
    )
    # Non-interactive runs use the same availability defaults as the GUI:
    # vision/draft/thinking on when available and n-gram on unless explicitly
    # disabled. --ngram remains a compatible explicit-on spelling.
    # ``draft_default`` False (profile ``auto_draft: false``) keeps a paired
    # external drafter off unless the user asks for it; embedded MTP is
    # unaffected.
    external_draft_off = not draft_default and draft_model is not None
    if non_interactive:
        return (
            model.mmproj is not None,
            draft_available and not external_draft_off,
            thinking_available,
            bool(force_ngram or not disable_ngram),
            draft_model,
        )

    # ── Vision ───────────────────────────────────────────────────────
    use_vision = False
    if model.mmproj is not None:
        use_vision = _confirm(
            f"Vision aktivieren? ({model.mmproj.name})",
            default_yes=True,
        )
        if not use_vision:
            model.mmproj = None

    # ── Draft Model ──────────────────────────────────────────────────
    use_draft = False
    effective_draft = draft_model
    if draft_available:
        detail = effective_draft.name if effective_draft is not None else "embedded MTP"
        use_draft = _confirm(
            f"Draft aktivieren? ({detail})", default_yes=not external_draft_off
        )
        if not use_draft:
            effective_draft = None

    # ── Thinking / Reasoning ────────────────────────────────────────
    # Read the chat template from GGUF metadata (the authoritative source);
    # fall back to filename heuristics only when no template is available.
    use_thinking = False
    if thinking_available:
        use_thinking = _confirm(
            "Thinking/Reasoning aktivieren? (<|think|> / <|reserved_special_token>)",
            default_yes=True,
        )

    # ── n-gram self-speculation ─────────────────────────────────────
    # Model-agnostic and default-on for every model. --no-ngram is the
    # explicit opt-out; --ngram remains a compatible explicit-on spelling.
    if disable_ngram:
        use_ngram = False
    elif force_ngram:
        use_ngram = True
    else:
        use_ngram = _confirm(
            "n-gram self-speculative decoding aktivieren?",
            default_yes=True,
        )

    return use_vision, use_draft, use_thinking, use_ngram, effective_draft


def _ocr_options_for_selection(
    model: ModelEntry,
    args: argparse.Namespace,
    *,
    use_vision: bool,
    non_interactive: bool,
) -> Optional[OcrJobOptions]:
    """Build the same OCR job options exposed by the Qt dialog."""
    requested = bool(getattr(args, "ocr_input", None))
    if not requested:
        if non_interactive or not is_ocr_model(model):
            return None
        if not _confirm("OCR document workflow for this model?", default_yes=True):
            return None

    if not use_vision or model.mmproj is None:
        raise OcrWorkflowError(
            "OCR requires the model's matching mmproj; do not use --novision."
        )

    preset = ocr_model_preset(model)
    projector_warning = ocr_projector_warning(model)
    if projector_warning:
        print(f"[AutoTuner] OCR projector warning: {projector_warning}")
    raw_inputs = list(getattr(args, "ocr_input", None) or [])
    if not raw_inputs:
        try:
            raw = input(
                "OCR input file/folder (separate several paths with ';'): "
            ).strip()
        except EOFError as exc:
            raise OcrWorkflowError("No OCR input was provided") from exc
        raw_inputs = [part.strip() for part in raw.split(";") if part.strip()]
    if not raw_inputs:
        raise OcrWorkflowError("No OCR input was provided")

    output_raw = getattr(args, "ocr_output", None)
    if not output_raw and not non_interactive:
        default_output = str(Path.cwd() / "ocr-output")
        try:
            entered = input(f"OCR output folder [{default_output}]: ").strip()
        except EOFError:
            entered = ""
        output_raw = entered or default_output
    output_dir = Path(output_raw or (Path.cwd() / "ocr-output"))

    prompt = getattr(args, "ocr_prompt", None)
    if prompt is None and not non_interactive:
        try:
            entered = input(f"OCR prompt [{preset.prompt}]: ").strip()
        except EOFError:
            entered = ""
        prompt = entered or preset.prompt
    prompt = preset.prompt if prompt is None else str(prompt)

    return OcrJobOptions(
        inputs=[Path(value) for value in raw_inputs],
        output_dir=output_dir,
        prompt=prompt,
        max_tokens=int(getattr(args, "ocr_max_tokens", preset.max_tokens)),
        dpi=int(getattr(args, "ocr_dpi", 220)),
        page_range=str(getattr(args, "ocr_pages", "") or ""),
        output_format=str(getattr(args, "ocr_format", "markdown")),
        keep_rendered=bool(getattr(args, "ocr_keep_rendered", False)),
        strip_grounding=(
            False
            if bool(getattr(args, "ocr_keep_grounding", False))
            else preset.strip_grounding
        ),
        request_timeout_seconds=int(getattr(args, "ocr_timeout", 1200)),
        stop_server_when_done=True,
    ).normalized()


def _run_ocr_terminal_job(
    cmd: List[str],
    cfg: TunedConfig,
    model: ModelEntry,
    profile: ModelProfile,
    args: argparse.Namespace,
    options: OcrJobOptions,
) -> int:
    """Start a job-owned server, process documents, then release VRAM."""
    from server_process import ServerProcess

    build = probe_binary_build_number(cmd[0])
    preset = ocr_model_preset(model)
    required_build = max(
        int(getattr(profile, "min_llama_build", 0) or 0), preset.min_llama_build
    )
    if required_build and build is not None and build < required_build:
        print(
            f"[AutoTuner] OCR requires llama.cpp b{required_build}+ for "
            f"{preset.label}; selected binary reports b{build}."
        )
        return 2
    if required_build and build is None:
        print(
            f"[AutoTuner] Warning: could not verify the selected binary's build. "
            f"{preset.label} is validated on b{required_build}+."
        )

    server = ServerProcess(cmd, env_overrides=cfg.env_overrides)

    def progress(stage: str, current: int, total: int, message: str) -> None:
        counter = f" [{current}/{total}]" if total else ""
        print(f"[OCR:{stage}]{counter} {message}")

    base_url = client_base_url(args.host, args.port)
    runner = OcrJobRunner(
        base_url,
        model.name,
        options,
        model_name=model.name,
        llama_build=build,
        server_command=cmd,
        progress=progress,
    )
    inference_started = False
    try:
        job_dir, total_pages = runner.prepare()
        print(f"[OCR:prepare] Prepared {total_pages} page(s); job directory: {job_dir}")
        if tcp_port_in_use(args.host, args.port):
            raise OcrWorkflowError(
                f"Port {args.port} is already in use; refusing to send OCR "
                "documents to an unverified endpoint."
            )
        server.start()
        wait_for_server(
            base_url,
            timeout_seconds=300,
            progress=progress,
            process_is_running=lambda: (
                server.proc is not None and server.proc.poll() is None
            ),
            expected_model_alias=model.name,
        )
        inference_started = True
        result = runner.run()
        print(_BAR)
        print(f"  OCR output   : {result.job_dir}")
        print(f"  Combined file: {result.combined_output}")
        print(
            f"  Pages        : {result.completed_pages}/{result.total_pages} complete, "
            f"{result.failed_pages} failed"
        )
        print(f"  Manifest     : {result.manifest_path}")
        print(_BAR)
        if result.cancelled:
            return 130
        return 1 if result.failed_pages else 0
    except OcrCancelled as exc:
        if not inference_started:
            runner.finalize_prepared("cancelled", str(exc))
        print(f"[AutoTuner] OCR cancelled: {exc}")
        return 130
    except (OcrWorkflowError, OSError, ValueError) as exc:
        if not inference_started:
            runner.finalize_prepared("failed", str(exc))
        print(f"[AutoTuner] OCR failed: {exc}")
        logs = server.get_logs()
        if logs:
            print("[AutoTuner] Last llama-server output:")
            for line in logs[-20:]:
                print("  " + line.rstrip())
        return 1
    except KeyboardInterrupt:
        runner.cancel()
        if not inference_started:
            runner.finalize_prepared("cancelled", "OCR cancelled by user")
        print("\n[AutoTuner] OCR cancelled by user.")
        return 130
    finally:
        server.stop()


def _pick_model(
    flat: List[ModelEntry], cli_query: Optional[str]
) -> tuple[Optional[ModelEntry], List[str]]:
    if cli_query:
        parts = cli_query.split()
        query_parts = []
        flags = []
        for p in parts:
            if p.startswith("--") or p.lower() in (
                "novision",
                "nodraft",
                "nothinking",
                "ngram",
                "no-ngram",
                "nongram",
            ):
                flags.append(p.lower().lstrip("-"))
            else:
                query_parts.append(p)

        q = " ".join(query_parts).lower()
        matches = [e for e in flat if q in e.name.lower()]
        if not matches:
            print(f"[AutoTuner] No model matched --model '{cli_query}'.")
            return None, []
        if len(matches) > 1:
            print(f"[AutoTuner] '{cli_query}' is ambiguous — matches:")
            for e in matches:
                print(f"    - {e.name}")
            return None, []
        return matches[0], flags

    while True:
        try:
            raw = input(f"Select a model [1-{len(flat)}, q to quit]: ").strip()
        except EOFError:
            return None, []
        if raw.lower() in ("q", "quit", "exit"):
            return None, []

        parts = raw.split()
        model_idx_str = None
        flags = []
        for p in parts:
            if p.startswith("--") or p.lower() in (
                "novision",
                "nodraft",
                "nothinking",
                "ngram",
                "no-ngram",
                "nongram",
            ):
                flags.append(p.lower().lstrip("-"))
            elif model_idx_str is None and p.isdigit():
                model_idx_str = p
            else:
                flags.append(p.lower().lstrip("-"))

        if model_idx_str is None:
            print(
                "  please enter a number (optionally followed by flags like '--novision')."
            )
            continue
        n = int(model_idx_str)
        if not 1 <= n <= len(flat):
            print(f"  number must be between 1 and {len(flat)}.")
            continue
        return flat[n - 1], flags


# ---------------------------------------------------------------------------
# llama-server discovery


def _native_binary_suffixes() -> Tuple[str, ...]:
    """Executable suffixes to auto-discover on this OS, preferred first."""
    if os.name == "nt":
        return (".exe",)
    # Never auto-select Windows .exe builds on Linux/macOS. Shared model/build
    # folders can contain both Windows and native artifacts; launching the .exe
    # from Ubuntu fails with PermissionError/Exec format error and looks like a
    # model-load crash.
    return ("",)


def _binary_subpaths(binary_name: str) -> List[str]:
    stem = binary_name[:-4] if binary_name.lower().endswith(".exe") else binary_name
    bases = [
        "build/bin/Release/",
        "build/bin/Debug/",
        "build/bin/",
        "build/",
        "",
    ]
    out: List[str] = []
    for base in bases:
        for suffix in _native_binary_suffixes():
            candidate = f"{base}{stem}{suffix}"
            if candidate not in out:
                out.append(candidate)
    return out


_SERVER_SUBPATHS = _binary_subpaths("llama-server")


def _is_runnable_binary(path: Path) -> bool:
    """True for binaries this OS can launch directly."""
    try:
        if not path.is_file():
            return False
    except OSError:
        return False
    if os.name == "nt":
        # Shared dual-boot build folders can hold a Linux ELF "llama-server"
        # right next to "llama-server.exe" — CreateProcess cannot run the
        # extension-less ELF, so only accept Windows-executable suffixes.
        return path.suffix.lower() in (".exe", ".bat", ".cmd", ".com")
    if path.suffix.lower() == ".exe":
        return False
    return os.access(path, os.X_OK)


# Diffusion models run through the single-shot llama-diffusion-cli example
# binary, not llama-server. Same build tree, different target name.
# DiffusionGemma (PR #24427) ships its OWN fork binaries
# llama-diffusion-gemma-cli / -gemma-server; mainline Dream/LLaDA/RND1 use
# the generic llama-diffusion-cli. _DIFFUSION_BINARIES lists the candidate
# binary names in PREFERENCE order — the resolver tries each until one is
# found in the fork tree.
_DIFFUSION_BINARIES = [
    "llama-diffusion-gemma-cli",  # DiffusionGemma fork (PR #24427)
    "llama-diffusion-cli",  # mainline Dream/LLaDA/RND1
]


def _diffusion_binary_for_arch(arch: Optional[str]) -> str:
    """Return the preferred diffusion binary name for a model architecture.

    DiffusionGemma (arch 'diffusion-gemma', fork-only PR #24427) ships its
    own llama-diffusion-gemma-cli, which is NOT interchangeable with the
    generic llama-diffusion-cli (different model loader + diffusion flags).
    All other diffusion archs (dream / llada / rnd1) use the generic CLI.
    """
    if arch and "gemma" in arch.lower():
        return "llama-diffusion-gemma-cli"
    return "llama-diffusion-cli"


def _diffusion_subpaths_for(binary_name: str) -> List[str]:
    """Build native candidate subpaths for a given diffusion binary name."""
    return _binary_subpaths(binary_name)


def _candidate_search_roots() -> List[Path]:
    """Folders to look in for a llama.cpp / 1b_llama.cpp checkout."""
    roots: List[Path] = []
    seen: set = set()

    def add(p) -> None:
        try:
            rp = Path(p).expanduser().resolve()
            _debug_print(f"Checking path: {rp}")
        except (OSError, RuntimeError):
            return
        if rp in seen or not rp.exists():
            return
        seen.add(rp)
        roots.append(rp)

    env_dir = os.environ.get("LLAMA_CPP_DIR")
    _debug_print(f"LLAMA_CPP_DIR: {env_dir}")
    if env_dir:
        add(env_dir)
        parent = Path(env_dir).expanduser()
        # Also add siblings: other forks next to the selected one
        try:
            preferred_backend = _fork_backend(parent.name)
            siblings = [
                sibling
                for sibling in parent.parent.iterdir()
                if sibling.is_dir() and re.search(r"llama", sibling.name, re.IGNORECASE)
            ]
            siblings.sort(
                key=lambda sibling: _fork_name_sort_key(
                    sibling.name, preferred_backend=preferred_backend
                )
            )
            for sibling in siblings:
                add(sibling)
        except (OSError, PermissionError):
            pass
        # If env_dir is a CONTAINER (a parent holding several *_llama.cpp
        # builds, e.g. H:/LAB/ai-local), enumerate its direct children so a
        # profile hint like '2b_llama/llama-server' resolves to a versioned
        # child dir like '2b_b8840_llama.cpp'. Without this the resolver only
        # sees the container, not the forks inside it.
        try:
            children = [
                child
                for child in parent.iterdir()
                if child.is_dir() and re.search(r"llama", child.name, re.IGNORECASE)
            ]
            children.sort(key=lambda child: _fork_name_sort_key(child.name))
            for child in children:
                add(child)
        except (OSError, PermissionError):
            pass
        add(parent.parent / "BitNet")

    bases = [Path(__file__).resolve().parent, Path.cwd()]
    # Fork hint list — current build scripts produce backend-qualified names
    # such as '{prefix}_b{NUM}_{vulkan|hip}_llama.cpp'; legacy installations
    # may still use '{prefix}_llama.cpp'. Versioned forms are not listed here
    # because they vary
    # per build; the fork discovery (_discover_llama_forks) finds those by
    # directory walk + binary probe. These bare names are only hint anchors
    # for the candidate-root search when a user still uses the unversioned
    # naming. Prefixes: 1b_ = 1-bit Bonsai, 2b_ = 2-bit/Ternary Bonsai,
    # tq_ = TurboQuant, ik_ = Gemma external drafter, d_ = Diffusion.
    common_subs = (
        "llama.cpp",
        "1b_llama.cpp",
        "2b_llama.cpp",
        "ik_llama.cpp",
        "tq_llama.cpp",
        "atq_llama.cpp",
        "d_llama.cpp",
        "BitNet",
        "ai-local/llama.cpp",
        "ai-local/1b_llama.cpp",
        "ai-local/2b_llama.cpp",
        "ai-local/ik_llama.cpp",
        "ai-local/tq_llama.cpp",
        "ai-local/d_llama.cpp",
    )
    for base in bases:
        chain = [base, *list(base.parents)[:5]]
        for p in chain:
            for sub in common_subs:
                add(p / sub)
    return roots


# Fork dir naming convention: backend-qualified current builds use
# '{prefix}_b{NUM}_{vulkan|hip}_llama.cpp' or
# '{version}_{vulkan|hip}_llama.cpp'. Legacy backend-neutral forms remain
# discoverable. To match a backend-neutral profile hint (for example
# '2b_llama') against either optimized sibling, strip only the complete
# version and terminal backend token:
#     '2b_b8840_vulkan_llama.cpp'       -> '2b_llama'
#     '2b_b8840_hip_llama.cpp'          -> '2b_llama'
#     'b10678_vulkan_llama.cpp'         -> 'llama'
#     'b10678_hip_llama.cpp'            -> 'llama'
#     'b10548_dev_a298422da_llama.cpp'  -> 'llama'  (legacy dev)
#     'tq_0.3.0_vulkan_llama.cpp'       -> 'tq_llama'
#     '1b_llama.cpp'                    -> '1b_llama' (legacy)
# This keeps the 1-bit ('1b_') and 2-bit/Ternary ('2b_') families distinct.
_FORK_VERSION_RE = re.compile(
    r"(?:^|_)(?:b\d+(?:_dev)?(?:_[0-9a-f]{7,40})?|v?\d+\.\d+\.\d+)(?=_|$)",
    re.IGNORECASE,
)
_FORK_BACKEND_RE = re.compile(
    r"(?:^|_)(vulkan|hip)(?=_llama(?:\.cpp)?$)", re.IGNORECASE
)


def _fork_backend(name: str) -> Optional[str]:
    """Return the explicit terminal backend token in a build name, if any."""
    match = _FORK_BACKEND_RE.search(name.lower())
    return match.group(1).lower() if match else None


def _fork_family(name: str) -> str:
    """Normalize a fork dir/name to its backend-neutral family form."""
    n = name.lower()
    if n.endswith(".cpp"):
        n = n[:-4]
    n = _FORK_VERSION_RE.sub("", n).lstrip("_")
    n = re.sub(r"(^|_)(?:vulkan|hip)(?=_llama$)", "", n)
    return n.lstrip("_")


def _fork_name_sort_key(name: str, preferred_backend: Optional[str] = None) -> tuple:
    """Deterministically group backend twins and prefer an active backend.

    An explicit/persisted selected backend wins. Otherwise Vulkan sorts before
    HIP to preserve the historically validated default; users can still select
    either clearly labeled sibling.
    """
    lower = name.lower()
    backend = _fork_backend(lower)
    if preferred_backend:
        backend_rank = 0 if backend == preferred_backend else 1 if backend else 2
    else:
        backend_rank = {"vulkan": 0, "hip": 1, None: 2}.get(backend, 3)
    backendless = _FORK_BACKEND_RE.sub("", lower)
    return (_fork_family(lower), backendless, backend_rank, lower)


_FORK_BUILD_NUMBER_RE = re.compile(r"(?:^|_)b(\d+)(?=_|$)", re.IGNORECASE)
_FORK_SEMVER_RE = re.compile(r"(?:^|_)v?(\d+)\.(\d+)\.(\d+)(?=_|$)", re.IGNORECASE)


def _fork_newest_first_key(name: str, preferred_backend: Optional[str] = None) -> tuple:
    """Choose among builds of one fork family: backend first, then newest.

    The recipes never delete an older pinned tree, so e.g.
    '2b_b10687_vulkan_llama.cpp' and '2b_b10743_vulkan_llama.cpp' coexist. A
    backend-neutral profile hint must land on the newest build, not on the
    lexicographically first one; the display order keeps _fork_name_sort_key.
    """
    family, _backendless, backend_rank, lower = _fork_name_sort_key(
        name, preferred_backend=preferred_backend
    )
    build = _FORK_BUILD_NUMBER_RE.search(lower)
    semver = _FORK_SEMVER_RE.search(lower)
    if build:
        version: tuple = (0, -int(build.group(1)))
    elif semver:
        version = (1, *(-int(part) for part in semver.groups()))
    else:
        version = (2,)
    return (family, backend_rank, version, lower)


#: Family of plain mainline/stable builds ('b11249_vulkan_llama.cpp',
#: '0.5.0_hip_llama.cpp'); the default runtime when no saved choice applies.
_MAINLINE_FORK_FAMILY = "llama"


def _default_fork_index(
    forks: List[Tuple[str, Path]], preferred: Optional[Path] = None
) -> int:
    """Index of the build to select when no saved choice is available.

    The display order groups builds by family name, and the Ternary/Bonsai
    family '2b_llama' sorts before plain 'llama'. Falling back to index 0
    therefore selected the special-purpose Prism fork whenever the saved build
    had been replaced (e.g. b11224 deleted after building b11249), and the
    oldest mainline build otherwise. An existing ``preferred`` build still
    wins; a missing one keeps its family and backend and takes that family's
    newest build. Without a preference a bare legacy 'llama.cpp' checkout
    stays the default, else the newest mainline build (Vulkan before HIP,
    numbered builds before stable releases).
    """
    if not forks:
        return 0
    names = [name for name, _path in forks]
    family, backend = _MAINLINE_FORK_FAMILY, None
    if preferred is not None:
        try:
            wanted = os.path.normcase(str(Path(preferred).resolve(strict=False)))
        except (OSError, RuntimeError):
            wanted = os.path.normcase(str(preferred))
        for index, (_name, path) in enumerate(forks):
            try:
                if os.path.normcase(str(Path(path).resolve(strict=False))) == wanted:
                    return index
            except (OSError, RuntimeError):
                continue
        family = _fork_family(Path(preferred).name)
        backend = _fork_backend(Path(preferred).name)
    else:
        for index, name in enumerate(names):
            if name.lower() == "llama.cpp":
                return index
    for wanted_family in (family, _MAINLINE_FORK_FAMILY):
        members = [
            i for i, name in enumerate(names) if _fork_family(name) == wanted_family
        ]
        if members:
            return min(members, key=lambda i: _fork_newest_first_key(names[i], backend))
    return 0


def _resolve_server_binary(user_value: str) -> str:
    """Turn a user-provided server name/path into something runnable."""
    p = Path(user_value).expanduser()
    if p.is_absolute() and _is_runnable_binary(p):
        return str(p)

    has_sep = os.sep in user_value or (
        os.altsep is not None and os.altsep in user_value
    )

    if has_sep and not p.is_absolute():
        parts = list(p.parts)
        fork_name = parts[0].lower() if parts else ""
        inner = Path(*parts[1:]) if len(parts) > 1 else None

        if inner is not None and fork_name:
            roots = _candidate_search_roots()
            requested_backend = _fork_backend(fork_name)
            # An explicitly selected fork (LLAMA_CPP_DIR) keeps priority. Among
            # the other builds of the family the selected backend, then the
            # newest build wins (older pinned trees are never deleted).
            selected: Optional[Path] = None
            env_dir = os.environ.get("LLAMA_CPP_DIR")
            if env_dir:
                try:
                    selected = Path(env_dir).expanduser().resolve()
                except (OSError, RuntimeError):
                    selected = None
            preferred_backend = requested_backend or (
                _fork_backend(selected.name) if selected is not None else None
            )
            roots.sort(
                key=lambda root: (
                    root != selected,
                    _fork_newest_first_key(
                        root.name, preferred_backend=preferred_backend
                    ),
                )
            )
            for root in roots:
                root_base = root.name.lower()
                root_backend = _fork_backend(root_base)
                if (
                    requested_backend
                    and root_backend
                    and root_backend != requested_backend
                ):
                    continue
                if root_base.endswith(".cpp"):
                    root_base = root_base[:-4]
                if (
                    root_base.startswith(fork_name)
                    or fork_name.startswith(root_base)
                    # Fuzzy fallback: normalize version + terminal backend,
                    # so '2b_llama' resolves to either source-identical
                    # '2b_bNNNN_vulkan_llama.cpp' / '_hip_llama.cpp' sibling.
                    or _fork_family(root_base).startswith(_fork_family(fork_name))
                    or _fork_family(fork_name).startswith(_fork_family(root_base))
                ):
                    candidate = root / inner
                    if _is_runnable_binary(candidate):
                        _debug_print(f"Found candidate: {candidate}")
                        return str(candidate)
                    for sub in _SERVER_SUBPATHS:
                        candidate = root / sub
                        if _is_runnable_binary(candidate):
                            _debug_print(
                                f"Found candidate in fork subpath: {candidate}"
                            )
                            return str(candidate)
                        candidate_with_inner = (root / sub) / inner
                        if _is_runnable_binary(candidate_with_inner):
                            _debug_print(
                                f"Found candidate in fork subpath with inner: {candidate_with_inner}"
                            )
                            return str(candidate_with_inner)

    anchors: List[Path] = []
    seen: set = set()

    def add_anchor(a: Path) -> None:
        try:
            ra = a.resolve()
        except (OSError, RuntimeError):
            return
        if ra in seen:
            return
        seen.add(ra)
        anchors.append(ra)

    for base in (Path(__file__).resolve().parent, Path.cwd()):
        chain = [base, *list(base.parents)[:5]]
        for a in chain:
            add_anchor(a)

    for a in anchors:
        candidate = a / p
        if _is_runnable_binary(candidate):
            _debug_print(f"Found candidate in anchors: {candidate}")
            return str(candidate)

    if not has_sep:
        which = shutil.which(user_value)
        if which:
            return which

    name = Path(user_value).name or "llama-server"
    if name in ("llama-server", "llama-server.exe"):
        candidate_subpaths = list(_SERVER_SUBPATHS)
    else:
        candidate_subpaths = [
            f"build/bin/Release/{name}",
            f"build/bin/Debug/{name}",
            f"build/bin/{name}",
            f"build/{name}",
            name,
        ]

    for root in _candidate_search_roots():
        for sub in candidate_subpaths:
            candidate = root / sub
            if _is_runnable_binary(candidate):
                _debug_print(f"Found candidate in subpaths: {candidate}")
                return str(candidate)

    _debug_print(f"Defaulting to user value: {user_value}")
    return user_value


def _server_binary_in_fork(root: Path) -> Optional[str]:
    """Return the first native llama-server inside one discovered build."""
    for subpath in _SERVER_SUBPATHS:
        candidate = root / subpath
        if _is_runnable_binary(candidate):
            return str(candidate)
    return None


def _find_compatible_draft_server(
    draft_model: Optional[ModelEntry],
    preferred_binary: str,
    *,
    discovered_forks: Optional[List[Tuple[str, Path]]] = None,
) -> Optional[str]:
    """Find a DFlash2-capable sibling build without changing global selection.

    Mainline b10658+ and normal drafters retain ``preferred_binary``. A
    DFlash2 sidecar selected with an older identified build may still fall back
    to a reviewed legacy PR #27342 sibling, preserving old installations while
    current mainline needs no special fork.
    """
    allowed, message, _build = check_draft_model_build(draft_model, preferred_binary)
    needs_capability_search = bool(
        draft_model is not None
        and draft_model.is_dflash2_drafter
        and (not allowed or message)
    )
    if not needs_capability_search:
        return preferred_binary

    preferred_key = os.path.normcase(str(Path(preferred_binary)))
    forks = (
        discovered_forks if discovered_forks is not None else _discover_llama_forks()
    )
    for _name, root in forks:
        candidate = _server_binary_in_fork(root)
        if not candidate or os.path.normcase(candidate) == preferred_key:
            continue
        candidate_allowed, _candidate_message, _candidate_build = (
            check_draft_model_build(draft_model, candidate)
        )
        if candidate_allowed and not _candidate_message:
            return candidate
    return preferred_binary if allowed else None


def _resolve_diffusion_binary(user_value: str, arch: Optional[str] = None) -> str:
    """Resolve the diffusion binary (fork or mainline build).

    A diffusion model needs a diffusion CLI (llama-diffusion-cli for
    mainline Dream/LLaDA/RND1, or llama-diffusion-gemma-cli for the
    DiffusionGemma fork PR #24427), which lives in the same build tree
    as llama-server.

    ``user_value`` may be:
      * a bare name ("llama-diffusion-cli") — searched in all known roots,
      * a fork-relative path ("d_b96.../llama-diffusion-cli") — the fork is
        matched by name like any other server_binary,
      * an absolute path — used as-is if it exists.

    ``arch`` (the GGUF general.architecture) selects the PREFERRED binary
    name: DiffusionGemma wants llama-diffusion-gemma-cli, everything else
    the generic llama-diffusion-cli. When ``user_value`` is a bare default
    ("llama-diffusion-cli") we still try the arch-preferred name first so
    the DiffusionGemma fork's dedicated binary is found automatically.
    """
    # If the user passed an explicit binary name/path, honour it directly.
    resolved = _resolve_server_binary(user_value)
    # If the generic resolver already found a real file, use it.
    if _is_runnable_binary(Path(resolved)):
        return resolved

    # Build the ordered list of binary names to search for. An EXPLICIT
    # request (e.g. "llama-diffusion-gemma-server") wins over the
    # arch-derived default, then the arch-preferred name, then the generic
    # llama-diffusion-cli as the portable fallback.
    requested_name = Path(user_value).name if user_value else ""
    preferred = _diffusion_binary_for_arch(arch)
    search_names: List[str] = []
    for name in (requested_name, preferred, "llama-diffusion-cli"):
        if name and name not in search_names:
            search_names.append(name)

    # Scan the diffusion-specific subpaths across all roots, trying each
    # candidate binary name in preference order. The outer loop is the
    # binary name so the PREFERRED name (e.g. llama-diffusion-gemma-cli for
    # DiffusionGemma) is searched across ALL roots before falling back to
    # the generic llama-diffusion-cli — otherwise a different fork's
    # generic CLI would win over the correct fork's dedicated binary.
    for binary_name in search_names:
        for root in _candidate_search_roots():
            for sub in _diffusion_subpaths_for(binary_name):
                candidate = root / sub
                if _is_runnable_binary(candidate):
                    _debug_print(f"Found diffusion binary: {candidate}")
                    return str(candidate)
    # Last resort: a bare name might be on PATH.
    for name in search_names:
        which = shutil.which(name)
        if which:
            return which
    return resolved


# ---------------------------------------------------------------------------
# llama.cpp fork discovery


def _discover_llama_forks() -> List[Tuple[str, Path]]:
    """Scan the filesystem for llama.cpp fork directories.

    A fork directory is any dir whose name matches ``*llama.cpp`` and that
    contains at least one runnable llama-server binary inside.

    When LLAMA_CPP_DIR points to a parent folder (e.g. C:\\LAB\\ai-local),
    this function recursively walks that tree to find all subdirectories
    matching ``*llama.cpp*`` — supporting nested structures like
    ``C:\\LAB\\ai-local\\1b_llama.cpp\\build\\…``.

    Returns a list of ``(display_name, resolved_path)`` tuples sorted so that
    the bare ``llama.cpp`` comes first, then alphabetical.
    """
    seen: set = set()
    forks: List[Tuple[str, Path]] = []

    # Collect candidate parent directories to scan
    parents: List[Path] = []

    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        try:
            p = Path(env_dir).expanduser().resolve()
            if p.exists():
                # If the env dir itself matches, add it too
                if re.search(r"llama\.cpp", p.name, re.IGNORECASE):
                    parents.append(p)
                # Always scan the env dir as a root — forks may be direct
                # children or nested deeper.
                parents.append(p)
                # Also scan parent in case env points to e.g. C:\LAB
                parents.append(p.parent)
        except (OSError, RuntimeError):
            pass

    for base in (Path(__file__).resolve().parent, Path.cwd()):
        for ancestor in [base, *list(base.parents)[:6]]:
            parents.append(ancestor)

    # Documented workspace convention: the README and .bat launchers keep
    # all *_llama.cpp builds side-by-side under '<drive>/ai-local' or
    # '<drive>/LAB/ai-local'. Add those for every drive the script / cwd
    # lives on so fork discovery works even without LLAMA_CPP_DIR set.
    # (This is what makes the terminal launcher find H:/LAB/ai-local when
    # run from H:/GitHub/Auto Tuner.)
    _WORKSPACE_RELS = ("ai-local", "LAB/ai-local", "LAB\\ai-local")
    drive_roots: set = set()
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        try:
            anchor = base.resolve()
        except (OSError, RuntimeError):
            continue
        # Walk up to the drive root (Windows) or '/' (POSIX).
        # NB: anchor.drive / anchor.root are plain strings on POSIX, so
        # wrap them in Path() — drive_roots must hold Path objects only
        # (consumed below via `root / rel`).
        if anchor.drive:
            drive_roots.add(Path(anchor.drive + os.sep))
        elif anchor.root:
            drive_roots.add(Path(anchor.root))
        else:
            drive_roots.add(anchor)
    for root in drive_roots:
        for rel in _WORKSPACE_RELS:
            cand = root / rel
            if cand not in parents:
                parents.append(cand)

    # Deduplicate while preserving order
    deduped: List[Path] = []
    for p in parents:
        try:
            rp = p.resolve()
        except (OSError, RuntimeError):
            continue
        if rp not in seen:
            seen.add(rp)
            deduped.append(rp)
    parents = deduped

    for parent in parents:
        if not parent.exists():
            continue
        try:
            children = list(parent.iterdir())
        except (OSError, PermissionError):
            continue
        for child in children:
            if not child.is_dir():
                continue
            name = child.name
            # Match: "llama.cpp", "*_llama.cpp", "llama" (plain),
            # "llama-b9150-bin-win-vulkan-x64" (GitHub release ZIPs),
            # "1b_llama", etc.  The binary check below is the real guard
            # against false-positives (e.g. "llama-3-8b" model folders
            # won't contain llama-server.exe and are skipped silently).
            if not re.search(
                r"(?:(?:^|[-_.])llama(?:[-_.]|$)|llama\.cpp)", name, re.IGNORECASE
            ):
                continue
            try:
                rp = child.resolve()
            except (OSError, RuntimeError):
                continue
            if rp in seen:
                continue
            # Must actually contain a runnable binary — otherwise it's a
            # stale checkout or a source-only clone without a build.
            has_binary = any(_is_runnable_binary(rp / sub) for sub in _SERVER_SUBPATHS)
            if not has_binary:
                # Debug aid: surface WHY a matching dir was skipped, so a
                # missing fork in the menu is immediately diagnosable (e.g.
                # build not finished yet, or a non-standard build layout).
                debug_cat(
                    "llama_cpp",
                    f"fork-skip: {name} matched the name pattern but has "
                    "no llama-server binary under any of: "
                    + ", ".join(_SERVER_SUBPATHS),
                )
                continue
            debug_cat("llama_cpp", f"fork-found: {name}")
            seen.add(rp)
            forks.append((name, rp))

    # Bare legacy llama.cpp first, then family/version pairs with Vulkan before
    # HIP. Persisted GUI/CLI selections still override this fallback order.
    forks.sort(
        key=lambda item: (
            item[0].lower() != "llama.cpp",
            *_fork_name_sort_key(item[0]),
        )
    )
    return forks


def _pick_fork(
    forks: List[Tuple[str, Path]], non_interactive: bool = False
) -> Optional[Path]:
    """Show the fork menu and return the chosen fork directory.

    Returns ``None`` only when no forks were discovered at all.
    Handles ``EOFError`` gracefully (non-TTY / CI context) by defaulting to
    the standard build (see ``_default_fork_index``).
    """
    if not forks:
        return None

    default = forks[_default_fork_index(forks)]
    if len(forks) == 1 or non_interactive:
        print(f"[AutoTuner] Using default llama.cpp fork: {default[0]}")
        return default[1]

    print("\n" + _BAR)
    print("  LLAMA.CPP FORK SELECTION")
    print(_BAR)
    print("  Found the following llama.cpp forks:\n")
    for i, (name, path) in enumerate(forks, 1):
        print(f"  {i}. {name:<30} {path}")
    print()
    print("  Enter a number to select, or press Enter for the default.")
    print(_BAR)

    try:
        raw = input(
            f"Select fork [1-{len(forks)}] (default {forks.index(default) + 1}): "
        ).strip()
    except EOFError:
        raw = ""

    if not raw:
        print(f"[AutoTuner] Using fork: {default[0]}")
        return default[1]

    if raw.isdigit():
        n = int(raw)
        if 1 <= n <= len(forks):
            selected = forks[n - 1]
            print(f"[AutoTuner] Using fork: {selected[0]}")
            return selected[1]

    print(f"[AutoTuner] Invalid choice '{raw}' — using default: {default[0]}")
    return default[1]


def _required_fork_name(profile: ModelProfile) -> Optional[str]:
    """Extract the required fork directory name from ``profile.server_binary``.

    Examples:
      ``"1b_llama/llama-server"``  → ``"1b_llama.cpp"``
      ``"ik_llama.cpp/llama-server"`` → ``"ik_llama.cpp"``
      ``None`` → ``None``
    """
    if not profile.server_binary:
        return None
    parts = Path(profile.server_binary).parts
    if not parts:
        return None
    first = parts[0]
    # Normalize: add .cpp suffix when missing
    if re.search(r"llama", first, re.IGNORECASE) and not first.endswith(".cpp"):
        return first + ".cpp"
    return first


# ---------------------------------------------------------------------------
# Client settings hint


def _print_client_settings(host: str, port: int, ctx: int, model: ModelEntry) -> None:
    """Print a copy-pasteable block for OpenAI-API clients."""
    base_url = f"http://{host}:{port}/v1"
    print()
    print(_BAR)
    print("  Client settings (Roo-Code, Continue, Cline, Open WebUI, …)")
    print(_BAR)
    print(f"    Base URL          : {base_url}")
    print("    API key           : sk-no-key   (any non-empty string works)")
    print(f"    Model name        : {model.name}")
    hint = (
        "<- set this in your client" if _RENDERER.plain else "← set this in your client"
    )
    print(f"    Context window    : {ctx:,} tokens   {hint}")
    print(_BAR)


# ---------------------------------------------------------------------------
# Argument parsing


def _server_was_explicit(argv: List[str]) -> bool:
    """Recognize AutoTuner --server options before llama-server passthrough."""
    try:
        argv = argv[: argv.index("--")]
    except ValueError:
        pass
    return any(
        option == "--server" or option.startswith("--server=") for option in argv
    )


def _parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse AutoTuner options while preserving real llama-server passthrough."""
    try:
        separator_index = argv.index("--")
    except ValueError:
        before, separator, passthrough = argv, False, []
    else:
        before, separator, passthrough = (
            argv[:separator_index],
            True,
            argv[separator_index + 1 :],
        )
    plain_help = "--plain" in before
    target_help = describe_targets()
    if plain_help:
        target_help = _ascii_safe(target_help)
    p = argparse.ArgumentParser(
        prog="auto_tuner",
        description="Interactive launcher for llama-server with auto-tuned "
        "config based on free RAM/VRAM.",
        epilog="Pass extra llama-server arguments after --, for example: "
        "auto_tuner --model Qwen -- --threads 8",
    )
    p.add_argument(
        "--models-path",
        default=os.environ.get("AUTOTUNER_MODELS", "./models"),
        help="Folder to scan for *.gguf models "
        "(default: ./models, env AUTOTUNER_MODELS)",
    )
    p.add_argument(
        "--settings-path",
        default=str(Path(__file__).parent / "settings"),
        help="Folder with per-model YAML profiles (default: ./settings)",
    )
    p.add_argument(
        "--server",
        default=os.environ.get("LLAMA_SERVER", "llama-server"),
        help="Path to the llama-server binary "
        "(default: llama-server, env LLAMA_SERVER)",
    )
    p.add_argument(
        "--llama-cpp-dir",
        default=os.environ.get("LLAMA_CPP_DIR"),
        help="Path to your llama.cpp checkout (env LLAMA_CPP_DIR). "
        "All forks in the same parent folder are discovered automatically. "
        "Useful when llama.cpp lives outside the standard search paths "
        "(e.g. C:\\LAB\\ai-local\\llama.cpp).",
    )
    p.add_argument(
        "--host", default="127.0.0.1", help="Server bind host (default: 127.0.0.1)"
    )
    p.add_argument("--port", type=int, default=1234, help="Server port (default: 1234)")
    p.add_argument(
        "--ctx",
        type=int,
        default=None,
        help="Override context length (otherwise auto-tuned)",
    )
    p.add_argument(
        "--model", default=None, help="Pick model by substring without showing the menu"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command but don't start the server",
    )
    p.add_argument(
        "--yes", "-y", action="store_true", help="Skip the launch confirmation prompt"
    )
    p.add_argument(
        "--novision",
        action="store_true",
        help="Disable vision (mmproj) even if available",
    )
    p.add_argument(
        "--nodraft",
        action="store_true",
        help="Disable speculative decoding/draft model",
    )
    p.add_argument(
        "--nothinking", action="store_true", help="Disable thinking/reasoning output"
    )
    p.add_argument(
        "--ngram",
        action="store_true",
        help="Explicitly enable n-gram self-speculative decoding (already on by "
        "default for every model).",
    )
    p.add_argument(
        "--no-ngram",
        action="store_true",
        help="Disable the default n-gram self-speculative decoding path.",
    )
    p.add_argument(
        "--no-mmproj-offload",
        action="store_true",
        help="Keep an active vision projector in system RAM instead of VRAM. "
        "The AutoTuner moves its estimated footprint to the RAM budget.",
    )
    p.add_argument(
        "--ocr-input",
        action="append",
        metavar="PATH",
        default=[],
        help="Run the shared OCR workflow for a file or folder. May be repeated; "
        "PDF pages are rendered and Office files are converted with LibreOffice.",
    )
    p.add_argument(
        "--ocr-output",
        metavar="DIR",
        default=None,
        help="OCR output parent directory (default: ./ocr-output).",
    )
    p.add_argument(
        "--ocr-prompt",
        default=None,
        help="Override the detected OCR model family's verified task prompt.",
    )
    p.add_argument(
        "--ocr-pages",
        default="",
        metavar="RANGE",
        help="PDF page selection such as 1-3,5 (default: all pages).",
    )
    p.add_argument(
        "--ocr-dpi",
        type=int,
        default=220,
        metavar="N",
        help="PDF render DPI, clamped to 72-600 (default: 220).",
    )
    p.add_argument(
        "--ocr-max-tokens",
        type=int,
        default=4096,
        metavar="N",
        help="Maximum generated tokens per page (default: 4096).",
    )
    p.add_argument(
        "--ocr-format",
        choices=("markdown", "text"),
        default="markdown",
        help="Combined/page output format (default: markdown).",
    )
    p.add_argument(
        "--ocr-timeout",
        type=int,
        default=1200,
        metavar="SECONDS",
        help="llama-server request timeout per page (default: 1200).",
    )
    p.add_argument(
        "--ocr-keep-rendered",
        action="store_true",
        help="Keep normalized/rendered page PNG files beside OCR results.",
    )
    p.add_argument(
        "--ocr-keep-grounding",
        action="store_true",
        help="Keep <|ref|>/<|det|> coordinate markup instead of cleaning it.",
    )
    p.add_argument(
        "--no-prompt-cache",
        action="store_true",
        dest="no_prompt_cache",
        help="Disable host-memory prompt caching (--cache-ram 0). By default "
        "prompt caching is auto-enabled; Vision caching requires llama.cpp "
        "b10045+ and safely falls back to off on older/unprobeable builds.",
    )
    p.add_argument(
        "--cache-ram-mib",
        type=int,
        default=None,
        metavar="MIB",
        help="Maximum host prompt-cache size in MiB (default: saved GUI value; "
        "-1 unlimited, 0 disabled). --no-prompt-cache wins.",
    )
    p.add_argument(
        "--no-metrics",
        action="store_true",
        dest="no_metrics",
        help="Do not pass --metrics; disables the GET /metrics endpoint.",
    )
    p.add_argument(
        "--slots-api",
        action="store_true",
        dest="slots_api",
        help="Enable GET /slots. AutoTuner requests --slots when enabled and "
        "--no-slots when disabled; unsupported flags are compatibility-filtered.",
    )
    p.add_argument(
        "--force-mlock",
        action="store_true",
        help="Force a locking --load-mode (mlock/mmap+mlock) even when the "
        "auto-tuner would otherwise skip it; requires sufficient free "
        "RAM/VRAM and OS privileges, and a b10151+ build on GPU systems",
    )
    p.add_argument(
        "--mode",
        choices=("chat", "coding"),
        default=None,
        help="Sampling mode (default: saved GUI preference or chat).",
    )
    p.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt; requires an unambiguous --model and uses safe availability defaults.",
    )
    p.add_argument(
        "--plain",
        action="store_true",
        help="Use ASCII-only terminal output (safe for logs and legacy consoles).",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color even in a supported terminal.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable all debug output.",
    )
    p.add_argument(
        "--debug-category",
        action="append",
        choices=("hardware", "scanner", "llama_cpp", "config"),
        default=[],
        help="Enable a debug category; may be repeated.",
    )
    p.add_argument(
        "--performance-target",
        choices=list_target_names(),
        default=None,
        metavar="{safe,balanced,throughput,low_vram}",
        help="VRAM utilisation preset. Overrides any 'performance_target:' "
        "in the YAML profile. Tiers:\n" + target_help,
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="Open the Qt log-viewer window after the server starts "
        "(requires PyQt6; server stdout/stderr stream into the window).",
    )
    p.add_argument(
        "--diagnose",
        metavar="SUBSTR",
        nargs="?",
        const="",  # bare --diagnose (no arg) → empty string → all models
        default=None,
        help="Print a metadata diagnostic report for matching model(s) "
        "and exit without launching. With a substring: filters by name "
        "(case-insensitive contains-match). Without: reports all models.",
    )
    p.add_argument(
        "--gpu",
        "-g",
        metavar="NAME",
        default=None,
        help="Pin this server to ONE GPU exclusively (boots only on the "
        "named card, hides the rest). NAME is matched case-insensitively "
        "as a substring of the card name, e.g. 'R9700' or '9070'. Use when "
        "launching a second server so it lands on the still-free card "
        "instead of piling onto an already-full one. Overrides the "
        "persisted forced_gpu setting; omit to use auto selection.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Prompt text for diffusion models (llama-diffusion-cli is "
            "single-shot and needs a prompt). Ignored for server models."
        ),
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"AutoTuner v{VERSION}",
        help="Show the AutoTuner version and exit.",
    )
    args = p.parse_args(before)
    args.passthrough = passthrough if separator else []
    return args


# ---------------------------------------------------------------------------
# main


def main(argv: Optional[List[str]] = None) -> int:  # noqa: C901  (complex but intentional)
    raw_argv = argv if argv is not None else sys.argv[1:]
    args = _parse_args(raw_argv)
    _configure_renderer(args)
    if args.llama_cpp_dir:
        os.environ["LLAMA_CPP_DIR"] = args.llama_cpp_dir

    global _DEBUG_MODE
    _DEBUG_MODE = bool(args.debug)
    _DEBUG_CATEGORIES.clear()
    for category in args.debug_category:
        enable_debug_category(category)
    non_interactive = bool(
        args.non_interactive
        or not sys.stdin.isatty()
        or (args.model is not None and (args.dry_run or args.yes))
    )
    if non_interactive and not args.model and args.diagnose is None:
        print(
            "[AutoTuner] --non-interactive requires --model so selection cannot block."
        )
        return 2

    _print_banner()
    if _DEBUG_MODE or _DEBUG_CATEGORIES:
        enabled = "all" if _DEBUG_MODE else ", ".join(sorted(_DEBUG_CATEGORIES))
        print(f"  Debug: {enabled}")

    # ── llama.cpp fork discovery ────────────────────────────────────────────
    # Only show the fork menu when --server was NOT specified explicitly by the
    # user (i.e. it is still the default "llama-server" or from LLAMA_SERVER).
    # An explicit --server path overrides everything.
    user_specified_server = (
        _server_was_explicit(raw_argv) or "LLAMA_SERVER" in os.environ
    )

    discovered_forks: List[Tuple[str, Path]] = []
    selected_fork_path: Optional[Path] = None

    if not user_specified_server and args.diagnose is None:
        discovered_forks = _discover_llama_forks()
        selected_fork_path = _pick_fork(
            discovered_forks, non_interactive=non_interactive
        )
        if selected_fork_path is not None:
            # Point LLAMA_CPP_DIR at the chosen fork so that
            # _candidate_search_roots() finds binaries there (and siblings).
            os.environ["LLAMA_CPP_DIR"] = str(selected_fork_path)
            print(f"[AutoTuner] LLAMA_CPP_DIR → {selected_fork_path}\n")
        else:
            print(
                "[AutoTuner] No llama.cpp forks found on disk — "
                "set LLAMA_CPP_DIR or pass --server.\n"
            )

    # Keep the user's/default fork as the per-selection baseline. A profile can
    # temporarily select a specialized fork without leaking it to the next model.
    default_fork_path = selected_fork_path

    # ── System detection ────────────────────────────────────────────────────
    # Probe the selected/exact binary so CUDA/HIP/Vulkan/SYCL/Metal device
    # identities, memory, and supported-device intersection are authoritative.
    system = detect_system(_resolve_server_binary(args.server))
    _print_system(system)

    models_path = Path(args.models_path).expanduser()
    if not models_path.exists():
        print(f"\n[AutoTuner] Models folder not found: {models_path}")
        print("  Pass --models-path /path/to/models or set AUTOTUNER_MODELS.")
        return 2

    print(f"\n[AutoTuner] Scanning models in: {models_path}")
    entries = scan_models(models_path)
    if not entries:
        print("[AutoTuner] No *.gguf models found.")
        return 2

    profiles = load_profiles(Path(args.settings_path))
    print(f"[AutoTuner] Loaded {len(profiles)} profile(s) from {args.settings_path}")

    # ── Diagnose-only path ─────────────────────────────────────────────
    # `--diagnose` (with optional substring) prints a metadata audit for
    # the matching models and exits without launching anything. Reuses
    # the diagnostics module so CLI and (future) GUI button share the
    # same formatting and warning catalogue.
    if args.diagnose is not None:
        from diagnostics import format_diagnostic_report, find_model_by_substring

        matches = find_model_by_substring(entries, args.diagnose)
        if not matches:
            print(
                f"\n[AutoTuner] --diagnose: no model matches "
                f"'{args.diagnose}'. Scanned {len(entries)} file(s)."
            )
            return 2
        print(
            f"\n[AutoTuner] --diagnose: reporting on {len(matches)} "
            f"model(s) (of {len(entries)} scanned).\n"
        )
        for m in matches:
            print(format_diagnostic_report(m))
            print()
        return 0

    # ── Main loop ───────────────────────────────────────────────────────────
    first_iteration = True
    last_exit_code = 0

    while True:
        if not first_iteration:
            print()
            system = detect_system(_resolve_server_binary(args.server))
            _print_system(system)
            args.model = None
            args.ctx = None

        groups = group_entries(entries)
        flat = _print_menu(groups)

        try:
            model, picked_flags = _pick_model(flat, args.model)
        except KeyboardInterrupt:
            print("\n[AutoTuner] Aborted by user.")
            return 0

        if model is None:
            print("[AutoTuner] No model selected — exiting.")
            if first_iteration and args.model is not None:
                return 2
            return last_exit_code if first_iteration else 0

        # Inline menu flags apply only to this selection; CLI flags stay global.
        selection_flags = set(picked_flags)
        use_novision = args.novision or "novision" in selection_flags
        use_nodraft = args.nodraft or "nodraft" in selection_flags
        use_nothinking = args.nothinking or "nothinking" in selection_flags
        use_ngram_flag = args.ngram or "ngram" in selection_flags
        disable_ngram_flag = (
            args.no_ngram
            or "no-ngram" in selection_flags
            or "nongram" in selection_flags
        )
        model = __import__("copy").copy(model)

        if use_novision and model.mmproj is not None:
            print(
                f"[AutoTuner] Vision disabled per --novision "
                f"(ignoring {model.mmproj.name})"
            )
            model.mmproj = None

        if default_fork_path is not None:
            selected_fork_path = default_fork_path
            os.environ["LLAMA_CPP_DIR"] = str(selected_fork_path)
        profile = match_profile(
            model.name, profiles, getattr(model, "architecture", "")
        )

        # ── Fork / model mismatch check ─────────────────────────────────
        # Some models require a specific fork (e.g. bonsai → 1b_llama.cpp).
        # Warn the user if their selected fork differs and offer to switch.
        # This check is skipped when the user passed --server explicitly.
        if not user_specified_server and selected_fork_path is not None:
            required_fork = _required_fork_name(profile)
            if required_fork:
                selected_name = selected_fork_path.name.lower()
                req_lower = required_fork.lower()
                # Compare on the backend-neutral family form so both
                # '2b_bNNNN_vulkan_llama.cpp' and '_hip_llama.cpp' match the
                # profile requirement '2b_llama.cpp'.
                if _fork_family(selected_name) != _fork_family(req_lower):
                    print(
                        f"\n[AutoTuner] ⚠  Profile '{profile.display_name}' "
                        f"requires: {required_fork}"
                    )
                    print(f"             You selected:  {selected_fork_path.name}")
                    # Look for the required fork among already-discovered forks
                    matching = [
                        (n, p)
                        for n, p in discovered_forks
                        if _fork_family(n) == _fork_family(req_lower)
                    ]
                    selected_backend = _fork_backend(selected_name)
                    matching.sort(
                        key=lambda item: _fork_newest_first_key(
                            item[0], preferred_backend=selected_backend
                        )
                    )
                    if matching:
                        switch = non_interactive or _confirm(
                            f"Switch to {required_fork} for this model?",
                            default_yes=True,
                        )
                        if switch:
                            selected_fork_path = matching[0][1]
                            os.environ["LLAMA_CPP_DIR"] = str(selected_fork_path)
                            print(f"[AutoTuner] Switched to: {selected_fork_path.name}")
                        else:
                            print(
                                f"[AutoTuner] Keeping {selected_fork_path.name} "
                                "— the model may not load correctly."
                            )
                    else:
                        print(
                            f"[AutoTuner] ⚠  {required_fork} not found on this system."
                        )
                        print(
                            "             Continuing with current fork "
                            "— the model may not load correctly."
                        )

        # ── Draft model detection ────────────────────────────────────────
        # scanner.py already paired each main model with its assistant /
        # draft sibling (when present). We just wrap that path in a
        # ModelEntry — `compute_config` only needs `.path` and `.size_gb`.
        draft_model = None
        if profile.draft_max > 0 and not use_nodraft and model.draft is not None:
            try:
                draft_size = model.draft.stat().st_size
                draft_model = ModelEntry(
                    path=model.draft,
                    name=model.draft.stem,
                    group=model.group,
                    size_bytes=draft_size,
                    mmproj=None,
                    draft=None,
                    metadata=read_gguf_metadata(model.draft),
                    part_paths=[model.draft],
                )
                print(f"[AutoTuner] Found draft model: {draft_model.name}")
            except OSError as exc:
                print(f"[AutoTuner] Draft sibling unreadable ({exc}); skipping.")

        # ── Interactive feature chain ────────────────────────────────────
        (use_vision, use_draft, use_thinking, use_ngram, effective_draft) = (
            _ask_interactive_features(
                model,
                draft_model,
                args.settings_path,
                force_ngram=use_ngram_flag,
                disable_ngram=disable_ngram_flag,
                non_interactive=non_interactive,
                draft_available=(draft_model is not None or model.has_embedded_mtp)
                and not use_nodraft,
                thinking_available=model.supports_thinking and not use_nothinking,
                draft_default=bool(getattr(profile, "auto_draft", True)),
            )
        )
        if not use_draft:
            effective_draft = None
        if use_nothinking:
            use_thinking = False

        try:
            ocr_options = _ocr_options_for_selection(
                model,
                args,
                use_vision=use_vision,
                non_interactive=non_interactive,
            )
        except OcrWorkflowError as exc:
            print(f"[AutoTuner] OCR setup failed: {exc}")
            return 2

        def resolve_specialized_binary(
            profile: ModelProfile,
            use_draft_flag: bool,
            model_name: str,
        ) -> str:
            """Choose the exact server/fork before hardware tuning."""
            if profile.server_binary:
                return profile.server_binary
            if gemma_draft_needs_ik_fork(
                model_name, use_draft_flag, _resolve_server_binary(args.server)
            ):
                print(
                    "[AutoTuner] Selected build advertises no --spec-type "
                    "(pre-b9190) — Gemma-4 drafter falls back to ik_llama.cpp."
                )
                return "ik_llama.cpp/llama-server"
            return args.server

        # ── Config computation ───────────────────────────────────────────
        # Re-probe after model/profile fork selection. A profile may have
        # switched from the menu's default build to CUDA, SYCL, OpenVINO, or a
        # specialized fork; tuning against the old OS-visible GPU set would
        # produce invalid indices and memory assumptions.
        tuning_is_diffusion = (
            model.is_diffusion or profile.runner == "llama-diffusion-cli"
        ) and profile.runner != "llama-diffusion-gemma-server"
        if tuning_is_diffusion:
            tuning_request = profile.server_binary or "llama-diffusion-cli"
            tuning_binary = _resolve_diffusion_binary(
                tuning_request,
                arch=(model.metadata or {}).get("general.architecture"),
            )
        elif profile.runner == "llama-diffusion-gemma-server":
            tuning_binary = _resolve_diffusion_binary(
                "llama-diffusion-gemma-server",
                arch=(model.metadata or {}).get("general.architecture"),
            )
        else:
            tuning_binary = _resolve_server_binary(
                resolve_specialized_binary(profile, use_draft, model.name)
            )

        compatible_draft_binary = _find_compatible_draft_server(
            effective_draft, tuning_binary
        )
        if compatible_draft_binary is not None and os.path.normcase(
            compatible_draft_binary
        ) != os.path.normcase(tuning_binary):
            print(
                "[AutoTuner] DFlash2 runtime: selected stock build is "
                f"incompatible; using {compatible_draft_binary} for this launch."
            )
            tuning_binary = compatible_draft_binary

        draft_allowed, draft_message, _draft_build = check_draft_model_build(
            effective_draft,
            tuning_binary,
            target=model,
        )
        if draft_message:
            level = "Warning" if draft_allowed else "Error"
            print(f"[AutoTuner] {level}: {draft_message}")
        if not draft_allowed:
            return 2

        model_allowed, model_message, _model_build = check_model_build(
            model,
            tuning_binary,
        )
        if model_message:
            level = "Warning" if model_allowed else "Error"
            print(f"[AutoTuner] {level}: {model_message}")
        if not model_allowed:
            return 2

        system = detect_system(tuning_binary)

        # Resolve performance target: CLI > YAML profile > "balanced".
        perf_target = resolve_performance_target(
            cli_choice=getattr(args, "performance_target", None),
            profile_choice=getattr(profile, "performance_target", "") or None,
        )

        try:
            cfg = compute_config(
                model,
                system,
                profile,
                draft_model=effective_draft,
                user_ctx=args.ctx,
                force_mlock=getattr(args, "force_mlock", False),
                perf_target=perf_target,
                no_mmproj_offload=bool(
                    getattr(args, "no_mmproj_offload", False)
                    and model.mmproj is not None
                ),
                prompt_cache_ram_mib=_effective_prompt_cache_mib(args),
                mode=_effective_mode(args),
                gpu_priorities=app_settings.get_gpu_priorities(),
                force_gpu=getattr(args, "gpu", None) or app_settings.get_forced_gpu(),
            )
        except MemoryError as exc:
            # An exhausted budget is refused, never turned into a 2k plan.
            print(f"[AutoTuner] Error: {exc}")
            return 2

        print(f"\n  [mlock] decision: model={model.name}")
        print(
            f"         full_offload={cfg.full_offload}  "
            f"vram={cfg.estimated_model_vram_gb:.1f}GB  "
            f"ram={cfg.estimated_model_ram_gb:.1f}GB  "
            f"mapped={float(getattr(cfg, 'mapped_model_ram_gb', 0.0) or 0.0):.1f}GB  "
            f"mapped_active={float(getattr(cfg, 'mapped_model_resident_gb', 0.0) or 0.0):.1f}GB  "
            f"runtime_ram={float(getattr(cfg, 'runtime_ram_overhead_gb', 0.0) or 0.0):.1f}GB"
        )
        print(
            f"         sys: total_vram={system.total_vram_gb:.1f}GB  "
            f"free_vram={system.free_vram_gb:.1f}GB  "
            f"total_ram={system.total_ram_gb:.1f}GB  "
            f"free_ram={system.free_ram_gb:.1f}GB"
        )
        print(
            f"         force_mlock={getattr(args, 'force_mlock', False)}  "
            f"-> mlock={cfg.mlock}  no_mmap={cfg.no_mmap}"
        )

        _RENDERER.section("Session")
        selected_mode = _effective_mode(args)
        gpu_choice = args.gpu or app_settings.get_forced_gpu() or "Auto"
        print(
            f"  Mode: {selected_mode}  |  Target: {perf_target.name}  |  GPU: {gpu_choice}  "
            f"|  Prompt cache: {cfg.prompt_cache_ram_gb:.1f} GB"
        )
        _print_config(model, profile, cfg, system)

        # ── Runner selection: diffusion vs server ────────────────────────
        # Diffusion text models (Dream/LLaDA/RND1 mainline; DiffusionGemma
        # fork) are NOT served by llama-server — they run single-shot via
        # llama-diffusion-cli with --diffusion-* flags and no /health/API.
        # The scanner detects this from general.architecture; a profile may
        # also force it with `runner: llama-diffusion-cli`.
        is_diffusion_run = tuning_is_diffusion

        extra = list(args.passthrough or [])
        if ocr_options is not None:
            # Make the request's model id deterministic across GUI/TUI and
            # independent of llama-server's path-derived default alias.
            extra = ["-a", model.name, *extra]

        if is_diffusion_run:
            # Resolve the diffusion binary. A profile's server_binary field
            # can point at the fork (e.g. "d_b96.../llama-diffusion-cli");
            # otherwise we search for the diffusion binary in the build
            # tree. The architecture selects the preferred binary name:
            # DiffusionGemma (PR #24427) ships llama-diffusion-gemma-cli,
            # mainline Dream/LLaDA/RND1 use the generic llama-diffusion-cli.
            diff_request = profile.server_binary or "llama-diffusion-cli"
            diff_arch = (model.metadata or {}).get("general.architecture")
            diffusion_bin = _resolve_diffusion_binary(diff_request, arch=diff_arch)

            if _is_runnable_binary(Path(diffusion_bin)):
                print(f"[AutoTuner] Found diffusion binary: {diffusion_bin}")
            elif not shutil.which(diffusion_bin):
                print(
                    f"[AutoTuner] Warning: diffusion binary "
                    f"'{diffusion_bin}' not found."
                )
                print(
                    "  Diffusion models need llama-diffusion-cli (build it from "
                    "your diffusion-capable llama.cpp/fork checkout)."
                )
                print(
                    "  Point the profile's server_binary at it, e.g. "
                    'server_binary: "d_b96xxx/llama-diffusion-cli".'
                )

            diff_prompt = getattr(args, "prompt", None)
            cmd = build_diffusion_command(
                model=model,
                config=cfg,
                profile=profile,
                diffusion_binary=diffusion_bin,
                prompt=diff_prompt,
                extra_args=extra,
            )
        elif profile.runner == "llama-diffusion-gemma-server":
            # ── DiffusionGemma HTTP server (PR #24427) ───────────────────
            # Persistent OpenAI-compatible server with its own binary +
            # flag set. Uses the dedicated builder (the gemma-server's
            # manual arg parser rejects llama-server-only flags).
            diff_arch = (model.metadata or {}).get("general.architecture")
            gemma_server = _resolve_diffusion_binary(
                "llama-diffusion-gemma-server", arch=diff_arch
            )
            if _is_runnable_binary(Path(gemma_server)):
                print(f"[AutoTuner] Found diffusion-gemma-server: {gemma_server}")
            elif not shutil.which(gemma_server):
                print(
                    f"[AutoTuner] Warning: llama-diffusion-gemma-server "
                    f"'{gemma_server}' not found."
                )
                print(
                    "  Build a DiffusionGemma-capable fork (PR #24427) and "
                    "select it via LLAMA_CPP_DIR."
                )
            cmd = build_diffusion_server_command(
                model=model,
                config=cfg,
                profile=profile,
                server_binary=gemma_server,
                host=args.host,
                port=args.port,
                extra_args=extra,
                enable_metrics=not getattr(args, "no_metrics", False),
                enable_slots_api=bool(getattr(args, "slots_api", False)),
            )
        else:
            # ── Binary resolution (server path) ──────────────────────────
            raw_server = profile.server_binary or args.server
            server = tuning_binary

            if server != raw_server:
                print(f"[AutoTuner] Found server binary: {server}")
            elif not _is_runnable_binary(Path(server)) and not shutil.which(server):
                print(
                    f"[AutoTuner] Warning: server binary '{server}' not found "
                    "or not executable on this OS."
                )
                print("  Pass --server /path/to/llama-server, set LLAMA_SERVER, or")
                print("  set LLAMA_CPP_DIR to your llama.cpp checkout.")

            # ── Build command (server path) ──────────────────────────────
            try:
                cmd = build_command(
                    model=model,
                    config=cfg,
                    profile=profile,
                    draft_model=effective_draft,
                    server_binary=server,
                    host=args.host,
                    port=args.port,
                    extra_args=extra,
                    use_thinking=use_thinking,
                    # --nodraft disables external drafts AND embedded MTP.
                    # Draftless n-gram remains independent.
                    enable_speculative=use_draft,
                    enable_ngram=use_ngram,
                    enable_prompt_cache=not getattr(args, "no_prompt_cache", False),
                    enable_metrics=not getattr(args, "no_metrics", False),
                    enable_slots_api=bool(getattr(args, "slots_api", False)),
                )
            except ValueError as exc:
                print(f"[AutoTuner] Incompatible memory-plan settings: {exc}")
                return 2

        # Enforce profile-specific runtime gates before llama.cpp begins a
        # potentially multi-minute model load. OCR performs the same check in
        # its job-owned path because it also combines preset requirements.
        if ocr_options is None:
            build_allowed, build_message, _detected_build = check_profile_build(
                profile, cmd[0]
            )
            if build_message:
                level = "Warning" if build_allowed else "Error"
                print(f"[AutoTuner] {level}: {build_message}")
            if not build_allowed:
                return 2

        cmd, removed_args = prepare_command_for_binary(cmd)
        if removed_args:
            print(
                "[AutoTuner] Compatibility adjustments for the selected "
                "llama.cpp binary: " + "; ".join(removed_args)
            )

        if args.dry_run:
            if ocr_options is not None:
                print("[AutoTuner] --dry-run — not starting the OCR server/job.")
                print("Server command:")
                print("  " + " ".join(cmd))
                print("OCR inputs:")
                for input_path in ocr_options.inputs:
                    print(f"  - {input_path}")
                print(f"OCR output: {ocr_options.output_dir}")
                print(f"OCR prompt: {ocr_options.prompt}")
            elif is_diffusion_run:
                print("[AutoTuner] --dry-run — not starting diffusion generation.")
                print("Command:")
                print("  " + " ".join(cmd))
            else:
                print("[AutoTuner] --dry-run — not starting the server.")
                print("Command:")
                print("  " + " ".join(cmd))
                _print_client_settings(args.host, args.port, cfg.ctx, model)
            return 0

        try:
            prompt_label = (
                "Run OCR document job now?"
                if ocr_options is not None
                else (
                    "Run diffusion generation now?"
                    if is_diffusion_run
                    else "Launch llama-server now?"
                )
            )
            launch_now = non_interactive or args.yes or _confirm(prompt_label)
        except KeyboardInterrupt:
            print("\n[AutoTuner] Aborted by user.")
            return 0

        if not launch_now:
            print("[AutoTuner] Launch skipped — back to the model menu.")
            first_iteration = False
            continue

        if ocr_options is not None:
            print(
                "\n[AutoTuner] Starting the OCR model and processing documents "
                "page-by-page. Ctrl+C cancels and releases the server.\n"
            )
            return _run_ocr_terminal_job(cmd, cfg, model, profile, args, ocr_options)

        if is_diffusion_run:
            print(
                "\n[AutoTuner] Running llama-diffusion-cli (single-shot). "
                "Output prints below; the process exits when done.\n"
            )
            if not getattr(args, "prompt", None):
                print(
                    "[AutoTuner] Note: no --prompt given. llama-diffusion-cli "
                    'needs a prompt; pass --prompt "..." or add -p via '
                    "passthrough.\n"
                )
        else:
            _print_client_settings(args.host, args.port, cfg.ctx, model)
            print(
                f"\n[AutoTuner] Web UI will be available at "
                f"http://{args.host}:{args.port}\n"
            )

        # ── Launch (terminal or GUI) ─────────────────────────────────────
        # The GUI log viewer is built around a long-running server (health
        # state, chat endpoint). A diffusion run is single-shot CLI output,
        # so it always uses the terminal path even under --gui.
        if args.gui and not is_diffusion_run:
            # Lazy import: only when --gui is actually used.
            # This keeps the CI smoke-import clean even without PyQt6.
            try:
                from PyQt6.QtWidgets import QApplication
                from qt_log_viewer import LogViewerWindow
                from server_process import ServerProcess
            except ImportError as exc:
                print(f"[AutoTuner] --gui requires PyQt6 and qt_log_viewer.py: {exc}")
                print("[AutoTuner] Falling back to terminal mode.")
                last_exit_code = launch(cmd, env_overrides=cfg.env_overrides)
            else:
                srv = ServerProcess(cmd, env_overrides=cfg.env_overrides)
                try:
                    srv.start()
                except OSError as exc:
                    print(
                        f"[AutoTuner] ERROR: server binary '{cmd[0]}' could not "
                        f"be started: {exc}",
                        file=sys.stderr,
                    )
                    last_exit_code = 126
                    continue
                app = QApplication(sys.argv)
                window = LogViewerWindow(srv)
                window.show()
                sys.exit(app.exec())
        else:
            if args.gui and is_diffusion_run:
                print(
                    "[AutoTuner] --gui ignored for diffusion (single-shot CLI); "
                    "running in terminal mode."
                )
            last_exit_code = launch(cmd, env_overrides=cfg.env_overrides)

        if non_interactive:
            return last_exit_code

        print()
        try:
            keep_going = _confirm(
                "Server stopped. Pick another model?", default_yes=True
            )
        except KeyboardInterrupt:
            print("\n[AutoTuner] Goodbye.")
            return 0

        if not keep_going:
            print("[AutoTuner] Goodbye.")
            return 0

        first_iteration = False


if __name__ == "__main__":
    sys.exit(main())
