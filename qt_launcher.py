"""AutoTuner Qt Launcher — standalone GUI for model selection and server control.

llama-server opens in its own terminal window (visible, full output).
The Qt log panel shows AutoTuner-level status messages only.

Run with:
  python qt_launcher.py
  python qt_launcher.py --models-path D:/models
"""

from __future__ import annotations

import base64
import copy
import hashlib
import itertools
import json
import queue
import dataclasses
import os
import platform
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, cast, Tuple

from PyQt6.QtCore import (
    Qt,
    QByteArray,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRect,
    QThread,
    QTimer,
    QUrl,
    pyqtSignal,
    QSize,
)
from PyQt6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QIcon,
    QMouseEvent,
    QShowEvent,
    QStandardItemModel,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QSystemTrayIcon,
    QTextEdit,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hardware import detect_system, GPUInfo, SystemInfo
from scanner import (
    scan_models,
    group_entries,
    ModelEntry,
    read_gguf_metadata,
    metadata_cache_stats,
    is_mmproj_compatible,
    is_draft_compatible,
)
from settings_loader import load_profiles, match_profile, ModelProfile
from tuner import (
    _visibility_env_for_gpus,
    build_command,
    check_draft_model_build,
    check_model_build,
    check_profile_build,
    compute_config,
    effective_load_mode,
    extract_params_billion,
    gemma_draft_needs_ik_fork,
    kv_quant_factor,
    match_gpu_by_token,
    prepare_command_for_binary,
    probe_binary_build_info,
    probe_binary_build_number,
    veto_unsafe_mlock,
    TunedConfig,
)
from performance_target import (
    PERFORMANCE_TARGETS,
    list_target_names,
    resolve_performance_target,
    DEFAULT_TARGET_NAME,
)
import app_settings
import startup_manager
from ocr_workflow import (
    OcrJobOptions,
    OcrJobResult,
    OcrJobRunner,
    SUPPORTED_INPUT_EXTENSIONS,
    client_base_url,
    is_ocr_model,
    ocr_model_preset,
    ocr_projector_warning,
    server_model_ids,
)
from autotuner_version import VERSION, GITHUB_REPO, USER_AGENT
from model_benchmark import (
    BenchmarkCancelled,
    BenchmarkCandidate,
    BenchmarkFailure,
    BenchmarkLimits,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuiteJob,
    BenchmarkSuiteJobResult,
    BenchmarkSuiteResult,
    BenchmarkSuiteRunner,
    BENCHMARK_RECORD_SCHEMA,
    BENCHMARK_SEARCH_SCHEMA,
    baseline_candidate,
    shortlist_candidates_from_record,
)
from performance_report import write_performance_report
from theme_dialog import ThemeEditorDialog
from theme_manager import SYSTEM_THEME_ID, ThemeDefinition, ThemeManager
from localization import (
    setting_tooltip_html,
    translate as _tr,
    CUSTOM_LANGUAGE_ACTION,
    LanguageManager,
    LanguagePackError,
)
from single_instance import (
    ALLOW_MULTIPLE_ENV,
    SingleInstanceGuard,
    instance_key,
    multiple_instances_allowed,
)
from control_api import (
    ApiModel,
    ApiRuntime,
    ControlApiServer,
    ControlRequest,
    DISCOVERY_FILE_NAME,
    discovery_payload,
    write_discovery_file,
)


def _get_fork_tools():
    """Lazy import — never triggers auto_tuner.main()."""
    from auto_tuner import (
        _discover_llama_forks,
        _resolve_diffusion_binary,
        _resolve_server_binary,
    )

    return _discover_llama_forks, _resolve_server_binary, _resolve_diffusion_binary


def _bundled_resource(*parts: str) -> Path:
    """Return a source-tree or PyInstaller-bundled resource path."""
    return Path(__file__).resolve().parent.joinpath(*parts)


def _source_update_message(detail: str) -> str:
    """Keep source-update feedback as explicit about the running version as EXE updates."""
    return f"AutoTuner v{VERSION}: {detail}"


def _about_text() -> str:
    """Return static, network-free application information for the About dialog."""
    url = f"https://github.com/{GITHUB_REPO}"
    return f'<b>AutoTuner</b><br>Version v{VERSION}<br><a href="{url}">{url}</a>'


def _runtime_identity(value: str) -> str:
    """Return a stable absolute/case-normalized identity for one executable."""
    resolved = shutil.which(value) or value
    try:
        path = Path(resolved).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return os.path.normcase(str(resolved))
    return os.path.normcase(str(path))


@dataclass(frozen=True)
class _PerformanceRuntimeOption:
    """One explicitly selectable llama-server installation for a benchmark suite."""

    display_name: str
    binary: str
    root: Optional[Path]
    backend_hint: str = ""
    active: bool = False


_RUNTIME_BACKEND_ORDER = ("hip", "vulkan", "cuda", "metal", "sycl", "cpu")


def _benchmark_backend_key(
    runtime_binary: str,
    system: SystemInfo,
    build_name: str = "",
) -> str:
    """Classify the exact execution backend, preferring probed device evidence."""
    detected = {
        app_settings.normalise_performance_backend(gpu.runtime_backend)
        for gpu in system.gpus
        if gpu.runtime_backend
    }
    detected.discard("")
    if len(detected) == 1:
        return next(iter(detected))
    if detected:
        for backend in _RUNTIME_BACKEND_ORDER:
            if backend in detected:
                return backend
        return "other"

    identity = f"{build_name} {runtime_binary}".lower()
    aliases = (
        ("hip", ("_hip_", "-hip-", "rocm")),
        ("vulkan", ("_vulkan_", "-vulkan-", "vulkan")),
        ("cuda", ("_cuda_", "-cuda-", "cuda")),
        ("metal", ("_metal_", "-metal-", "metal")),
        ("sycl", ("_sycl_", "-sycl-", "oneapi")),
        ("cpu", ("_cpu_", "-cpu-", "cpu-only", "cpu_only")),
    )
    for backend, tokens in aliases:
        if any(token in identity for token in tokens):
            return backend
    return "cpu" if not system.gpus else "other"


def _benchmark_environment_fingerprint(
    runtime_binary: str,
    runtime_build: Optional[int],
    system: SystemInfo,
    baseline: TunedConfig,
) -> dict:
    """Describe every stable dimension that can invalidate a measured winner."""
    runtime_path = _runtime_identity(runtime_binary)
    try:
        runtime_stat = Path(runtime_path).stat()
        runtime_size = int(runtime_stat.st_size)
        runtime_mtime_ns = int(runtime_stat.st_mtime_ns)
    except OSError:
        runtime_size = 0
        runtime_mtime_ns = 0
    return {
        "search_schema": BENCHMARK_SEARCH_SCHEMA,
        "runtime_path": runtime_path,
        "runtime_build": int(runtime_build) if runtime_build is not None else None,
        "runtime_size": runtime_size,
        "runtime_mtime_ns": runtime_mtime_ns,
        "backend": _benchmark_backend_key(runtime_binary, system),
        "os": str(system.os_name),
        "cpu": str(system.cpu_name),
        "physical_cores": int(system.cpu_cores_physical),
        "logical_cores": int(system.cpu_cores_logical),
        "devices": [
            {
                "name": str(gpu.name),
                "backend": str(gpu.runtime_backend or ""),
                "device": str(gpu.runtime_device or ""),
                "vram_mb": int(gpu.total_vram_mb),
            }
            for gpu in system.gpus
        ],
        "quality": {
            "cache_k": str(baseline.cache_k),
            "cache_v": str(baseline.cache_v),
            "flash_attn": bool(baseline.flash_attn),
            "ctx": int(baseline.ctx),
            "ngl": int(baseline.ngl),
            "n_cpu_moe": (
                int(baseline.n_cpu_moe) if baseline.n_cpu_moe is not None else None
            ),
            "tensor_split": str(baseline.tensor_split or ""),
            "main_gpu": (
                int(baseline.main_gpu) if baseline.main_gpu is not None else None
            ),
            "no_kv_offload": bool(baseline.no_kv_offload),
            "rope_scaling": bool(baseline.rope_scaling),
            "rope_factor": float(baseline.rope_scale_factor),
            "parallel": 1,
        },
        "baseline_runtime": baseline_candidate(baseline).settings(),
    }


def _benchmark_limits_for_workload(
    benchmark_type: str, prompt_context_fraction: float
) -> BenchmarkLimits:
    """Return the bounded workload while keeping Quick's total runtime unlimited."""
    if benchmark_type == "fast":
        return BenchmarkLimits(
            max_candidates=7,
            confirmation_runs=1,
            samples_per_candidate=2,
            # A Quick suite may contain many slow models. Keep startup/request
            # safeguards but never abandon remaining candidates due to wall time.
            total_timeout_s=0.0,
            generated_tokens=128,
            min_prompt_tokens=1024,
            max_prompt_tokens=16384,
            prompt_context_fraction=prompt_context_fraction,
            max_draft_tokens=8,
        )
    custom = benchmark_type == "custom"
    return BenchmarkLimits(
        prompt_context_fraction=prompt_context_fraction,
        min_prompt_tokens=1 if custom else 4096,
        max_prompt_tokens=None if custom else 65536,
    )


def _application_theme_manager(
    app: QApplication, builtin_dir: Optional[Path] = None
) -> ThemeManager:
    """Return the single ThemeManager attached to this QApplication."""
    manager = getattr(app, "theme_manager", None)
    if isinstance(manager, ThemeManager):
        return manager
    manager = ThemeManager(builtin_dir)
    # QApplication is a C++ wrapper without arbitrary attributes in its stubs;
    # setattr/getattr deliberately retain this one process-wide manager.
    setattr(app, "theme_manager", manager)
    return manager


def _theme_replace_id(
    original: ThemeDefinition, edited: ThemeDefinition
) -> Optional[str]:
    """Authorize replacement only for the selected user theme's unchanged id."""
    if original.source == "user" and edited.id.casefold() == original.id.casefold():
        return original.id
    return None


def _setting_tooltip(summary: str, technical: str) -> str:
    """Build consistent two-level hover help for beginner and expert users.

    The English (UK) text is the translation key. ``LanguageManager`` takes
    the generated HTML apart again and translates both sections live, so
    keep the two arguments plain text (no markup).
    """
    return setting_tooltip_html(summary, technical)


def _open_local_folder(path: Path) -> bool:
    """Open *path* in the platform's file manager via Qt."""
    folder = path.expanduser().resolve(strict=False)
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


def _prepare_application_log() -> Optional[Path]:
    """Return a bounded persistent GUI log, avoiding home writes in tests."""
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get(
        "AUTOTUNER_DATA_DIR"
    ):
        return None
    path = app_settings.app_data_dir() / "logs" / "autotuner-app.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 4 * 1024 * 1024:
            previous = path.with_suffix(path.suffix + ".1")
            try:
                previous.unlink()
            except FileNotFoundError:
                pass
            path.replace(previous)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n=== AutoTuner v{VERSION} GUI "
                f"{datetime.now().isoformat(timespec='seconds')} ===\n"
            )
        return path
    except OSError:
        return None


_NATIVE_ICON_HANDLES: List[Tuple[int, int, int]] = []


def _set_windows_native_window_icon(window: QMainWindow, ico_path: Path) -> None:
    """Set HWND icons explicitly; Qt 6 can leave WM_GETICON empty when frozen."""
    if os.name != "nt" or not ico_path.is_file():
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.LoadImageW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        user32.LoadImageW.restype = ctypes.c_void_p
        user32.SendMessageW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        user32.SendMessageW.restype = ctypes.c_ssize_t

        hwnd = ctypes.c_void_p(int(window.winId()))
        image_icon = 1
        lr_load_from_file = 0x0010
        wm_seticon = 0x0080
        for icon_kind, size in ((1, 32), (0, 16)):
            handle = user32.LoadImageW(
                None,
                str(ico_path),
                image_icon,
                size,
                size,
                lr_load_from_file,
            )
            if handle:
                user32.SendMessageW(hwnd, wm_seticon, icon_kind, handle)
                _NATIVE_ICON_HANDLES.append(
                    (int(hwnd.value or 0), icon_kind, int(handle))
                )
    except Exception:
        pass


def _release_windows_native_icons() -> None:
    if os.name != "nt" or not _NATIVE_ICON_HANDLES:
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.SendMessageW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        user32.SendMessageW.restype = ctypes.c_ssize_t
        user32.DestroyIcon.argtypes = [ctypes.c_void_p]
        user32.DestroyIcon.restype = ctypes.c_int
        for hwnd, icon_kind, handle in _NATIVE_ICON_HANDLES:
            user32.SendMessageW(
                ctypes.c_void_p(hwnd), 0x0080, icon_kind, ctypes.c_void_p()
            )
            user32.DestroyIcon(ctypes.c_void_p(handle))
    except Exception:
        pass
    finally:
        _NATIVE_ICON_HANDLES.clear()


def _default_settings_path() -> Path:
    # When frozen (PyInstaller), ``__file__`` resolves into the throw-away
    # ``_MEIPASS`` extraction folder where bundled read-only resources live.
    return _bundled_resource("settings")


def _system_tray_supported() -> bool:
    """Return whether this desktop can host a notification-area icon.

    Windows and macOS always provide the native host. Qt may briefly report
    False there while Explorer/Finder is restarting; a QSystemTrayIcon created
    during that window is added automatically once the host returns. Linux
    desktop environments genuinely vary, so trust Qt's runtime probe there.
    """
    if sys.platform in ("win32", "darwin"):
        return True
    return QSystemTrayIcon.isSystemTrayAvailable()


def _default_models_path() -> Path:
    """Resolve default models folder.

    Preference order:
      1. Persisted choice (autotuner_settings.json)
      2. AUTOTUNER_MODELS environment variable
      3. <script_dir>/models or <script_dir>/../models if either exists
      4. <script_dir>/models (placeholder; user will be prompted)
    """
    saved = app_settings.get_models_path()
    if saved is not None:
        return saved
    env = os.environ.get("AUTOTUNER_MODELS", "")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    script_dir = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent
    )
    for c in (script_dir / "models", script_dir.parent / "models"):
        if c.exists():
            return c
    return script_dir / "models"


# ---------------------------------------------------------------------------
# Terminal process — spawns llama-server detached from the GUI


def _send_ctrl_break_to_new_console(pid: int) -> bool:
    """Best-effort CTRL_BREAK to a child that owns its own console (Windows).

    ``GenerateConsoleCtrlEvent`` only reaches process groups attached to the
    caller's console, and the GUI spawns llama-server with
    ``CREATE_NEW_CONSOLE``. A process without a console of its own (the
    frozen ``--windowed`` build) can attach to the child's console for the
    duration of the call. A process that owns a console (source run from a
    terminal) must not free it, so it keeps relying on the kill escalation.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        if kernel32.GetConsoleWindow():
            return False
        if not kernel32.AttachConsole(int(pid)):
            return False
        try:
            kernel32.SetConsoleCtrlHandler(None, True)
            return bool(kernel32.GenerateConsoleCtrlEvent(1, int(pid)))  # CTRL_BREAK
        finally:
            kernel32.FreeConsole()
            kernel32.SetConsoleCtrlHandler(None, False)
    except Exception:
        return False


class _ServerProbeWorker:
    """Run the readiness and slot probes of server records off the GUI thread.

    ``_poll_server`` used to call ``urlopen`` with 0.3-0.4 s timeouts every
    500 ms on Qt's thread, which stalled the window for up to that long per
    tick while a model was loading. Jobs are now queued here; results come
    back through a Qt signal, so the GUI only applies state changes.
    """

    def __init__(self, on_result: Callable[[dict], None]) -> None:
        self._on_result = on_result
        self._queue: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._pending: set = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="AutoTunerServerProbe", daemon=True
        )
        self._thread.start()

    def submit(self, job: dict) -> bool:
        """Queue one probe unless the same (token, kind) is still in flight."""
        key = (job.get("token"), job.get("kind"))
        with self._lock:
            if key in self._pending:
                return False
            self._pending.add(key)
        self._queue.put(job)
        return True

    def stop(self, timeout: float = 1.5) -> None:
        self._queue.put(None)
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @staticmethod
    def _probe(job: dict) -> dict:
        result = dict(job)
        url = str(job.get("url") or "").rstrip("/")
        if job.get("kind") == "slots":
            try:
                with urllib.request.urlopen(f"{url}/slots", timeout=0.4) as resp:
                    result["payload"] = json.loads(
                        resp.read().decode("utf-8", errors="replace") or "[]"
                    )
            except Exception as exc:
                result["error"] = str(exc)
            return result
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=0.3) as resp:
                result["ready"] = resp.status == 200
        except Exception:
            result["ready"] = False
        if result["ready"] and job.get("verify_models"):
            result["served_ids"] = server_model_ids(url, timeout_seconds=0.3)
        return result

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                result = self._probe(job)
            except Exception as exc:  # never let the worker die silently
                result = dict(job, error=str(exc), ready=False)
            finally:
                with self._lock:
                    self._pending.discard((job.get("token"), job.get("kind")))
            try:
                self._on_result(result)
            except Exception:
                pass


_PROBE_TOKENS = itertools.count(1)


class _TerminalProcess:
    """Spawn llama-server in an independent process group.

    Windows opens a separate console so the user sees native server output.
    On Linux/macOS the server output is streamed live: every line goes to a
    per-launch log file (recoverable after a crash), to our own stdout (so a
    terminal that launched the GUI shows tokens/s, prompt-processing progress
    and generation timings as before) and to an optional ``on_output``
    callback (the GUI log panel). Desktop launches without a terminal still
    have the log file + panel.
    """

    def __init__(
        self,
        cmd: List[str],
        env_overrides: Optional[dict] = None,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.cmd = cmd
        self.env_overrides: dict = env_overrides or {}
        self.on_output = on_output
        self.proc: Optional[subprocess.Popen] = None
        self.log_path: Optional[Path] = None
        self._stopped_event = threading.Event()
        self._stopped_event.set()

    # Stops whose signal/kill escalation is still running on a daemon thread.
    # main() waits for them before ending the process, so a Quit can never
    # leave llama-server behind with the model in VRAM.
    _pending_stops: "set[_TerminalProcess]" = set()
    _pending_lock = threading.Lock()

    @classmethod
    def wait_pending_stops(cls, timeout: float) -> List["_TerminalProcess"]:
        """Wait (bounded) for every in-flight stop; return those still pending."""
        deadline = time.monotonic() + timeout
        while True:
            with cls._pending_lock:
                pending = list(cls._pending_stops)
            if not pending or time.monotonic() >= deadline:
                return pending
            time.sleep(0.1)

    def _open_posix_log(self):
        log_dir = app_settings.app_data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.log_path = log_dir / f"llama-server-{stamp}.log"
        return self.log_path.open("a", encoding="utf-8", buffering=1)

    def start(self) -> None:
        env = os.environ.copy()
        if self.env_overrides:
            env.update(self.env_overrides)
        self._stopped_event.clear()
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
            try:
                self.proc = subprocess.Popen(self.cmd, creationflags=flags, env=env)
            except BaseException:
                self._stopped_event.set()
                raise
            self._watch_exit(self.proc)
            return
        log_fh = self._open_posix_log()
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                env=env,
            )
        except BaseException:
            self._stopped_event.set()
            log_fh.close()
            raise
        self._watch_exit(self.proc)
        threading.Thread(
            target=self._pump_output,
            args=(self.proc, log_fh),
            daemon=True,
        ).start()

    def _watch_exit(self, proc: subprocess.Popen) -> None:
        def watch() -> None:
            try:
                proc.wait()
            except Exception:
                return
            if self.proc is proc or self.proc is None:
                self._stopped_event.set()

        threading.Thread(target=watch, daemon=True).start()

    def _pump_output(self, proc: subprocess.Popen, log_fh) -> None:
        """Mirror every server line to log file, own stdout and callback.

        Runs on a daemon thread; reading a single merged pipe (stderr=STDOUT)
        avoids the two-pipe deadlock. The callback must be thread-safe — the
        GUI passes a pyqtSignal.emit, which queues into the Qt main thread.
        """
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                try:
                    log_fh.write(line)
                except (OSError, ValueError):
                    pass
                try:
                    # Frozen --windowed builds can have sys.stdout = None.
                    if sys.stdout is not None:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                except (OSError, ValueError):
                    pass
                if self.on_output is not None:
                    try:
                        self.on_output(line.rstrip("\n"))
                    except Exception:
                        pass
        finally:
            try:
                log_fh.close()
            except Exception:
                pass

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def returncode(self) -> Optional[int]:
        return self.proc.returncode if self.proc is not None else None

    def has_stopped(self) -> bool:
        """Whether the most recently started process has fully exited."""
        return self._stopped_event.is_set()

    def wait_stopped(self, timeout: Optional[float] = None) -> bool:
        return self._stopped_event.wait(timeout)

    def stop(self) -> None:
        """Non-blocking signal + background wait, observable via has_stopped()."""
        if self.proc is None:
            self._stopped_event.set()
            return
        try:
            if os.name == "nt":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                _send_ctrl_break_to_new_console(self.proc.pid)
            else:
                os.kill(-self.proc.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

        # Capture in a local variable BEFORE clearing self.proc — the daemon
        # thread runs after self.proc is already None.
        proc = self.proc
        assert proc is not None
        self.proc = None
        with _TerminalProcess._pending_lock:
            _TerminalProcess._pending_stops.add(self)

        def _wait() -> None:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "nt":
                        proc.kill()
                    else:
                        os.kill(-proc.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            finally:
                if proc.poll() is not None:
                    self._stopped_event.set()
                with _TerminalProcess._pending_lock:
                    _TerminalProcess._pending_stops.discard(self)

        threading.Thread(target=_wait, daemon=True).start()


class _PathListDialog(QDialog):
    """Small overlay dialog for enabling/disabling multiple folder roots."""

    def __init__(
        self,
        parent: QWidget,
        title: str,
        paths: List[Tuple[Path, bool]],
        pick_title: str,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._pick_title = pick_title
        self._updating_master = False
        self.resize(720, 420)

        layout = QVBoxLayout(self)
        self._master = QCheckBox("Enable all")
        self._master.setToolTip(
            _setting_tooltip(
                "Includes or excludes every saved folder from scanning in one step.",
                "This toggles each list item's enabled state without deleting its path. "
                "Only enabled roots are searched; pressing Cancel discards the dialog "
                "changes, while OK persists the resulting path/state pairs.",
            )
        )
        self._master.toggled.connect(self._toggle_all)
        layout.addWidget(self._master)

        self._list = QListWidget()
        self._list.setToolTip(
            _setting_tooltip(
                "Lists saved folders. Check a folder to scan it, or uncheck it to keep "
                "the path saved but temporarily ignore it.",
                "Paths are normalized and duplicate roots are removed. Model folders "
                "are scanned recursively for GGUF files; llama-build roots are scanned "
                "for runnable platform-compatible server binaries.",
            )
        )
        self._list.itemChanged.connect(self._sync_master)
        layout.addWidget(self._list, 1)

        buttons_row = QHBoxLayout()
        self._btn_add = QPushButton("Add…")
        self._btn_edit = QPushButton("Edit…")
        self._btn_remove = QPushButton("Remove")
        self._btn_add.setToolTip(
            _setting_tooltip(
                "Adds another folder to this list and enables it immediately.",
                "The selected directory is resolved to an absolute path, deduplicated "
                "against existing entries, and persisted only after OK is pressed.",
            )
        )
        self._btn_edit.setToolTip(
            _setting_tooltip(
                "Replaces the selected entry with a different folder.",
                "The replacement keeps the item's enabled state, stores a resolved "
                "absolute path, and removes any duplicate created by the change.",
            )
        )
        self._btn_remove.setToolTip(
            _setting_tooltip(
                "Removes the selected folder from the saved list.",
                "This removes only AutoTuner's path entry; it never deletes the folder, "
                "models, llama.cpp build, or any files on disk.",
            )
        )
        self._btn_add.clicked.connect(self._add_path)
        self._btn_edit.clicked.connect(self._edit_path)
        self._btn_remove.clicked.connect(self._remove_path)
        for btn in (self._btn_add, self._btn_edit, self._btn_remove):
            buttons_row.addWidget(btn)
        buttons_row.addStretch()
        layout.addLayout(buttons_row)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

        for path, enabled in paths:
            self._append_item(path, enabled)
        self._sync_master()

    def _append_item(self, path: Path, enabled: bool = True) -> None:
        item = QListWidgetItem(str(path))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(
            Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked
        )
        item.setData(Qt.ItemDataRole.UserRole, Path(path))
        self._list.addItem(item)

    def _pick_directory(self, start: Optional[Path]) -> Optional[Path]:
        directory = str(start) if start is not None else str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, self._pick_title, directory)
        return Path(folder).resolve() if folder else None

    def _add_path(self) -> None:
        selected = self._pick_directory(None)
        if selected is not None:
            self._append_item(selected, True)
            self._dedupe_items()
            self._sync_master()

    def _edit_path(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        current = item.data(Qt.ItemDataRole.UserRole)
        start = current if isinstance(current, Path) else Path(str(item.text()))
        selected = self._pick_directory(start)
        if selected is None:
            return
        item.setText(str(selected))
        item.setData(Qt.ItemDataRole.UserRole, selected)
        self._dedupe_items()
        self._sync_master()

    def _remove_path(self) -> None:
        row = self._list.currentRow()
        if row >= 0:
            self._list.takeItem(row)
            self._sync_master()

    def _toggle_all(self, checked: bool) -> None:
        if self._updating_master:
            return
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )

    def _sync_master(self) -> None:
        total = self._list.count()
        enabled = sum(
            1
            for i in range(total)
            if self._list.item(i).checkState() == Qt.CheckState.Checked
        )
        self._updating_master = True
        try:
            self._master.setChecked(total > 0 and enabled == total)
            if total == 0:
                self._master.setText(_tr("Enable all"))
            elif enabled == total:
                self._master.setText(f"{_tr('All enabled')} ({enabled}/{total})")
            elif enabled == 0:
                self._master.setText(f"{_tr('All disabled')} (0/{total})")
            else:
                self._master.setText(f"{_tr('Partially enabled')} ({enabled}/{total})")
        finally:
            self._updating_master = False

    def _dedupe_items(self) -> None:
        seen: set[str] = set()
        for i in reversed(range(self._list.count())):
            item = self._list.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            try:
                key = os.path.normcase(str(Path(path).resolve(strict=False)))
            except (OSError, RuntimeError, TypeError, ValueError):
                key = os.path.normcase(item.text())
            if key in seen:
                self._list.takeItem(i)
            else:
                seen.add(key)

    def paths(self) -> List[Tuple[Path, bool]]:
        out: List[Tuple[Path, bool]] = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(path, Path):
                path = Path(str(item.text()))
            out.append((path, item.checkState() == Qt.CheckState.Checked))
        return out


class _ApplicationSettingsDialog(QDialog):
    """Small application-level settings dialog (all options are opt-in)."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMaximumWidth(800)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Appearance and application behaviour for this installation. Startup and "
            "notification-area options are disabled by default."
        )
        intro.setWordWrap(True)
        intro.setMaximumWidth(760)
        layout.addWidget(intro)

        appearance = QGroupBox("Appearance")
        appearance.setMaximumWidth(760)
        appearance_layout = QGridLayout(appearance)
        theme_label = QLabel("Theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.setMinimumWidth(160)
        self.theme_combo.setMaximumWidth(500)
        self.theme_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.theme_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.theme_combo.setAccessibleName("Theme")
        theme_label.setBuddy(self.theme_combo)
        self.theme_combo.setToolTip(
            _setting_tooltip(
                "Previews and selects the application's color and font theme.",
                "Built-ins stay read-only; edited built-ins are saved as new '-user' themes, while selected user themes can be updated in the per-installation theme folder.",
            )
        )
        app = cast(Optional[QApplication], QApplication.instance())
        if app is not None:
            manager = _application_theme_manager(app)
            widest = 0
            for theme in manager.available():
                text = f"{theme.name} ({theme.source})"
                self.theme_combo.addItem(text, theme.qualified_id)
                self.theme_combo.setItemData(
                    self.theme_combo.count() - 1, text, Qt.ItemDataRole.ToolTipRole
                )
                widest = max(
                    widest, self.theme_combo.fontMetrics().horizontalAdvance(text)
                )
            self.theme_combo.setMinimumWidth(min(500, max(160, widest + 48)))
            index = self.theme_combo.findData(manager.current_id)
            self.theme_combo.setCurrentIndex(max(0, index))
        self.reload_themes_button = QPushButton("Reload")
        self.customize_theme_button = QPushButton("Customise…")
        self.open_themes_button = QPushButton("Open folder")
        self.about_button = QPushButton("About AutoTuner")
        self.about_button.setAccessibleName("About AutoTuner")
        self.about_button.setToolTip(
            _setting_tooltip(
                "Shows the running AutoTuner version and GitHub project page.",
                "The dialog is local and does not contact GitHub; use Update to check for newer source or release versions.",
            )
        )
        self.reload_themes_button.setToolTip(
            _setting_tooltip(
                "Reloads JSON files copied into the user theme folder.",
                "Invalid files are ignored and listed in a warning; the selected valid theme is reapplied.",
            )
        )
        self.customize_theme_button.setToolTip(
            _setting_tooltip(
                "Opens the selected theme in the safe color and font editor.",
                "Built-ins are copied to a draft with '-user' appended to the ID and name; Save validates it. Saving an existing user theme with the same ID atomically updates it.",
            )
        )
        self.open_themes_button.setToolTip(
            _setting_tooltip(
                "Opens the folder used for your own themes.",
                "Files in this persistent folder survive source updates and compiled-binary swaps; click Reload after copying a JSON file.",
            )
        )
        appearance_layout.addWidget(theme_label, 0, 0)
        appearance_layout.addWidget(self.theme_combo, 0, 1)
        appearance_layout.addWidget(self.reload_themes_button, 1, 0, 1, 2)
        appearance_layout.addWidget(self.customize_theme_button, 2, 0, 1, 2)
        appearance_layout.addWidget(self.open_themes_button, 3, 0, 1, 2)
        appearance_layout.addWidget(self.about_button, 4, 0, 1, 2)
        layout.addWidget(appearance)

        self.autostart_checkbox = QCheckBox("Start after login")
        self.autostart_was_enabled = startup_manager.is_autostart_enabled()
        self.autostart_checkbox.setChecked(self.autostart_was_enabled)
        self.autostart_checkbox.setToolTip(
            _setting_tooltip(
                "Opens AutoTuner automatically after you sign in, so it is ready "
                "without a manual launch.",
                "This creates a per-user startup entry only; it does not install a "
                "service and needs no administrator rights. Windows uses HKCU Run, "
                "Linux an XDG autostart file, and macOS a user LaunchAgent. Turning "
                "the option off removes that entry.",
            )
        )
        layout.addWidget(self.autostart_checkbox)

        self.start_minimized_checkbox = QCheckBox("Start minimized after login")
        self.start_minimized_checkbox.setChecked(
            app_settings.get_start_minimized_at_login()
        )
        # Until the user saves an explicit choice, the option mirrors
        # "Hide on close" so both login options together start in the tray.
        self.start_minimized_explicit = (
            app_settings.start_minimized_at_login_is_explicit()
        )
        self.start_minimized_touched = False
        self.start_minimized_checkbox.clicked.connect(self._mark_start_minimized)
        self.start_minimized_checkbox.setEnabled(self.autostart_was_enabled)
        self.autostart_checkbox.toggled.connect(
            self.start_minimized_checkbox.setEnabled
        )
        self.start_minimized_checkbox.setToolTip(
            _setting_tooltip(
                "When AutoTuner starts because you signed in, it opens without a "
                "window: in the notification area if Hide on close is on, "
                "otherwise minimized on the taskbar.",
                "Applies only to the login entry, which passes --autostart; manual "
                "starts always show the window. Until you save a choice here it "
                "follows Hide on close. Servers, the control API and background "
                "scans start as usual; restore the window from the tray icon, the "
                "taskbar, or by launching AutoTuner again.",
            )
        )
        start_minimized_row = QHBoxLayout()
        start_minimized_row.setContentsMargins(24, 0, 0, 0)
        start_minimized_row.addWidget(self.start_minimized_checkbox)
        start_minimized_row.addStretch(1)
        layout.addLayout(start_minimized_row)

        self.minimize_checkbox = QCheckBox("Hide on close")
        tray_available = _system_tray_supported()
        self.minimize_checkbox.setChecked(
            app_settings.get_minimize_on_close() and tray_available
        )
        self.minimize_checkbox.setEnabled(tray_available)
        if tray_available:
            self.minimize_checkbox.setToolTip(
                _setting_tooltip(
                    "The window disappears when you click X, but AutoTuner and its "
                    "servers keep running in the notification area.",
                    "The close event is redirected to QSystemTrayIcon.hide() instead "
                    "of ending the process. Restore the window from the tray icon; "
                    "use its Quit action or AutoTuner's Quit button for a full, "
                    "graceful shutdown of managed servers.",
                )
            )
        else:
            self.minimize_checkbox.setText("Hide on close (unavailable)")
            self.minimize_checkbox.setToolTip(
                _setting_tooltip(
                    "This option cannot be used because no notification area was "
                    "detected on this desktop.",
                    "Qt reports that QSystemTrayIcon is unavailable, so hiding the "
                    "window would leave no reliable way to restore it. AutoTuner "
                    "therefore disables the setting and keeps normal X-to-close "
                    "behaviour.",
                )
            )
        layout.addWidget(self.minimize_checkbox)
        self.minimize_checkbox.toggled.connect(self._follow_hide_on_close)

        self.debug_checkbox = QCheckBox("Debug mode (verbose AutoTuner log)")
        self.debug_checkbox.setChecked(app_settings.get_debug_mode())
        self.debug_checkbox.setToolTip(
            _setting_tooltip(
                "Records extra AutoTuner discovery and configuration details for support.",
                "This affects AutoTuner's own debug categories and rotating application "
                "log only. It does not enable llama-server --verbose, capture prompts or "
                "API payloads, or include credentials. A restart is not required.",
            )
        )
        layout.addWidget(self.debug_checkbox)

        control_api = QGroupBox("External control API")
        control_api.setMaximumWidth(760)
        api_layout = QGridLayout(control_api)
        api_help = QLabel(
            "Use this loopback API to list AutoTuner models, switch the active "
            "model with saved settings, and proxy OpenAI-compatible requests."
        )
        api_help.setWordWrap(True)
        api_layout.addWidget(api_help, 0, 0, 1, 3)

        self.control_api_checkbox = QCheckBox(
            "Enable local OpenAI-compatible control API"
        )
        self.control_api_checkbox.setChecked(app_settings.get_control_api_enabled())
        self.control_api_checkbox.setToolTip(
            _setting_tooltip(
                "Lets trusted programs on this computer select and use your "
                "AutoTuner models through one stable OpenAI-compatible address.",
                "The gateway binds only to 127.0.0.1, requires a cryptographically "
                "random bearer token, serializes model switches, reuses saved launch "
                "settings, and stops only the previous API-managed model.",
            )
        )
        api_layout.addWidget(self.control_api_checkbox, 1, 0, 1, 3)

        api_port_label = QLabel("Port:")
        self.control_api_port = QSpinBox()
        self.control_api_port.setRange(1024, 65535)
        self.control_api_port.setValue(app_settings.get_control_api_port())
        self.control_api_port.setToolTip(
            _setting_tooltip(
                "Chooses the local port used by external clients.",
                "The listener is always loopback-only. The default is 1233 so it "
                "does not collide with llama-server's default base port 1234.",
            )
        )
        api_port_label.setBuddy(self.control_api_port)
        api_layout.addWidget(api_port_label, 2, 0)
        api_layout.addWidget(self.control_api_port, 2, 1)

        self.control_api_endpoint = QLabel()
        self.control_api_endpoint.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        api_layout.addWidget(self.control_api_endpoint, 2, 2)

        api_key_label = QLabel("API key:")
        initial_token = app_settings.get_control_api_token() or secrets.token_urlsafe(
            32
        )
        self.control_api_token = QLineEdit(initial_token)
        self.control_api_token.setReadOnly(True)
        self.control_api_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.control_api_token.setAccessibleName("API key")
        self.control_api_token.setToolTip(
            _setting_tooltip(
                "The secret that authorised local clients must send as a bearer token.",
                "It contains at least 256 bits of random material and is stored in the "
                "per-user settings file. Regenerating it immediately invalidates old "
                "client configurations after you accept this dialog.",
            )
        )
        api_key_label.setBuddy(self.control_api_token)
        api_layout.addWidget(api_key_label, 3, 0)
        api_layout.addWidget(self.control_api_token, 3, 1, 1, 2)

        api_buttons = QHBoxLayout()
        self.control_api_copy_key_button = QPushButton("Copy")
        self.control_api_regenerate_button = QPushButton("Regenerate")
        self.control_api_copy_pi_button = QPushButton("Copy Pi setup")
        self.control_api_copy_key_button.setToolTip(
            _setting_tooltip(
                "Copies only the current API key to the clipboard.",
                "The clipboard contains a credential until another application "
                "replaces it; paste it only into a trusted local client.",
            )
        )
        self.control_api_regenerate_button.setToolTip(
            _setting_tooltip(
                "Creates a new random API key in this dialog.",
                "The old key remains active until OK is clicked. Environment-owned "
                "keys cannot be regenerated here.",
            )
        )
        self.control_api_copy_pi_button.setToolTip(
            _setting_tooltip(
                "Copies the endpoint and key names used by AutoTuner's Pi extension.",
                "The two newline-separated environment assignments are "
                "AUTOTUNER_API_URL and AUTOTUNER_API_KEY; the copied key is a secret.",
            )
        )
        self.control_api_copy_key_button.clicked.connect(self._copy_api_key)
        self.control_api_regenerate_button.clicked.connect(self._regenerate_api_key)
        self.control_api_copy_pi_button.clicked.connect(self._copy_pi_setup)
        api_buttons.addWidget(self.control_api_copy_key_button)
        api_buttons.addWidget(self.control_api_regenerate_button)
        api_buttons.addWidget(self.control_api_copy_pi_button)
        api_layout.addLayout(api_buttons, 4, 0, 1, 3)
        if app_settings.control_api_token_is_overridden():
            self.control_api_regenerate_button.setEnabled(False)
            self.control_api_token.setToolTip(
                self.control_api_token.toolTip()
                + "<br><br>AUTOTUNER_CONTROL_API_KEY currently overrides this value."
            )
        self.control_api_port.valueChanged.connect(self._update_api_endpoint)
        self._update_api_endpoint()
        layout.addWidget(control_api)

        profiles = QGroupBox("Performance profiles")
        profiles.setMaximumWidth(760)
        profile_layout = QVBoxLayout(profiles)
        profile_help = QLabel(
            "Export or import measured per-model/performance-mode profiles. The "
            "portable bundle matches moved models by GGUF filename and file size; "
            "appearance, paths, ports, and unrelated application settings are excluded."
        )
        profile_help.setWordWrap(True)
        profile_layout.addWidget(profile_help)
        profile_buttons = QHBoxLayout()
        self.export_profiles_button = QPushButton("Export profiles…")
        self.import_profiles_button = QPushButton("Import profiles…")
        self.export_profiles_button.setToolTip(
            _setting_tooltip(
                "Saves measured tuning profiles to a portable JSON backup.",
                "Exports validated Expert snapshots and benchmark evidence for all "
                "four performance targets. Atomic writing prevents partial bundles.",
            )
        )
        self.import_profiles_button.setToolTip(
            _setting_tooltip(
                "Restores performance profiles from an AutoTuner JSON backup.",
                "Strictly validates the bundle schema, merges valid target profiles, "
                "and maps local GGUFs by exact filename plus byte size.",
            )
        )
        profile_buttons.addWidget(self.export_profiles_button)
        profile_buttons.addWidget(self.import_profiles_button)
        profile_layout.addLayout(profile_buttons)
        layout.addWidget(profiles)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _mark_start_minimized(self, _checked: bool = False) -> None:
        """Remember that the user chose the login-start option directly."""
        self.start_minimized_touched = True

    def _follow_hide_on_close(self, checked: bool) -> None:
        """Mirror Hide on close until the login-start option is chosen."""
        if not (self.start_minimized_explicit or self.start_minimized_touched):
            self.start_minimized_checkbox.setChecked(checked)

    def _update_api_endpoint(self) -> None:
        self.control_api_endpoint.setText(
            f"http://127.0.0.1:{self.control_api_port.value()}/v1"
        )

    def _copy_api_key(self) -> None:
        app = cast(Optional[QApplication], QApplication.instance())
        if app is not None:
            app.clipboard().setText(self.control_api_token.text())

    def _regenerate_api_key(self) -> None:
        if app_settings.control_api_token_is_overridden():
            return
        self.control_api_token.setText(secrets.token_urlsafe(32))

    def _copy_pi_setup(self) -> None:
        app = cast(Optional[QApplication], QApplication.instance())
        if app is None:
            return
        app.clipboard().setText(
            f"AUTOTUNER_API_URL=http://127.0.0.1:{self.control_api_port.value()}\n"
            f"AUTOTUNER_API_KEY={self.control_api_token.text()}"
        )

    def sizeHint(self) -> QSize:  # noqa: N802
        """Keep the settings dialog bounded across platform font metrics."""
        hint = super().sizeHint()
        return QSize(min(hint.width(), 800), hint.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        """Allow layouts to compress to the same documented 800px bound."""
        hint = super().minimumSizeHint()
        return QSize(min(hint.width(), 800), hint.height())


# ---------------------------------------------------------------------------
# Shared OCR setup/progress UI


class _OcrSetupDialog(QDialog):
    """Collect document inputs and the shared OCR pipeline options."""

    def __init__(self, model: ModelEntry, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.model = model
        self.preset = ocr_model_preset(model)
        self.setWindowTitle(f"OCR – {self.preset.label}")
        self.setModal(True)
        self.resize(760, 660)

        layout = QVBoxLayout(self)
        title = QLabel(
            f"<b>{escape(model.name)}</b><br>"
            "Images are normalized directly; PDF pages are rendered locally; "
            "Word/Office files are converted through LibreOffice first."
        )
        title.setWordWrap(True)
        layout.addWidget(title)

        if self.preset.notes:
            note = QLabel(self.preset.notes)
            note.setWordWrap(True)
            note.setObjectName("mutedLabel")
            layout.addWidget(note)

        input_group = QGroupBox("Input files and folders")
        input_layout = QVBoxLayout(input_group)
        self.input_list = QListWidget()
        self.input_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.input_list.setAccessibleName("OCR input files and folders")
        input_layout.addWidget(self.input_list, 1)
        input_buttons = QHBoxLayout()
        add_files = QPushButton("Add files…")
        add_folder = QPushButton("Add folder…")
        remove = QPushButton("Remove selected")
        clear = QPushButton("Clear")
        add_files.clicked.connect(self._add_files)
        add_folder.clicked.connect(self._add_folder)
        remove.clicked.connect(self._remove_selected)
        clear.clicked.connect(self.input_list.clear)
        for button in (add_files, add_folder, remove, clear):
            input_buttons.addWidget(button)
        input_buttons.addStretch(1)
        input_layout.addLayout(input_buttons)
        layout.addWidget(input_group, 1)

        options_group = QGroupBox("Output and OCR options")
        grid = QGridLayout(options_group)
        self.output_edit = QLineEdit(str(self._default_output_dir()))
        self.output_edit.setAccessibleName("OCR output folder")
        browse_output = QPushButton("Browse…")
        browse_output.clicked.connect(self._browse_output)
        grid.addWidget(QLabel("Output folder:"), 0, 0)
        grid.addWidget(self.output_edit, 0, 1, 1, 3)
        grid.addWidget(browse_output, 0, 4)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlainText(self.preset.prompt)
        self.prompt_edit.setMaximumHeight(72)
        self.prompt_edit.setAccessibleName("OCR prompt")
        grid.addWidget(QLabel("Prompt:"), 1, 0, Qt.AlignmentFlag.AlignTop)
        grid.addWidget(self.prompt_edit, 1, 1, 1, 4)

        self.pages_edit = QLineEdit()
        self.pages_edit.setPlaceholderText("all pages, or e.g. 1-3,5")
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(220)
        self.dpi_spin.setSuffix(" DPI")
        self.tokens_spin = QSpinBox()
        self.tokens_spin.setRange(1, 32768)
        self.tokens_spin.setValue(self.preset.max_tokens)
        self.tokens_spin.setSingleStep(512)
        self.format_combo = QComboBox()
        self.format_combo.addItem("Markdown (.md)", "markdown")
        self.format_combo.addItem("Plain text (.txt)", "text")
        grid.addWidget(QLabel("PDF pages:"), 2, 0)
        grid.addWidget(self.pages_edit, 2, 1)
        grid.addWidget(QLabel("Render:"), 2, 2)
        grid.addWidget(self.dpi_spin, 2, 3)
        grid.addWidget(QLabel("Max tokens/page:"), 3, 0)
        grid.addWidget(self.tokens_spin, 3, 1)
        grid.addWidget(QLabel("Format:"), 3, 2)
        grid.addWidget(self.format_combo, 3, 3)

        self.keep_rendered = QCheckBox("Keep rendered/normalized page PNG files")
        self.keep_grounding = QCheckBox(
            "Keep model grounding tags and bounding-box coordinates"
        )
        self.keep_grounding.setChecked(not self.preset.strip_grounding)
        self.stop_server = QCheckBox("Stop the OCR server when the job finishes")
        self.stop_server.setChecked(True)
        grid.addWidget(self.keep_rendered, 4, 0, 1, 5)
        grid.addWidget(self.keep_grounding, 5, 0, 1, 5)
        grid.addWidget(self.stop_server, 6, 0, 1, 5)
        layout.addWidget(options_group)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        start_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        start_button.setText("Start OCR")
        self.buttons.accepted.connect(self._validate_and_accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    @staticmethod
    def _default_output_dir() -> Path:
        documents = Path.home() / "Documents"
        root = documents if documents.is_dir() else Path.home()
        return root / "AutoTuner-OCR"

    def _add_path(self, path: str) -> None:
        normalized = os.path.normcase(str(Path(path).expanduser()))
        for index in range(self.input_list.count()):
            current = os.path.normcase(self.input_list.item(index).text())
            if current == normalized:
                return
        self.input_list.addItem(str(Path(path).expanduser()))

    def _add_files(self) -> None:
        extensions = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_INPUT_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select OCR documents",
            str(Path.home()),
            f"OCR documents ({extensions});;All files (*)",
        )
        for path in files:
            self._add_path(path)

    def _add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select OCR input folder", str(Path.home())
        )
        if path:
            self._add_path(path)

    def _remove_selected(self) -> None:
        for item in self.input_list.selectedItems():
            self.input_list.takeItem(self.input_list.row(item))

    def _browse_output(self) -> None:
        current = self.output_edit.text().strip() or str(self._default_output_dir())
        path = QFileDialog.getExistingDirectory(
            self, "Select OCR output folder", current
        )
        if path:
            self.output_edit.setText(path)

    def _validate_and_accept(self) -> None:
        if self.input_list.count() == 0:
            QMessageBox.warning(self, "OCR", "Add at least one input file or folder.")
            return
        if not self.output_edit.text().strip():
            QMessageBox.warning(self, "OCR", "Choose an output folder.")
            return
        if not self.prompt_edit.toPlainText().strip() and self.preset.prompt:
            answer = QMessageBox.question(
                self,
                "Empty OCR prompt",
                "The prompt is empty. Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.accept()

    def options(self) -> OcrJobOptions:
        return OcrJobOptions(
            inputs=[
                Path(self.input_list.item(index).text())
                for index in range(self.input_list.count())
            ],
            output_dir=Path(self.output_edit.text().strip()),
            prompt=self.prompt_edit.toPlainText().strip(),
            max_tokens=self.tokens_spin.value(),
            dpi=self.dpi_spin.value(),
            page_range=self.pages_edit.text().strip(),
            output_format=str(self.format_combo.currentData()),
            keep_rendered=self.keep_rendered.isChecked(),
            strip_grounding=not self.keep_grounding.isChecked(),
            stop_server_when_done=self.stop_server.isChecked(),
        ).normalized()


class _OcrProgressDialog(QDialog):
    cancel_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AutoTuner OCR")
        self.setModal(False)
        self.setMinimumWidth(560)
        self._finished = False
        layout = QVBoxLayout(self)
        self.status_label = QLabel("Starting llama-server and loading the OCR model…")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)
        self.cancel_button = QPushButton("Cancel OCR")
        self.cancel_button.clicked.connect(self._request_cancel)
        layout.addWidget(self.cancel_button, 0, Qt.AlignmentFlag.AlignRight)

    def _request_cancel(self) -> None:
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancelling OCR and releasing resources…")
        self.cancel_requested.emit()

    def update_progress(
        self, _stage: str, current: int, total: int, message: str
    ) -> None:
        self.status_label.setText(message)
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        else:
            self.progress_bar.setRange(0, 0)

    def mark_finished(self) -> None:
        self._finished = True
        self.cancel_button.setEnabled(False)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self._finished:
            self._request_cancel()
            event.ignore()
            return
        super().closeEvent(event)


class _OcrPrepareWorker(QObject):
    progress = pyqtSignal(str, int, int, str)
    prepared = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, runner: OcrJobRunner) -> None:
        super().__init__()
        self.runner = runner

    def run(self) -> None:
        try:
            self.runner.progress = self.progress.emit
            self.runner.prepare()
            self.prepared.emit(self.runner)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self.runner.cancel()


class _OcrWorker(QObject):
    progress = pyqtSignal(str, int, int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, runner: OcrJobRunner) -> None:
        super().__init__()
        self.runner = runner

    def run(self) -> None:
        try:
            self.finished.emit(self.runner.run())
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self.runner.cancel()


class _PerformanceTuneSetupDialog(QDialog):
    """Collect scope and optional expensive axes for a performance suite."""

    def __init__(
        self,
        model_name: str,
        default_context: int,
        maximum_context: int,
        model_count: int,
        parent: Optional[QWidget] = None,
        *,
        runtime_options: Optional[Sequence[_PerformanceRuntimeOption]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Performance test options")
        self.setModal(True)
        self.setMinimumWidth(620)
        self._runtime_options = list(runtime_options or [])
        layout = QVBoxLayout(self)

        intro = QLabel(
            f"Tune <b>{escape(model_name)}</b>. Every selected performance mode "
            "gets its own saved Perform profile. By default, every model, llama "
            "build, and performance mode uses its own safe maximum context, so a "
            "smaller next model cannot inherit an unsafe context from the first. "
            "Quick pass uses at most 3.125% context and a 128-token decode window "
            "without a global runtime cutoff, so every selected model and mode can "
            "finish. Standard uses 12.5% (capped at 65,536 prompt tokens); Custom "
            "accepts 0.01–100% without that cap."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        context_group = QGroupBox("Context and validation")
        context_layout = QGridLayout(context_group)
        context_layout.addWidget(QLabel("Context policy:"), 0, 0)
        self.context_mode_combo = QComboBox()
        self.context_mode_combo.addItem(
            "Safe maximum for each model / build / mode (recommended)", "adaptive"
        )
        self.context_mode_combo.addItem("One fixed context (advanced)", "fixed")
        self.context_mode_combo.setToolTip(
            _setting_tooltip(
                "Adaptive mode prevents a large first model or Safe profile from "
                "forcing the same context onto a smaller model or Throughput profile.",
                "For every suite job AutoTuner recomputes placement, KV precision, "
                "runtime buffers, native/YaRN limits, and the safe context against "
                "the exact selected llama build and its detected backend.",
            )
        )
        context_layout.addWidget(self.context_mode_combo, 0, 1)
        context_layout.addWidget(QLabel("Fixed context:"), 1, 0)
        self.context_spin = QSpinBox()
        self.context_spin.setRange(512, max(512, min(10_000_000, maximum_context)))
        self.context_spin.setSingleStep(1024)
        self.context_spin.setValue(
            max(512, min(default_context, self.context_spin.maximum()))
        )
        self.context_spin.setEnabled(False)
        self.context_spin.setToolTip(
            "Used only in fixed-context mode. Each model is still clamped to its "
            "native or explicitly enabled YaRN limit."
        )
        context_layout.addWidget(self.context_spin, 1, 1)
        context_layout.addWidget(QLabel("Test workload:"), 2, 0)
        self.test_length_combo = QComboBox()
        self.test_length_combo.addItem(
            "Quick pass — ≤3.125% context (recommended)", "fast"
        )
        self.test_length_combo.addItem("Standard test — 12.5% context", "quick")
        self.test_length_combo.addItem("Custom context percentage", "custom")
        self.test_length_combo.setToolTip(
            _setting_tooltip(
                "Quick pass is the default short search; Standard validates the "
                "12.5% workload; Custom uses the exact percentage below.",
                "Quick pass uses at most 16,384 prompt tokens and 128 decode tokens "
                "with no overall model/suite deadline. Startup and individual HTTP "
                "timeouts still detect stuck servers. Standard uses a 65,536 prompt "
                "cap. Custom values from 0.01% through 100% deliberately ignore it.",
            )
        )
        context_layout.addWidget(self.test_length_combo, 2, 1)
        context_layout.addWidget(QLabel("Custom context:"), 3, 0)
        self.custom_percent_spin = QDoubleSpinBox()
        self.custom_percent_spin.setRange(0.01, 100.0)
        self.custom_percent_spin.setDecimals(2)
        self.custom_percent_spin.setSingleStep(0.01)
        self.custom_percent_spin.setSuffix(" %")
        self.custom_percent_spin.setValue(25.0)
        self.custom_percent_spin.setEnabled(False)
        self.custom_percent_spin.setToolTip(
            "Custom prompt fraction. Its prompt is limited only by the selected "
            "context minus the 256-token decode window and safety margin."
        )
        context_layout.addWidget(self.custom_percent_spin, 3, 1)
        self.test_length_combo.currentIndexChanged.connect(
            self._update_test_option_states
        )
        self.context_mode_combo.currentIndexChanged.connect(
            self._update_context_option_states
        )
        self.real_validation = QCheckBox(
            "Try the exact fixed context in the isolated test server"
        )
        self.real_validation.setChecked(False)
        self.real_validation.setToolTip(
            "Fixed mode normally clamps the request to the static safe estimate. "
            "This advanced option lets the private benchmark server try the exact "
            "fixed value above that estimate; a bounded allocation failure affects "
            "only that job and the suite continues."
        )
        context_layout.addWidget(self.real_validation, 4, 0, 1, 2)
        self.enable_yarn = QCheckBox("Enable YaRN when context exceeds native context")
        self.enable_yarn.setChecked(False)
        context_layout.addWidget(self.enable_yarn, 5, 0, 1, 2)
        layout.addWidget(context_group)

        builds_group = QGroupBox("llama builds (backend-specific profiles)")
        builds_layout = QVBoxLayout(builds_group)
        self.runtime_scope_combo = QComboBox()
        self.runtime_scope_combo.addItem(
            "Test only the active selected build", "active"
        )
        self.runtime_scope_combo.addItem(
            "Choose one or more installed builds", "multiple"
        )
        self.runtime_scope_combo.setEnabled(len(self._runtime_options) > 1)
        builds_layout.addWidget(self.runtime_scope_combo)
        self.runtime_list = QListWidget()
        self.runtime_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.runtime_list.setMaximumHeight(150)
        for index, option in enumerate(self._runtime_options):
            backend_label = app_settings.performance_backend_label(option.backend_hint)
            label = option.display_name
            if option.backend_hint:
                label = f"{backend_label} · {label}"
            if option.active:
                label += " · active"
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if option.active else Qt.CheckState.Unchecked
            )
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setToolTip(option.binary)
            self.runtime_list.addItem(item)
        if self._runtime_options and not any(
            option.active for option in self._runtime_options
        ):
            self.runtime_list.item(0).setCheckState(Qt.CheckState.Checked)
        self.runtime_list.setEnabled(False)
        builds_layout.addWidget(self.runtime_list)
        self.runtime_scope_combo.currentIndexChanged.connect(
            self._update_runtime_option_states
        )
        builds_help = QLabel(
            "HIP, Vulkan, CPU, and other execution backends are measured "
            "independently. Results are saved as separate Perform profiles and "
            "are never reused across backends."
        )
        builds_help.setWordWrap(True)
        builds_layout.addWidget(builds_help)
        layout.addWidget(builds_group)

        modes_group = QGroupBox("Performance modes (independent saved profiles)")
        modes_layout = QGridLayout(modes_group)
        self.mode_checks: Dict[str, QCheckBox] = {}
        for index, name in enumerate(list_target_names()):
            check = QCheckBox(name)
            check.setChecked(True)
            check.setToolTip(PERFORMANCE_TARGETS[name].description)
            self.mode_checks[name] = check
            modes_layout.addWidget(check, index // 2, index % 2)
        layout.addWidget(modes_group)

        scope_group = QGroupBox("Optional extended search")
        scope_layout = QVBoxLayout(scope_group)
        self.tune_mtp = QCheckBox("Tune MTP/draft n-max (adds draft-depth runs)")
        self.tune_mtp.setChecked(False)
        scope_layout.addWidget(self.tune_mtp)
        self.test_external_drafters = QCheckBox(
            "Test every compatible external drafter (Q4/Q8/BF16 etc.)"
        )
        self.test_external_drafters.setChecked(False)
        self.test_external_drafters.setEnabled(False)
        self.tune_mtp.toggled.connect(self.test_external_drafters.setEnabled)
        scope_layout.addWidget(self.test_external_drafters)
        self.try_best_settings = QCheckBox(
            "Try only best Settings from a stable Quick pass"
        )
        self.try_best_settings.setChecked(False)
        self.try_best_settings.setEnabled(False)
        self.try_best_settings.setToolTip(
            "Opt-in conservative shortlist: baseline, top finalists, and thread/batch "
            "representatives are remeasured at the full workload. Missing, noisy, or "
            "single-sample Quick evidence automatically falls back to the full search."
        )
        scope_layout.addWidget(self.try_best_settings)
        self.all_models = QCheckBox(
            f"Run all benchmarkable scanned models ({max(1, model_count)} found)"
        )
        self.all_models.setChecked(False)
        scope_layout.addWidget(self.all_models)
        self.rerun_all_models = QCheckBox("Reset old measured data, then rerun all")
        self.rerun_all_models.setChecked(False)
        self.rerun_all_models.setEnabled(False)
        self.rerun_all_models.setToolTip(
            "After final confirmation, atomically deletes all old Quick, Standard, "
            "and Custom evidence plus measured Perform backend profiles for the "
            "selected models/modes before any llama-server starts. Custom user "
            "profiles remain. Interruption leaves only newly completed checkpoints."
        )
        self.all_models.toggled.connect(self._update_all_models_options)
        scope_layout.addWidget(self.rerun_all_models)
        warning = QLabel(
            "All modes or all models can require many model reloads. Runs are "
            "strictly sequential; every completed model/mode/drafter combination is "
            "saved immediately and skipped next time unless Rerun is enabled. A "
            "confirmed rerun resets old measured data first; cancellation never "
            "restores it."
        )
        warning.setWordWrap(True)
        scope_layout.addWidget(warning)
        layout.addWidget(scope_group)
        self._update_test_option_states()
        self._update_context_option_states()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setText("Prepare performance test")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_targets(self) -> List[str]:
        return [name for name, check in self.mode_checks.items() if check.isChecked()]

    def _update_runtime_option_states(self, _index: int = -1) -> None:
        multiple = str(self.runtime_scope_combo.currentData() or "active") == "multiple"
        self.runtime_list.setEnabled(multiple and len(self._runtime_options) > 1)

    def selected_runtime_options(self) -> List[_PerformanceRuntimeOption]:
        if not self._runtime_options:
            return []
        multiple = str(self.runtime_scope_combo.currentData() or "active") == "multiple"
        if not multiple:
            active = [option for option in self._runtime_options if option.active]
            return active[:1] or self._runtime_options[:1]
        selected: List[_PerformanceRuntimeOption] = []
        for row in range(self.runtime_list.count()):
            item = self.runtime_list.item(row)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            try:
                index = int(item.data(Qt.ItemDataRole.UserRole))
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(self._runtime_options):
                selected.append(self._runtime_options[index])
        return selected

    def _update_all_models_options(self, checked: bool) -> None:
        self.rerun_all_models.setEnabled(bool(checked))
        if not checked:
            self.rerun_all_models.setChecked(False)

    def _update_test_option_states(self, _index: int = -1) -> None:
        benchmark_type = self.benchmark_type()
        self.custom_percent_spin.setEnabled(benchmark_type == "custom")
        self.try_best_settings.setEnabled(benchmark_type in ("quick", "custom"))
        if benchmark_type == "fast":
            self.try_best_settings.setChecked(False)

    def benchmark_type(self) -> str:
        value = str(self.test_length_combo.currentData() or "fast")
        return value if value in ("fast", "quick", "custom") else "fast"

    def _update_context_option_states(self, _index: int = -1) -> None:
        fixed = str(self.context_mode_combo.currentData() or "adaptive") == "fixed"
        self.context_spin.setEnabled(fixed)
        self.real_validation.setEnabled(fixed)

    def desired_context(self) -> int:
        """Return zero for per-job adaptive capacity, otherwise the fixed pin."""
        if str(self.context_mode_combo.currentData() or "adaptive") != "fixed":
            return 0
        return max(512, int(self.context_spin.value()))

    def prompt_context_fraction(self) -> float:
        if self.benchmark_type() == "custom":
            return max(0.0001, min(1.0, self.custom_percent_spin.value() / 100.0))
        if self.benchmark_type() == "fast":
            return 0.03125
        return 0.125

    def accept(self) -> None:
        if not self.selected_targets():
            QMessageBox.information(
                self, "Select a mode", "Select at least one performance mode to test."
            )
            return
        if self._runtime_options and not self.selected_runtime_options():
            QMessageBox.information(
                self,
                "Select a llama build",
                "Select at least one installed llama build to test.",
            )
            return
        super().accept()


def _performance_records_for_model(
    records_by_test: Dict[str, List[dict]], entry: ModelEntry
) -> Dict[str, List[dict]]:
    """Keep complete candidate evidence for exactly one selected model."""
    exact_key = app_settings.favorite_model_key(entry.path)
    expected_name = entry.path.name.casefold()
    expected_sizes = {max(0, int(entry.size_bytes or 0))}
    try:
        expected_sizes.add(max(0, int(entry.path.stat().st_size)))
    except OSError:
        pass
    expected_sizes.discard(0)
    selected: Dict[str, List[dict]] = {key: [] for key in ("fast", "quick", "custom")}
    for test_type in selected:
        raw_records = records_by_test.get(test_type, [])
        for record in raw_records if isinstance(raw_records, list) else []:
            if not isinstance(record, dict):
                continue
            raw_path = str(record.get("model_path", "") or "").strip()
            if (
                raw_path
                and app_settings.favorite_model_key(Path(raw_path)) == exact_key
            ):
                selected[test_type].append(record)
                continue
            try:
                record_size = max(0, int(record.get("model_size", 0) or 0))
            except (TypeError, ValueError):
                record_size = 0
            record_filename = Path(raw_path).name.casefold() if raw_path else ""
            if (
                record_size > 0
                and record_size in expected_sizes
                and record_filename == expected_name
            ):
                selected[test_type].append(record)
                continue
            record_name = str(record.get("model_name", "") or "").strip()
            if (
                not raw_path
                and record_size > 0
                and record_size in expected_sizes
                and record_name.casefold() == entry.name.casefold()
            ):
                selected[test_type].append(record)
    return selected


class _PerformanceAnalysisDialog(QDialog):
    """Selected-model view with every successful and failed candidate run."""

    _TEST_ORDER = ("fast", "quick", "custom")
    _TEST_TITLES = {
        "fast": "⚡ Quick pass · ≤3.125% context",
        "quick": "✅ Standard performance test · 12.5% context",
        "custom": "🧪 Custom context performance test",
    }
    _METRIC_HELP = {
        "prompt": (
            "Prompt processing (PP): llama.cpp's native prompt_per_second timing "
            "after an excluded warm-up (or prompt_n / prompt_ms when the direct "
            "field is unavailable). The deterministic prompt uses each run's saved "
            "context fraction. Only Standard is capped at 65,536 prompt tokens; "
            "Custom is uncapped. Prompt caching is disabled."
        ),
        "decode": (
            "n_decode: llama.cpp's native predicted_per_second timing (or "
            "predicted_n / predicted_ms fallback). AutoTuner requests 256 "
            "deterministic decode tokens with EOS ignored and uses the median of "
            "the accepted samples."
        ),
        "overall": (
            "End-to-end: (prompt tokens + generated tokens) divided by native "
            "prompt time plus native decode time for this exact workload. This is "
            "the score used to rank candidates and performance modes."
        ),
    }

    def __init__(
        self,
        records_by_test: Dict[str, List[dict]],
        parent: Optional[QWidget] = None,
        *,
        html_path: Optional[Path] = None,
        selected_model_name: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Performance analysis")
        self.resize(1100, 760)
        self.setMinimumSize(760, 520)
        self.test_tiles: Dict[str, QFrame] = {}
        self.record_count_by_test: Dict[str, int] = {}
        self.candidate_count_by_test: Dict[str, int] = {}
        self.failed_candidate_count_by_test: Dict[str, int] = {}
        self.model_names_by_test: Dict[str, List[str]] = {}

        layout = QVBoxLayout(self)
        scope = escape(selected_model_name or "selected model")
        intro = QLabel(
            f"<b>Selected model only: {scope}</b><br>Quick pass, Standard 12.5%, "
            "and Custom records are separated below. Every candidate run is shown, "
            "including unsuccessful settings and its stored error. The detailed HTML "
            "report remains global and includes every model."
        )
        intro.setTextFormat(Qt.TextFormat.RichText)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        contents = QWidget()
        contents_layout = QVBoxLayout(contents)
        contents_layout.setSpacing(14)
        for test_type in self._TEST_ORDER:
            raw_records = records_by_test.get(test_type, [])
            records = [item for item in raw_records if isinstance(item, dict)]
            tile = self._build_test_tile(test_type, records)
            self.test_tiles[test_type] = tile
            self.record_count_by_test[test_type] = len(records)
            candidates = [
                candidate
                for record in records
                for candidate in (
                    record.get("candidates")
                    if isinstance(record.get("candidates"), list)
                    else []
                )
                if isinstance(candidate, dict)
            ]
            self.candidate_count_by_test[test_type] = len(candidates)
            self.failed_candidate_count_by_test[test_type] = sum(
                bool(str(candidate.get("error", "") or "").strip())
                for candidate in candidates
            )
            contents_layout.addWidget(tile)
        contents_layout.addStretch(1)
        scroll.setWidget(contents)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        if html_path is not None:
            report_button = QPushButton("Open detailed HTML report")
            report_button.setObjectName("openPerformanceHtmlReport")
            report_button.clicked.connect(
                lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(html_path)))
            )
            buttons.addButton(report_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)
        close_button = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_button is not None:
            close_button.clicked.connect(self.accept)
        layout.addWidget(buttons)

    @staticmethod
    def _positive_float(value: object) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        return parsed if parsed > 0.0 and parsed < float("inf") else 0.0

    @staticmethod
    def _nonnegative_int(value: object) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _winner_metrics(cls, record: dict) -> Tuple[float, float, float]:
        winner_id = str(record.get("winner_id", ""))
        candidates = record.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                if winner_id and str(candidate.get("id", "")) != winner_id:
                    continue
                return (
                    cls._positive_float(candidate.get("prompt_tps")),
                    cls._positive_float(candidate.get("generation_tps")),
                    cls._positive_float(candidate.get("overall_tps")),
                )
        return (
            cls._positive_float(record.get("prompt_tps")),
            cls._positive_float(record.get("generation_tps")),
            cls._positive_float(record.get("overall_tps")),
        )

    @classmethod
    def _winner_workload(cls, record: dict) -> Tuple[int, int, int]:
        winner_id = str(record.get("winner_id", ""))
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            return 0, 0, 0
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if winner_id and str(candidate.get("id", "")) != winner_id:
                continue
            samples = candidate.get("samples")
            if not isinstance(samples, list):
                return 0, 0, 0
            valid = [sample for sample in samples if isinstance(sample, dict)]
            if not valid:
                return 0, 0, 0
            return (
                cls._nonnegative_int(valid[0].get("prompt_tokens")),
                cls._nonnegative_int(valid[0].get("generated_tokens")),
                len(valid),
            )
        return 0, 0, 0

    @staticmethod
    def _model_identity(record: dict) -> str:
        path = str(record.get("model_path", "") or "").strip()
        name = str(record.get("model_name", "") or "").strip()
        return os.path.normcase(path) if path else name.casefold() or "unknown-model"

    @staticmethod
    def _model_name(record: dict) -> str:
        name = str(record.get("model_name", "") or "").strip()
        if name:
            return name
        path = str(record.get("model_path", "") or "").strip()
        return Path(path).stem if path else "Unknown model"

    @staticmethod
    def _metric_bar(value: float, maximum: float, help_text: str) -> QProgressBar:
        bar = QProgressBar()
        bar.setRange(0, 1000)
        ratio = value / maximum if value > 0.0 and maximum > 0.0 else 0.0
        bar.setValue(max(0, min(1000, int(round(ratio * 1000)))))
        bar.setFormat(f"{value:.1f} tok/s" if value > 0.0 else "no timing")
        bar.setTextVisible(True)
        bar.setMinimumWidth(180)
        bar.setToolTip(_setting_tooltip(help_text, help_text))
        return bar

    @staticmethod
    def _winner_settings_text(record: dict) -> str:
        settings = record.get("winner_settings")
        if not isinstance(settings, dict):
            return "No winner settings were stored."
        ordered = ("threads", "batch_threads", "batch", "ubatch", "draft_n_max")
        parts = [f"{key}={settings[key]}" for key in ordered if key in settings]
        return ", ".join(parts) or "No winner settings were stored."

    @staticmethod
    def _candidate_settings_text(candidate: dict) -> str:
        settings = candidate.get("settings")
        if not isinstance(settings, dict):
            return "—"
        ordered = ("threads", "batch_threads", "batch", "ubatch", "draft_n_max")
        keys = [key for key in ordered if key in settings]
        keys.extend(sorted(key for key in settings if key not in keys))
        return ", ".join(f"{key}={settings[key]}" for key in keys) or "—"

    @classmethod
    def _candidate_table(cls, record: dict) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 8)
        candidates = [
            candidate
            for candidate in (
                record.get("candidates")
                if isinstance(record.get("candidates"), list)
                else []
            )
            if isinstance(candidate, dict)
        ]
        winner_id = str(record.get("winner_id", "") or "")
        successful = [
            candidate
            for candidate in candidates
            if not str(candidate.get("error", "") or "").strip()
        ]
        failed_count = len(candidates) - len(successful)
        overall_values = [
            cls._positive_float(candidate.get("overall_tps"))
            for candidate in successful
            if cls._positive_float(candidate.get("overall_tps")) > 0.0
        ]
        range_text = (
            f" · end-to-end {min(overall_values):.1f}–{max(overall_values):.1f} tok/s"
            if overall_values
            else ""
        )
        stats = QLabel(
            f"Candidate runs: {len(candidates)} · successful {len(successful)} · "
            f"failed {failed_count}{range_text}"
        )
        stats.setObjectName("performanceCandidateStats")
        layout.addWidget(stats)

        tree = QTreeWidget()
        tree.setObjectName("performanceCandidateRuns")
        tree.setProperty("candidateCount", len(candidates))
        tree.setProperty("failedCandidateCount", failed_count)
        tree.setColumnCount(8)
        tree.setHeaderLabels(
            [
                "Status",
                "Candidate",
                "Settings",
                "PP tok/s",
                "Decode tok/s",
                "End-to-end",
                "Samples",
                "Error",
            ]
        )
        tree.setRootIsDecorated(False)
        tree.setAlternatingRowColors(True)
        for candidate in candidates:
            identifier = str(
                candidate.get("label") or candidate.get("id") or "candidate"
            )
            error = str(candidate.get("error", "") or "").strip()
            is_winner = bool(winner_id and str(candidate.get("id", "")) == winner_id)
            status = "Failed" if error else "Winner" if is_winner else "Measured"
            samples = candidate.get("samples")
            sample_count = len(samples) if isinstance(samples, list) else 0
            item = QTreeWidgetItem(
                [
                    status,
                    identifier,
                    cls._candidate_settings_text(candidate),
                    f"{cls._positive_float(candidate.get('prompt_tps')):.2f}",
                    f"{cls._positive_float(candidate.get('generation_tps')):.2f}",
                    f"{cls._positive_float(candidate.get('overall_tps')):.2f}",
                    str(sample_count),
                    error or "—",
                ]
            )
            log_tail = str(candidate.get("log_tail", "") or "").strip()
            if log_tail:
                item.setToolTip(7, log_tail)
            tree.addTopLevelItem(item)
        if not candidates:
            tree.addTopLevelItem(
                QTreeWidgetItem(["Unavailable", "No candidate details stored"])
            )
        tree.setMinimumHeight(min(280, max(100, 54 + 28 * max(1, len(candidates)))))
        for column in range(tree.columnCount()):
            tree.resizeColumnToContents(column)
        layout.addWidget(tree)
        return container

    def _build_test_tile(self, test_type: str, records: List[dict]) -> QFrame:
        tile = QFrame()
        tile.setObjectName(f"performanceAnalysisTile-{test_type}")
        tile.setProperty("benchmarkType", test_type)
        tile.setFrameShape(QFrame.Shape.StyledPanel)
        tile_layout = QVBoxLayout(tile)
        title = QLabel(f"<h2>{escape(self._TEST_TITLES[test_type])}</h2>")
        title.setTextFormat(Qt.TextFormat.RichText)
        tile_layout.addWidget(title)

        grouped: Dict[str, List[dict]] = {}
        display_names: Dict[str, str] = {}
        for record in records:
            identity = self._model_identity(record)
            grouped.setdefault(identity, []).append(record)
            display_names.setdefault(identity, self._model_name(record))
        self.model_names_by_test[test_type] = sorted(
            display_names.values(), key=str.casefold
        )
        total_candidates = sum(
            len(record.get("candidates"))
            for record in records
            if isinstance(record.get("candidates"), list)
        )
        summary = QLabel(
            f"{len(grouped)} selected model(s) · {len(records)} mode/backend "
            f"record(s) · {total_candidates} candidate run(s). Winner bars are "
            "normalized only inside this tile and only against the same metric."
        )
        summary.setWordWrap(True)
        tile_layout.addWidget(summary)

        if not records:
            empty = QLabel(
                "No saved results for this test yet. Run this test type from "
                "Performance test; results for the other tile remain untouched."
            )
            empty.setWordWrap(True)
            empty.setObjectName("performanceAnalysisEmpty")
            tile_layout.addWidget(empty)
        else:
            metrics = {id(record): self._winner_metrics(record) for record in records}
            maxima = tuple(
                max((values[index] for values in metrics.values()), default=0.0)
                for index in range(3)
            )
            target_order = {
                name: index for index, name in enumerate(list_target_names())
            }
            for identity in sorted(
                grouped, key=lambda key: display_names[key].casefold()
            ):
                model_records = sorted(
                    grouped[identity],
                    key=lambda item: target_order.get(
                        str(item.get("performance_target", "")), 999
                    ),
                )
                card = QFrame()
                card.setFrameShape(QFrame.Shape.StyledPanel)
                card_layout = QVBoxLayout(card)
                model_title = QLabel(f"<b>{escape(display_names[identity])}</b>")
                card_layout.addWidget(model_title)
                model_path = str(model_records[0].get("model_path", "") or "")
                if model_path:
                    path_label = QLabel(model_path)
                    path_label.setObjectName("mutedLabel")
                    path_label.setTextInteractionFlags(
                        Qt.TextInteractionFlag.TextSelectableByMouse
                    )
                    card_layout.addWidget(path_label)

                grid = QGridLayout()
                for column, label in enumerate(
                    ("Mode / context", "Prompt processing", "n_decode", "End-to-end")
                ):
                    header = QLabel(f"<b>{label}</b>")
                    if column > 0:
                        header.setToolTip(
                            _setting_tooltip(
                                "Hover for how this metric is collected.",
                                self._METRIC_HELP[
                                    ("prompt", "decode", "overall")[column - 1]
                                ],
                            )
                        )
                    grid.addWidget(header, 0, column)

                fastest_record_id = max(
                    (id(item) for item in model_records),
                    key=lambda record_id: metrics[record_id][2],
                )
                for record_index, record in enumerate(model_records):
                    row = 1 + record_index * 2
                    prompt_tps, decode_tps, overall_tps = metrics[id(record)]
                    target = str(record.get("performance_target", "unknown"))
                    drafter = str(
                        record.get("drafter_label", "No drafter") or "No drafter"
                    )
                    backend = app_settings.performance_backend_label(
                        str(record.get("benchmark_backend", ""))
                    )
                    context = self._nonnegative_int(record.get("desired_context", 0))
                    prompt_tokens, generated_tokens, sample_count = (
                        self._winner_workload(record)
                    )
                    marker = (
                        "★ "
                        if id(record) == fastest_record_id and overall_tps > 0
                        else ""
                    )
                    workload = f"ctx {context:,}"
                    fraction = self._positive_float(
                        record.get("prompt_context_fraction")
                    )
                    if fraction > 0.0:
                        workload += f" · {fraction * 100:.2f}%"
                    if prompt_tokens or generated_tokens:
                        workload += (
                            f" · prompt {prompt_tokens:,} + decode {generated_tokens:,}"
                        )
                    runtime_label = str(record.get("runtime_label", "") or "").strip()
                    backend_text = runtime_label or backend
                    mode_label = QLabel(
                        f"{marker}{escape(target)} · {escape(backend_text)} · "
                        f"{escape(drafter)}<br><small>{workload}</small>"
                    )
                    detail = self._winner_settings_text(record)
                    if sample_count:
                        detail += f"; accepted samples={sample_count}"
                    mode_label.setToolTip(
                        _setting_tooltip(
                            "Winning runtime settings and measured workload.",
                            detail,
                        )
                    )
                    grid.addWidget(mode_label, row, 0)
                    grid.addWidget(
                        self._metric_bar(
                            prompt_tps, maxima[0], self._METRIC_HELP["prompt"]
                        ),
                        row,
                        1,
                    )
                    grid.addWidget(
                        self._metric_bar(
                            decode_tps, maxima[1], self._METRIC_HELP["decode"]
                        ),
                        row,
                        2,
                    )
                    grid.addWidget(
                        self._metric_bar(
                            overall_tps, maxima[2], self._METRIC_HELP["overall"]
                        ),
                        row,
                        3,
                    )
                    grid.addWidget(self._candidate_table(record), row + 1, 0, 1, 4)
                card_layout.addLayout(grid)
                tile_layout.addWidget(card)

        explanation = QLabel(
            "<b>How the metrics are collected</b><br>"
            f"• {escape(self._METRIC_HELP['prompt'])}<br>"
            f"• {escape(self._METRIC_HELP['decode'])}<br>"
            f"• {escape(self._METRIC_HELP['overall'])}<br>"
            "★ marks the highest winner end-to-end value inside this test tile. "
            "Candidate tables retain every measured and failed setting; use the "
            "Status and Error columns rather than color alone."
        )
        explanation.setWordWrap(True)
        explanation.setTextFormat(Qt.TextFormat.RichText)
        tile_layout.addWidget(explanation)
        return tile


class _PerformanceTuneDialog(QDialog):
    """Non-blocking progress surface for the real server benchmark suite."""

    cancel_requested = pyqtSignal()
    stop_after_model_requested = pyqtSignal()
    stop_after_mode_requested = pyqtSignal()

    def __init__(
        self,
        summary: str,
        job_count: int,
        parent: Optional[QWidget] = None,
        *,
        allow_model_stop: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("AutoTuner performance test")
        self.setModal(False)
        self.setMinimumWidth(620)
        self._finished = False
        layout = QVBoxLayout(self)
        title = QLabel(
            f"{summary}\n{job_count} model/mode profile(s) will be measured "
            "sequentially. Each candidate starts a fresh private llama-server."
        )
        title.setWordWrap(True)
        layout.addWidget(title)
        self.status_label = QLabel("Preparing deterministic candidates…")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)
        actions = QHBoxLayout()
        self.stop_after_model_button = QPushButton("Stop after Model")
        self.stop_after_model_button.setEnabled(bool(allow_model_stop))
        self.stop_after_model_button.setToolTip(
            "Finish all remaining performance modes for the model currently "
            "being tested, save each one, then stop before the next model."
        )
        self.stop_after_model_button.clicked.connect(self._request_stop_after_model)
        actions.addWidget(self.stop_after_model_button)
        self.stop_after_mode_button = QPushButton("Stop after Performance Mode")
        self.stop_after_mode_button.setToolTip(
            "Finish and save the active model/performance-mode run, then stop."
        )
        self.stop_after_mode_button.clicked.connect(self._request_stop_after_mode)
        actions.addWidget(self.stop_after_mode_button)
        actions.addStretch(1)
        self.cancel_button = QPushButton("Cancel performance test")
        self.cancel_button.clicked.connect(self._request_cancel)
        actions.addWidget(self.cancel_button)
        layout.addLayout(actions)

    def _request_stop_after_model(self) -> None:
        if self._finished or not self.stop_after_model_button.isEnabled():
            return
        self.stop_after_model_button.setEnabled(False)
        self.status_label.setText(
            "Stop requested: finishing and saving all modes for the current model…"
        )
        self.stop_after_model_requested.emit()

    def _request_stop_after_mode(self) -> None:
        if self._finished or not self.stop_after_mode_button.isEnabled():
            return
        self.stop_after_mode_button.setEnabled(False)
        self.stop_after_model_button.setEnabled(False)
        self.status_label.setText(
            "Stop requested: finishing and saving the active performance mode…"
        )
        self.stop_after_mode_requested.emit()

    def _request_cancel(self) -> None:
        if self._finished or not self.cancel_button.isEnabled():
            return
        self.cancel_button.setEnabled(False)
        self.status_label.setText(
            "Cancelling and stopping the active benchmark server…"
        )
        self.cancel_requested.emit()

    def update_progress(self, completed: int, total: int, message: str) -> None:
        self.status_label.setText(message)
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(max(0, min(completed, total)))
        else:
            self.progress_bar.setRange(0, 0)

    def mark_finished(self) -> None:
        self._finished = True
        self.stop_after_model_button.setEnabled(False)
        self.stop_after_mode_button.setEnabled(False)
        self.cancel_button.setEnabled(False)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self._finished:
            self._request_cancel()
            event.ignore()
            return
        super().closeEvent(event)


class _PerformanceTuneWorker(QObject):
    progress = pyqtSignal(int, int, str)
    checkpointed = pyqtSignal(object)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        runner: BenchmarkSuiteRunner,
        checkpoint_callback: Optional[Callable[[object], dict]] = None,
    ) -> None:
        super().__init__()
        self.runner = runner
        self._checkpoint_callback = checkpoint_callback
        if checkpoint_callback is not None:
            self.runner.checkpoint = self._checkpoint

    def _checkpoint(self, outcome: object) -> None:
        try:
            assert self._checkpoint_callback is not None
            payload = self._checkpoint_callback(outcome)
        except Exception as exc:
            payload = {
                "key": getattr(getattr(outcome, "job", None), "key", ""),
                "saved": False,
                "error": f"checkpoint failed: {exc}",
            }
        self.checkpointed.emit(payload)
        if not payload.get("saved"):
            raise BenchmarkFailure(
                str(payload.get("error") or "benchmark checkpoint could not be saved")
            )

    def run(self) -> None:
        try:
            self.runner.progress = self.progress.emit
            self.finished.emit(self.runner.run())
        except BenchmarkCancelled:
            self.cancelled.emit()
        except BenchmarkFailure as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self.runner.cancel()

    def stop_after_model(self) -> None:
        self.runner.stop_after_model()

    def stop_after_performance_mode(self) -> None:
        self.runner.stop_after_performance_mode()


# ---------------------------------------------------------------------------
# Hardware detection worker with global timeout


class _HwDetectWorker(QObject):
    """Runs detect_system() in a background thread with a global timeout."""

    finished = pyqtSignal(object, str)  # SystemInfo|None, error_msg

    def __init__(
        self, timeout: float = 30.0, llama_binary: Optional[str] = None
    ) -> None:
        super().__init__()
        self._timeout = timeout
        self._llama_binary = llama_binary

    def run(self) -> None:
        result: list = [None, ""]  # [SystemInfo|None, error_str]

        def _detect() -> None:
            try:
                result[0] = detect_system(self._llama_binary)
            except Exception as exc:
                result[1] = str(exc)

        t = threading.Thread(target=_detect, daemon=True)
        t.start()
        t.join(self._timeout)

        if t.is_alive():
            # Detection timed out — emit with whatever partial result exists.
            # result[0] may still be None if detect_system() never returned.
            self.finished.emit(
                result[0], "Hardware detection timed out (partial result)."
            )
        elif result[1]:
            self.finished.emit(None, result[1])
        else:
            self.finished.emit(result[0], "")


# ---------------------------------------------------------------------------
# Background scanner


class _ScanWorker(QObject):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, roots: List[Path]) -> None:
        super().__init__()
        self._roots = roots
        self.stats: Dict[str, object] = {}

    def run(self) -> None:
        started = time.monotonic()
        cache_before = metadata_cache_stats()
        try:
            entries: List[ModelEntry] = []
            seen: set[str] = set()
            for root in self._roots:
                for entry in scan_models(root):
                    try:
                        key = os.path.normcase(str(entry.path.resolve(strict=False)))
                    except (OSError, RuntimeError):
                        key = os.path.normcase(str(entry.path))
                    if key in seen:
                        continue
                    seen.add(key)
                    entries.append(entry)
            cache_after = metadata_cache_stats()
            self.stats = {
                "elapsed_s": time.monotonic() - started,
                "entries": cache_after["entries"],
                "hits": max(0, cache_after["hits"] - cache_before["hits"]),
                "misses": max(0, cache_after["misses"] - cache_before["misses"]),
                "workers": cache_after["workers"],
            }
            self.finished.emit(entries)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Full metadata diagnostic export worker


class _MetadataDiagnosticWorker(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(str, int)
    error = pyqtSignal(str)

    def __init__(self, roots: List[Path], output_path: Path) -> None:
        super().__init__()
        self._roots = list(roots)
        self._output_path = output_path

    def run(self) -> None:
        try:
            from get_metadata import write_metadata_report

            output, count = write_metadata_report(
                self._roots,
                self._output_path,
                progress=self.progress.emit,
            )
            self.finished.emit(str(output), count)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# GitHub update worker


class _UpdateWorker(QObject):
    """Update a git checkout or a downloaded GitHub ZIP without losing settings."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    _SETTINGS_NAME = "autotuner_settings.json"
    _SETTINGS_BACKUP_NAME = "autotuner_settings.json.update-backup"
    _UPDATE_STATE_NAME = ".autotuner_update.json"
    _GITHUB_REPO = GITHUB_REPO
    _GITHUB_BRANCH = "main"
    _ARCHIVE_SKIP_DIRS = {
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        ".venv_linux",
        "venv",
        "env",
        "ENV",
    }
    _ARCHIVE_SKIP_FILES = {
        _SETTINGS_NAME,
        _SETTINGS_BACKUP_NAME,
        _UPDATE_STATE_NAME,
    }

    def __init__(self, repo_root: Path) -> None:
        super().__init__()
        # Skript-Verzeichnis merken für verständliche Fehlermeldungen.
        try:
            self._script_dir = repo_root.resolve()
        except (OSError, RuntimeError):
            self._script_dir = repo_root
        # Echten Git-Root suchen: vom Skript-Verzeichnis aus aufwärts, bis ein
        # `.git`-Marker gefunden wird. Funktioniert auch, wenn qt_launcher.py
        # in einem Unterordner liegt. Wenn kein Marker existiert, behandeln wir
        # den Ordner als GitHub-Archiv/Release-ZIP und aktualisieren per
        # heruntergeladenem Source-ZIP statt per git pull.
        self._repo_root = self._find_git_root(self._script_dir)
        # App code lives next to qt_launcher.py; user settings live centrally
        # under ~/.autotuner. Git commands still run from _repo_root, while
        # archive updates must never target an unrelated
        # parent repo that happens to contain a .git directory.
        self._app_root = self._script_dir
        # Git-Executable plattformübergreifend auflösen. `shutil.which` reicht
        # normalerweise; unter Windows suchen wir zusätzlich in den üblichen
        # "Git for Windows"-Installationspfaden, weil pythonw-basierte Starter
        # teilweise einen reduzierten PATH haben. Für ZIP-Installationen ist Git
        # nicht nötig.
        self._git_bin = shutil.which("git") or self._find_git_windows_fallback()

    # Maximale Suchtiefe beim Aufwärtslaufen nach `.git`. qt_launcher.py liegt
    # normalerweise direkt im Repo-Root; 5 Ebenen decken auch verschachtelte
    # Layouts ab, verhindern aber, dass wir in völlig unrelated Eltern-Repos
    # (z. B. ein ~/.git für Dotfile-Management) landen.
    _GIT_ROOT_MAX_DEPTH = 5

    @classmethod
    def _find_git_root(cls, start: Path) -> Optional[Path]:
        """Sucht aufwärts nach einem `.git`-Verzeichnis oder -File (Worktree).

        Liefert None, wenn innerhalb von `_GIT_ROOT_MAX_DEPTH` Ebenen kein
        Marker gefunden wurde.
        """
        try:
            cur = start.resolve()
        except (OSError, RuntimeError):
            cur = start
        for _ in range(cls._GIT_ROOT_MAX_DEPTH):
            try:
                if (cur / ".git").exists():
                    return cur
            except OSError:
                pass
            if cur.parent == cur:
                return None
            cur = cur.parent
        return None

    @staticmethod
    def _find_git_windows_fallback() -> Optional[str]:
        """Sucht Git for Windows in typischen Installationspfaden.

        Wird nur gebraucht, wenn `git` nicht auf PATH ist (z. B. bei Start über
        pythonw.exe mit reduziertem PATH). Auf Nicht-Windows liefert die
        Funktion None.
        """
        if os.name != "nt":
            return None
        candidate_roots: List[Path] = []
        for env_var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
            val = os.environ.get(env_var)
            if val:
                candidate_roots.append(Path(val) / "Git")
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            candidate_roots.append(Path(local_appdata) / "Programs" / "Git")
        for sub in ("cmd", "bin", "mingw64\\bin", "mingw32\\bin"):
            for root in candidate_roots:
                cand = root / sub / "git.exe"
                try:
                    if cand.is_file():
                        return str(cand)
                except OSError:
                    continue
        return None

    def _run_git(self, *args: str, check: bool = True, timeout: float = 180.0) -> str:
        if self._repo_root is None:
            raise RuntimeError(
                "Git update requested, but this folder has no .git metadata."
            )
        if not self._git_bin:
            raise RuntimeError(
                "Git executable not found. Please install Git "
                "(https://git-scm.com) and restart AutoTuner."
            )
        return self._run(
            [self._git_bin, *args],
            check=check,
            timeout=timeout,
            cwd=self._repo_root,
        )

    def _run(
        self,
        cmd: List[str],
        check: bool = True,
        timeout: float = 600.0,
        cwd: Optional[Path] = None,
    ) -> str:
        pretty = " ".join(cmd)
        self.progress.emit(f"[Update] $ {pretty}")
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        cp = subprocess.run(
            cmd,
            cwd=cwd or self._repo_root or self._app_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            **kwargs,
        )
        out = (cp.stdout or "").strip()
        if out:
            if len(out) > 4000:
                out = out[-4000:]
            for line in out.splitlines():
                self.progress.emit(f"[Update] {line}")
        if check and cp.returncode != 0:
            raise RuntimeError(f"Command failed ({cp.returncode}): {pretty}\n{out}")
        return out

    @staticmethod
    def _status_path(line: str) -> str:
        # Porcelain v1: two status columns, a space, then the path. Renames are
        # shown as "old -> new"; the destination is what matters for safety.
        p = line[3:] if len(line) > 3 else ""
        if " -> " in p:
            p = p.split(" -> ", 1)[1]
        return p.strip().strip('"')

    def _launcher_is_tracked_by_git(self) -> bool:
        if self._repo_root is None:
            return False
        try:
            launcher_rel = (
                (self._script_dir / "qt_launcher.py")
                .resolve()
                .relative_to(self._repo_root.resolve())
                .as_posix()
            )
        except (OSError, RuntimeError, ValueError):
            return False
        tracked = self._run_git("ls-files", "--", launcher_rel, check=False)
        return any(
            line.strip().replace("\\", "/") == launcher_rel
            for line in tracked.splitlines()
        )

    def _backup_settings(self) -> Dict[Path, Optional[bytes]]:
        paths = {self._app_root / self._SETTINGS_NAME}
        try:
            paths.add(app_settings._settings_file())  # portable path or home fallback
        except Exception:
            pass
        backups: Dict[Path, Optional[bytes]] = {}
        for p in paths:
            try:
                data = p.read_bytes() if p.exists() else None
                backups[p] = data
                if p == self._app_root / self._SETTINGS_NAME and data is not None:
                    # Crash-safe legacy backup stays in the shared user-data
                    # directory; updates must not create fresh state in source.
                    backup = app_settings.app_data_dir() / self._SETTINGS_BACKUP_NAME
                    backup.write_bytes(data)
            except OSError:
                backups[p] = None
        return backups

    def _restore_settings(self, backups: Dict[Path, Optional[bytes]]) -> None:
        for p, data in backups.items():
            try:
                if data is None:
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            except OSError as exc:
                self.progress.emit(f"[Update] Warning: could not restore {p}: {exc}")
        for backup in (
            app_settings.app_data_dir() / self._SETTINGS_BACKUP_NAME,
            self._app_root / self._SETTINGS_BACKUP_NAME,
        ):
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _read_bytes(path: Path) -> Optional[bytes]:
        try:
            return path.read_bytes() if path.exists() else None
        except OSError:
            return None

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def _github_request(self, url: str) -> urllib.request.Request:
        headers = {
            "Accept": "application/vnd.github+json"
            if "api.github.com" in url
            else "application/octet-stream",
            "User-Agent": "AutoTuner-updater",
        }
        if "api.github.com" in url:
            headers["X-GitHub-Api-Version"] = "2022-11-28"
        return urllib.request.Request(url, headers=headers)

    def _fetch_json(self, url: str, timeout: float = 30.0) -> Dict[str, object]:
        with urllib.request.urlopen(self._github_request(url), timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            data = resp.read().decode(charset, errors="replace")
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"GitHub returned unexpected JSON for {url}")
        return parsed

    def _github_archive_info(self) -> Tuple[str, Optional[str], str]:
        branch = self._GITHUB_BRANCH
        sha: Optional[str] = None
        api_root = f"https://api.github.com/repos/{self._GITHUB_REPO}"

        try:
            repo_info = self._fetch_json(api_root)
            default_branch = repo_info.get("default_branch")
            if isinstance(default_branch, str) and default_branch:
                branch = default_branch
        except Exception as exc:
            self.progress.emit(
                f"[Update] Warning: could not query GitHub default branch: {exc}"
            )

        try:
            branch_info = self._fetch_json(
                f"{api_root}/branches/{urllib.parse.quote(branch, safe='')}"
            )
            commit = branch_info.get("commit")
            if isinstance(commit, dict):
                commit_sha = commit.get("sha")
                if isinstance(commit_sha, str) and commit_sha:
                    sha = commit_sha
        except Exception as exc:
            self.progress.emit(
                f"[Update] Warning: could not query GitHub branch SHA: {exc}"
            )

        archive_url = (
            f"https://github.com/{self._GITHUB_REPO}/archive/refs/heads/"
            f"{urllib.parse.quote(branch, safe='')}.zip"
        )
        return branch, sha, archive_url

    def _read_update_state(self) -> Dict[str, str]:
        state_path = app_settings.app_data_dir() / self._UPDATE_STATE_NAME
        try:
            parsed = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {str(k): str(v) for k, v in parsed.items() if isinstance(v, str)}

    def _write_update_state(self, branch: str, sha: Optional[str]) -> None:
        state: Dict[str, str] = {
            "repo": self._GITHUB_REPO,
            "branch": branch,
            "installed_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        }
        if sha:
            state["sha"] = sha
        try:
            (app_settings.app_data_dir() / self._UPDATE_STATE_NAME).write_text(
                json.dumps(state, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            self.progress.emit(f"[Update] Warning: could not write update state: {exc}")

    def _download_file(self, url: str, destination: Path) -> None:
        self.progress.emit(f"[Update] Downloading {url}")
        try:
            with urllib.request.urlopen(
                self._github_request(url), timeout=300.0
            ) as resp:
                with destination.open("wb") as fh:
                    shutil.copyfileobj(resp, fh)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not download update archive: {exc}") from exc

    def _safe_extract_zip(self, zip_path: Path, extract_dir: Path) -> Path:
        extract_dir.mkdir(parents=True, exist_ok=True)
        base = extract_dir.resolve()
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                target = (extract_dir / member.filename).resolve()
                if not self._is_relative_to(target, base):
                    raise RuntimeError(f"Unsafe path in update ZIP: {member.filename}")
            zf.extractall(extract_dir)

        children = [p for p in extract_dir.iterdir() if p.name != "__MACOSX"]
        if len(children) == 1 and children[0].is_dir():
            return children[0]
        return extract_dir

    def _copy_archive_tree(self, source_root: Path) -> int:
        copied = 0
        for src in source_root.rglob("*"):
            rel = src.relative_to(source_root)
            if any(part in self._ARCHIVE_SKIP_DIRS for part in rel.parts):
                continue
            if (
                rel.as_posix() in self._ARCHIVE_SKIP_FILES
                or rel.name in self._ARCHIVE_SKIP_FILES
            ):
                continue

            dst = self._app_root / rel
            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                continue
            if not src.is_file():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        return copied

    def _run_archive_update(self, backups: Dict[Path, Optional[bytes]]) -> bool:
        self.progress.emit(
            "[Update] No usable AutoTuner git checkout found; using GitHub "
            "source archive updater."
        )
        branch, sha, archive_url = self._github_archive_info()
        state = self._read_update_state()
        if sha and state.get("repo") == self._GITHUB_REPO and state.get("sha") == sha:
            self._restore_settings(backups)
            self.finished.emit(
                True,
                _source_update_message(
                    "Source files already match the current GitHub branch."
                ),
            )
            return True

        old_requirements = self._read_bytes(self._app_root / "requirements.txt")
        with tempfile.TemporaryDirectory(prefix="autotuner-update-") as td:
            tmp = Path(td)
            zip_path = tmp / "autotuner-source.zip"
            extract_dir = tmp / "extract"
            self._download_file(archive_url, zip_path)
            source_root = self._safe_extract_zip(zip_path, extract_dir)
            copied = self._copy_archive_tree(source_root)
            self.progress.emit(f"[Update] Copied {copied} file(s) from GitHub archive.")

        # Restore before installing dependencies so the GUI can keep using the
        # user's current settings even if pip takes a while.
        self._restore_settings(backups)
        new_requirements = self._read_bytes(self._app_root / "requirements.txt")
        if new_requirements is not None and new_requirements != old_requirements:
            self.progress.emit(
                "[Update] requirements.txt changed; installing dependencies …"
            )
            self._run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                timeout=900.0,
                cwd=self._app_root,
            )

        self._write_update_state(branch, sha)
        short = sha[:8] if sha else branch
        self.finished.emit(
            True,
            _source_update_message(
                f"GitHub archive source update installed ({short}). Local settings "
                "were restored. Please restart AutoTuner."
            ),
        )
        return True

    def run(self) -> None:
        backups: Dict[Path, Optional[bytes]] = {}
        settings_restored = False
        try:
            if self._repo_root is None:
                backups = self._backup_settings()
                settings_restored = self._run_archive_update(backups)
                return
            if not self._git_bin:
                raise RuntimeError(
                    "Git executable not found. Please install Git "
                    "(https://git-scm.com) and restart AutoTuner."
                )
            if self._run_git("rev-parse", "--is-inside-work-tree").strip() != "true":
                raise RuntimeError(
                    f"`git rev-parse` reports {self._repo_root} is not inside a "
                    "work tree. Refusing to continue."
                )
            if not self._launcher_is_tracked_by_git():
                self.progress.emit(
                    "[Update] Found a parent .git directory, but this AutoTuner "
                    "folder is not tracked by it; using GitHub source archive updater."
                )
                backups = self._backup_settings()
                settings_restored = self._run_archive_update(backups)
                return

            branch = self._run_git("rev-parse", "--abbrev-ref", "HEAD").strip()
            if branch == "HEAD":
                raise RuntimeError("Detached HEAD: automatic update is not supported.")

            upstream = self._run_git(
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{u}",
                check=False,
            ).strip()
            if not upstream:
                upstream = f"origin/{branch}"

            dirty = self._run_git(
                "status", "--porcelain", "--untracked-files=no", check=False
            )
            dirty_lines = [ln for ln in dirty.splitlines() if ln.strip()]
            unsafe_dirty = [
                ln
                for ln in dirty_lines
                if self._status_path(ln).replace("\\", "/") != self._SETTINGS_NAME
            ]
            if unsafe_dirty:
                raise RuntimeError(
                    "Local code changes would make the update unsafe. "
                    "Commit/stash them first:\n" + "\n".join(unsafe_dirty[:12])
                )

            backups = self._backup_settings()

            # Older clones accidentally still track autotuner_settings.json.
            # To make a fast-forward possible, temporarily restore the tracked
            # copy, then write the user's exact bytes back in finally.
            settings_dirty = any(
                self._status_path(ln).replace("\\", "/") == self._SETTINGS_NAME
                for ln in dirty_lines
            )
            tracked_settings = bool(
                self._run_git(
                    "ls-files", "--error-unmatch", self._SETTINGS_NAME, check=False
                ).strip()
            )
            if settings_dirty and tracked_settings:
                self.progress.emit(
                    "[Update] Backing up local settings before git fast-forward …"
                )
                self._run_git("checkout", "--", self._SETTINGS_NAME)

            self._run_git("fetch", "--prune", "origin", timeout=300.0)
            counts = self._run_git(
                "rev-list", "--left-right", "--count", f"HEAD...{upstream}"
            ).split()
            ahead = int(counts[0]) if counts else 0
            behind = int(counts[1]) if len(counts) > 1 else 0
            if ahead:
                raise RuntimeError(
                    f"Local branch has {ahead} commit(s) not on {upstream}; "
                    "refusing to auto-merge."
                )
            if behind == 0:
                if backups:
                    self._restore_settings(backups)
                    settings_restored = True
                self.finished.emit(
                    True,
                    _source_update_message(
                        "Source files already match the current Git branch."
                    ),
                )
                return

            old_head = self._run_git("rev-parse", "HEAD").strip()
            self._run_git("pull", "--ff-only", timeout=300.0)
            new_head = self._run_git("rev-parse", "HEAD").strip()
            changed = self._run_git(
                "diff", "--name-only", f"{old_head}..{new_head}", check=False
            ).splitlines()

            # Restore before installing dependencies so the GUI can keep using
            # the user's current settings even if pip takes a while.
            self._restore_settings(backups)
            settings_restored = True

            if (
                "requirements.txt" in changed
                and (self._repo_root / "requirements.txt").exists()
            ):
                self.progress.emit(
                    "[Update] requirements.txt changed; installing dependencies …"
                )
                self._run(
                    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                    timeout=900.0,
                )

            short = new_head[:8]
            self.finished.emit(
                True,
                _source_update_message(
                    f"Source update from Git branch installed ({short}). Local "
                    "settings were restored. Please restart AutoTuner."
                ),
            )
        except Exception as exc:
            if backups and not settings_restored:
                self._restore_settings(backups)
            self.finished.emit(False, str(exc))


class _BinaryUpdateWorker(QObject):
    """Self-update for a compiled AutoTuner build (PyInstaller onefile).

    The source-based ``_UpdateWorker`` replaces ``.py`` files next to the
    script — that is meaningless for a frozen binary, where the code is
    embedded in the ``.exe`` / Linux ELF. Instead this worker:

      1. asks GitHub for the latest Release of ``GITHUB_REPO``;
      2. compares the Release ``tag_name`` against :data:`VERSION`;
      3. picks the asset matching the host OS (Windows ``.exe`` / Linux
         binary / macOS app or binary) — AutoTuner runs on Windows, Ubuntu
         and macOS, so the asset selection must be OS-aware;
      4. downloads it to a temp file next to the running binary (same
         volume → atomic ``move``);
      5. writes a tiny swap shim (``.bat`` on Windows, ``.sh`` on POSIX)
         that waits for this process to exit, replaces the binary, and
         relaunches it — because Windows locks the running ``.exe`` and
         the process cannot overwrite itself;
      6. launches the shim detached, then emits ``finished``. The GUI
         quits immediately afterwards so the shim can complete the swap.

    User state survives untouched in the shared ``~/.autotuner`` directory,
    independently of where the replaced binary is installed.
    """

    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str, bool)  # (ok, message, needs_restart)

    _GITHUB_REPO = GITHUB_REPO

    def __init__(self) -> None:
        super().__init__()
        self._exe_path = Path(sys.executable).resolve()
        self._data_dir = app_settings.app_data_dir()

    # -- helpers ----------------------------------------------------------
    def _github_request(self, url: str) -> urllib.request.Request:
        headers = {
            "Accept": (
                "application/vnd.github+json"
                if "api.github.com" in url
                else "application/octet-stream"
            ),
            "User-Agent": USER_AGENT,
        }
        if "api.github.com" in url:
            headers["X-GitHub-Api-Version"] = "2022-11-28"
        return urllib.request.Request(url, headers=headers)

    def _fetch_json(self, url: str, timeout: float = 30.0) -> Dict[str, object]:
        with urllib.request.urlopen(self._github_request(url), timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            data = resp.read().decode(charset, errors="replace")
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"GitHub returned unexpected JSON for {url}")
        return parsed

    @staticmethod
    def _parse_version(tag: str) -> Tuple[int, ...]:
        """Parse a release tag into a comparable int tuple.

        Accepts ``v1.2.0``, ``1.2.0``, ``v1.2.0-beta`` (pre-release suffix
        ignored). Non-numeric leading segments are dropped.
        """
        clean = tag.strip().lstrip("vV")
        parts: List[int] = []
        for tok in clean.split("."):
            digits = ""
            for ch in tok:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                parts.append(int(digits))
            else:
                break
        return tuple(parts) if parts else (0,)

    def _pick_asset(
        self, assets: List[Dict[str, object]]
    ) -> Optional[Dict[str, object]]:
        """Choose the release asset for the current OS.

        Prefers a per-OS **zip** asset (e.g. ``AutoTuner-Windows-x64.zip``,
        ``AutoTuner-Linux-x64.zip``, ``AutoTuner-macOS-arm64.zip``) — a
        single asset that serves BOTH the beginner download AND the in-app
        auto-update (the worker extracts the binary from the zip). Falls back
        to a raw binary asset for older releases that ship an unpacked
        ``.exe`` / Linux / macOS binary.

        Windows → name contains ``windows`` or ends ``.exe``.
        Linux   → name contains ``linux`` (zip or raw ELF).
        macOS   → name contains ``macos``, ``darwin`` or ``osx``.
        """
        system = platform.system().lower()
        machine = platform.machine().strip().lower()
        if machine in {"amd64", "x86_64", "x64"}:
            arch = "x64"
        elif machine in {"arm64", "aarch64"}:
            arch = "arm64"
        else:
            arch = machine

        def _asset_arch(name: str) -> str:
            lower = name.lower()
            if any(token in lower for token in ("arm64", "aarch64")):
                return "arm64"
            if any(token in lower for token in ("x86_64", "amd64", "x64")):
                return "x64"
            return ""  # legacy/generic asset

        def _pick_arch(
            candidates: List[Dict[str, object]],
        ) -> Optional[Dict[str, object]]:
            for asset in candidates:
                if _asset_arch(str(asset.get("name", ""))) == arch:
                    return asset
            # Backward compatibility for old releases with one generic OS asset.
            for asset in candidates:
                if not _asset_arch(str(asset.get("name", ""))):
                    return asset
            return None

        if system == "windows":
            candidates = [
                asset
                for asset in assets
                if "windows" in str(asset.get("name", "")).lower()
            ]
            picked = _pick_arch(candidates)
            if picked is not None:
                return picked
            raw = [
                asset
                for asset in assets
                if str(asset.get("name", "")).lower().endswith(".exe")
            ]
            return _pick_arch(raw)
        if system == "darwin":
            candidates = [
                asset
                for asset in assets
                if any(
                    token in str(asset.get("name", "")).lower()
                    for token in ("macos", "darwin", "osx")
                )
            ]
            return _pick_arch(candidates)

        # Linux / other POSIX: never fall through here on macOS.
        candidates = [
            asset for asset in assets if "linux" in str(asset.get("name", "")).lower()
        ]
        picked = _pick_arch(candidates)
        if picked is not None:
            return picked
        raw = [
            asset
            for asset in assets
            if not str(asset.get("name", "")).lower().endswith(".exe")
            and "." not in str(asset.get("name", ""))
        ]
        return _pick_arch(raw)

    def _extract_binary_from_zip(self, zip_path: Path) -> Path:
        """Pull the AutoTuner binary out of a downloaded release zip.

        Accepts the binary at the archive root OR inside one top folder
        (the common ``AutoTuner-Windows-x64/AutoTuner.exe`` layout that
        Compress-Archive / zip produce). On Windows the member must end in
        ``.exe``; on Linux it is the ``AutoTuner-Linux`` / ``AutoTuner`` ELF;
        on macOS it is the app bundle's inner Mach-O or a raw
        ``AutoTuner-macOS`` binary. Extracts next to the running binary (same
        volume → atomic swap).
        """
        member_name: Optional[str] = None
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                base = Path(info.filename).name.lower()
                if platform.system() == "Windows":
                    if base.endswith(".exe"):
                        member_name = info.filename
                        break
                elif platform.system() == "Darwin":
                    if base in ("autotuner-macos", "autotuner") or base.endswith(
                        ("-macos", "-darwin")
                    ):
                        member_name = info.filename
                        break
                else:
                    if base in ("autotuner-linux", "autotuner") or base.endswith(
                        "-linux"
                    ):
                        member_name = info.filename
                        break
            if member_name is None:
                raise RuntimeError(f"No AutoTuner binary found inside {zip_path.name}")
            suffix = ".exe" if os.name == "nt" else ".new"
            out_fd, out_path = tempfile.mkstemp(
                prefix="autotuner_update_bin_", suffix=suffix, dir=str(self._data_dir)
            )
            os.close(out_fd)
            out_file = Path(out_path)
            with zf.open(member_name) as src, out_file.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        return out_file

    # -- swap shims -------------------------------------------------------
    def _write_windows_shim(self, new_exe: Path) -> Path:
        """A detached ``.bat`` that retries the replace then relaunches.

        Windows locks the running ``.exe``, so the move fails until the GUI
        process has exited. The shim polls (1 s × 60 ≈ 1 min), then launches
        the new binary and removes itself. ``ping`` is used as the delay
        because the shim runs without a console (``DETACHED_PROCESS``) where
        ``timeout`` errors out.
        """
        exe = self._exe_path
        shim = self._data_dir / ".autotuner_update.bat"
        bat = (
            "@echo off\r\n"
            f'set "EXE={exe}"\r\n'
            f'set "NEW={new_exe}"\r\n'
            "for /l %%i in (1,1,60) do (\r\n"
            '  move /Y "%NEW%" "%EXE%" >nul 2>&1 && goto done\r\n'
            "  ping -n 2 127.0.0.1 >nul\r\n"
            ")\r\n"
            "exit /b 1\r\n"
            ":done\r\n"
            'start "" "%EXE%"\r\n'
            '(del "%~f0")\r\n'
        )
        shim.write_text(bat, encoding="ascii", errors="ignore")
        return shim

    def _write_linux_shim(self, new_bin: Path) -> Path:
        """A detached ``.sh`` that replaces the binary and relaunches.

        Linux does not lock the running ELF — ``mv`` over it succeeds at
        once (the running process keeps the old inode) — so this is a
        single attempt followed by a relaunch.
        """
        exe = self._exe_path
        shim = self._data_dir / ".autotuner_update.sh"
        sh = (
            "#!/bin/sh\n"
            f'EXE="{exe}"\n'
            f'NEW="{new_bin}"\n'
            'if mv -f "$NEW" "$EXE" 2>/dev/null; then\n'
            '  chmod +x "$EXE"\n'
            '  nohup "$EXE" >/dev/null 2>&1 &\n'
            "fi\n"
            'rm -f "$0"\n'
        )
        shim.write_text(sh, encoding="utf-8")
        shim.chmod(0o755)
        return shim

    def _launch_shim(self, shim: Path) -> None:
        if os.name == "nt":
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: no console,
            # survives the parent GUI quitting.
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            DETACHED_PROCESS = 0x00000008
            subprocess.Popen(
                ["cmd", "/c", str(shim)],
                cwd=str(self._data_dir),
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                close_fds=True,
            )
        else:
            subprocess.Popen(
                ["/bin/sh", str(shim)],
                cwd=str(self._data_dir),
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )

    # -- main -------------------------------------------------------------
    def run(self) -> None:
        try:
            self.progress.emit("[Update] Checking latest GitHub Release …")
            api = f"https://api.github.com/repos/{self._GITHUB_REPO}/releases/latest"
            release = self._fetch_json(api)
            tag = str(release.get("tag_name") or "").strip()
            if not tag:
                raise RuntimeError("latest release has no tag_name")
            self.progress.emit(f"[Update] Latest release: {tag} (running v{VERSION})")

            latest = self._parse_version(tag)
            current = self._parse_version(VERSION)
            if latest <= current:
                self.finished.emit(
                    True, f"AutoTuner v{VERSION} is up to date (latest {tag}).", False
                )
                return

            raw_assets = release.get("assets", [])
            if not isinstance(raw_assets, list):
                raw_assets = []
            assets: List[Dict[str, object]] = [
                a for a in raw_assets if isinstance(a, dict)
            ]
            asset = self._pick_asset(assets)
            if asset is None:
                names = ", ".join(str(a.get("name", "?")) for a in assets) or "(none)"
                raise RuntimeError(
                    f"Release {tag} has no asset for this OS "
                    f"({platform.system()}). Assets: {names}"
                )
            url = str(asset.get("browser_download_url") or "")
            name = str(asset.get("name", "update"))
            if not url:
                raise RuntimeError(f"asset {name} has no download URL")
            size_raw = asset.get("size") or 0
            size = int(size_raw) if isinstance(size_raw, (int, float)) else 0
            self.progress.emit(
                f"[Update] Downloading {name} ({size / 1048576:.1f} MB) …"
            )

            # Temp file on the SAME volume as the binary → atomic move.
            suffix = (
                ".zip"
                if name.lower().endswith(".zip")
                else (".exe" if os.name == "nt" else ".new")
            )
            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix="autotuner_update_", suffix=suffix, dir=str(self._data_dir)
            )
            os.close(tmp_fd)
            tmp_file = Path(tmp_path)
            with urllib.request.urlopen(
                self._github_request(url), timeout=600.0
            ) as resp:
                with tmp_file.open("wb") as fh:
                    shutil.copyfileobj(resp, fh)
            if tmp_file.stat().st_size == 0:
                raise RuntimeError(f"downloaded {name} is empty")

            # Release assets ship as a per-OS zip (one asset serves the
            # beginner download AND this auto-update). Extract the binary;
            # older releases with a raw .exe / ELF work unchanged.
            if name.lower().endswith(".zip"):
                self.progress.emit(f"[Update] Extracting binary from {name} …")
                bin_path = self._extract_binary_from_zip(tmp_file)
                try:
                    tmp_file.unlink()
                except OSError:
                    pass
            else:
                bin_path = tmp_file

            if os.name == "nt":
                shim = self._write_windows_shim(bin_path)
            else:
                shim = self._write_linux_shim(bin_path)
            self._launch_shim(shim)

            self.finished.emit(
                True,
                f"Update to {tag} downloaded. AutoTuner will restart to "
                "complete the swap.",
                True,
            )
        except Exception as exc:
            self.finished.emit(False, str(exc), False)


# ---------------------------------------------------------------------------
# Draft-model detection helper (mirrors auto_tuner.py logic)

# ---------------------------------------------------------------------------
# Draft-model lookup
#
# scanner.py already pairs each main model with its assistant/draft
# sibling (when present) and stores the path in `entry.draft`. We just
# wrap that path in a ModelEntry so the rest of the launcher (which
# expects a ModelEntry with `.path` and `.size_gb`) keeps working.


def _make_draft_entry(p: Path, group: str) -> Optional[ModelEntry]:
    """Build a ModelEntry for a draft GGUF at ``p``.

    Metadata is read so the drafter's :attr:`is_standalone_drafter` resolves
    correctly — that flag drives whether the launcher emits
    ``--spec-type draft-mtp`` (required for Gemma 4 gemma4-assistant heads,
    which do not auto-detect from ``-md`` alone). Returns None if the file
    has vanished.
    """
    try:
        size = p.stat().st_size
    except OSError:
        return None
    return ModelEntry(
        path=p,
        name=p.stem,
        group=group,
        size_bytes=size,
        mmproj=None,
        draft=None,
        metadata=read_gguf_metadata(p),
    )


def _find_draft_model(
    entry: ModelEntry, all_entries: List[ModelEntry]
) -> Optional[ModelEntry]:
    """Return a ModelEntry for `entry`'s auto-paired draft, or None."""
    if entry.draft is None:
        return None
    return _make_draft_entry(entry.draft, entry.group)


def _drafter_profile_key(value: object) -> str:
    """Return the stable profile axis for none, embedded, or external heads."""
    if value is None:
        return app_settings.NO_DRAFTER_PROFILE_KEY
    if isinstance(value, ModelEntry):
        path = value.path
    else:
        text = str(value or "").strip()
        if not text or text == app_settings.DRAFT_NONE_SENTINEL:
            return app_settings.NO_DRAFTER_PROFILE_KEY
        if text == app_settings.DRAFT_EMBEDDED_SENTINEL:
            return app_settings.EMBEDDED_DRAFTER_PROFILE_KEY
        path = Path(text)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return f"external:{path.name.casefold()}|{max(0, int(size))}"


# Capability markers shown next to the model name in the list. Keep
# Terminal and GUI in sync — both pull from this single source.
#
#   👁  vision     (mmproj projector found)
#   ⚡  draft      (assistant/draft sibling found → speculative decoding)
#   🧠  thinking   (chat template emits <think> / reasoning_content)
#   🛠  tool-use   (chat template advertises tool_calls / function_call)


def _capability_markers(entry: ModelEntry) -> str:
    """Return a small symbol string summarising what this model supports."""
    syms: List[str] = []
    if entry.has_vision:
        syms.append("👁")
    if entry.has_speculative_draft:  # covers both external GGUF and embedded MTP
        syms.append("⚡")
    if entry.supports_thinking:
        syms.append("🧠")
    if entry.supports_tool_use:
        syms.append("🛠")
    return " ".join(syms)


_FAVORITE_ROLE = int(Qt.ItemDataRole.UserRole) + 1
_TREE_PATH_ROLE = int(Qt.ItemDataRole.UserRole) + 2


class _FavoriteStarDelegate(QStyledItemDelegate):
    """Draw and handle the favorite star without replacing list-row widgets."""

    favoriteToggled = pyqtSignal(object, bool)

    @staticmethod
    def _star_rect(option) -> QRect:
        return QRect(
            option.rect.left() + 3,
            option.rect.top(),
            24,
            option.rect.height(),
        )

    @staticmethod
    def _text_rect(option) -> QRect:
        # Reserve a fixed device-independent area instead of relying on spaces,
        # whose width varies across fonts, DPI settings, and operating systems.
        return option.rect.adjusted(31, 0, 0, 0)

    def paint(self, painter, option, index) -> None:
        # Folder/header rows in the tree carry no ModelEntry. Render those with
        # the native Qt delegate so indentation and expand arrows remain intact.
        if index.data(Qt.ItemDataRole.UserRole) is None:
            super().paint(painter, option, index)
            return
        text_option = QStyleOptionViewItem(option)
        text_option.rect = self._text_rect(option)
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        # Let Qt render focus, text, and platform theming in the remaining area.
        super().paint(painter, text_option, index)
        favorite = bool(index.data(_FAVORITE_ROLE))
        painter.save()
        font = painter.font()
        font.setBold(favorite)
        if font.pointSizeF() > 0:
            font.setPointSizeF(max(13.0, font.pointSizeF()))
        painter.setFont(font)
        manager = getattr(QApplication.instance(), "theme_manager", None)
        color = (
            manager.favorite_color(favorite)
            if isinstance(manager, ThemeManager)
            else "#777777"
        )
        painter.setPen(QColor(color))
        painter.drawText(
            self._star_rect(option),
            Qt.AlignmentFlag.AlignCenter,
            "★",
        )
        painter.restore()

    def editorEvent(self, event, model, option, index) -> bool:
        if (
            event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
            and self._star_rect(option).contains(event.position().toPoint())
        ):
            entry = index.data(Qt.ItemDataRole.UserRole)
            if entry is not None:
                self.favoriteToggled.emit(entry, not bool(index.data(_FAVORITE_ROLE)))
                return True
        return super().editorEvent(event, model, option, index)


def _sort_model_entries(
    entries: List[ModelEntry], favorite_models: set[str]
) -> List[ModelEntry]:
    """Keep the established group/name order, with all favorites first."""
    groups = group_entries(entries)
    ordered = [
        entry
        for group_name in sorted(groups.keys())
        for entry in sorted(groups[group_name], key=lambda model: model.name.lower())
    ]
    return sorted(
        ordered,
        key=lambda entry: (
            app_settings.favorite_model_key(entry.path) not in favorite_models
        ),
    )


def _model_display_text(entry: ModelEntry) -> str:
    """Return the shared model-row label used by list and folder views."""
    marks = _capability_markers(entry)
    tail = f"  ({entry.size_gb:.1f} GB)"
    return f"{entry.name}  {marks}{tail}" if marks else f"{entry.name}{tail}"


MODEL_TOOLTIP_FAVORITE = "Favourite: click the star to remove it"
MODEL_TOOLTIP_NOT_FAVORITE = "Not a favourite: click the star to mark it"
MODEL_TOOLTIP_THINKING = "Thinking: the chat template emits <think>"
MODEL_TOOLTIP_TOOL_USE = "Tool use: the chat template supports tool_calls"
MODEL_TOOLTIP_VISION = "Vision:"
MODEL_TOOLTIP_DRAFT = "Draft:"


def _model_tooltip(
    entry: ModelEntry,
    favorite: bool,
    translate: Optional[Callable[[str], str]] = None,
) -> str:
    """Describe favorite state and paired capabilities for one model row.

    Item tooltips are plain strings on ``QListWidgetItem``/``QTreeWidgetItem``
    rows, which the live widget retranslation cannot reach, so the caller
    passes the active language's ``translate`` callable.
    """
    tr = translate or (lambda text: text)
    state = tr(MODEL_TOOLTIP_FAVORITE) if favorite else tr(MODEL_TOOLTIP_NOT_FAVORITE)
    lines = [entry.name, str(entry.path), "", f"★  {state}"]
    if entry.mmproj is not None:
        lines.append(f"👁  {tr(MODEL_TOOLTIP_VISION)} {entry.mmproj.name}")
    if entry.draft is not None:
        lines.append(f"⚡  {tr(MODEL_TOOLTIP_DRAFT)} {entry.draft.name}")
    if entry.supports_thinking:
        lines.append(f"🧠  {tr(MODEL_TOOLTIP_THINKING)}")
    if entry.supports_tool_use:
        lines.append(f"🛠  {tr(MODEL_TOOLTIP_TOOL_USE)}")
    return "\n".join(lines)


def _model_folder_parts(entry: ModelEntry, roots: List[Path]) -> Tuple[str, ...]:
    """Return stable display folders for an entry under one or more scan roots.

    A single configured model root is omitted because its children are already
    the useful vendor/family hierarchy. With multiple roots, a unique root
    label is prepended so equally named folders on different drives never merge.
    """
    try:
        parent = entry.path.parent.resolve(strict=False)
    except (OSError, RuntimeError):
        parent = entry.path.parent

    normalized_roots: List[Path] = []
    for root in roots:
        try:
            normalized_roots.append(root.resolve(strict=False))
        except (OSError, RuntimeError):
            normalized_roots.append(root)

    for root in normalized_roots:
        try:
            relative = parent.relative_to(root)
        except ValueError:
            continue
        parts = list(relative.parts)
        if len(normalized_roots) > 1:
            label = root.name or str(root)
            duplicate = (
                sum(
                    1
                    for candidate in normalized_roots
                    if (candidate.name or str(candidate)) == label
                )
                > 1
            )
            parts.insert(0, str(root) if duplicate else label)
        return tuple(part for part in parts if part not in ("", "."))

    # Defensive fallback for synthetic entries and paths moved after scanning.
    return tuple(
        part for part in re.split(r"[\\/]", entry.group) if part not in ("", ".")
    )


def _clean_model_name(name: str) -> str:
    """Strip quant/distributor suffixes for a clean --alias name."""
    import re as _re

    clean = _re.sub(
        r"[-_]?(?:iq\d+(?:_+[a-z\d]+)*(?:[-_]\d+[.\d]*bpw)?|"
        r"q\d+(?:_+[a-z\d]+)*|tf\d+|bf16|f16|f32)$",
        "",
        name,
        flags=_re.IGNORECASE,
    ).strip("-_")
    return _re.sub(r"[-_](?:ud|unsloth)$", "", clean, flags=_re.IGNORECASE).strip("-_")


def _extract_server_api_key(command: Sequence[object]) -> Optional[str]:
    """Return the llama-server credential without logging or exposing its file."""
    args = [str(value) for value in command]
    for index, arg in enumerate(args):
        if arg == "--api-key" and index + 1 < len(args):
            return args[index + 1] or None
        if arg.startswith("--api-key="):
            return arg.split("=", 1)[1] or None
        if arg == "--api-key-file" and index + 1 < len(args):
            path = Path(args[index + 1])
            try:
                if path.stat().st_size > 1024 * 1024:
                    return None
                for line in path.read_text(encoding="utf-8").splitlines():
                    candidate = line.strip()
                    if candidate and not candidate.startswith("#"):
                        return candidate
            except (OSError, UnicodeError):
                return None
    return None


#: Launch arguments whose value is a credential. ``--connect-code`` (b12187,
#: PR #28277) is the 40-character code that opens the server's peer-to-peer
#: tunnel to anyone who knows it.
_CREDENTIAL_FLAGS = (
    "--api-key",
    "--api-key-file",
    "--connect-code",
    "--hf-token",
    "-hft",
)


def _redacted_command(command: Sequence[object]) -> str:
    """Format a launch command while removing direct credential arguments."""
    args = [str(value) for value in command]
    redacted: List[str] = []
    hide_next = False
    for arg in args:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        flag = arg.split("=", 1)[0]
        if flag in _CREDENTIAL_FLAGS and "=" in arg:
            redacted.append(f"{flag}=<redacted>")
        elif arg in _CREDENTIAL_FLAGS:
            redacted.append(arg)
            hide_next = True
        else:
            redacted.append(arg)
    return " ".join(redacted)


def _control_model_id(entry: ModelEntry) -> str:
    """Return a readable, path-stable external ID for one GGUF entry."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", entry.name).strip("-._").lower()
    slug = (slug or "model")[:96].rstrip("-._") or "model"
    try:
        identity = os.path.normcase(str(entry.path.resolve(strict=False)))
    except (OSError, RuntimeError):
        identity = os.path.normcase(str(entry.path))
    digest = hashlib.sha256(identity.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{slug}--{digest}"


_CONTROL_QUANT_RE = re.compile(
    r"(?<![A-Za-z0-9])((?:UD-)?(?:i\d+-)?(?:IQ\d+(?:_[A-Z0-9]+)*|Q\d+(?:_[A-Z0-9]+)*"
    r"|MXFP4(?:_MOE)?|NVFP4|TQ\d+_\d+|BF16|F16|F32))(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _control_quant_label(name: str) -> str:
    """Return the trailing quantisation token of a model name, if any."""
    matches = _CONTROL_QUANT_RE.findall(str(name or ""))
    return matches[-1].upper() if matches else ""


def _control_runtime_id(label: str) -> str:
    """Return the stable slug external clients use to select a llama build."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(label or "").lower()).strip("-")
    return slug[:96].rstrip("-") or "runtime"


def _control_runtime_backend(label: str, binary: str) -> str:
    """Classify a build by its directory/binary identity without probing it."""
    identity = f"{label} {binary}".lower()
    aliases = (
        ("hip", ("_hip_", "-hip-", "rocm", "hip_llama", "hip/")),
        ("vulkan", ("_vulkan_", "-vulkan-", "vulkan")),
        ("cuda", ("_cuda_", "-cuda-", "cuda")),
        ("sycl", ("_sycl_", "-sycl-", "oneapi", "sycl")),
        ("metal", ("_metal_", "-metal-", "metal")),
        ("opencl", ("_opencl_", "-opencl-", "opencl")),
        ("cpu", ("_cpu_", "-cpu-", "cpu-only", "cpu_only")),
    )
    for backend, tokens in aliases:
        if any(token in identity for token in tokens):
            return backend
    return "unknown"


def _command_value(command: Sequence[object], *flags: str) -> Optional[str]:
    """Return the argument following the last occurrence of any *flags*."""
    args = [str(value) for value in command]
    found: Optional[str] = None
    for index, arg in enumerate(args):
        if arg in flags and index + 1 < len(args):
            found = args[index + 1]
        else:
            for flag in flags:
                if arg.startswith(flag + "="):
                    found = arg[len(flag) + 1 :]
    return found


def _redacted_command_list(command: Sequence[object]) -> List[str]:
    """List form of :func:`_redacted_command` for JSON consumers."""
    return _redacted_command(command).split(" ") if command else []


class _LongMessageDialog(QDialog):
    """Scrollable replacement for oversized ``QMessageBox`` reports."""

    def __init__(
        self,
        title: str,
        message: str,
        icon: QMessageBox.Icon = QMessageBox.Icon.Information,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        icon_label = QLabel()
        pixmap_icon = self.style().standardIcon(
            {
                QMessageBox.Icon.Warning: QStyle.StandardPixmap.SP_MessageBoxWarning,
                QMessageBox.Icon.Critical: QStyle.StandardPixmap.SP_MessageBoxCritical,
                QMessageBox.Icon.Question: QStyle.StandardPixmap.SP_MessageBoxQuestion,
            }.get(icon, QStyle.StandardPixmap.SP_MessageBoxInformation)
        )
        icon_label.setPixmap(pixmap_icon.pixmap(32, 32))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        header.addWidget(icon_label, 0)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlainText(message)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.text.setFrameShape(QFrame.Shape.NoFrame)
        header.addWidget(self.text, 1)
        layout.addLayout(header)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._buttons = buttons

        # Size to the content but never beyond ~75% of the available screen.
        metrics = self.text.fontMetrics()
        lines = message.count("\n") + 1
        longest = max(
            (metrics.horizontalAdvance(line) for line in message.splitlines()),
            default=0,
        )
        wanted_width = min(max(520, longest + 96), 1100)
        wanted_height = min(max(200, lines * metrics.lineSpacing() + 110), 100_000)
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            wanted_width = min(wanted_width, int(available.width() * 0.9))
            wanted_height = min(wanted_height, int(available.height() * 0.75))
        self.resize(wanted_width, wanted_height)
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setDefault(True)
            ok_button.setFocus()


def _control_api_catalogue(
    entries: Sequence[ModelEntry],
    profiles: Sequence[ModelProfile],
    default_runtime_id: str = "",
) -> List[ApiModel]:
    """Describe every scanned model and mark non-server runners explicitly."""
    models: List[ApiModel] = []
    used_ids: set[str] = set()
    profile_list = list(profiles)
    for entry in entries:
        profile = match_profile(
            entry.name, profile_list, getattr(entry, "architecture", "")
        )
        runner = str(profile.runner or "llama-server")
        extra = {str(arg).strip().casefold() for arg in profile.extra_args}
        reason = ""
        if entry.is_standalone_drafter:
            reason = "Standalone draft models cannot serve requests by themselves."
        elif entry.is_diffusion and runner != "llama-diffusion-gemma-server":
            reason = "This diffusion model uses a single-shot CLI rather than an HTTP server."
        elif "--embeddings" in extra or "--embedding" in extra:
            reason = "This profile exposes embeddings rather than chat completions."

        model_id = _control_model_id(entry)
        if model_id in used_ids:
            # A 40-bit path digest collision is extraordinarily unlikely, but
            # never let it silently remove a model from the external catalogue.
            suffix = hashlib.sha256(str(entry.path).encode("utf-8")).hexdigest()[:16]
            model_id = f"{model_id.rsplit('--', 1)[0]}--{suffix}"
        used_ids.add(model_id)

        limits = [
            int(value)
            for value in (entry.native_context, profile.max_context)
            if isinstance(value, int) and value > 0
        ]
        context_window = min(limits) if limits else 8192
        max_tokens = max(256, min(16_384, context_window // 2))
        overrides = app_settings.get_model_overrides(entry.name)
        remembered_mmproj = app_settings.get_mmproj_selection(entry.name)
        vision = bool(entry.mmproj)
        if remembered_mmproj == app_settings.MMPROJ_NONE_SENTINEL or (
            remembered_mmproj is None and overrides.get("vision") is False
        ):
            vision = False

        models.append(
            ApiModel(
                id=model_id,
                name=entry.name,
                path=str(entry.path),
                context_window=context_window,
                max_tokens=max_tokens,
                reasoning=entry.supports_thinking,
                input_types=("text", "image") if vision else ("text",),
                runnable=not reason,
                unavailable_reason=reason,
                size_bytes=int(entry.size_bytes or 0),
                quant=_control_quant_label(entry.name),
                params_b=float(extract_params_billion(entry.name) or 0.0),
                family=str(profile.display_name or ""),
                architecture=str(getattr(entry, "architecture", "") or ""),
                default_runtime_id=str(default_runtime_id or ""),
            )
        )
    return models


# ---------------------------------------------------------------------------
# Expert-panel value helpers (widget-state ↔ config)
#
# The Expert panel edits a flat set of widgets (spinboxes, combos,
# checkboxes, a line edit). Two consumers need the SAME translation
# logic:
#
#   • the live panel — reads widgets, writes a TunedConfig at launch;
#   • the persisted snapshot — a model's saved Expert state is applied
#     for preview/launch WITHOUT the panel being open.
#
# Factoring the translation into free functions that operate on a plain
# ``values`` dict (instead of on widgets) lets both paths share one
# implementation, so the on-screen panel and the disk-restored config
# can never drift apart.
#
# ``values`` keys: ctx, cache_k, cache_v, ngl, n_cpu_moe, threads,
# batch_threads, batch, ubatch, flash_attn, load_mode, jinja,
# verbose, numa, rope_scaling, rope_factor, temperature, top_k,
# top_p, min_p, repeat_penalty, presence_penalty, reasoning,
# think_budget, reasoning_preserve, parallel_enabled, parallel_count,
# metrics_enabled, slots_api_enabled, extras.


def _expert_sampling_from_values(vals: dict) -> dict:
    """Build the sampling dict the launcher expects from a values snapshot."""
    return {
        "temperature": float(vals.get("temperature", 0.7)),
        "top_k": int(vals.get("top_k", 40)),
        "top_p": float(vals.get("top_p", 0.9)),
        "min_p": float(vals.get("min_p", 0.05)),
        "repeat_penalty": float(vals.get("repeat_penalty", 1.05)),
        "presence_penalty": float(vals.get("presence_penalty", 0.0)),
    }


def _reasoning_flags_from_values(reasoning, think_budget) -> List[str]:
    """Translate a reasoning-effort label + think budget into CLI flags.

    Mirror of ``ExpertPanel._reasoning_flags_from_widgets`` but driven by
    plain values so it works for both the live panel and a disk snapshot.
    See that method for the full mapping rules.
    """
    out: List[str] = []
    choice = str(reasoning or "auto").strip().lower()
    if choice == "off":
        out += ["--reasoning", "off"]
    elif choice and choice != "auto":
        # "extra_high" intentionally kept with underscore — that's the
        # spelling Qwen3.6 community templates use.
        payload = '{"reasoning_effort":"' + choice + '"}'
        out += ["--chat-template-kwargs", payload]
    try:
        budget = int(think_budget if think_budget is not None else -1)
    except (TypeError, ValueError):
        budget = -1
    if budget >= 0:
        # llama.cpp renamed --think-budget → --reasoning-budget at b9625.
        out += ["--reasoning-budget", str(budget)]
    return out


def _expert_extras_from_values(vals: dict) -> List[str]:
    """Rebuild the extra_cli_flags list from a values snapshot."""
    extras: List[str] = []
    if vals.get("jinja"):
        extras.append("--jinja")
    if vals.get("verbose"):
        extras.append("--verbose")
    extras.extend(
        _reasoning_flags_from_values(
            vals.get("reasoning", "auto"), vals.get("think_budget", -1)
        )
    )
    if vals.get("reasoning_preserve"):
        extras.append("--reasoning-preserve")
    free = (vals.get("extras") or "").strip()
    if free:
        extras.extend(free.split())
    return extras


_EXPERT_LOAD_MODES = {"auto", "none", "mmap", "mlock", "mmap+mlock", "dio"}


def _expert_load_mode_from_values(cfg: TunedConfig, vals: dict) -> str:
    """Normalize new load-mode snapshots and migrate the old checkboxes."""
    if "load_mode" in vals:
        mode = str(vals.get("load_mode") or "auto").strip().lower()
        return mode if mode in _EXPERT_LOAD_MODES else "auto"
    if "mlock" in vals or "no_mmap" in vals:
        locked = bool(vals.get("mlock", getattr(cfg, "mlock", False)))
        no_mmap = bool(vals.get("no_mmap", getattr(cfg, "no_mmap", False)))
        if locked:
            return "mlock" if no_mmap else "mmap+mlock"
        return "none" if no_mmap else "auto"
    return effective_load_mode(cfg) or "auto"


def _expert_cascading_pins(snapshot: dict) -> dict:
    """Include legacy FA choices in the memory/KV cascade, not a late overlay."""
    pins = {k: v for k, v in (snapshot.get("pins") or {}).items() if v is not None}
    values = snapshot.get("values") or {}
    if "force_flash_attn" not in pins and "flash_attn" in values:
        pins["force_flash_attn"] = bool(values["flash_attn"])
    return pins


def apply_expert_values(cfg: TunedConfig, vals: dict) -> TunedConfig:
    """Overlay the NON-cascading values onto ``cfg`` (in place + returned).

    Used both by the live panel (``_apply_noncascading``) and by the
    override-aware launch path: after compute_config runs with the saved
    auto-mode pins, the user's threads / batch / flags / sampling /
    reasoning choices are stamped back on so they survive the recompute.
    Cascading fields (ctx, KV quants, ngl, n_cpu_moe, rope) are left
    untouched — those belong to compute_config.
    """
    planned_parallel = cfg.n_parallel
    try:
        if vals.get("threads"):
            cfg.threads = int(vals["threads"]) or cfg.threads
        if vals.get("batch_threads"):
            cfg.batch_threads = int(vals["batch_threads"]) or cfg.batch_threads
        if vals.get("batch"):
            cfg.batch = int(vals["batch"]) or cfg.batch
        if vals.get("ubatch"):
            cfg.ubatch = int(vals["ubatch"]) or cfg.ubatch
        load_mode = _expert_load_mode_from_values(cfg, vals)
        cfg.load_mode = load_mode
        # Keep the legacy fields synchronized for external callers and the
        # conservative mlock safety gate. Command generation uses load_mode.
        cfg.mlock = load_mode in {"mlock", "mmap+mlock"}
        cfg.no_mmap = load_mode in {"none", "mlock"}
        numa_choice = str(vals.get("numa", "off") or "off")
        cfg.numa = None if numa_choice == "off" else numa_choice
        cfg.sampling = _expert_sampling_from_values(vals)
        cfg.extra_cli_flags = _expert_extras_from_values(vals)
        try:
            cfg.draft_n_max = max(
                0, int(vals.get("draft_n_max", getattr(cfg, "draft_n_max", 0)) or 0)
            )
        except (TypeError, ValueError):
            cfg.draft_n_max = 0
        cfg.metrics_enabled = bool(
            vals.get("metrics_enabled", getattr(cfg, "metrics_enabled", True))
        )
        cfg.slots_api_enabled = bool(
            vals.get("slots_api_enabled", getattr(cfg, "slots_api_enabled", False))
        )
        # Parallel-slots override (--parallel / -np). When enabled, pin
        # the count and mark it forced so the panel can render the
        # checkbox state; when disabled, leave whatever compute_config
        # derived from the performance target and clear the flag.
        if vals.get("parallel_enabled"):
            try:
                cfg.n_parallel = max(
                    1, int(vals.get("parallel_count", cfg.n_parallel) or cfg.n_parallel)
                )
            except (TypeError, ValueError):
                pass
            cfg.n_parallel_forced = True
        else:
            cfg.n_parallel_forced = False
    except Exception:
        pass
    if cfg.adaptive_memory:
        cfg.batch, cfg.ubatch = min(cfg.batch, 256), min(cfg.ubatch, 128)
        cfg.load_mode, cfg.no_mmap, cfg.mlock = "none", True, False
        cfg.n_parallel = planned_parallel
        cfg.extra_cli_flags = [
            arg
            for arg in cfg.extra_cli_flags
            if arg not in {"--op-offload", "--no-op-offload"}
        ]
        cfg.extra_cli_flags.append("--no-op-offload")
    return cfg


def expert_cfg_from_values(base: TunedConfig, vals: dict) -> TunedConfig:
    """Build a frozen (manual) TunedConfig from ``base`` + a values snapshot.

    Every editable field is taken from ``vals``; unmodelled fields
    (tensor_split, main_gpu, env_overrides, VRAM estimates, …) are
    inherited from ``base`` so the result is still a complete config.
    This is the disk equivalent of ``ExpertPanel._build_manual_config``.
    """
    cfg = copy.copy(base)
    base_ctx = max(1, int(base.ctx or 1))
    base_parallel = max(1, int(base.n_parallel or 1))
    cfg.ctx = int(vals.get("ctx", base.ctx))
    cfg.cache_k = str(vals.get("cache_k", base.cache_k))
    cfg.cache_v = str(vals.get("cache_v", base.cache_v))
    cfg.flash_attn = bool(vals.get("flash_attn", base.flash_attn))
    cfg.ngl = int(vals.get("ngl", base.ngl))
    try:
        n_cpu = int(vals.get("n_cpu_moe", 0) or 0)
    except (TypeError, ValueError):
        n_cpu = 0
    cfg.n_cpu_moe = n_cpu if n_cpu > 0 else None
    cfg.rope_scaling = bool(vals.get("rope_scaling", base.rope_scaling))
    try:
        cfg.rope_scale_factor = float(
            vals.get("rope_factor", base.rope_scale_factor) or 1.0
        )
    except (TypeError, ValueError):
        cfg.rope_scale_factor = float(base.rope_scale_factor or 1.0)
    # Non-cascading overlay (threads / batch / flags / sampling / reasoning)
    apply_expert_values(cfg, vals)
    # Manual snapshots own their context/slots/quants. Scale K and V
    # separately, including unequal MLA heads, rather than inheriting the
    # Q8 estimate when a frozen snapshot actually requests F16/BF16.
    key_share = max(0.0, min(1.0, base.kv_key_fraction))
    key_scale = key_share * kv_quant_factor(cfg.cache_k) / kv_quant_factor(base.cache_k)
    value_scale = (
        (1.0 - key_share) * kv_quant_factor(cfg.cache_v) / kv_quant_factor(base.cache_v)
    )
    quant_scale = key_scale + value_scale
    cfg.kv_key_fraction = key_scale / quant_scale if quant_scale > 0 else 0.5
    kv_scale = (
        quant_scale
        * (max(1, int(cfg.ctx or 1)) * max(1, int(cfg.n_parallel or 1)))
        / (base_ctx * base_parallel)
    )
    if abs(kv_scale - 1.0) > 1e-9:
        cfg.estimated_kv_gb = max(0.0, base.estimated_kv_gb * kv_scale)
        cfg.kv_vram_gb = max(0.0, base.kv_vram_gb * kv_scale)
        cfg.kv_ram_gb = max(0.0, base.kv_ram_gb * kv_scale)
    cfg.kv_quant_strategy = "manual"
    return cfg


# ---------------------------------------------------------------------------
# Expert panel — editable settings overlay
# ---------------------------------------------------------------------------


_TURBO_KV_TYPES = frozenset(
    {"turbo2", "turbo2_tcq", "turbo3", "turbo3_tcq", "turbo4", "tq3_0"}
)


def _is_turbo_kv_type(value: str) -> bool:
    return str(value or "").strip().lower() in _TURBO_KV_TYPES


def _show_turbo_kv_fork_warning(parent: QWidget) -> bool:
    """Warn once per Expert-panel session about fork-only cache formats."""
    if app_settings.get_turbo_kv_warning_suppressed():
        return False
    box = QMessageBox(parent)
    box.setWindowTitle("TurboQuant fork required")
    box.setIcon(QMessageBox.Icon.Warning)
    box.setText("Turbo KV-cache formats are not available in mainline llama.cpp yet.")
    box.setInformativeText(
        "Select a compatible TurboQuant fork before launching with turbo2, "
        "turbo3, or turbo4. A normal mainline build will reject these cache "
        "types during startup."
    )
    dismiss_button = box.addButton("Dismiss", QMessageBox.ButtonRole.AcceptRole)
    never_button = box.addButton(
        "Never Show Again", QMessageBox.ButtonRole.DestructiveRole
    )
    box.setDefaultButton(dismiss_button)
    box.exec()
    if box.clickedButton() is never_button:
        app_settings.set_turbo_kv_warning_suppressed(True)
    return True


class ExpertPanel(QWidget):
    """Editable replacement for the read-only config preview.

    Lives inside a ``QStackedWidget`` paired with the preview, so toggling
    Expert mode just switches the visible page — the surrounding layout
    (Launch options below, log panel underneath) does not move.

    Two sub-modes:

    * **Auto** — every widget edit recomputes the rest via ``compute_config``
      with the matching ``force_*`` parameter. The view re-populates from
      the new config so cascade effects are visible immediately. The
      Expert override values are kept in ``self._user_pins`` and reapplied
      on every recompute (so pinning ctx=32k then changing K-quant keeps
      ctx pinned).
    * **Manual** — edits go straight into the local widget state and are
      assembled into a ``TunedConfig`` at launch time. No cascade, no
      recompute. The user owns the consequences.

    A signal is emitted when the user wants to leave Expert mode entirely
    (the parent swaps the stacked widget back to the preview page).
    """

    # Emitted with the current configuration after any cascading recompute,
    # so the parent window can refresh its memory-estimate footer.
    configChanged = pyqtSignal(object)  # TunedConfig
    # Emitted with the new mode name when the user toggles Auto/Manual.
    modeChanged = pyqtSignal(str)  # "auto" | "manual"
    # Emitted when the user clicks the close (×) button.
    closeRequested = pyqtSignal()
    # Emitted (debounced) with a full panel snapshot whenever the user
    # edits any Expert widget in either mode, so the parent can persist
    # the state per model. The snapshot shape is
    #   {"mode": str, "pins": dict, "values": dict, "saved_at": str}.
    # Programmatic population (load / reset) does NOT emit this.
    stateChanged = pyqtSignal(dict)
    # Emitted when the user clicks the Reset button — the parent clears
    # the saved override for the current model and reloads pure Auto.
    resetRequested = pyqtSignal()

    # Mainline types are followed by fork-only TurboQuant formats. BF16 has
    # the same memory footprint as F16 and is available for exact Expert use;
    # Auto prefers Q8_0, with Q4_0 only when the target exceeds its budget.
    # Selecting a Turbo type shows a special-fork warning.
    _KV_QUANT_OPTIONS = [
        "q8_0",
        "q4_0",
        "q4_1",
        "iq4_nl",
        "q5_0",
        "q5_1",
        "f16",
        "bf16",
        "turbo4",
        "turbo3",
        "turbo2",
    ]
    _NUMA_OPTIONS = ["off", "distribute", "isolate", "numactl"]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # Recompute callback: parent sets this so we can call
        # compute_config with the current model/system/profile in Auto
        # mode. Signature: (force_overrides: dict) -> Optional[TunedConfig]
        self._recompute_cb = None

        # Persistent overrides the user has pinned in Auto mode. Keys
        # are compute_config kwarg names ("force_cache_k", "user_ctx",
        # …). A None entry means "release this pin" — equivalent to
        # popping the key, but kept distinct so we can show in the
        # log what the user explicitly released.
        self._user_pins: dict = {}

        # Cached last config we displayed — needed by Manual mode to
        # build the final TunedConfig at launch time.
        self._last_cfg: Optional[TunedConfig] = None

        # Hardware snapshot used to clamp ctx slider etc. Set by parent
        # on every mode switch.
        self._system: Optional[SystemInfo] = None
        self._native_ctx: int = 0  # native_context from GGUF (0 = unknown)
        self._profile_max: int = 8192  # YAML max_context

        # Guard flag — when True we are programmatically setting widget
        # values inside `_populate_from_cfg`, so the valueChanged signals
        # must NOT trigger a recompute (which would either be a no-op
        # echo or an infinite loop) NOR schedule a debounced save (which
        # would persist the just-loaded state back over itself).
        self._populating = False
        self._turbo_warning_shown = False

        # Debounced auto-save. Any real user edit (in either mode) arms a
        # single-shot 300 ms timer; when it fires we emit `stateChanged`
        # with a fresh snapshot so the parent can persist it per model.
        # 300 ms collapses a spinbox drag / rapid typing into one write.
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self._emit_state_changed)

        # Transient "✓ gespeichert" confirmation. Shown briefly after a
        # real save fires (so the user sees their tweak is remembered),
        # then auto-hidden. The hide timer is single-shot so a rapid
        # follow-up edit just re-shows + re-schedules the hide.
        self._hide_saved_timer = QTimer(self)
        self._hide_saved_timer.setSingleShot(True)
        self._hide_saved_timer.setInterval(1500)
        self._hide_saved_timer.timeout.connect(self._hide_saved)

        # ── Mode toggle row + close button ─────────────────────────────
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 4)
        mode_row.setSpacing(6)

        self._btn_auto = QPushButton("⚙ Auto")
        self._btn_auto.setCheckable(True)
        self._btn_auto.setChecked(True)
        self._btn_auto.setToolTip(
            _setting_tooltip(
                "Change one value while AutoTuner safely adjusts related values to "
                "keep the configuration practical for your hardware.",
                "Each edited field becomes a pinned override and compute_config() "
                "re-runs around it. Context, KV-cache precision, memory placement, "
                "and other dependent values may change so the resulting VRAM/RAM "
                "plan still fits.",
            )
        )
        self._btn_auto.clicked.connect(lambda: self._set_mode("auto"))
        mode_row.addWidget(self._btn_auto)

        self._btn_manual = QPushButton("✎ Manual")
        self._btn_manual.setCheckable(True)
        self._btn_manual.setToolTip(
            _setting_tooltip(
                "Keeps every value exactly as you enter it, without automatic safety "
                "adjustments.",
                "The panel assembles a TunedConfig directly from the widget values; "
                "compute_config() is not re-run. Invalid or overcommitted memory, "
                "thread, batch, or fork-specific combinations can therefore make "
                "llama-server slow down or fail to start.",
            )
        )
        self._btn_manual.clicked.connect(lambda: self._set_mode("manual"))
        mode_row.addWidget(self._btn_manual)

        # Reset — drops the saved Expert state for this model and reloads
        # the AutoTuner's automatically-best config. Sits next to
        # Auto/Manual so the "back to Auto" path is one click.
        self._btn_reset = QPushButton("⟲ Reset")
        self._btn_reset.setToolTip(
            _setting_tooltip(
                "Removes this model's saved Expert changes and returns to the fully "
                "automatic recommendation.",
                "The per-model Expert snapshot and pinned overrides are deleted, "
                "then compute_config() is run again from the current model metadata, "
                "hardware state, performance target, and launch options.",
            )
        )
        self._btn_reset.clicked.connect(self.resetRequested.emit)
        mode_row.addWidget(self._btn_reset)

        # "✓ gespeichert" flash — confirms a per-model autosave just
        # landed. Hidden until the first real edit; never shown for
        # programmatic load / restore / reset (those bypass
        # `_emit_state_changed` via the `_populating` guard).
        self._saved_lbl = QLabel("✓ saved")
        self._saved_lbl.setProperty("themeRole", "saved")
        self._saved_lbl.setVisible(False)
        mode_row.addWidget(self._saved_lbl)

        mode_row.addStretch(1)

        self._btn_close = QPushButton("✕")
        self._btn_close.setFixedWidth(28)
        self._btn_close.setToolTip(
            _setting_tooltip(
                "Closes the Expert editor and returns to the normal configuration "
                "preview.",
                "This only changes the visible panel. Saved Expert overrides remain "
                "associated with the selected model until you use Reset; no server "
                "is started or stopped.",
            )
        )
        self._btn_close.clicked.connect(self.closeRequested.emit)
        mode_row.addWidget(self._btn_close)

        # ── Editable widgets (created once, populated per model) ───────
        self._widgets_created = False
        self._build_widgets()

        # ── Layout ─────────────────────────────────────────────────────
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(2)
        outer.addLayout(mode_row)

        # The scroll area keeps the panel usable when the user shrinks
        # the window or picks a tiny font.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._inner)
        outer.addWidget(scroll, 1)

        self._mode = "auto"

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        """Create the grid of editable widgets (once, reused per model)."""
        self._inner = QWidget()
        grid = QGridLayout(self._inner)
        grid.setContentsMargins(4, 0, 4, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(3)

        row = 0

        def _add(label: str, widget: QWidget, tip: str = "") -> None:
            nonlocal row
            label_widget = QLabel(label)
            label_widget.setProperty("themeRole", "muted")
            grid.addWidget(label_widget, row, 0)
            grid.addWidget(widget, row, 1)
            if tip:
                widget.setToolTip(tip)
                label_widget.setToolTip(tip)
            row += 1

        def _section(title: str) -> None:
            nonlocal row
            section_label = QLabel(f"── {title} ──")
            section_label.setProperty("themeRole", "section")
            grid.addWidget(section_label, row, 0, 1, 2)
            row += 1

        # Context length
        _section("Context & KV cache")
        self._sp_ctx = QSpinBox()
        self._sp_ctx.setRange(1024, 4_194_304)
        self._sp_ctx.setSingleStep(1024)
        self._sp_ctx.setGroupSeparatorShown(True)
        self._sp_ctx.valueChanged.connect(lambda _: self._on_edit("user_ctx"))
        _add(
            "Context tokens",
            self._sp_ctx,
            _setting_tooltip(
                "Sets how much recent text, code, and conversation the model can "
                "consider at once. More context helps with large tasks but needs "
                "more memory.",
                "Passed as llama-server --ctx-size. KV-cache memory grows roughly "
                "linearly with context length and with the number of parallel slots. "
                "In Auto mode, changing it triggers a new KV precision and CPU/GPU "
                "placement plan so the requested window fits when possible.",
            ),
        )

        self._cb_cache_k = QComboBox()
        self._cb_cache_k.addItems(self._KV_QUANT_OPTIONS)
        self._cb_cache_k.currentTextChanged.connect(
            lambda value: self._on_kv_quant_changed("force_cache_k", value)
        )
        _add(
            "K-quant",
            self._cb_cache_k,
            _setting_tooltip(
                "Chooses the quality and memory use of the model's attention-key "
                "memory. Higher precision is safer for recall; lower precision saves "
                "VRAM or RAM.",
                "This maps to --cache-type-k. f16 uses the most memory, q8_0 is a "
                "high-quality compromise, and q4/q5 or TurboQuant formats compress "
                "more aggressively. Flash Attention is required for quantized KV "
                "types on supported llama.cpp builds.",
            ),
        )

        self._cb_cache_v = QComboBox()
        self._cb_cache_v.addItems(self._KV_QUANT_OPTIONS)
        self._cb_cache_v.currentTextChanged.connect(
            lambda value: self._on_kv_quant_changed("force_cache_v", value)
        )
        _add(
            "V-quant",
            self._cb_cache_v,
            _setting_tooltip(
                "Chooses the quality and memory use of the attention-value cache. "
                "Lower precision can free substantial memory for a longer context.",
                "This maps to --cache-type-v. With Flash Attention, compatible "
                "AMD/Vulkan builds can use an asymmetric K/V plan, often keeping K "
                "at higher precision than V. Some backends or older builds require "
                "matching types and may reject unsupported combinations.",
            ),
        )

        # Layer placement
        _section("Layer placement")
        self._sp_ngl = QSpinBox()
        self._sp_ngl.setRange(0, 999)
        self._sp_ngl.valueChanged.connect(lambda _: self._on_edit("force_ngl"))
        _add(
            "GPU layers (ngl)",
            self._sp_ngl,
            _setting_tooltip(
                "Controls how much of a regular dense model runs on the GPU. More "
                "GPU layers are usually faster but consume more VRAM.",
                "Passed as --n-gpu-layers. 0 keeps layers on CPU; 999 requests full "
                "offload and lets llama.cpp clamp to the model's actual layer count. "
                "Mixture-of-Experts placement is governed separately by n_cpu_moe.",
            ),
        )

        self._sp_ncpumoe = QSpinBox()
        self._sp_ncpumoe.setRange(0, 999)
        self._sp_ncpumoe.valueChanged.connect(
            lambda _: self._on_edit("force_n_cpu_moe")
        )
        _add(
            "n_cpu_moe",
            self._sp_ncpumoe,
            _setting_tooltip(
                "For Mixture-of-Experts models, moves expert layers to system RAM "
                "when VRAM is limited. More CPU expert layers save VRAM but are "
                "usually slower.",
                "Passed as --n-cpu-moe. It affects MoE expert tensors rather than the "
                "dense shared layers controlled by --n-gpu-layers. 0 keeps eligible "
                "experts GPU-resident; larger values progressively offload them.",
            ),
        )

        # Threads & batching
        _section("Threads & batching")
        self._sp_threads = QSpinBox()
        self._sp_threads.setRange(1, 256)
        _add(
            "threads",
            self._sp_threads,
            _setting_tooltip(
                "Sets how many CPU threads generate tokens. AutoTuner normally picks "
                "a sensible value for your processor.",
                "Passed as --threads / -t for the token-generation phase. Too few "
                "threads can bottleneck CPU work; too many can increase scheduling, "
                "cache, and hybrid-core overhead without improving speed.",
            ),
        )

        self._sp_batch_threads = QSpinBox()
        self._sp_batch_threads.setRange(1, 256)
        _add(
            "batch threads",
            self._sp_batch_threads,
            _setting_tooltip(
                "Sets CPU parallelism while the initial prompt is processed. A good "
                "value can reduce time to the first generated token.",
                "Passed as --threads-batch / -tb. Prompt ingestion can use different "
                "parallelism from token generation; excessive threads may hurt on "
                "hybrid CPUs or when several servers share the machine.",
            ),
        )

        self._sp_batch = QSpinBox()
        self._sp_batch.setRange(1, 16384)
        self._sp_batch.setSingleStep(64)
        _add(
            "batch",
            self._sp_batch,
            _setting_tooltip(
                "Controls how many prompt tokens llama.cpp may prepare together. "
                "Larger batches can process prompts faster but need more temporary "
                "memory.",
                "Passed as --batch-size / -b. This is the logical maximum batch and "
                "must be at least as large as the physical micro-batch. Very high "
                "values can increase compute-buffer VRAM/RAM or fail on constrained "
                "backends.",
            ),
        )

        self._sp_ubatch = QSpinBox()
        self._sp_ubatch.setRange(1, 16384)
        self._sp_ubatch.setSingleStep(64)
        _add(
            "ubatch",
            self._sp_ubatch,
            _setting_tooltip(
                "Sets the smaller chunks actually sent through the model during "
                "prompt processing. Smaller chunks use less peak memory; larger "
                "chunks may be faster.",
                "Passed as --ubatch-size / -ub. llama.cpp divides the logical batch "
                "into these physical micro-batches. It should not exceed batch size, "
                "and backend-specific memory limits often determine the best value.",
            ),
        )

        # Parallelism (llama-server --parallel N, short: -np N)
        # Each slot gets its own KV-cache window, so Auto mode re-fits
        # the context length around the chosen slot count (same math the
        # performance target uses internally). Off = keep the
        # performance-target default (one slot for desktop presets).
        _section("Parallelism")
        # Create the spinbox first so the checkbox's toggled signal can
        # safely reference it (it enables/disables the spinbox directly,
        # which must also work in Manual mode where _on_edit is a no-op).
        self._sp_parallel = QSpinBox()
        self._sp_parallel.setRange(1, 32)
        self._sp_parallel.setEnabled(False)
        self._sp_parallel.valueChanged.connect(
            lambda _: self._on_edit("force_n_parallel")
        )

        self._chk_parallel = QCheckBox("Parallel slots (--parallel / -np)")
        self._chk_parallel.toggled.connect(self._sp_parallel.setEnabled)
        self._chk_parallel.toggled.connect(lambda _: self._on_edit("force_n_parallel"))
        _add(
            "",
            self._chk_parallel,
            _setting_tooltip(
                "Allows several requests or agents to generate at the same time. "
                "Leave it off unless you need concurrency, because every slot uses "
                "additional memory.",
                "Enables a manual --parallel / -np override. llama-server uses "
                "continuous batching and allocates a KV window per slot, so total KV "
                "memory scales with slot count. Off delegates the count to the chosen "
                "performance target (one slot by default).",
            ),
        )
        _add(
            "parallel slots",
            self._sp_parallel,
            _setting_tooltip(
                "Chooses the exact number of simultaneous request slots after the "
                "parallel override is enabled.",
                "Passed as --parallel N. Each slot receives its own context/KV-cache "
                "capacity, so raising N can reduce the context that fits or cause "
                "memory pressure. AutoTuner suggests 3 with at least 24 GiB free on "
                "the largest GPU and otherwise 2.",
            ),
        )

        # Flags
        _section("Flags")
        self._chk_fa = QCheckBox("flash attention (-fa)")
        _add(
            "",
            self._chk_fa,
            _setting_tooltip(
                "Uses a faster, more memory-efficient attention implementation when "
                "the selected backend supports it.",
                "Passed as --flash-attn / -fa. It is required by llama.cpp for "
                "quantized KV-cache types and can reduce attention memory traffic. "
                "Unsupported GPU/backend combinations may ignore or reject it.",
            ),
        )

        self._cb_load_mode = QComboBox()
        self._cb_load_mode.addItem("auto (runtime/device default)", "auto")
        self._cb_load_mode.addItem("none (normal reads)", "none")
        self._cb_load_mode.addItem("mmap", "mmap")
        self._cb_load_mode.addItem("mlock (without mmap, b10151+)", "mlock")
        self._cb_load_mode.addItem("mmap + mlock", "mmap+mlock")
        self._cb_load_mode.addItem("direct I/O", "dio")
        _add(
            "model load mode",
            self._cb_load_mode,
            _setting_tooltip(
                "Chooses whether model weights are mapped, read normally, locked in "
                "RAM, or loaded through direct I/O. Auto keeps llama.cpp's default.",
                "Passed as --load-mode MODE. Since b10151, mlock means locking without "
                "mmap and mmap+mlock explicitly combines both. none disables mapping "
                "without locking; dio requests DirectIO when the platform supports it. "
                "Locking still requires sufficient RAM and OS privileges. AutoTuner "
                "blocks it on old or unprobeable GPU builds with the historic Vulkan "
                "host-buffer crash.",
            ),
        )

        self._chk_jinja = QCheckBox("--jinja")
        _add(
            "",
            self._chk_jinja,
            _setting_tooltip(
                "Lets llama-server format messages using the chat template stored "
                "with the model, including supported reasoning output handling.",
                "Passed as --jinja. llama.cpp renders the GGUF/Jinja template for the "
                "OpenAI-compatible chat API and can separate supported <think> traces "
                "into reasoning_content. A missing or broken template may require a "
                "different template override.",
            ),
        )

        self._chk_verbose = QCheckBox("--verbose")
        _add(
            "",
            self._chk_verbose,
            _setting_tooltip(
                "Prints much more server detail for troubleshooting. Leave it off "
                "for a quieter terminal during normal use.",
                "Passed as --verbose. The llama-server terminal includes additional "
                "request, scheduler, model, and backend diagnostics; this can produce "
                "large logs and may expose prompt/request metadata during debugging.",
            ),
        )

        self._chk_metrics = QCheckBox("--metrics (/metrics)")
        _add(
            "",
            self._chk_metrics,
            _setting_tooltip(
                "Adds a monitoring page that tools such as Prometheus can read. It "
                "does not change model quality.",
                "Passed as --metrics and exposes GET /metrics on the same server host "
                "and port. The endpoint reports operational counters and timings; if "
                "you bind beyond localhost, protect it with the same network controls "
                "as the inference API.",
            ),
        )

        self._chk_slots_api = QCheckBox("--slots (/slots API)")
        _add(
            "",
            self._chk_slots_api,
            _setting_tooltip(
                "Lets AutoTuner show how many request slots are busy or available. "
                "It is mainly useful when serving concurrent clients.",
                "Passed as --slots and enables llama-server's GET /slots endpoint. "
                "AutoTuner polls compatible builds for busy/total state. The endpoint "
                "can reveal runtime details, so avoid exposing it on an untrusted "
                "network.",
            ),
        )

        self._cb_numa = QComboBox()
        self._cb_numa.addItems(self._NUMA_OPTIONS)
        _add(
            "NUMA",
            self._cb_numa,
            _setting_tooltip(
                "Controls how memory and CPU work are spread on systems with more "
                "than one memory node or CPU socket. Most desktop PCs should leave "
                "this off.",
                "Maps to llama.cpp --numa distribute, isolate, or numactl. These modes "
                "change thread and memory locality on NUMA hardware; a wrong policy "
                "can increase cross-node traffic and reduce performance.",
            ),
        )

        self._chk_rope = QCheckBox("RoPE scaling (YaRN)")
        self._chk_rope.toggled.connect(lambda _: self._on_edit("force_rope_scale"))
        _add(
            "",
            self._chk_rope,
            _setting_tooltip(
                "Extends supported models beyond their original context length. It "
                "can handle longer input, but may reduce quality if the model or "
                "profile was not designed for it.",
                "Forces YaRN/RoPE scaling instead of following the matched YAML "
                "profile. AutoTuner derives the required factor from requested versus "
                "native context and emits the corresponding rope-scaling arguments. "
                "Turning it off keeps the native positional scale.",
            ),
        )

        self._sp_rope_factor = QDoubleSpinBox()
        self._sp_rope_factor.setRange(1.0, 32.0)
        self._sp_rope_factor.setSingleStep(0.5)
        self._sp_rope_factor.setDecimals(1)
        _add(
            "RoPE factor",
            self._sp_rope_factor,
            _setting_tooltip(
                "Sets how far the model's positional range is stretched. 1.0 means "
                "no extension; higher values target proportionally longer context.",
                "Used as the YaRN/RoPE scale factor when scaling is active. A factor "
                "that is too low cannot cover the requested context; an unnecessarily "
                "high factor can harm short-context accuracy. Manual mode accepts the "
                "value exactly as entered.",
            ),
        )

        # Sampling
        _section("Sampling")
        self._sp_temp = QDoubleSpinBox()
        self._sp_temp.setRange(0.0, 5.0)
        self._sp_temp.setSingleStep(0.05)
        self._sp_temp.setDecimals(2)
        _add(
            "temperature",
            self._sp_temp,
            _setting_tooltip(
                "Controls randomness. Lower values are more predictable; higher "
                "values are more varied but can become less reliable.",
                "Passed as --temp and rescales token logits before sampling. 0 is "
                "effectively deterministic in common llama.cpp sampling paths; the "
                "best nonzero value depends on the model and the other samplers.",
            ),
        )

        self._sp_top_k = QSpinBox()
        self._sp_top_k.setRange(0, 1000)
        _add(
            "top_k",
            self._sp_top_k,
            _setting_tooltip(
                "Limits each next-token choice to the K most likely candidates. "
                "Smaller values are more focused; 0 disables this filter.",
                "Passed as --top-k. The sampler removes every token outside the top K "
                "before later probability filters run. Very small K can make output "
                "repetitive; large K approaches an unrestricted candidate set.",
            ),
        )

        self._sp_top_p = QDoubleSpinBox()
        self._sp_top_p.setRange(0.0, 1.0)
        self._sp_top_p.setSingleStep(0.01)
        self._sp_top_p.setDecimals(3)
        _add(
            "top_p",
            self._sp_top_p,
            _setting_tooltip(
                "Keeps the smallest group of likely tokens whose combined chance "
                "reaches this value. Lower values are safer and narrower.",
                "Passed as --top-p for nucleus sampling. At 1.0 it removes almost "
                "nothing; lower thresholds dynamically trim the probability tail. It "
                "interacts with temperature, top-k, and min-p in sampler order.",
            ),
        )

        self._sp_min_p = QDoubleSpinBox()
        self._sp_min_p.setRange(0.0, 1.0)
        self._sp_min_p.setSingleStep(0.01)
        self._sp_min_p.setDecimals(3)
        _add(
            "min_p",
            self._sp_min_p,
            _setting_tooltip(
                "Drops tokens that are far less likely than the current best choice. "
                "Higher values make output more focused.",
                "Passed as --min-p. A candidate must meet a probability floor relative "
                "to the highest-probability token, making the filter adapt to model "
                "confidence. 0 disables it; excessive values can reduce diversity.",
            ),
        )

        self._sp_rep = QDoubleSpinBox()
        self._sp_rep.setRange(0.5, 2.5)
        self._sp_rep.setSingleStep(0.01)
        self._sp_rep.setDecimals(3)
        _add(
            "repeat_penalty",
            self._sp_rep,
            _setting_tooltip(
                "Discourages the model from repeating recent words or patterns. 1.0 "
                "means no penalty; slightly higher values reduce loops.",
                "Passed as --repeat-penalty and modifies logits for tokens found in "
                "the repeat window. Values that are too high can damage code, names, "
                "formatting, and intentional repetition.",
            ),
        )

        self._sp_presence = QDoubleSpinBox()
        self._sp_presence.setRange(-2.0, 2.0)
        self._sp_presence.setSingleStep(0.1)
        self._sp_presence.setDecimals(2)
        _add(
            "presence_penalty",
            self._sp_presence,
            _setting_tooltip(
                "Encourages introducing new tokens instead of reusing ones already "
                "present. Positive values increase novelty; 0 leaves it neutral.",
                "Passed as --presence-penalty. Unlike a count-based frequency penalty, "
                "it applies based on whether a token appeared at all. Negative values "
                "encourage reuse; strong positive values can hurt coherent code or "
                "terminology.",
            ),
        )

        # Speculative decoding — cascading because hybrid/recurrent targets
        # allocate one rollback-state snapshot per proposed draft token.
        _section("Speculative decoding")
        self._sp_draft_n_max = QSpinBox()
        self._sp_draft_n_max.setRange(0, 64)
        self._sp_draft_n_max.setSpecialValueText("Profil (auto)")
        self._sp_draft_n_max.valueChanged.connect(
            lambda _: self._on_edit("force_draft_n_max")
        )
        _add(
            "draft n-max",
            self._sp_draft_n_max,
            _setting_tooltip(
                "Limits how many tokens the faster draft path may propose per step. "
                "0 uses the tested model profile; larger values are not automatically "
                "faster.",
                "Passed as --spec-draft-n-max N when an external -md drafter or "
                "embedded MTP head is active. 0 uses draft_max from the matched YAML "
                "profile (fallback 2). The main model must verify every proposal, and "
                "on AMD/Vulkan values above roughly 2–3 often add overhead.",
            ),
        )

        # Reasoning controls (llama-server b9118 era).
        # The five settings here cover three different mechanisms the
        # server understands, all wired to the same dropdown to keep the
        # UI simple:
        #   "auto"        — emit no reasoning flag; model/template decide
        #   "off"         — --reasoning off  (silence thinking traces)
        #   "minimal"     — --chat-template-kwargs '{"reasoning_effort":"minimal"}'
        #   "low"/"med"/"high"/"extra_high" — same kwarg with that value
        # "extra_high" is not standardised upstream but several Qwen3.6
        # community templates accept it; falls back to "high" on builds
        # that reject it.
        _section("Reasoning / thinking")
        self._cb_reasoning = QComboBox()
        self._cb_reasoning.addItems(
            ["auto", "off", "minimal", "low", "medium", "high", "extra_high"]
        )
        _add(
            "Effort",
            self._cb_reasoning,
            _setting_tooltip(
                "Chooses how much internal reasoning a compatible model should use. "
                "Auto leaves the decision to the model; Off requests no reasoning.",
                "Minimal through extra_high are sent as reasoning_effort via "
                "--chat-template-kwargs. Off emits --reasoning off; Auto emits no "
                "override. Support and accepted names depend on the model's Jinja "
                "template and llama.cpp build.",
            ),
        )

        self._sp_think_budget = QSpinBox()
        self._sp_think_budget.setRange(-1, 1_048_576)
        self._sp_think_budget.setSingleStep(256)
        self._sp_think_budget.setValue(-1)
        self._sp_think_budget.setGroupSeparatorShown(True)
        _add(
            "Think budget",
            self._sp_think_budget,
            _setting_tooltip(
                "Caps how many tokens a reasoning model may spend thinking before it "
                "must answer. -1 leaves it unlimited; 0 asks it to skip thinking.",
                "Positive values emit --reasoning-budget N on compatible llama.cpp "
                "builds. -1 emits no budget flag. This budget consumes context/output "
                "tokens and is separate from the qualitative reasoning-effort hint.",
            ),
        )

        self._chk_reasoning_preserve = QCheckBox(
            "Preserve reasoning history (--reasoning-preserve)"
        )
        _add(
            "",
            self._chk_reasoning_preserve,
            _setting_tooltip(
                "Keeps earlier assistant thinking traces available in later turns "
                "when the model supports that behaviour.",
                "Passed as --reasoning-preserve. Compatible chat templates retain "
                "reasoning content in conversation history instead of stripping it. "
                "This can improve continuity but increases prompt/context usage and "
                "may expose prior reasoning to clients. Since llama.cpp b10786 the "
                "server enables preserved reasoning by default whenever the template "
                "supports it; unticked leaves that default alone. Add "
                "--no-reasoning-preserve to Extra CLI flags to force it off.",
            ),
        )

        # Extra free-form CLI flags
        _section("Extra CLI flags")
        self._le_extra = QLineEdit()
        self._le_extra.setPlaceholderText(
            'e.g.  --chat-template-kwargs \'{"reasoning_effort":"high"}\''
        )
        _add(
            "extras",
            self._le_extra,
            _setting_tooltip(
                "Adds advanced llama-server options that AutoTuner does not expose "
                "elsewhere. Use this only when you know the exact flag syntax.",
                "The text is parsed into arguments and appended to the generated "
                "command after modeled settings. Unsupported, duplicated, or unsafe "
                "flags can override assumptions or make startup fail; normal shell "
                "expansion is not used.",
            ),
        )

        grid.setRowStretch(row, 1)
        self._widgets_created = True

        # ── Autosave wiring ───────────────────────────────────────────
        # Every editable widget schedules a debounced snapshot persist,
        # independent of whether it is a cascading widget (those ALSO
        # drive `_on_edit` → recompute). The `_populating` guard inside
        # `_schedule_save` keeps programmatic population (load / reset)
        # from firing a spurious save.
        for sp in (
            self._sp_ctx,
            self._sp_ngl,
            self._sp_ncpumoe,
            self._sp_threads,
            self._sp_batch_threads,
            self._sp_batch,
            self._sp_ubatch,
            self._sp_rope_factor,
            self._sp_temp,
            self._sp_top_k,
            self._sp_top_p,
            self._sp_min_p,
            self._sp_rep,
            self._sp_presence,
            self._sp_think_budget,
            self._sp_parallel,
            self._sp_draft_n_max,
        ):
            sp.valueChanged.connect(self._schedule_save)
        for cb in (
            self._cb_cache_k,
            self._cb_cache_v,
            self._cb_load_mode,
            self._cb_numa,
            self._cb_reasoning,
        ):
            cb.currentTextChanged.connect(self._schedule_save)
        self._chk_fa.toggled.connect(lambda _: self._on_edit("force_flash_attn"))
        for chk in (
            self._chk_fa,
            self._chk_jinja,
            self._chk_verbose,
            self._chk_metrics,
            self._chk_slots_api,
            self._chk_reasoning_preserve,
            self._chk_rope,
            self._chk_parallel,
        ):
            chk.toggled.connect(self._schedule_save)
        self._le_extra.textChanged.connect(self._schedule_save)
        self._btn_auto.clicked.connect(self._schedule_save)
        self._btn_manual.clicked.connect(self._schedule_save)

    # ------------------------------------------------------------------
    # Mode toggling
    # ------------------------------------------------------------------
    def _set_mode(self, mode: str) -> None:
        if mode not in ("auto", "manual"):
            return
        self._mode = mode
        self._btn_auto.setChecked(mode == "auto")
        self._btn_manual.setChecked(mode == "manual")
        # Switching from Manual → Auto drops any stale pins so the
        # cascade starts from the current model's auto-defaults.
        if mode == "auto":
            self._user_pins.clear()
            self._recompute(force_overrides={})
        self.modeChanged.emit(mode)

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # Public API — called by the parent window
    # ------------------------------------------------------------------
    def configure_for_model(
        self,
        cfg: TunedConfig,
        system: SystemInfo,
        native_ctx: int,
        profile_max: int,
        recompute_cb,
    ) -> None:
        """Bind the panel to a specific model selection.

        ``recompute_cb`` takes a dict of ``force_*`` kwargs and returns a
        fresh ``TunedConfig`` (or None on failure). Called from Auto
        mode whenever the user edits a cascading widget.
        """
        self._system = system
        self._native_ctx = native_ctx
        self._profile_max = profile_max
        self._recompute_cb = recompute_cb
        # New model → drop pins, repaint from the fresh cfg.
        self._user_pins.clear()
        self._populate_from_cfg(cfg)

    def current_config(self) -> Optional[TunedConfig]:
        """Return the configuration to launch with.

        Auto mode: the last cascaded config.
        Manual mode: assembled from the live widget values.
        """
        if self._mode == "auto":
            return self._last_cfg
        return self._build_manual_config()

    # ------------------------------------------------------------------
    # Widget ↔ cfg bridging
    # ------------------------------------------------------------------
    def _populate_from_cfg(self, cfg: TunedConfig) -> None:
        """Mirror cfg values into widgets without firing recompute."""
        self._last_cfg = cfg
        self._populating = True
        try:
            # Context
            ctx_max = max(self._profile_max, self._native_ctx, cfg.ctx, 8192)
            self._sp_ctx.setMaximum(ctx_max)
            self._sp_ctx.setValue(cfg.ctx)

            # KV quants
            self._set_combo(self._cb_cache_k, cfg.cache_k)
            self._set_combo(self._cb_cache_v, cfg.cache_v)

            # Layer placement
            self._sp_ngl.setValue(min(999, cfg.ngl))
            self._sp_ncpumoe.setValue(cfg.n_cpu_moe or 0)

            # Threads & batching
            self._sp_threads.setValue(cfg.threads)
            self._sp_batch_threads.setValue(cfg.batch_threads)
            self._sp_batch.setValue(cfg.batch)
            self._sp_ubatch.setValue(cfg.ubatch)

            # Parallel slots — checkbox reflects an active override
            # (n_parallel_forced, set in both Auto and Manual mode); the
            # spinbox shows the live count when forced, otherwise the
            # hardware-suggested default so enabling yields a sane value.
            parallel_on = bool(getattr(cfg, "n_parallel_forced", False))
            self._chk_parallel.setChecked(parallel_on)
            self._sp_parallel.setEnabled(parallel_on)
            self._sp_parallel.setValue(
                int(cfg.n_parallel) if parallel_on else self._suggested_parallel_count()
            )

            # Flags
            self._chk_fa.setChecked(cfg.flash_attn)
            load_mode = effective_load_mode(cfg) or "auto"
            load_mode_idx = self._cb_load_mode.findData(load_mode)
            if load_mode_idx >= 0:
                self._cb_load_mode.setCurrentIndex(load_mode_idx)
            extras_in = list(cfg.extra_cli_flags or [])
            self._chk_jinja.setChecked("--jinja" in extras_in)
            self._chk_verbose.setChecked("--verbose" in extras_in)
            self._chk_metrics.setChecked(
                bool(getattr(cfg, "metrics_enabled", True)) or "--metrics" in extras_in
            )
            self._chk_slots_api.setChecked(
                bool(getattr(cfg, "slots_api_enabled", False)) or "--slots" in extras_in
            )
            self._chk_reasoning_preserve.setChecked("--reasoning-preserve" in extras_in)
            self._set_combo(self._cb_numa, cfg.numa or "off")

            self._chk_rope.setChecked(cfg.rope_scaling)
            self._sp_rope_factor.setValue(
                float(cfg.rope_scale_factor) if cfg.rope_scale_factor > 0 else 1.0
            )

            # Sampling
            s = cfg.sampling or {}
            self._sp_temp.setValue(float(s.get("temperature", 0.7)))
            self._sp_top_k.setValue(int(s.get("top_k", 40)))
            self._sp_top_p.setValue(float(s.get("top_p", 0.9)))
            self._sp_min_p.setValue(float(s.get("min_p", 0.05)))
            self._sp_rep.setValue(float(s.get("repeat_penalty", 1.05)))
            self._sp_presence.setValue(float(s.get("presence_penalty", 0.0)))

            # Speculative decoding (0 = Profil-Default, kein Override)
            self._sp_draft_n_max.setValue(int(getattr(cfg, "draft_n_max", 0) or 0))

            # Reasoning + think-budget: parse them out of extra_cli_flags
            # so the dedicated dropdowns show the right state and the
            # free-form field below doesn't display the raw flags.
            reasoning_value, think_budget_value, leftover_extras = (
                self._parse_reasoning_from_extras(extras_in)
            )
            self._set_combo(self._cb_reasoning, reasoning_value)
            self._sp_think_budget.setValue(think_budget_value)

            # Extra CLI: filter out the flags we already model as
            # checkboxes / dedicated widgets so they don't appear twice.
            modeled = {
                "--jinja",
                "--verbose",
                "--metrics",
                "--slots",
                "--reasoning-preserve",
            }
            free_flags = [f for f in leftover_extras if f not in modeled]
            self._le_extra.setText(" ".join(free_flags))
        finally:
            self._populating = False

    @staticmethod
    def _parse_reasoning_from_extras(
        extras: List[str],
    ) -> Tuple[str, int, List[str]]:
        """Pull reasoning + reasoning-budget out of a flat CLI-flags list.

        Returns (reasoning_value, think_budget_value, leftover_extras)
        where leftover_extras drops every flag we successfully decoded.

        Recognises these shapes:
          * ``--reasoning off`` / ``--reasoning on`` / ``--reasoning auto``
          * ``--chat-template-kwargs '{"reasoning_effort":"high"}'``
          * ``--reasoning-budget N``  (b9625+ name)
          * ``--think-budget N``  (legacy name, still read from old settings)
          * ``--think 0``  (synonym for budget=0)
        Anything we cannot parse is preserved verbatim in leftover.
        """
        reasoning = "auto"
        budget = -1
        leftover: List[str] = []

        i = 0
        n = len(extras)
        while i < n:
            arg = extras[i]
            low = arg.lower()
            if low in ("--reasoning", "--think") and i + 1 < n:
                val = extras[i + 1].strip().lower()
                if low == "--reasoning":
                    if val in ("off", "false", "0", "no", "disable"):
                        reasoning = "off"
                    # We intentionally collapse on/auto into "auto" —
                    # the GUI only distinguishes "off" from "leave it
                    # to the template", which "auto" expresses.
                else:  # --think
                    try:
                        budget = int(val)
                    except ValueError:
                        leftover.extend([arg, extras[i + 1]])
                i += 2
                continue
            if low in ("--reasoning-budget", "--think-budget") and i + 1 < n:
                try:
                    budget = int(extras[i + 1])
                except ValueError:
                    leftover.extend([arg, extras[i + 1]])
                i += 2
                continue
            if low == "--chat-template-kwargs" and i + 1 < n:
                payload = extras[i + 1]
                # Quick-and-dirty extraction without a full JSON parse:
                # the canonical form is '{"reasoning_effort":"<value>"}'.
                m = re.search(r'"reasoning_effort"\s*:\s*"([^"]+)"', payload)
                if m:
                    candidate = m.group(1).strip().lower()
                    valid = {
                        "off",
                        "none",
                        "minimal",
                        "low",
                        "medium",
                        "high",
                        "extra_high",
                    }
                    if candidate in valid:
                        reasoning = "off" if candidate == "none" else candidate
                    i += 2
                    continue
                # Not a reasoning kwarg — keep the original flag pair.
                leftover.extend([arg, payload])
                i += 2
                continue
            leftover.append(arg)
            i += 1
        return reasoning, budget, leftover

    @staticmethod
    def _set_combo(combo: QComboBox, value: str) -> None:
        """Select ``value`` in ``combo``; insert it if missing (Turbo quants)."""
        idx = combo.findText(value)
        if idx < 0:
            combo.addItem(value)
            idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Auto-cascade
    # ------------------------------------------------------------------
    def _on_kv_quant_changed(self, kind: str, value: str) -> None:
        """Handle Expert K/V edits and warn about fork-only Turbo types."""
        if (
            not self._populating
            and not self._turbo_warning_shown
            and _is_turbo_kv_type(value)
        ):
            self._turbo_warning_shown = _show_turbo_kv_fork_warning(self)
        self._on_edit(kind)

    def _on_edit(self, kind: str) -> None:
        """A cascading widget was edited.

        Only acts in Auto mode and only when we are not in the middle
        of programmatically populating widgets.
        """
        if self._populating or self._mode != "auto":
            return
        # Update the pin set for this widget kind.
        if kind == "user_ctx":
            self._user_pins["user_ctx"] = self._sp_ctx.value()
        elif kind == "force_cache_k":
            self._user_pins["force_cache_k"] = self._cb_cache_k.currentText()
        elif kind == "force_cache_v":
            self._user_pins["force_cache_v"] = self._cb_cache_v.currentText()
        elif kind == "force_ngl":
            self._user_pins["force_ngl"] = self._sp_ngl.value()
        elif kind == "force_n_cpu_moe":
            v = self._sp_ncpumoe.value()
            self._user_pins["force_n_cpu_moe"] = v if v > 0 else None
        elif kind == "force_draft_n_max":
            value = self._sp_draft_n_max.value()
            self._user_pins["force_draft_n_max"] = value if value > 0 else None
        elif kind == "force_n_parallel":
            # Pinning the parallel-slot count makes Auto mode re-fit ctx
            # around N slots; unchecking releases the pin so the
            # performance-target default (one slot) takes over again.
            if self._chk_parallel.isChecked():
                self._user_pins["force_n_parallel"] = max(1, self._sp_parallel.value())
            else:
                self._user_pins["force_n_parallel"] = None
        elif kind == "force_rope_scale":
            self._user_pins["force_rope_scale"] = self._chk_rope.isChecked()
        elif kind == "force_flash_attn":
            self._user_pins["force_flash_attn"] = self._chk_fa.isChecked()

        self._recompute(force_overrides=dict(self._user_pins))

    def _recompute(
        self, force_overrides: dict, *, overlay_widgets: bool = True
    ) -> None:
        """Ask the parent to rebuild the config with these overrides.

        ``overlay_widgets`` re-stamps the live non-cascading widget values
        (threads / batch / flags / sampling / reasoning / extras) onto the
        cascaded result so user edits survive a rebuild. Draft n-max is a
        cascading pin because it changes recurrent-state memory.
        Reset passes ``False`` — there the whole point is to DROP those
        edits and repaint every widget from the pure Auto config.
        """
        if self._recompute_cb is None:
            return
        cfg = self._recompute_cb(force_overrides)
        if cfg is None:
            return
        if overlay_widgets:
            # Apply the live (non-cascading) widget values on top of the
            # cascaded result so the user's batch/thread/flag/sampling edits
            # survive the rebuild.
            cfg = self._apply_noncascading(cfg)
        self._populate_from_cfg(cfg)
        self.configChanged.emit(cfg)

    def _apply_noncascading(self, cfg: TunedConfig) -> TunedConfig:
        """Overlay the widget values that do not feed back into compute_config."""
        return apply_expert_values(cfg, self._widgets_to_values())

    def _build_manual_config(self) -> Optional[TunedConfig]:
        """Construct a TunedConfig from widget values without compute_config."""
        base = self._last_cfg
        if base is None:
            return None
        return expert_cfg_from_values(base, self._widgets_to_values())

    def _suggested_parallel_count(self) -> int:
        """Hardware-aware default for the parallel-slots override.

        3 when the largest GPU has plenty of free VRAM (≥24 GB), else 2.
        Falls back to 2 on CPU-only systems. This is only the *initial*
        spinbox suggestion shown while the override is off — once the
        user picks a value it is persisted in the Expert snapshot.
        """
        sysinfo = self._system
        if sysinfo and getattr(sysinfo, "gpus", None):
            biggest_free = max((g.free_vram_gb for g in sysinfo.gpus), default=0.0)
            return 3 if biggest_free >= 24.0 else 2
        return 2

    # ------------------------------------------------------------------
    # Reasoning helper
    # ------------------------------------------------------------------
    def _reasoning_flags_from_widgets(self) -> List[str]:
        """Translate the two reasoning widgets into llama-server flags.

        Thin wrapper over the shared ``_reasoning_flags_from_values`` so
        the live panel and the disk-snapshot path stay in lock-step.
        See the free function for the full mapping rules:
          dropdown == "auto"   → no flag (let the template decide)
          dropdown == "off"    → --reasoning off  (silence thinking)
          dropdown == anything else → --chat-template-kwargs
                                       '{"reasoning_effort":"<value>"}'
          spinbox  == -1        → no flag
          spinbox  >=  0        → --reasoning-budget <N>
        """
        return _reasoning_flags_from_values(
            self._cb_reasoning.currentText(), int(self._sp_think_budget.value())
        )

    # ------------------------------------------------------------------
    # Snapshot / autosave / restore
    # ------------------------------------------------------------------
    def _widgets_to_values(self) -> dict:
        """Read every editable widget into a JSON-serialisable values dict.

        This is the single source of truth for both the debounced save
        (→ ``stateChanged``) and the two config builders
        (``_apply_noncascading`` / ``_build_manual_config``) via the free
        helpers, so a widget can never be added without also being
        persisted.
        """
        return {
            "ctx": self._sp_ctx.value(),
            "cache_k": self._cb_cache_k.currentText(),
            "cache_v": self._cb_cache_v.currentText(),
            "ngl": self._sp_ngl.value(),
            "n_cpu_moe": self._sp_ncpumoe.value(),
            "threads": self._sp_threads.value(),
            "batch_threads": self._sp_batch_threads.value(),
            "batch": self._sp_batch.value(),
            "ubatch": self._sp_ubatch.value(),
            "flash_attn": self._chk_fa.isChecked(),
            "load_mode": str(self._cb_load_mode.currentData() or "auto"),
            "jinja": self._chk_jinja.isChecked(),
            "verbose": self._chk_verbose.isChecked(),
            "metrics_enabled": self._chk_metrics.isChecked(),
            "slots_api_enabled": self._chk_slots_api.isChecked(),
            "numa": self._cb_numa.currentText(),
            "rope_scaling": self._chk_rope.isChecked(),
            "rope_factor": self._sp_rope_factor.value(),
            "temperature": self._sp_temp.value(),
            "top_k": self._sp_top_k.value(),
            "top_p": self._sp_top_p.value(),
            "min_p": self._sp_min_p.value(),
            "repeat_penalty": self._sp_rep.value(),
            "presence_penalty": self._sp_presence.value(),
            "reasoning": self._cb_reasoning.currentText(),
            "think_budget": self._sp_think_budget.value(),
            "reasoning_preserve": self._chk_reasoning_preserve.isChecked(),
            "parallel_enabled": self._chk_parallel.isChecked(),
            "parallel_count": self._sp_parallel.value(),
            "draft_n_max": self._sp_draft_n_max.value(),
            "extras": self._le_extra.text().strip(),
        }

    def _make_snapshot(self) -> dict:
        """Full persisted state: mode + auto-mode pins + widget values."""
        return {
            "mode": self._mode,
            "pins": {k: v for k, v in self._user_pins.items()},
            "values": self._widgets_to_values(),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _schedule_save(self, *_) -> None:
        """Arm the debounced save timer (no-op while populating/resetting).

        Accepts and ignores the signal payload (value/index/text) so it
        can be connected to every widget signal type uniformly.
        """
        if self._populating:
            return
        self._save_timer.start()  # (re)starts the 300 ms countdown

    def _emit_state_changed(self) -> None:
        """Timer fired → emit a fresh snapshot for the parent to persist.

        Also flashes the "✓ gespeichert" confirmation so the user sees
        their tweak was saved. Guarded by `_populating` so a programmatic
        load / restore / reset never flashes a false confirmation.
        """
        if self._populating:
            return
        self.stateChanged.emit(self._make_snapshot())
        self._flash_saved()

    def _flash_saved(self) -> None:
        """Show the "✓ gespeichert" label and (re)arm the 1.5 s hide timer."""
        self._saved_lbl.setVisible(True)
        self._hide_saved_timer.start()

    def _hide_saved(self) -> None:
        """Hide timer fired → drop the "gespeichert" confirmation."""
        self._saved_lbl.setVisible(False)

    def flush_pending_save(self) -> None:
        """Immediately persist if a debounced save is still pending.

        Called by the parent before it repopulates the panel (model switch,
        checkbox toggle while open) so an in-flight edit is not lost.
        """
        if self._save_timer.isActive():
            self._save_timer.stop()
            self._emit_state_changed()

    def restore_from_snapshot(self, snap: dict) -> None:
        """Apply a saved Expert snapshot to the live panel.

        Sets mode + pins, paints the widget values, and — in Auto mode —
        re-runs the cascade from the saved pins so the displayed cascading
        fields (ctx / KV / ngl / n_cpu_moe) match what the user pinned.
        Manual mode just paints the frozen values. Emits NO save (the whole
        point is to reproduce a saved state, not re-record it).
        """
        if not isinstance(snap, dict) or "values" not in snap:
            return
        vals = snap.get("values") or {}
        mode = snap.get("mode", "auto")
        if mode not in ("auto", "manual"):
            mode = "auto"

        base = self._last_cfg

        # 1. Paint every widget from the saved values. Done under the
        #    populating guard so the valueChanged flood does not trigger
        #    a recompute or a save.
        self._populating = True
        try:
            self._user_pins = _expert_cascading_pins(snap)
            self._mode = mode
            self._btn_auto.setChecked(mode == "auto")
            self._btn_manual.setChecked(mode == "manual")
            if vals and base is not None:
                painted = expert_cfg_from_values(base, vals)
                self._populate_from_cfg(painted)
        finally:
            self._populating = False

        # 2. Re-derive the effective config.
        if mode == "auto":
            # Cascade from the saved pins (read non-cascading widgets = the
            # just-painted values), then repaint. _recompute sets _last_cfg.
            self._recompute(force_overrides=dict(self._user_pins))
        else:
            # Manual: the frozen config IS the painted values.
            self._last_cfg = self._build_manual_config()
        if self._last_cfg is not None:
            self.configChanged.emit(self._last_cfg)
            if not self._turbo_warning_shown and (
                _is_turbo_kv_type(self._last_cfg.cache_k)
                or _is_turbo_kv_type(self._last_cfg.cache_v)
            ):
                self._turbo_warning_shown = _show_turbo_kv_fork_warning(self)

    def reset_to_auto(self) -> None:
        """Reload the AutoTuner's automatically-best config (Reset button).

        Clears pins, forces Auto mode and re-cascades from empty pins.
        Must NOT emit a save — the parent has just cleared the override,
        and re-persisting the freshly-loaded Auto state would undo that.
        """
        self._populating = True
        try:
            self._user_pins.clear()
            self._mode = "auto"
            self._btn_auto.setChecked(True)
            self._btn_manual.setChecked(False)
        finally:
            self._populating = False
        # _recompute runs with empty pins → pure Auto; its internal
        # _populate_from_cfg is guarded, so no save fires.
        # overlay_widgets=False: without it the stale non-cascading widget
        # values (threads / batch / flags / sampling / reasoning / extras /
        # draft n-max) were stamped straight back onto the fresh Auto config,
        # so Reset only ever reset the cascading fields — the "Reset doesn't
        # reset everything" bug.
        self._recompute(force_overrides={}, overlay_widgets=False)


# ---------------------------------------------------------------------------
# Main window


class _ResponsiveSystemBar(QWidget):
    """Wrap hardware details without imposing their full text width on the window."""

    _COMPACT_MIN_WIDTH = 640

    def __init__(
        self, labels: Sequence[QLabel], parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._labels = list(labels)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(6, 2, 6, 2)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(2)
        self._columns = 0
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        for label in self._labels:
            label.setWordWrap(True)
            label.setMinimumWidth(0)
            # Hardware names are informative text, not a window-width contract.
            # Ignoring their horizontal size hint lets the grid give each label
            # a narrower cell; QLabel's height-for-width support then wraps it.
            label.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Preferred,
            )
        self._reflow(len(self._labels))
        # Reserve the initial one-line height before the first style/layout
        # polish pass. Otherwise Qt can initially collapse a zero-width hint to
        # height 0, then steal space from the user's splitters on the next theme
        # refresh when the same bar becomes visible.
        self.setMinimumHeight(self.minimumSizeHint().height())

    @property
    def column_count(self) -> int:
        """Current number of hardware fields per row (used by UI regression tests)."""
        return self._columns

    def _single_row_width(self) -> int:
        margins = self._grid.contentsMargins()
        spacing = max(0, self._grid.horizontalSpacing())
        text_width = sum(
            label.fontMetrics().horizontalAdvance(label.text())
            for label in self._labels
        )
        return (
            margins.left()
            + margins.right()
            + text_width
            + spacing * max(0, len(self._labels) - 1)
        )

    def _columns_for_width(self, width: int) -> int:
        if width >= self._single_row_width():
            return max(1, len(self._labels))
        if width >= self._COMPACT_MIN_WIDTH:
            return min(2, max(1, len(self._labels)))
        return 1

    def _reflow(self, columns: int) -> None:
        columns = max(1, min(int(columns), max(1, len(self._labels))))
        if columns == self._columns:
            return
        for label in self._labels:
            self._grid.removeWidget(label)
        for index, label in enumerate(self._labels):
            self._grid.addWidget(label, index // columns, index % columns)
        for column in range(max(1, len(self._labels))):
            self._grid.setColumnStretch(column, 1 if column < columns else 0)
        self._columns = columns
        self._grid.invalidate()
        self.updateGeometry()

    def _wrapped_height(self, width: int, columns: Optional[int] = None) -> int:
        columns = columns or self._columns_for_width(width)
        margins = self._grid.contentsMargins()
        horizontal_spacing = max(0, self._grid.horizontalSpacing())
        vertical_spacing = max(0, self._grid.verticalSpacing())
        cell_width = max(
            1,
            (
                width
                - margins.left()
                - margins.right()
                - horizontal_spacing * max(0, columns - 1)
            )
            // columns,
        )
        row_heights: List[int] = []
        for start in range(0, len(self._labels), columns):
            row_heights.append(
                max(
                    label.heightForWidth(cell_width)
                    if label.hasHeightForWidth()
                    else label.sizeHint().height()
                    for label in self._labels[start : start + columns]
                )
            )
        return (
            margins.top()
            + margins.bottom()
            + sum(row_heights)
            + vertical_spacing * max(0, len(row_heights) - 1)
        )

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._wrapped_height(max(1, width))

    def sizeHint(self) -> QSize:  # noqa: N802
        # QWidget/QGridLayout otherwise asks every horizontally-Ignored label
        # for its height at an almost-zero width. That can produce a bogus
        # hundreds-of-pixels vertical hint even though the real two-column bar
        # needs only a few lines at its current width.
        width = max(1, self.width())
        return QSize(0, self._wrapped_height(width, self._columns_for_width(width)))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        margins = self._grid.contentsMargins()
        line_height = max(
            (label.fontMetrics().height() for label in self._labels), default=0
        )
        return QSize(0, margins.top() + margins.bottom() + line_height)

    def refresh_layout(self) -> None:
        """Re-evaluate wrapping after hardware text or font metrics change."""
        self._reflow(self._columns_for_width(max(0, self.width())))
        self.updateGeometry()

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._reflow(self._columns_for_width(event.size().width()))
        super().resizeEvent(event)


class MainWindow(QMainWindow):
    # Signal carrying SystemInfo updates from the background sysinfo thread.
    # Qt widgets are NOT thread-safe — touching a QLabel from a daemon
    # thread produced sporadic random crashes ("GUI just closed itself").
    # Background work emits this signal; the slot runs on the GUI thread.
    _sysinfo_ready = pyqtSignal(object)  # SystemInfo
    _bg_log = pyqtSignal(str)  # log message from background thread
    _control_request_ready = pyqtSignal(object)  # ControlRequest from HTTP threads
    _probe_result_ready = pyqtSignal(object)  # server probe result from the worker
    _control_log_ready = pyqtSignal(str)  # gateway log line from HTTP threads

    _FORK_TOOLTIP_SUMMARY = (
        "Selects which llama.cpp build starts the server. AutoTuner can choose a "
        "more compatible build for a specific model when needed."
    )
    _FORK_TOOLTIP_TECHNICAL = (
        "The selected directory supplies llama-server or the matching diffusion "
        "binary. Model YAML profiles may override this default for required forks "
        "or features. Build discovery scans the enabled llama.cpp roots configured "
        "with the adjacent llama Builds button."
    )
    _FORK_COMBO_BASE_TOOLTIP = _setting_tooltip(
        _FORK_TOOLTIP_SUMMARY, _FORK_TOOLTIP_TECHNICAL
    )
    _FORK_COMBO_MIN_WIDTH = 220
    _FORK_COMBO_TEXT_PADDING = 72
    _WIN_SETTINGS_COMMAND_ID = 0x1FFE
    _WIN_ABOUT_COMMAND_ID = 0x1FFD

    def __init__(
        self,
        models_path: Path,
        settings_path: Path,
        *,
        start_background: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"AutoTuner v{VERSION}")
        # Hard-coded default size — only kicks in when no persisted
        # geometry exists (first launch on this machine, or the JSON
        # was wiped). `restoreGeometry` below replaces this when a
        # blob is on disk.
        self.resize(1320, 840)
        self._restore_window_geometry()

        self.models_path = models_path
        configured_model_paths = app_settings.get_model_paths()
        self.model_paths: List[Tuple[Path, bool]] = (
            configured_model_paths if configured_model_paths else [(models_path, True)]
        )
        self.settings_path = settings_path

        self._server: Optional[_TerminalProcess] = None
        # Multi-server registry. Each entry tracks one running llama-server
        # instance so we can (a) auto-assign ports 1234, 1235, 1236… and
        # reclaim them when a server stops, and (b) account for the VRAM a
        # previously-launched model already holds when placing the next one.
        # Shape per entry:
        #   {
        #     "proc": _TerminalProcess,
        #     "port": int,
        #     "base_url": str,
        #     "ready": bool,
        #     "model": str,          # display name
        #     "gpu": Optional[str],  # GPU name it was steered onto (if any)
        #     "vram_gb": float,      # estimated GPU footprint
        #   }
        self._servers: List[dict] = []
        # Monotonic counter so each server gets a stable identifier for the
        # switcher dropdown (ports can be reused after a stop, so port alone
        # is not a durable key).
        self._next_server_id: int = 1
        # GPU name the most recent launch was pinned to (for the registry).
        self._last_pinned_gpu: Optional[str] = None
        # Base port for the first server; subsequent ones get base+1, base+2…
        # Restored from settings so a non-default port survives restarts.
        self._base_port: int = app_settings.get_base_port()
        # /health handshake state: base URL of the running server and a
        # latch that flips once GET /health returns 200 (model loaded).
        self._server_base_url: Optional[str] = None
        self._server_ready: bool = False
        self._all_entries: List[ModelEntry] = []
        self._control_api: Optional[ControlApiServer] = None
        self._control_model_paths: Dict[str, Path] = {}
        self._control_runtime_paths: Dict[str, Path] = {}
        self._control_api_record: Optional[dict] = None
        self._control_closing = False
        self._system: Optional[SystemInfo] = None
        self._profiles: List[ModelProfile] = []
        self._forks: List[Tuple[str, Path]] = []
        self._fork_roots: List[Tuple[Path, bool]] = app_settings.get_llama_build_paths()
        self._fork_path: Optional[Path] = None  # manueller Fork-Ordner

        # Currently selected model + its draft (set in _show_config)
        self._current_entry: Optional[ModelEntry] = None
        self._current_draft: Optional[ModelEntry] = None

        # Per-model override cache for launch-option checkboxes. mmproj and
        # draft are controlled exclusively by their dropdown selections;
        # legacy vision/draft booleans are only read once as migration input.
        # Shape: { "<model_name>": {"thinking": bool, "ngram": bool, ...} }
        self._option_overrides: dict = {}
        # Internal, process-local Expert clipboard. It deliberately carries the
        # source performance target so a copied safe/throughput profile cannot
        # be pasted into a different target by accident.
        self._expert_settings_clipboard: Optional[Dict[str, object]] = None
        self._favorite_models = app_settings.get_favorite_models()
        self._model_view_mode = app_settings.get_model_view_mode()
        # New folders start expanded. Explicit collapses survive filtering,
        # rescans, view switches, favorite changes, and application restarts.
        self._tree_collapsed_paths = app_settings.get_model_tree_collapsed_paths()
        self._tree_native_toggle_item: Optional[QTreeWidgetItem] = None
        self._tree_manual_toggle = False
        # Rebuilding a QTreeWidget synchronously inside its item delegate's
        # mouse event can invalidate Qt's active QModelIndex and crash the
        # process. Favorite changes therefore coalesce into a zero-delay refresh.
        self._favorite_refresh_pending = False

        # Track whether the user has manually overridden the fork selection
        self._fork_manual_override = False

        # Remember the *container* the user pointed at via "📂 Fork" so
        # restarts still show every sibling build. This stays distinct
        # from the currently active fork in `self._fork_path`.
        self._fork_container: Optional[Path] = None

        self._scan_thread: Optional[QThread] = None
        self._scan_worker: Optional[_ScanWorker] = None
        self._diagnostic_thread: Optional[QThread] = None
        self._diagnostic_worker: Optional[_MetadataDiagnosticWorker] = None
        self._update_thread: Optional[QThread] = None
        # Either the source-based updater (dev/source installs) or the
        # binary-swap updater (frozen builds); both are QObjects moved to
        # ``_update_thread``.
        self._update_worker: Optional[QObject] = None
        self._ocr_thread: Optional[QThread] = None
        self._ocr_worker: Optional[_OcrWorker | _OcrPrepareWorker] = None
        self._ocr_progress_dialog: Optional[_OcrProgressDialog] = None
        self._ocr_server_record: Optional[dict] = None
        self._ocr_prepared_runner: Optional[OcrJobRunner] = None
        self._ocr_prepare_error: str = ""
        self._ocr_locked_states: Dict[QWidget, bool] = {}
        self._benchmark_thread: Optional[QThread] = None
        self._benchmark_worker: Optional[_PerformanceTuneWorker] = None
        self._benchmark_dialog: Optional[_PerformanceTuneDialog] = None
        self._benchmark_base_config: Optional[TunedConfig] = None
        self._benchmark_entry: Optional[ModelEntry] = None
        self._benchmark_system: Optional[SystemInfo] = None
        # Per-job checkpoint payloads are populated synchronously by the
        # benchmark worker before it starts the next performance mode.
        self._benchmark_checkpoints: Dict[str, dict] = {}
        self._benchmark_rerun_reset = False
        self._benchmark_locked_states: Dict[QWidget, bool] = {}
        # Expert autosave belongs to the exact target/profile/drafter loaded
        # into the panel, even if a selector changes before debounce fires.
        self._expert_loaded_performance_target = self._current_performance_target_name()
        self._expert_loaded_profile_slot = app_settings.PROFILE_AUTO
        self._expert_loaded_drafter_key = app_settings.NO_DRAFTER_PROFILE_KEY
        self._expert_loaded_performance_backend = ""
        self._setting_profile_refreshing = False
        self._sysinfo_busy = False
        # Persisted font size — falls back to 10pt on first launch.
        self._font_size = app_settings.get_font_size()
        self._debug_mode = app_settings.get_debug_mode()
        self._app_log_path = _prepare_application_log()
        self._set_internal_debug_mode(self._debug_mode)
        # Explicit Quit bypasses the optional X→minimize behaviour. The flag
        # remains False for ordinary title-bar close requests.
        self._force_quit = False
        self._tray_icon: Optional[QSystemTrayIcon] = None
        self._tray_menu: Optional[QMenu] = None
        self._tray_hint_shown = False
        self._tray_restore_maximized = False
        self._native_window_setup_done = False
        app = cast(Optional[QApplication], QApplication.instance())
        if app is None:  # QMainWindow requires a QApplication in normal Qt use.
            raise RuntimeError("MainWindow requires a QApplication")
        self._theme_manager = _application_theme_manager(app)
        self._language_manager = LanguageManager(
            _bundled_resource("assets", "languages"),
            app_settings.app_data_dir() / "languages",
            self,
        )
        selected_language = self._language_manager.select(
            app_settings.get_language_id()
        )
        if selected_language != app_settings.get_language_id():
            app_settings.set_language_id(selected_language)
        self._language_manager.install(app)

        self._build_ui()
        # Wire background → GUI signals BEFORE the first scan kicks off,
        # so a fast hardware probe can't fire its result into a slot
        # that hasn't been connected yet (one of the crash patterns).
        self._sysinfo_ready.connect(self._update_sysinfo_labels)
        self._bg_log.connect(self._log)
        self._control_request_ready.connect(self._handle_control_request)
        self._control_log_ready.connect(self._log)
        self._probe_result_ready.connect(self._on_probe_result)
        self._probe_worker = _ServerProbeWorker(self._probe_result_ready.emit)
        if start_background:
            self._configure_control_api()
            QTimer.singleShot(0, self._startup_load)

        # Server crash-detection (lightweight poll — no stdout read)
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_server)
        if start_background:
            self._poll_timer.start(500)

        # Sysinfo refresh (non-blocking — daemon thread)
        self._sysinfo_timer = QTimer(self)
        self._sysinfo_timer.timeout.connect(self._sysinfo_async)
        if start_background:
            self._sysinfo_timer.start(6000)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # ── Primary toolbar ─────────────────────────────────────────────
        # Keep the high-frequency model/runtime controls on one compact row.
        # Font, language, update, and application settings deliberately live
        # in a second persistent row toggled by the final ellipsis button.
        tb = QToolBar("Main")
        tb.setObjectName("mainToolbar")
        tb.setMovable(False)
        tb.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self._main_toolbar = tb
        self.addToolBar(tb)

        toolbar_actions = (
            (
                "📂 Models folder",
                self._browse_models,
                _setting_tooltip(
                    "Choose one or more folders that contain your GGUF models.",
                    "Opens the model-path manager. Enabled paths are persisted per "
                    "operating system and scanned recursively; disabling a path keeps "
                    "it saved but excludes it from discovery.",
                ),
            ),
            (
                "🔄 Refresh",
                self._start_scan,
                _setting_tooltip(
                    "Scans the configured model folders again and refreshes the list.",
                    "Runs GGUF discovery and metadata matching in a background worker, "
                    "then rebuilds model, mmproj, draft/MTP, and profile associations "
                    "without freezing the interface.",
                ),
            ),
        )
        for label, slot, tooltip in toolbar_actions:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            btn.setToolTip(tooltip)
            if label.startswith("📂"):
                self._btn_models_folder = btn
            else:
                self._btn_refresh = btn
            tb.addWidget(btn)

        tb.addSeparator()
        tb.addWidget(QLabel("Fork:"))
        self._fork_combo = QComboBox()
        self._fork_combo.setMinimumWidth(self._FORK_COMBO_MIN_WIDTH)
        self._fork_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._fork_combo.setToolTip(self._FORK_COMBO_BASE_TOOLTIP)
        self._fork_combo.currentIndexChanged.connect(self._on_fork_changed)
        tb.addWidget(self._fork_combo)

        self._btn_fork_folder = QPushButton("llama Builds")
        self._btn_fork_folder.setToolTip(
            _setting_tooltip(
                "Adds, removes, or temporarily disables folders containing llama.cpp "
                "builds.",
                "AutoTuner scans enabled roots for runnable llama-server binaries and "
                "compatible sibling fork directories. Paths are stored per operating "
                "system so one settings file can be shared across dual-boot installs.",
            )
        )
        self._btn_fork_folder.clicked.connect(self._browse_fork_folder)
        tb.addWidget(self._btn_fork_folder)

        tb.addSeparator()
        tb.addWidget(QLabel("Performance:"))
        self._perf_combo = QComboBox()
        self._perf_combo.setMinimumWidth(120)
        # Build technical details from the registry so future tiers auto-appear.
        tip_lines = []
        for tname in list_target_names():
            target = PERFORMANCE_TARGETS[tname]
            tip_lines.append(f"• {tname}: {target.description}")
        self._perf_combo.setToolTip(
            _setting_tooltip(
                "Chooses the balance between speed, context capacity, and memory "
                "headroom. Balanced is the best starting point for most users.",
                "The preset changes VRAM/RAM safety reserves, KV placement, context "
                "fitting, and default parallel slots before model-specific tuning:\n"
                + "\n".join(tip_lines),
            )
        )
        for tname in list_target_names():
            self._perf_combo.addItem(tname)
        # Restore persisted choice (may be None → default).
        persisted_perf = app_settings.get_performance_target()
        initial_perf = persisted_perf or DEFAULT_TARGET_NAME
        idx = self._perf_combo.findText(initial_perf)
        if idx < 0:
            idx = self._perf_combo.findText(DEFAULT_TARGET_NAME)
        self._perf_combo.setCurrentIndex(max(0, idx))
        self._perf_combo.currentIndexChanged.connect(self._on_perf_changed)
        tb.addWidget(self._perf_combo)

        # ── Mode (chat / coding) ───────────────────────────────────────
        tb.addSeparator()
        tb.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.setMinimumWidth(90)
        self._mode_combo.setToolTip(
            _setting_tooltip(
                "Selects conversational or code-focused generation behaviour. Chat "
                "is more varied; Coding is usually more deterministic.",
                "The matched model profile supplies mode-specific temperature, top-k, "
                "top-p, min-p, repetition, and presence settings. Profiles without a "
                "coding block fall back to chat values; model placement and memory "
                "planning are unchanged.",
            )
        )
        for m in ("chat", "coding"):
            self._mode_combo.addItem(m)
        persisted_mode = app_settings.get_mode() or "chat"
        idx = self._mode_combo.findText(persisted_mode)
        self._mode_combo.setCurrentIndex(max(0, idx))
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        tb.addWidget(self._mode_combo)

        # ── GPU pin (Auto / per-card) ──────────────────────────────────
        tb.addSeparator()
        tb.addWidget(QLabel("GPU:"))
        self._gpu_combo = QComboBox()
        self._gpu_combo.setMinimumWidth(110)
        self._gpu_combo.setToolTip(
            _setting_tooltip(
                "Auto chooses a GPU from current free memory. Selecting a card forces "
                "the next server onto only that GPU, which is useful for running a "
                "second model on another card.",
                "A card selection persists as forced_gpu and mirrors CLI --gpu. The "
                "launch environment hides the other devices and sets the matching "
                "backend/main-GPU index. Auto keeps all eligible cards visible and "
                "uses live VRAM-aware placement.",
            )
        )
        self._gpu_combo.addItem("Auto", None)
        self._gpu_combo.currentIndexChanged.connect(self._on_gpu_changed)
        tb.addWidget(self._gpu_combo)

        tb.addSeparator()
        self._btn_more = QPushButton("⋯")
        self._btn_more.setObjectName("moreToolbarButton")
        self._btn_more.setAccessibleName("More controls")
        self._btn_more.setCheckable(True)
        self._btn_more.setFixedWidth(38)
        self._btn_more.setToolTip(
            _setting_tooltip(
                "Shows or hides language, font, update, and application settings.",
                "The secondary toolbar is click-persistent: it remains open while the "
                "pointer moves elsewhere and closes only when this ellipsis is clicked "
                "again.",
            )
        )
        self._btn_more.toggled.connect(self._toggle_more_toolbar)
        tb.addWidget(self._btn_more)

        # ── Click-persistent secondary toolbar ─────────────────────────
        more = QToolBar("More controls")
        more.setObjectName("moreToolbar")
        more.setMovable(False)
        more.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self._more_toolbar = more
        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, more)

        more.addWidget(QLabel("Language:"))
        self._language_combo = QComboBox()
        self._language_combo.setObjectName("languageSelector")
        self._language_combo.setMinimumWidth(180)
        self._language_combo.setToolTip(
            _setting_tooltip(
                "Changes AutoTuner's interface language immediately.",
                "Built-in JSON packs fall back to English (UK) for unknown strings. "
                "Custom packs are validated and copied into ~/.autotuner/languages.",
            )
        )
        self._populate_language_combo()
        self._language_combo.currentIndexChanged.connect(self._on_language_changed)
        more.addWidget(self._language_combo)

        self._btn_language_folder = QPushButton("Open language folder")
        self._btn_language_folder.setToolTip(
            _setting_tooltip(
                "Opens the folder for your own JSON language packs.",
                "AutoTuner creates an English-based custom-language-template.json "
                "there once. Edit its ID, name, locale, and translations, then choose "
                "Custom language pack… to validate and activate it.",
            )
        )
        self._btn_language_folder.clicked.connect(self._open_language_folder)
        more.addWidget(self._btn_language_folder)

        more.addSeparator()
        more.addWidget(QLabel("Font:"))
        for delta, label, attribute in (
            (-1, "A−", "_btn_font_smaller"),
            (+1, "A+", "_btn_font_larger"),
        ):
            button = QPushButton(label)
            button.setFixedWidth(36)
            button.clicked.connect(
                lambda _checked=False, amount=delta: self._change_font(amount)
            )
            button.setToolTip(
                _setting_tooltip(
                    "Makes all interface text smaller or larger.",
                    "Changes the QApplication font size by one point within the 7–22 "
                    "point range, reapplies it to the monospace preview/log widgets, "
                    "and persists the result for the next launch.",
                )
            )
            setattr(self, attribute, button)
            more.addWidget(button)

        more.addSeparator()
        self._btn_update = QPushButton("⬆ Update")
        self._btn_update.clicked.connect(self._start_update)
        self._btn_update.setToolTip(
            _setting_tooltip(
                "Checks GitHub for a newer AutoTuner version and installs it while "
                "keeping your personal settings.",
                "Source checkouts update through git or a source ZIP. Frozen builds "
                "download the OS-specific GitHub Release ZIP and use a restart swap "
                "helper. autotuner_settings.json is backed up and restored around "
                "either update path.",
            )
        )
        more.addWidget(self._btn_update)

        self._btn_settings = QPushButton("⚙ Settings")
        self._btn_settings.clicked.connect(self._open_application_settings)
        self._btn_settings.setToolTip(
            _setting_tooltip(
                "Opens application-wide startup and window behaviour options.",
                "These opt-in preferences are shared by source and frozen builds "
                "through ~/.autotuner. They do not change model tuning, generated "
                "llama-server arguments, or per-model Expert overrides.",
            )
        )
        more.addWidget(self._btn_settings)
        more.hide()

        # ── Sysinfo bar ────────────────────────────────────────────────
        self._cpu_lbl = QLabel("CPU: —")
        self._vram_lbl = QLabel("VRAM: —")
        self._ram_lbl = QLabel("RAM: —")
        self._gpu_lbl = QLabel("GPU: —")
        hardware_labels = (
            self._cpu_lbl,
            self._vram_lbl,
            self._ram_lbl,
            self._gpu_lbl,
        )
        for lbl in hardware_labels:
            lbl.setProperty("themeRole", "sysbar")
        self._sysbar = _ResponsiveSystemBar(hardware_labels)
        self._sysbar.setProperty("themeRole", "sysbar")

        # ── Filter + model list ────────────────────────────────────────
        fr = QWidget()
        frl = QHBoxLayout(fr)
        frl.setContentsMargins(2, 2, 2, 2)
        frl.addWidget(QLabel("Filter:"))
        self._search = QLineEdit()
        self._search.setPlaceholderText("type to filter…")
        self._search.setToolTip(
            _setting_tooltip(
                "Filters the model list while you type so large collections are easier "
                "to navigate.",
                "The case-insensitive text filter changes only which already-scanned "
                "model entries are visible. It does not rescan folders, modify files, "
                "or change the currently selected model's saved settings.",
            )
        )
        self._search.textChanged.connect(self._apply_filter)
        frl.addWidget(self._search, 1)

        self._btn_list_view = QPushButton("☷ List")
        self._btn_list_view.setCheckable(True)
        self._btn_list_view.setToolTip(
            _setting_tooltip(
                "Shows every model in the familiar flat list.",
                "Favorites stay sorted first. Switching views keeps the active filter "
                "and selected model and does not rescan or move any files.",
            )
        )
        self._btn_list_view.clicked.connect(
            lambda _checked=False: self._set_model_view("list")
        )
        frl.addWidget(self._btn_list_view)

        self._btn_tree_view = QPushButton("🌳 Folders")
        self._btn_tree_view.setCheckable(True)
        self._btn_tree_view.setToolTip(
            _setting_tooltip(
                "Groups models by their real folder and subfolder hierarchy.",
                "Click folder rows to expand or collapse them. Favorites are also "
                "shown in a separate section at the top while remaining visible in "
                "their original folders.",
            )
        )
        self._btn_tree_view.clicked.connect(
            lambda _checked=False: self._set_model_view("tree")
        )
        frl.addWidget(self._btn_tree_view)

        self._model_list = QListWidget()
        self._favorite_delegate = _FavoriteStarDelegate(self._model_list)
        self._favorite_delegate.favoriteToggled.connect(self._set_model_favorite)
        self._model_list.setItemDelegate(self._favorite_delegate)
        self._model_list.currentItemChanged.connect(self._on_selection_changed)
        self._model_list.itemDoubleClicked.connect(self._on_model_activated)
        self._model_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._model_list.customContextMenuRequested.connect(
            lambda position: self._show_model_context_menu(position, self._model_list)
        )

        self._model_tree = QTreeWidget()
        self._model_tree.setHeaderHidden(True)
        self._model_tree.setUniformRowHeights(True)
        self._model_tree.setAnimated(True)
        self._tree_favorite_delegate = _FavoriteStarDelegate(self._model_tree)
        self._tree_favorite_delegate.favoriteToggled.connect(self._set_model_favorite)
        self._model_tree.setItemDelegate(self._tree_favorite_delegate)
        self._model_tree.currentItemChanged.connect(self._on_selection_changed)
        self._model_tree.itemClicked.connect(self._on_tree_item_clicked)
        self._model_tree.itemDoubleClicked.connect(
            lambda item, _column: self._on_model_activated(item)
        )
        self._model_tree.itemExpanded.connect(
            lambda item: self._on_tree_expansion_changed(item, True)
        )
        self._model_tree.itemCollapsed.connect(
            lambda item: self._on_tree_expansion_changed(item, False)
        )
        self._model_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._model_tree.customContextMenuRequested.connect(
            lambda position: self._show_model_context_menu(position, self._model_tree)
        )

        self._model_view_stack = QStackedWidget()
        self._model_view_stack.addWidget(self._model_list)
        self._model_view_stack.addWidget(self._model_tree)
        self._set_model_view(self._model_view_mode, persist=False, repopulate=False)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(2)
        ll.addWidget(fr)
        ll.addWidget(self._model_view_stack)

        # ── Config preview / Expert panel (stacked) ────────────────────
        self._config_preview = QTextEdit()
        self._config_preview.setReadOnly(True)
        self._config_preview.setPlaceholderText("Select a model to see its config…")
        self._apply_mono_font(self._config_preview)

        # The Expert panel lives in the same area as the read-only
        # preview; switching is a single setCurrentIndex() call so the
        # surrounding layout stays put (no relayout / no flicker).
        self._expert_panel = ExpertPanel()
        self._expert_panel.configChanged.connect(self._on_expert_cfg_changed)
        self._expert_panel.modeChanged.connect(self._on_expert_mode_changed)
        self._expert_panel.closeRequested.connect(self._exit_expert_mode)
        # Persist the live Expert state per model (debounced) and handle
        # the Reset button (clear saved override → reload Auto).
        self._expert_panel.stateChanged.connect(self._on_expert_state_changed)
        self._expert_panel.resetRequested.connect(self._on_expert_reset)

        self._config_stack = QStackedWidget()
        self._config_stack.addWidget(self._config_preview)  # index 0 — preview
        self._config_stack.addWidget(self._expert_panel)  # index 1 — expert
        self._config_stack.setCurrentIndex(0)

        # ── Fast setting-profile bank (model × performance mode × drafter) ─
        self._setting_profile_row = QWidget()
        profile_layout = QHBoxLayout(self._setting_profile_row)
        profile_layout.setContentsMargins(4, 2, 4, 2)
        profile_layout.addWidget(QLabel("Settings profile:"))
        self._setting_profile_combo = QComboBox()
        self._setting_profile_combo.setMinimumWidth(180)
        self._setting_profile_combo.addItem("Auto", app_settings.PROFILE_AUTO)
        for slot in app_settings.PROFILE_PERFORM_SLOTS:
            self._setting_profile_combo.addItem(
                app_settings.setting_profile_label(slot), slot
            )
        self._setting_profile_combo.addItem(
            app_settings.setting_profile_label(app_settings.PROFILE_PERFORM),
            app_settings.PROFILE_PERFORM,
        )
        for index, slot in enumerate(app_settings.CUSTOM_PROFILE_SLOTS, start=1):
            self._setting_profile_combo.addItem(f"Custom {index}", slot)
        self._setting_profile_combo.setToolTip(
            _setting_tooltip(
                "Switches instantly between untouched Auto tuning, the measured "
                "performance winner, and four personal setting profiles.",
                "Profiles are isolated by model, performance mode, selected draft "
                "head, and execution backend. Perform Vulkan, Perform HIP, Perform "
                "CPU, and other detected backends stay independent and disabled "
                "until measured. Editing Auto or Perform in Expert settings forks "
                "the change into a Custom slot instead of overwriting either source.",
            )
        )
        self._setting_profile_combo.currentIndexChanged.connect(
            self._on_setting_profile_changed
        )
        profile_layout.addWidget(self._setting_profile_combo, 1)
        self._btn_rename_setting_profile = QPushButton("Rename")
        self._btn_rename_setting_profile.setEnabled(False)
        self._btn_rename_setting_profile.setToolTip(
            "Rename the selected Custom profile for this model and performance mode."
        )
        self._btn_rename_setting_profile.clicked.connect(self._rename_setting_profile)
        profile_layout.addWidget(self._btn_rename_setting_profile)
        self._setting_profile_row.setEnabled(False)

        # ── Expert button row (sits between preview and Launch options) ─
        # In normal mode this row shows a "🔧 Expert" button plus a
        # "🔍 Diagnose" button. When Expert mode is active the Expert
        # button is replaced by an [Auto] [Manual] pair (the Expert
        # panel itself owns those toggle buttons — see ExpertPanel — but
        # we still mirror the state here in the bottom row for parallel
        # access). The Diagnose button stays visible in both modes.
        self._btn_expert = QPushButton("🔧 Expert settings")
        self._btn_expert.setToolTip(
            _setting_tooltip(
                "Opens detailed controls for context, memory placement, sampling, "
                "reasoning, parallel requests, and server flags.",
                "Expert values are stored per model. Auto mode re-fits dependent "
                "values around your overrides; Manual mode constructs the launch "
                "configuration directly and can create unsupported or unsafe "
                "combinations.",
            )
        )
        self._btn_expert.clicked.connect(self._enter_expert_mode)
        self._btn_diagnose = QPushButton("🔍 Diagnose")
        self._btn_diagnose.setToolTip(
            _setting_tooltip(
                "Explains what AutoTuner detected about the selected model and why it "
                "chose the shown configuration.",
                "The report includes GGUF metadata inputs, architecture and MoE/hybrid "
                "classification, KV-size math, context and memory capacity estimates, "
                "matched profile data, and warnings useful for technical debugging.",
            )
        )
        self._btn_diagnose.clicked.connect(self._show_diagnostic_report)
        self._btn_diagnose.setEnabled(False)  # disabled until a model is picked
        self._btn_benchmark = QPushButton("🚀 Performance test")
        self._btn_benchmark.setEnabled(False)
        self._btn_benchmark.setToolTip(
            _setting_tooltip(
                "Measures realistic prompt and generation speed and saves an "
                "independent optimized profile for every selected performance mode.",
                "Runs fresh private loopback llama-server instances with either the "
                "standard 12.5% prompt (65,536-token cap) or a custom 0.01–100% "
                "uncapped prompt, plus 256 n_decode tokens. It can test all four "
                "performance targets, "
                "all benchmarkable models, optional YaRN, and optional MTP/draft "
                "depths. CPU threads and batch settings are swept and confirmed; the "
                "fastest mode is reported, remembered per model, and selected. Exact "
                "contexts above the static safe estimate are saved only after the real "
                "server proves them. No hardware overclock setting is changed.",
            )
        )
        self._btn_benchmark.clicked.connect(self._start_performance_tuning)
        self._btn_performance_analysis = QPushButton("📊 Performance report")
        self._btn_performance_analysis.setToolTip(
            _setting_tooltip(
                "Shows summaries in-app and opens a detailed browser report for every run.",
                "Quick pass, Standard 12.5%, and Custom evidence is grouped separately; "
                "legacy 25% records migrate to Custom. The self-contained HTML expands "
                "every candidate's settings, samples, PP/decode/end-to-end charts, "
                "drafter variant, and drafted-token acceptance without external assets.",
            )
        )
        self._btn_performance_analysis.clicked.connect(self._show_performance_analysis)
        self._btn_expert_row = QWidget()
        bex = QHBoxLayout(self._btn_expert_row)
        bex.setContentsMargins(0, 0, 0, 0)
        bex.addStretch(1)
        bex.addWidget(self._btn_expert)
        bex.addWidget(self._btn_diagnose)
        bex.addWidget(self._btn_benchmark)
        bex.addWidget(self._btn_performance_analysis)
        bex.addStretch(1)

        # ── Launch options (checkboxes) ────────────────────────────────
        opts = QGroupBox("Launch options")
        self._launch_options_group = opts
        ol = QVBoxLayout(opts)
        ol.setSpacing(4)

        # ── mmproj (vision projector) selector ──────────────────────────
        # Always-on manual override. Lists EVERY projector in the model's
        # folder so the user can switch precision (bf16 / f16 / f32), force a
        # projector the auto-logic didn't pair, or pick "none". Files the
        # scanner considers incompatible with this model are prefixed with a
        # warning marker (not hidden) so experimenting is possible; launching
        # with one just logs a warning. The choice is remembered per model.
        self._mmproj_row = QWidget()
        _mmproj_l = QHBoxLayout(self._mmproj_row)
        _mmproj_l.setContentsMargins(0, 0, 0, 0)
        _mmproj_l.setSpacing(4)
        _mmproj_l.addWidget(QLabel("mmproj:"))
        self._cb_mmproj = QComboBox()
        mmproj_tip = _setting_tooltip(
            "Chooses the vision projector that lets a multimodal model understand "
            "images. Choose none for text-only use.",
            "Every projector found beside the model is listed. AutoTuner marks "
            "metadata or filename mismatches with ⚠ but still allows experiments. "
            "The per-model selection supplies --mmproj and can change VRAM/RAM use "
            "depending on projector precision and CPU-offload choice.",
        )
        self._cb_mmproj.setToolTip(mmproj_tip)
        self._mmproj_row.setToolTip(mmproj_tip)
        self._cb_mmproj.currentIndexChanged.connect(self._on_mmproj_changed)
        _mmproj_l.addWidget(self._cb_mmproj, 1)
        self._mmproj_row.setVisible(True)
        ol.addWidget(self._mmproj_row)

        # ── draft (speculative-decoding head) selector ──────────────────
        # Parallel to the mmproj dropdown. Always shown; lists every draft /
        # assistant GGUF in the model's folder, plus a "none" entry and (when
        # the GGUF carries an embedded MTP head) an "embedded MTP" entry.
        # Incompatible drafts are flagged with '⚠'. Selecting here overrides
        # the scanner's auto pick and is remembered per model.
        self._draft_row = QWidget()
        _draft_l = QHBoxLayout(self._draft_row)
        _draft_l.setContentsMargins(0, 0, 0, 0)
        _draft_l.setSpacing(4)
        _draft_l.addWidget(QLabel("draft:"))
        self._cb_draft = QComboBox()
        draft_tip = _setting_tooltip(
            "Chooses a smaller draft or assistant model that may speed up generation "
            "by proposing tokens for the main model to verify.",
            "Lists external draft, EAGLE-3, DFlash, and supported embedded MTP paths. "
            "AutoTuner marks likely incompatibilities with ⚠ but keeps them selectable. "
            "Selecting a draft enables it immediately; selecting no draft disables "
            "model-based speculative decoding. The choice is remembered per model.",
        )
        self._cb_draft.setToolTip(draft_tip)
        self._draft_row.setToolTip(draft_tip)
        self._cb_draft.currentIndexChanged.connect(self._on_draft_combo_changed)
        _draft_l.addWidget(self._cb_draft, 1)
        self._draft_row.setVisible(True)
        ol.addWidget(self._draft_row)

        self._chk_mmproj_cpu = QCheckBox("Keep mmproj in RAM (--no-mmproj-offload)")
        self._chk_mmproj_cpu.setToolTip(
            _setting_tooltip(
                "Keeps the vision projector in system memory. This frees GPU memory "
                "for the model or context, but image processing becomes slower.",
                "Emits --no-mmproj-offload. The projector's estimated footprint moves "
                "from VRAM to RAM in AutoTuner's budget; text-token generation is "
                "mostly unaffected after image encoding, while each image prefill uses "
                "CPU execution.",
            )
        )
        # n-gram (ngram-mod) self-speculative decoding. Unlike Draft, this
        # needs no draft model and works on ANY GGUF (builds a rolling-hash
        # lookup table from the live context, ~16 MB). It is therefore always
        # available — never greyed out — and independent of the Draft toggle.
        self._chk_ngram = QCheckBox("n-gram speculative (ngram-mod)")
        self._chk_ngram.setToolTip(
            _setting_tooltip(
                "Speeds up repetitive text or code by reusing patterns already found "
                "in the current context. It needs no separate draft model.",
                "Enables ngram-mod self-speculative decoding, which builds a rolling "
                "hash lookup table of roughly 16 MiB and proposes matching token "
                "sequences. Gains are workload-dependent and strongest for code edits, "
                "summaries, or repeated reasoning patterns.",
            )
        )
        # Host-memory prompt caching (--cache-ram / -cram). Auto-ON and
        # user-toggleable. Current mainline (b10045+) can reuse multimodal
        # prompts too; older or unprobeable builds are kept safe by forcing
        # --cache-ram 0 when Vision is active.
        self._chk_prompt_cache = QCheckBox("Prompt caching (host RAM, -cram)")
        self._chk_prompt_cache.setToolTip(
            _setting_tooltip(
                "Keeps reusable prompt beginnings in system RAM so repeated long "
                "instructions can reach the first generated token faster.",
                "Enables llama-server --cache-ram. Matching system prompts, RAG "
                "scaffolds, or coding-agent preambles can skip part of prompt "
                "evaluation. The configured MiB limit is included in RAM planning; "
                "multimodal prompt reuse requires llama.cpp b10045 or newer.",
            )
        )
        self._sp_prompt_cache_mib = QSpinBox()
        self._sp_prompt_cache_mib.setRange(-1, 65536)
        self._sp_prompt_cache_mib.setSpecialValueText("Unlimited (-1)")
        self._sp_prompt_cache_mib.setSuffix(" MiB cache limit")
        self._sp_prompt_cache_mib.setSingleStep(256)
        self._sp_prompt_cache_mib.setValue(app_settings.get_prompt_cache_ram_mib())
        self._sp_prompt_cache_mib.setToolTip(
            _setting_tooltip(
                "Limits how much system memory prompt caching may use. 2048 MiB is a "
                "bounded default; 0 disables the cache and -1 allows it to grow.",
                "Passed as --cache-ram / -cram. Positive values cap the host cache in "
                "MiB and are added to AutoTuner's RAM estimate. Unlimited mode keeps "
                "a safety reserve but can still grow until llama-server eviction or "
                "system memory pressure intervenes.",
            )
        )
        self._chk_thinking = QCheckBox("Thinking / Reasoning")
        self._chk_thinking.setToolTip(
            _setting_tooltip(
                "Allows supported reasoning models to use their thinking mode. Turn it "
                "off for shorter, direct answers when the template supports that.",
                "This per-model launch option controls whether AutoTuner applies the "
                "model/profile reasoning path. Exact flags and output separation depend "
                "on the selected chat template and llama.cpp build; Expert settings can "
                "further choose effort, token budget, and history preservation.",
            )
        )

        for chk in (
            self._chk_mmproj_cpu,
            self._chk_ngram,
            self._chk_prompt_cache,
            self._chk_thinking,
        ):
            chk.setEnabled(False)
            ol.addWidget(chk)
        ol.addWidget(self._sp_prompt_cache_mib)

        # Checkbox toggles → persist the override AND refresh the
        # context / memory estimates. Each slot knows which option it owns.
        self._chk_mmproj_cpu.toggled.connect(self._on_mmproj_cpu_toggled)
        self._chk_ngram.toggled.connect(self._on_ngram_toggled)
        self._chk_prompt_cache.toggled.connect(self._on_prompt_cache_toggled)
        self._sp_prompt_cache_mib.valueChanged.connect(
            self._on_prompt_cache_limit_changed
        )
        self._chk_thinking.toggled.connect(self._on_thinking_toggled)

        # Let the group use its style/font-dependent size hint. A fixed cap
        # clips launch controls with large fonts or themed group-box titles.

        right = QWidget()
        rl2 = QVBoxLayout(right)
        rl2.setContentsMargins(0, 0, 0, 0)
        rl2.setSpacing(4)
        rl2.addWidget(self._setting_profile_row)
        rl2.addWidget(self._config_stack, 1)
        rl2.addWidget(self._btn_expert_row)
        rl2.addWidget(opts)

        # ── Top HSplitter ──────────────────────────────────────────────
        top_split = QSplitter(Qt.Orientation.Horizontal)
        top_split.setObjectName("top_split")
        top_split.setChildrenCollapsible(False)
        top_split.addWidget(left)
        top_split.addWidget(right)
        top_split.setSizes([370, 650])

        # ── Log panel ──────────────────────────────────────────────────
        self._log_panel = QTextEdit()
        self._log_panel.setReadOnly(True)
        self._log_panel.setMinimumHeight(0)
        self._apply_mono_font(self._log_panel)
        self._log_panel.setPlaceholderText(
            "AutoTuner status messages appear here.\n"
            "Server output is shown in the separate terminal window."
        )

        main_split = QSplitter(Qt.Orientation.Vertical)
        main_split.setObjectName("main_split")
        main_split.setChildrenCollapsible(True)
        main_split.addWidget(top_split)
        main_split.addWidget(self._log_panel)
        main_split.setSizes([560, 240])
        self._main_split = main_split

        # Allow the log panel to be completely collapsed (min size 0)
        # and prevent the top half from collapsing; only the log panel should
        # be hideable. The previous version pinned top_split to a 400px
        # *minimum* which fought the splitter and stopped the bottom panel
        # from ever reaching size 0 — the panel could only be shrunk, never
        # fully retracted. We instead set collapse policy per index: the top
        # half cannot collapse, the log panel can.
        self._log_panel.setMinimumSize(QSize(0, 0))
        top_split.setMinimumHeight(0)
        main_split.setCollapsible(0, False)  # top half: never collapse
        main_split.setCollapsible(1, True)  # log panel: fully retractable
        # A slightly wider handle makes the bottom edge easy to grab and drag
        # all the way down to nothing.
        main_split.setHandleWidth(6)

        # Keep references so the inner pane arrangement can be persisted /
        # restored independently of the outer window geometry (QMainWindow
        # saveState() does not round-trip plain central-widget splitters).
        self._splitters: List[QSplitter] = [top_split, main_split]

        # ── Button row ─────────────────────────────────────────────────
        btn_row = QWidget()
        bl = QHBoxLayout(btn_row)
        bl.setContentsMargins(6, 4, 6, 4)

        bl.addWidget(QLabel("Host:"))
        self._host_edit = QLineEdit("127.0.0.1")
        self._host_edit.setFixedWidth(120)
        self._host_edit.setToolTip(
            _setting_tooltip(
                "Sets which network address the model server listens on. Keep "
                "127.0.0.1 if only programs on this computer should connect.",
                "Passed as llama-server --host. 127.0.0.1 is loopback-only; 0.0.0.0 "
                "or a LAN address can expose the OpenAI-compatible API, metrics, and "
                "slots endpoints to other devices. AutoTuner does not add authentication "
                "or a firewall rule.",
            )
        )
        bl.addWidget(self._host_edit)

        bl.addWidget(QLabel("Base port:"))
        self._port_edit = QLineEdit(str(self._base_port))
        self._port_edit.setFixedWidth(60)
        self._port_edit.setToolTip(
            _setting_tooltip(
                "Sets the starting network port for model servers. The first server "
                "normally uses 1234 and additional servers receive the next free port.",
                "The validated value is persisted as base_port and passed as --port "
                "after adding the manual offset and checking active server collisions. "
                "Stopped servers release their port for reuse; valid ports are in the "
                "1–65535 range.",
            )
        )
        # Persist on focus loss / Enter so the chosen port is remembered even
        # without launching (matches fork_path / font_size behaviour).
        self._port_edit.editingFinished.connect(self._persist_base_port)
        bl.addWidget(self._port_edit)

        bl.addWidget(QLabel("Offset:"))
        self._port_offset_combo = QComboBox()
        self._port_offset_combo.setFixedWidth(60)
        self._port_offset_combo.setToolTip(
            _setting_tooltip(
                "Adds 0–10 to the base port, making it easy to keep separate AutoTuner "
                "instances or environments on predictable ports.",
                "The persisted offset is added before AutoTuner searches for collisions "
                "with its managed servers. For example, base 1234 plus offset 2 starts "
                "at 1236, then later concurrent servers use subsequent free ports.",
            )
        )
        for i in range(11):  # 0 to 10
            self._port_offset_combo.addItem(str(i))
        # Restore the persisted offset selection (clamped to the combo range).
        self._port_offset_combo.setCurrentIndex(
            max(
                0,
                min(
                    self._port_offset_combo.count() - 1, app_settings.get_port_offset()
                ),
            )
        )
        self._port_offset_combo.currentIndexChanged.connect(
            lambda _i: app_settings.set_port_offset(
                int(self._port_offset_combo.currentText())
            )
        )
        bl.addWidget(self._port_offset_combo)

        bl.addStretch()

        # ── Multi-server switcher ──────────────────────────────────────
        # Lets the user target a SPECIFIC running server (to stop just that
        # one) instead of only ever the most-recent. Repopulated whenever the
        # server registry changes (launch / stop / crash poll).
        bl.addWidget(QLabel("Server:"))
        self._server_combo = QComboBox()
        self._server_combo.setMinimumWidth(220)
        self._server_combo.setToolTip(
            _setting_tooltip(
                "Selects which running model server the status and Stop controls refer "
                "to.",
                "Entries track a managed process, model, port, readiness, slot state, "
                "GPU placement, and estimated VRAM footprint. Changing the selection "
                "does not stop or restart any server.",
            )
        )
        bl.addWidget(self._server_combo)

        self._btn_toggle_log = QPushButton("▾ Log")
        self._btn_toggle_log.setFixedHeight(32)
        self._btn_toggle_log.setCheckable(True)
        self._btn_toggle_log.setChecked(True)
        self._btn_toggle_log.setToolTip(
            _setting_tooltip(
                "Shows or completely hides AutoTuner's bottom status panel.",
                "Moves the vertical splitter to restore its previous size or collapse "
                "the panel to zero. It affects only the interface layout; the separate "
                "llama-server terminal and log generation continue unchanged.",
            )
        )
        self._btn_toggle_log.clicked.connect(self._toggle_log_panel)
        bl.addWidget(self._btn_toggle_log)

        self._btn_ocr = QPushButton("📄 OCR…")
        self._btn_ocr.setFixedHeight(32)
        self._btn_ocr.setEnabled(False)
        self._btn_ocr.setVisible(False)
        self._btn_ocr.setToolTip(
            _setting_tooltip(
                "Opens the document workflow for the selected OCR model.",
                "Accepts images, PDF, Word, OpenDocument, presentation, and spreadsheet "
                "inputs. Office files are converted with LibreOffice, PDF pages are "
                "rendered with PyMuPDF, and normalized images are sent to the same "
                "llama-server chat-completions API used by the terminal workflow.",
            )
        )
        self._btn_ocr.clicked.connect(self._open_ocr_workflow)
        bl.addWidget(self._btn_ocr)

        self._btn_launch = QPushButton("▶ Launch")
        self._btn_launch.setFixedHeight(32)
        self._btn_launch.setEnabled(False)
        self._btn_launch.setToolTip(
            _setting_tooltip(
                "Starts the selected model with the shown Auto or Expert configuration. "
                "Existing servers can keep running.",
                "AutoTuner resolves a compatible binary, validates memory and flags, "
                "builds the llama-server command, selects the requested or emptier GPU, "
                "assigns the next free port, and tracks the new process and health "
                "endpoint.",
            )
        )
        self._btn_launch.clicked.connect(self._launch_server)
        bl.addWidget(self._btn_launch)

        self._btn_stop = QPushButton("■ Stop")
        self._btn_stop.setFixedHeight(32)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setToolTip(
            _setting_tooltip(
                "Stops only the server currently selected in the Server list.",
                "Requests a graceful termination of the tracked terminal process, then "
                "uses the platform-specific fallback if needed. Other managed servers "
                "remain active and the released port becomes reusable.",
            )
        )
        self._btn_stop.clicked.connect(self._stop_server)
        bl.addWidget(self._btn_stop)

        self._btn_stop_all = QPushButton("■ Stop all")
        self._btn_stop_all.setFixedHeight(32)
        self._btn_stop_all.setEnabled(False)
        self._btn_stop_all.setToolTip(
            _setting_tooltip(
                "Stops every model server that this AutoTuner window launched.",
                "Iterates over the managed server registry and performs the same "
                "graceful, platform-aware process shutdown for each entry. Unrelated "
                "llama-server processes started elsewhere are not targeted.",
            )
        )
        self._btn_stop_all.clicked.connect(self._stop_all_clicked)
        bl.addWidget(self._btn_stop_all)

        self._btn_quit = QPushButton("Quit")
        self._btn_quit.setFixedHeight(32)
        self._btn_quit.setToolTip(
            _setting_tooltip(
                "Closes AutoTuner completely instead of hiding it in the notification "
                "area.",
                "Sets the explicit-quit path, asks for confirmation when managed servers "
                "are active, shuts them down, saves window state, removes the tray icon, "
                "and exits the application process.",
            )
        )
        self._btn_quit.clicked.connect(self._request_quit)
        bl.addWidget(self._btn_quit)

        # ── Root ───────────────────────────────────────────────────────
        root = QWidget()
        root_l = QVBoxLayout(root)
        root_l.setContentsMargins(4, 0, 4, 0)
        root_l.setSpacing(0)
        root_l.addWidget(self._sysbar)
        root_l.addWidget(main_split, 1)
        root_l.addWidget(btn_row)
        self.setCentralWidget(root)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Starting…")

        # Translate the complete first-built widget tree. Later dialogs are
        # handled by LanguageManager's application event filter on Show.
        self._language_manager.apply_to(self)
        for error in self._language_manager.errors:
            self._log(f"[Language] Ignored pack: {error}")

        # Re-apply the inner pane arrangement now that every splitter exists.
        self._restore_splitter_states()

    def _toggle_more_toolbar(self, expanded: bool) -> None:
        """Keep the secondary toolbar open until the ellipsis is toggled again."""
        self._more_toolbar.setVisible(bool(expanded))

    def _populate_language_combo(self) -> None:
        """List every validated built-in/custom pack plus the import action."""
        combo = self._language_combo
        combo.blockSignals(True)
        combo.clear()
        for pack in self._language_manager.available():
            label = pack.name if pack.source == "builtin" else f"★ {pack.name}"
            combo.addItem(label, pack.qualified_id)
            combo.setItemData(
                combo.count() - 1,
                f"{pack.locale} · {pack.path}",
                Qt.ItemDataRole.ToolTipRole,
            )
        combo.insertSeparator(combo.count())
        combo.addItem(
            self._language_manager.translate("Custom language pack…"),
            CUSTOM_LANGUAGE_ACTION,
        )
        index = combo.findData(self._language_manager.current_id)
        combo.setCurrentIndex(max(0, index))
        combo.blockSignals(False)

    def _restore_language_combo(self) -> None:
        index = self._language_combo.findData(self._language_manager.current_id)
        self._language_combo.blockSignals(True)
        self._language_combo.setCurrentIndex(max(0, index))
        self._language_combo.blockSignals(False)

    def _activate_language(self, language_id: str) -> None:
        selected = self._language_manager.select(language_id)
        app_settings.set_language_id(selected)
        self._populate_language_combo()
        self._language_manager.apply_all()
        # The configuration preview embeds the model profile's explanation,
        # so re-render it in the newly selected language.
        entry = getattr(self, "_current_entry", None)
        if entry is not None:
            try:
                self._show_config(entry)
            except Exception as exc:
                self._log(f"[Language] Could not refresh the preview: {exc}")
        # List/tree rows carry plain-string tooltips and the Favourites label,
        # which the widget event filter cannot retranslate.
        if getattr(self, "_all_entries", None):
            try:
                self._apply_filter(self._search.text())
            except Exception as exc:
                self._log(f"[Language] Could not refresh the model list: {exc}")
        self._status.showMessage(
            self._language_manager.translate("Language changed."), 3000
        )

    def _on_language_changed(self, _index: int) -> None:
        language_id = self._language_combo.currentData()
        if language_id == CUSTOM_LANGUAGE_ACTION:
            self._import_custom_language()
            return
        if isinstance(language_id, str):
            self._activate_language(language_id)

    def _import_custom_language(self) -> None:
        tr = self._language_manager.translate
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            tr("Select a custom AutoTuner language pack"),
            str(self._language_manager.user_dir),
            tr("AutoTuner language packs (*.json);;All files (*)"),
        )
        if not selected:
            self._restore_language_combo()
            return
        source = Path(selected)
        try:
            imported = self._language_manager.import_pack(source)
        except FileExistsError:
            reply = QMessageBox.question(
                self,
                tr("Language pack"),
                tr("A language pack with this ID already exists. Replace it?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._restore_language_combo()
                return
            try:
                imported = self._language_manager.import_pack(source, replace=True)
            except (LanguagePackError, OSError) as exc:
                QMessageBox.warning(self, tr("Could not load language pack"), str(exc))
                self._restore_language_combo()
                return
        except (LanguagePackError, OSError) as exc:
            QMessageBox.warning(self, tr("Could not load language pack"), str(exc))
            self._restore_language_combo()
            return

        self._populate_language_combo()
        self._activate_language(imported.qualified_id)
        self._log(f"[Language] Imported {imported.name}: {imported.path}")
        QMessageBox.information(
            self,
            self._language_manager.translate("Language pack imported"),
            self._language_manager.translate(
                "The language pack was imported and activated."
            ),
        )

    def _open_language_folder(self) -> None:
        try:
            template = self._language_manager.ensure_custom_template()
            _open_local_folder(self._language_manager.user_dir)
            self._log(f"[Language] Custom-pack template: {template}")
        except (LanguagePackError, OSError) as exc:
            QMessageBox.warning(
                self,
                self._language_manager.translate("Could not load language pack"),
                str(exc),
            )

    # ------------------------------------------------------------------
    # Authenticated external-control gateway
    # ------------------------------------------------------------------
    def _configure_control_api(self) -> bool:
        """Apply persisted API settings without ever exposing a non-loopback bind."""
        if not app_settings.get_control_api_enabled():
            self._stop_control_api()
            return True
        try:
            token = app_settings.ensure_control_api_token()
            port = app_settings.get_control_api_port()
        except (OSError, ValueError) as exc:
            self._log(f"[Control API] Could not prepare credentials: {exc}")
            return False

        current = self._control_api
        if (
            current is not None
            and current.running
            and current.port == port
            and current.token == token
        ):
            self._refresh_control_api_catalogue()
            return True
        self._stop_control_api()
        try:
            api = ControlApiServer(
                host="127.0.0.1",
                port=port,
                token=token,
                switch_callback=self._control_switch_from_http,
                stop_callback=self._control_stop_from_http,
                log_callback=self._control_log_ready.emit,
                switch_timeout_s=900.0,
                runtime_probe=self._control_probe_runtime,
            )
            self._control_api = api
            self._refresh_control_api_catalogue()
            endpoint = api.start()
            self._write_control_discovery(api)
            self._log(
                f"[Control API] OpenAI endpoint ready: {endpoint}/v1 "
                "(bearer authentication required)."
            )
            return True
        except (OSError, ValueError) as exc:
            self._control_api = None
            self._write_control_discovery(None)
            self._log(f"[Control API] Could not start on port {port}: {exc}")
            return False

    def _stop_control_api(self) -> None:
        api = self._control_api
        self._control_api = None
        if api is not None:
            api.stop()
        self._write_control_discovery(None)

    def _control_discovery_path(self) -> Path:
        # Next to the settings file, so test/portable overrides of the settings
        # location automatically relocate the discovery document as well.
        return Path(app_settings._settings_file()).parent / DISCOVERY_FILE_NAME

    def _write_control_discovery(self, api: Optional[ControlApiServer]) -> None:
        """Publish (or retract) the small client-discovery document.

        Pi and benchmark clients read ``control_api.json`` instead of parsing
        the complete settings file. The token is written only while the
        gateway is enabled and is rewritten immediately after regeneration.
        """
        try:
            if api is not None and api.running:
                payload = discovery_payload(
                    enabled=True,
                    port=api.port,
                    token=api.token,
                    version=VERSION,
                    pid=os.getpid(),
                    started_at=datetime.now(timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    host=api.host,
                )
            else:
                payload = discovery_payload(
                    enabled=False,
                    port=app_settings.get_control_api_port(),
                    version=VERSION,
                    pid=os.getpid(),
                )
            if not write_discovery_file(self._control_discovery_path(), payload):
                self._log(
                    "[Control API] Could not write the discovery file "
                    f"{self._control_discovery_path()}"
                )
        except Exception as exc:
            self._log(f"[Control API] Discovery file error: {exc}")

    def _control_runtime_catalogue(
        self,
    ) -> Tuple[List[ApiRuntime], str, Dict[str, Path]]:
        """Snapshot the toolbar's llama builds without probing any binary."""
        runtimes: List[ApiRuntime] = []
        paths: Dict[str, Path] = {}
        default_id = ""
        try:
            from auto_tuner import _server_binary_in_fork
        except Exception:
            return runtimes, default_id, paths
        combo = getattr(self, "_fork_combo", None)
        if combo is None:
            return runtimes, default_id, paths
        current = combo.currentIndex()
        used: set[str] = set()
        for index in range(combo.count()):
            path = combo.itemData(index)
            if path is None:
                continue
            label = str(combo.itemText(index)).strip()
            root = Path(path)
            runtime_id = _control_runtime_id(label)
            if runtime_id in used:
                suffix = 2
                while f"{runtime_id}-{suffix}" in used:
                    suffix += 1
                runtime_id = f"{runtime_id}-{suffix}"
            used.add(runtime_id)
            binary = _server_binary_in_fork(root) or ""
            available = bool(binary)
            runtimes.append(
                ApiRuntime(
                    id=runtime_id,
                    label=label,
                    server_binary=binary,
                    backend=_control_runtime_backend(label, binary or str(root)),
                    is_default=index == current,
                    available=available,
                    unavailable_reason=(
                        ""
                        if available
                        else "No runnable llama-server found in this build."
                    ),
                )
            )
            paths[runtime_id] = root
            if index == current:
                default_id = runtime_id
        return runtimes, default_id, paths

    @staticmethod
    def _control_probe_runtime(runtime: ApiRuntime) -> ApiRuntime:
        """Fill build identity on the HTTP thread using the cached probes."""
        if not runtime.server_binary:
            return runtime
        build = probe_binary_build_number(runtime.server_binary)
        info = probe_binary_build_info(runtime.server_binary)
        return dataclasses.replace(runtime, build=build, build_info=info)

    def _refresh_control_api_catalogue(self) -> None:
        try:
            runtimes, default_runtime_id, runtime_paths = (
                self._control_runtime_catalogue()
            )
            models = _control_api_catalogue(
                self._all_entries, self._profiles, default_runtime_id
            )
            model_paths = {
                model.id: Path(model.path) for model in models if model.runnable
            }
            if self._control_api is not None:
                self._control_api.update_runtimes(runtimes, default_runtime_id)
                self._control_api.update_models(models)
            self._control_model_paths = model_paths
            self._control_runtime_paths = runtime_paths
        except Exception as exc:
            self._log(f"[Control API] Could not refresh model catalogue: {exc}")

    def _control_switch_from_http(
        self,
        model_id: str,
        timeout_s: float,
        options: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        runtime_id = str((options or {}).get("runtime_id") or "")
        request = ControlRequest(
            action="switch",
            model_id=model_id,
            timeout_s=timeout_s,
            runtime_id=runtime_id,
        )
        self._control_request_ready.emit(request)
        return request.wait()

    def _control_stop_from_http(self, timeout_s: float) -> Dict[str, object]:
        request = ControlRequest(action="stop", timeout_s=timeout_s)
        self._control_request_ready.emit(request)
        return request.wait()

    def _handle_control_request(self, request: object) -> None:
        """Execute an HTTP-thread request on Qt's GUI thread."""
        if not isinstance(request, ControlRequest):
            return
        if self._control_closing:
            request.fail(
                "AutoTuner is shutting down.",
                status=503,
                code="shutting_down",
                backend_untouched=True,
            )
            return
        if request.action == "stop":
            record = self._control_api_record
            if record is not None and record in self._servers:
                process = record.get("proc")
                self._stop_specific_server(record)
                if isinstance(process, _TerminalProcess) and not process.has_stopped():
                    self._wait_for_control_process(
                        request,
                        process,
                        time.monotonic() + 15.0,
                        lambda: request.complete({"status": "stopped"}),
                    )
                    return
            else:
                self._control_api_record = None
                if self._control_api is not None:
                    self._control_api.clear_active()
            request.complete({"status": "stopped"})
            return
        if request.action != "switch":
            request.fail("Unknown control action.", status=400, code="invalid_action")
            return

        benchmark_busy = False
        try:
            benchmark_busy = bool(
                self._benchmark_thread is not None
                and self._benchmark_thread.isRunning()
            )
        except RuntimeError:
            benchmark_busy = False
        if benchmark_busy or self._ocr_thread is not None:
            request.fail(
                "AutoTuner is busy with an exclusive benchmark or OCR workflow.",
                status=409,
                code="autotuner_busy",
                backend_untouched=True,
            )
            return
        if self._system is None:
            request.fail(
                "Hardware detection is still running; retry in a moment.",
                status=503,
                code="hardware_pending",
                backend_untouched=True,
            )
            return

        path = self._control_model_paths.get(request.model_id)
        entry = next(
            (
                model
                for model in self._all_entries
                if path is not None and model.path == path
            ),
            None,
        )
        if entry is None:
            request.fail(
                f"The model {request.model_id!r} is no longer available.",
                status=404,
                code="model_not_found",
                backend_untouched=True,
            )
            return

        requested_runtime = str(request.runtime_id or "")
        if requested_runtime:
            runtime_root = self._control_runtime_paths.get(requested_runtime)
            if runtime_root is None:
                request.fail(
                    f"The llama-server runtime {requested_runtime!r} is not "
                    "available; read GET /api/v1/runtimes.",
                    status=409,
                    code="runtime_unavailable",
                    backend_untouched=True,
                )
                return
            if not self._select_control_runtime(runtime_root):
                request.fail(
                    f"The llama-server runtime {requested_runtime!r} could not "
                    "be selected in the toolbar.",
                    status=409,
                    code="runtime_unavailable",
                    backend_untouched=True,
                )
                return
        else:
            requested_runtime = self._current_control_runtime_id()

        active = self._control_api_record
        if active is not None and active in self._servers:
            if active.get("control_model_id") == request.model_id and (
                not requested_runtime
                or str(active.get("control_runtime_id") or "") == requested_runtime
            ):
                if active.get("ready"):
                    request.complete(self._control_record_result(active))
                else:
                    active.setdefault("control_requests", []).append(request)
                return
            process = active.get("proc")
            self._stop_specific_server(active)
            self._control_api_record = None
            if isinstance(process, _TerminalProcess) and not process.has_stopped():
                # _TerminalProcess.stop() is intentionally non-blocking for the
                # GUI. The queued transition waits until its process group has
                # actually exited, so old and new models never overlap in VRAM.
                self._wait_for_control_process(
                    request,
                    process,
                    time.monotonic() + 15.0,
                    lambda: self._launch_control_entry(request, entry),
                )
                return
        else:
            self._control_api_record = None
        self._launch_control_entry(request, entry)

    def _wait_for_control_process(
        self,
        request: ControlRequest,
        process: _TerminalProcess,
        deadline: float,
        on_stopped: Callable[[], None],
    ) -> None:
        """Poll process-group termination without blocking Qt's event loop."""
        if request.done:
            return
        if process.has_stopped():
            on_stopped()
            return
        if self._control_closing:
            request.fail(
                "AutoTuner is shutting down.", status=503, code="shutting_down"
            )
            return
        if time.monotonic() >= deadline:
            request.fail(
                "The previous llama-server did not exit; the new model was not started.",
                status=504,
                code="stop_timeout",
            )
            return
        QTimer.singleShot(
            50,
            lambda: self._wait_for_control_process(
                request, process, deadline, on_stopped
            ),
        )

    def _launch_control_entry(self, request: ControlRequest, entry: ModelEntry) -> None:
        if request.done:
            return
        # Reuse exactly the same per-model target/profile/draft/mmproj and
        # application-wide mode/GPU controls as a click launch. The external
        # API never constructs an independent TunedConfig.
        self._show_config(entry)
        self._select_model_path(entry.path, self._active_model_view())
        try:
            record = self._launch_server(interactive=False)
        except Exception as exc:
            self._log(f"[Control API] Launch raised {type(exc).__name__}: {exc}")
            request.fail(
                f"AutoTuner could not start the requested model: {exc}",
                status=500,
                code="launch_exception",
            )
            return
        if record is None:
            request.fail(
                self._last_launch_error
                or "AutoTuner could not start the requested model.",
                status=409,
                code="launch_failed",
            )
            return
        record["control_model_id"] = request.model_id
        record["control_runtime_id"] = self._current_control_runtime_id()
        record["control_requests"] = [request]
        # Expire slightly before the HTTP wait so Qt can stop an alive but
        # never-ready backend and deliver a structured timeout response.
        record["control_deadline"] = max(time.monotonic(), request.deadline - 0.5)
        self._control_api_record = record
        if record.get("ready"):
            self._complete_control_record(record)

    def _current_control_runtime_id(self) -> str:
        combo = getattr(self, "_fork_combo", None)
        if combo is None or combo.currentData() is None:
            return ""
        current = Path(combo.currentData())
        for runtime_id, root in self._control_runtime_paths.items():
            try:
                if root.resolve(strict=False) == current.resolve(strict=False):
                    return runtime_id
            except (OSError, RuntimeError):
                if root == current:
                    return runtime_id
        return _control_runtime_id(str(combo.currentText()))

    def _select_control_runtime(self, root: Path) -> bool:
        """Select *root* in the toolbar build dropdown, exactly like a click."""
        combo = getattr(self, "_fork_combo", None)
        if combo is None:
            return False
        try:
            wanted = root.resolve(strict=False)
        except (OSError, RuntimeError):
            wanted = root
        for index in range(combo.count()):
            path = combo.itemData(index)
            if path is None:
                continue
            try:
                same = Path(path).resolve(strict=False) == wanted
            except (OSError, RuntimeError):
                same = Path(path) == root
            if not same:
                continue
            if index != combo.currentIndex():
                self._log(
                    f"[Control API] Selecting llama build {combo.itemText(index)!r}"
                )
                combo.setCurrentIndex(index)
            return True
        return False

    def _control_record_result(self, record: dict) -> Dict[str, object]:
        return {
            "backend_url": str(
                record.get("client_base_url") or record.get("base_url") or ""
            ),
            "backend_api_key": _extract_server_api_key(record.get("command", [])),
            "alias": str(record.get("alias") or record.get("model") or ""),
            "detail": self._control_status_detail(record),
        }

    def _control_status_detail(self, record: dict) -> Dict[str, object]:
        """Describe the API-managed server for ``/api/v1/status`` consumers."""
        command = [str(value) for value in record.get("command", []) or []]
        cfg = record.get("cfg")
        entry = record.get("entry")
        if entry is None:
            # Records created outside _launch_server (tests, legacy sessions)
            # still resolve the scanned entry through the stable control ID.
            path = self._control_model_paths.get(
                str(record.get("control_model_id") or "")
            )
            entry = next(
                (
                    model
                    for model in self._all_entries
                    if path is not None and model.path == path
                ),
                None,
            )
        proc = record.get("proc")
        pid: Optional[int] = None
        log_path: Optional[str] = None
        try:
            raw_proc = getattr(proc, "proc", None)
            if raw_proc is not None and getattr(raw_proc, "pid", None):
                pid = int(raw_proc.pid)
            if getattr(proc, "log_path", None):
                log_path = str(proc.log_path)
        except Exception:
            pid, log_path = None, None

        binary = str(record.get("runtime_binary") or (command[0] if command else ""))
        runtime_id = str(record.get("control_runtime_id") or "")
        runtime_label = str(record.get("fork_label") or "")
        build = record.get("llama_build")
        backend = ""
        if self._system is not None and self._system.gpus:
            backends = {
                app_settings.normalise_performance_backend(gpu.runtime_backend)
                for gpu in self._system.gpus
                if gpu.runtime_backend
            }
            backends.discard("")
            if len(backends) == 1:
                backend = next(iter(backends))
        if not backend:
            backend = _control_runtime_backend(runtime_label, binary)
        runtime = {
            "id": runtime_id or None,
            "label": runtime_label or None,
            "server_binary": binary or None,
            "backend": backend,
            "build": f"b{int(build)}" if build else None,
            "build_number": int(build) if build else None,
            "build_info": probe_binary_build_info(binary) if binary else None,
        }

        model: Dict[str, object] = {
            "id": str(record.get("control_model_id") or "") or None,
            "name": str(record.get("model") or "") or None,
            "path": None,
            "quant": None,
            "ftype": None,
            "params_b": None,
            "size_bytes": None,
            "architecture": None,
            "draft_model_path": _command_value(command, "-md", "--model-draft"),
            "mmproj_path": _command_value(command, "--mmproj"),
            "profile": str(record.get("profile_name") or "") or None,
            "profile_file": str(record.get("profile_file") or "") or None,
        }
        if entry is not None:
            try:
                model["path"] = str(entry.path)
                model["quant"] = _control_quant_label(entry.name) or None
                model["ftype"] = model["quant"]
                params = float(extract_params_billion(entry.name) or 0.0)
                model["params_b"] = params if params > 0 else None
                model["size_bytes"] = int(entry.size_bytes or 0) or None
                model["architecture"] = str(entry.architecture or "") or None
            except Exception:
                pass
        if model["path"] is None:
            model["path"] = _command_value(command, "-m", "--model")

        launch: Dict[str, object] = {
            "ctx_size": None,
            "gpu_layers": None,
            "threads": None,
            "batch_threads": None,
            "batch": None,
            "ubatch": None,
            "kv_type_k": None,
            "kv_type_v": None,
            "flash_attention": _command_value(command, "-fa", "--flash-attn"),
            "spec_type": _command_value(command, "--spec-type"),
            "draft_n_max": None,
            "main_gpu": None,
            "tensor_split": None,
            "parallel": None,
            "thinking": record.get("thinking"),
            "profile": model["profile"],
            "performance_target": str(record.get("performance_target") or "") or None,
            "mode": str(record.get("mode") or "") or None,
            "load_mode": None,
            "n_cpu_moe": None,
            "port": record.get("port"),
        }
        for key, flags in (
            ("draft_n_max", ("--spec-draft-n-max",)),
            ("parallel", ("-np", "--parallel")),
        ):
            raw = _command_value(command, *flags)
            if raw is not None:
                try:
                    launch[key] = int(raw)
                except ValueError:
                    launch[key] = raw
        if cfg is not None:
            try:
                launch.update(
                    {
                        "ctx_size": int(cfg.ctx),
                        "gpu_layers": int(cfg.ngl),
                        "threads": int(cfg.threads),
                        "batch_threads": int(cfg.batch_threads),
                        "batch": int(cfg.batch),
                        "ubatch": int(cfg.ubatch),
                        "kv_type_k": str(cfg.cache_k),
                        "kv_type_v": str(cfg.cache_v),
                        "main_gpu": cfg.main_gpu,
                        "tensor_split": cfg.tensor_split,
                        "load_mode": str(cfg.load_mode),
                        "n_cpu_moe": cfg.n_cpu_moe,
                    }
                )
                if launch["flash_attention"] is None:
                    launch["flash_attention"] = "on" if cfg.flash_attn else "off"
            except Exception:
                pass
        else:
            for key, flags in (
                ("ctx_size", ("-c", "--ctx-size")),
                ("gpu_layers", ("-ngl", "--gpu-layers", "--n-gpu-layers")),
                ("threads", ("-t", "--threads")),
                ("batch_threads", ("-tb", "--threads-batch")),
                ("batch", ("-b", "--batch-size")),
                ("ubatch", ("-ub", "--ubatch-size")),
                ("kv_type_k", ("-ctk", "--cache-type-k")),
                ("kv_type_v", ("-ctv", "--cache-type-v")),
                ("main_gpu", ("-mg", "--main-gpu")),
                ("tensor_split", ("-ts", "--tensor-split")),
            ):
                raw = _command_value(command, *flags)
                if raw is None:
                    continue
                try:
                    launch[key] = int(raw)
                except ValueError:
                    launch[key] = raw

        devices: List[Dict[str, object]] = []
        if self._system is not None:
            for gpu in self._system.gpus:
                devices.append(
                    {
                        "index": int(gpu.index),
                        "name": str(gpu.name),
                        "vendor": str(gpu.vendor),
                        "backend": app_settings.normalise_performance_backend(
                            gpu.runtime_backend
                        )
                        or None,
                        "device": gpu.runtime_device,
                        "vram_mb": int(gpu.total_vram_mb),
                        "free_vram_mb": int(gpu.free_vram_mb),
                    }
                )

        env: Dict[str, str] = {}
        overrides = getattr(cfg, "env_overrides", None) if cfg is not None else None
        if isinstance(overrides, dict):
            for key, value in overrides.items():
                upper = str(key).upper()
                if any(
                    secret in upper for secret in ("KEY", "TOKEN", "SECRET", "PASS")
                ):
                    continue
                env[str(key)] = str(value)

        return {
            "pid": pid,
            "log_path": log_path,
            "runtime": runtime,
            "model": model,
            "launch": launch,
            "devices": devices,
            "env": env,
            "command_line": _redacted_command_list(command),
        }

    def _complete_control_record(self, record: dict) -> None:
        result = self._control_record_result(record)
        record.pop("control_deadline", None)
        pending = list(record.pop("control_requests", []))
        for request in pending:
            if isinstance(request, ControlRequest):
                request.complete(result)

    def _fail_control_record(
        self,
        record: dict,
        message: str,
        *,
        status: int = 502,
        code: str = "backend_exited",
    ) -> None:
        record.pop("control_deadline", None)
        pending = list(record.pop("control_requests", []))
        for request in pending:
            if isinstance(request, ControlRequest):
                request.fail(message, status=status, code=code)
        if record is self._control_api_record:
            self._control_api_record = None
            if self._control_api is not None:
                self._control_api.clear_active(
                    str(record.get("control_model_id") or "")
                )

    def _open_application_settings(self) -> None:
        """Preview appearance and persist selection only on confirmation.

        Theme-editor Save is intentionally immediate and independent from the
        outer settings dialog's selection/behaviour confirmation.
        """
        dialog = _ApplicationSettingsDialog(self)
        app = cast(Optional[QApplication], QApplication.instance())
        original = copy.deepcopy(self._theme_manager.current_definition)

        def refresh_widgets() -> None:
            self._apply_mono_font(self._config_preview)
            self._apply_mono_font(self._log_panel)
            for view in (self._model_list, self._model_tree):
                viewport = view.viewport()
                if viewport is not None:
                    viewport.update()

        def apply_definition(theme) -> None:
            self._apply_theme_definition(theme, app, refresh_widgets)

        def apply_selected() -> None:
            apply_definition(
                self._theme_manager.get(str(dialog.theme_combo.currentData()))
            )

        def rollback() -> None:
            apply_definition(original)

        def repopulate(selected: str) -> None:
            dialog.theme_combo.blockSignals(True)
            dialog.theme_combo.clear()
            widest = 0
            for theme in self._theme_manager.available():
                text = f"{theme.name} ({theme.source})"
                dialog.theme_combo.addItem(text, theme.qualified_id)
                dialog.theme_combo.setItemData(
                    dialog.theme_combo.count() - 1,
                    text,
                    Qt.ItemDataRole.ToolTipRole,
                )
                widest = max(
                    widest, dialog.theme_combo.fontMetrics().horizontalAdvance(text)
                )
            dialog.theme_combo.setMinimumWidth(min(500, max(160, widest + 48)))
            dialog.theme_combo.setCurrentIndex(
                max(0, dialog.theme_combo.findData(selected))
            )
            dialog.theme_combo.blockSignals(False)

        def reload_themes() -> None:
            self._theme_manager.reload()
            selected = str(dialog.theme_combo.currentData())
            if selected not in self._theme_manager.themes:
                selected = SYSTEM_THEME_ID
            repopulate(selected)
            apply_selected()
            if self._theme_manager.errors:
                QMessageBox.warning(
                    dialog,
                    "Theme files ignored",
                    "\n".join(self._theme_manager.errors[:8]),
                )

        def preview_editor(edited) -> None:
            apply_definition(edited)

        def customize() -> None:
            theme = self._theme_manager.get(str(dialog.theme_combo.currentData()))
            editor = ThemeEditorDialog(theme, dialog, preview_editor)
            if editor.exec() != QDialog.DialogCode.Accepted:
                apply_selected()
                return
            try:
                saved = editor.theme()
                replace_id = _theme_replace_id(theme, saved)
                self._theme_manager.save_user_theme(saved, replace_id=replace_id)
            except FileExistsError:
                QMessageBox.warning(
                    dialog,
                    "Theme exists",
                    "Choose a different ID. Only the selected user theme can be overwritten.",
                )
                apply_selected()
                return
            except Exception as exc:
                QMessageBox.warning(dialog, "Could not save theme", str(exc))
                apply_selected()
                return
            repopulate(saved.qualified_id)
            apply_selected()

        dialog.theme_combo.currentIndexChanged.connect(lambda _index: apply_selected())
        dialog.reload_themes_button.clicked.connect(reload_themes)
        dialog.customize_theme_button.clicked.connect(customize)

        def open_theme_folder() -> None:
            try:
                self._theme_manager.user_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                QMessageBox.warning(dialog, "Theme folder", str(exc))
                return
            _open_local_folder(self._theme_manager.user_dir)

        dialog.open_themes_button.clicked.connect(open_theme_folder)
        dialog.about_button.clicked.connect(lambda: self._show_about(dialog))

        def export_profiles() -> None:
            default = (
                app_settings.app_data_dir() / "AutoTuner-performance-profiles.json"
            )
            selected, _filter = QFileDialog.getSaveFileName(
                dialog,
                "Export performance profiles",
                str(default),
                "AutoTuner profiles (*.json);;All files (*)",
            )
            if not selected:
                return
            destination = Path(selected)
            if not destination.suffix:
                destination = destination.with_suffix(".json")
            ok, message, _count = app_settings.export_performance_profiles(destination)
            if ok:
                QMessageBox.information(dialog, "Profiles exported", message)
                self._log(f"[Profiles] {message} → {destination}")
            else:
                QMessageBox.warning(dialog, "Profile export failed", message)

        def import_profiles() -> None:
            selected, _filter = QFileDialog.getOpenFileName(
                dialog,
                "Import performance profiles",
                str(app_settings.app_data_dir()),
                "AutoTuner profiles (*.json);;All files (*)",
            )
            if not selected:
                return
            ok, message, _count = app_settings.import_performance_profiles(
                Path(selected), [entry.path for entry in self._all_entries]
            )
            if ok:
                self._refresh_config_preview()
                QMessageBox.information(dialog, "Profiles imported", message)
                self._log(f"[Profiles] {message} ← {selected}")
            else:
                QMessageBox.warning(dialog, "Profile import failed", message)

        dialog.export_profiles_button.clicked.connect(export_profiles)
        dialog.import_profiles_button.clicked.connect(import_profiles)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            rollback()
            return

        try:
            autostart_enabled = dialog.autostart_checkbox.isChecked()
            if autostart_enabled != dialog.autostart_was_enabled:
                startup_manager.set_autostart_enabled(autostart_enabled)
        except startup_manager.AutostartError as exc:
            rollback()
            QMessageBox.critical(self, "Autostart", str(exc))
            return

        minimize_to_tray = dialog.minimize_checkbox.isChecked()
        app_settings.set_minimize_on_close(minimize_to_tray)
        if dialog.start_minimized_explicit or dialog.start_minimized_touched:
            app_settings.set_start_minimized_at_login(
                dialog.start_minimized_checkbox.isChecked()
            )
        debug_enabled = dialog.debug_checkbox.isChecked()
        app_settings.set_debug_mode(debug_enabled)
        self._set_internal_debug_mode(debug_enabled)
        api_enabled = dialog.control_api_checkbox.isChecked()
        api_save_error = ""
        try:
            stored_token = (
                None
                if app_settings.control_api_token_is_overridden()
                else dialog.control_api_token.text()
            )
            app_settings.set_control_api_config(
                api_enabled, dialog.control_api_port.value(), stored_token
            )
            api_configured = self._configure_control_api()
        except (OSError, ValueError) as exc:
            api_configured = False
            api_save_error = str(exc)
            self._log(f"[Control API] Could not save settings: {exc}")
        if api_save_error:
            QMessageBox.warning(
                self,
                "External control API",
                "The control API settings were not saved; the previous configuration "
                f"is still active.\n\n{api_save_error}",
            )
        elif api_enabled and not api_configured:
            QMessageBox.warning(
                self,
                "External control API",
                "The loopback control API could not start. Check whether the selected "
                "port is already in use; details are in the AutoTuner log.",
            )
        selected_theme = str(dialog.theme_combo.currentData())
        if selected_theme in self._theme_manager.themes:
            app_settings.set_theme_id(selected_theme)
        if not minimize_to_tray:
            self._destroy_tray_icon()
        self._status.showMessage(
            "Settings saved"
            if not api_save_error
            else "Settings saved; control API settings unchanged",
            3000,
        )

    def _apply_theme_definition(
        self,
        theme: ThemeDefinition,
        app: Optional[QApplication],
        refresh_widgets: Callable[[], None],
    ) -> None:
        """Apply a preview while preserving the user's pane arrangement."""
        if app is None:
            return
        window_size = self.size()
        splitter_sizes = [(splitter, splitter.sizes()) for splitter in self._splitters]
        self._theme_manager.apply_definition(app, theme, self._font_size)
        refresh_widgets()

        def restore_layout() -> None:
            if not self.isMaximized():
                self.resize(window_size)
            for splitter, sizes in splitter_sizes:
                splitter.setSizes(sizes)
            central = self.centralWidget()
            if central is not None and central.layout() is not None:
                central.layout().activate()
            for splitter, sizes in splitter_sizes:
                splitter.setSizes(sizes)

        QTimer.singleShot(0, restore_layout)

    def _show_about(self, parent: Optional[QWidget] = None) -> None:
        """Show static version and repository information without a network request."""
        dialog = QMessageBox(parent or self)
        dialog.setWindowTitle("About AutoTuner")
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setTextFormat(Qt.TextFormat.RichText)
        dialog.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        dialog.setText(_about_text())
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.exec()

    def _ensure_tray_icon(self) -> bool:
        """Create and show the notification-area icon on demand."""
        if self._tray_icon is not None:
            if not self._tray_icon.isVisible():
                self._tray_icon.show()
            return True
        if not _system_tray_supported():
            return False

        icon = self.windowIcon()
        app = cast(Optional[QApplication], QApplication.instance())
        if icon.isNull() and app is not None:
            icon = app.windowIcon()
        if icon.isNull():
            fallback = _bundled_resource("assets", "AutoTuner.png")
            if fallback.is_file():
                icon = QIcon(str(fallback))
        if icon.isNull():
            return False

        tray = QSystemTrayIcon(icon, self)
        tray.setToolTip("AutoTuner")
        menu = QMenu()
        show_action = QAction("Show AutoTuner", menu)
        show_action.triggered.connect(self._restore_from_tray)
        menu.addAction(show_action)
        menu.addSeparator()
        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(self._request_quit)
        menu.addAction(quit_action)
        tray.setContextMenu(menu)
        tray.activated.connect(self._on_tray_activated)

        # Keep explicit references: on some PyQt/Python combinations the menu
        # can otherwise be garbage-collected while the native tray icon lives.
        self._tray_icon = tray
        self._tray_menu = menu
        tray.show()
        return True

    def _hide_to_tray(self) -> bool:
        """Hide the main window while keeping the process reachable via tray."""
        if not self._ensure_tray_icon():
            return False
        self._tray_restore_maximized = self.isMaximized()
        self.hide()
        if not self._tray_hint_shown and self._tray_icon is not None:
            if QSystemTrayIcon.supportsMessages():
                self._tray_icon.showMessage(
                    "AutoTuner is still running",
                    "Use the notification-area icon to restore or quit AutoTuner.",
                    QSystemTrayIcon.MessageIcon.Information,
                    4000,
                )
            self._tray_hint_shown = True
        return True

    def _start_in_notification_area(self) -> bool:
        """Keep a login-started window hidden behind the tray icon."""
        if not self._ensure_tray_icon():
            return False
        # restoreGeometry() already applied the saved maximized state.
        self._tray_restore_maximized = self.isMaximized()
        self._log(
            "Started from the login entry; AutoTuner is waiting in the "
            "notification area."
        )
        return True

    def _finish_native_window_setup(self) -> None:
        """Apply HWND icons and the system menu once a native window exists."""
        if self._native_window_setup_done:
            return
        self._native_window_setup_done = True
        _set_windows_native_window_icon(
            self, _bundled_resource("assets", "AutoTuner.ico")
        )
        self._install_windows_system_menu()

    def showEvent(self, a0: QShowEvent | None) -> None:  # noqa: N802
        super().showEvent(a0)
        # A tray-only login start skipped the post-show() native setup in
        # main(); run it after the first real show has created the HWND.
        if not self._native_window_setup_done:
            QTimer.singleShot(0, self._finish_native_window_setup)

    def _restore_from_tray(self) -> None:
        """Restore and focus the main window from the notification area."""
        if self._tray_restore_maximized:
            self.showMaximized()
        else:
            self.showNormal()
        self.raise_()
        self.activateWindow()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._restore_from_tray()

    def _destroy_tray_icon(self) -> None:
        """Remove the native tray icon and release its menu."""
        if self._tray_icon is not None:
            self._tray_icon.hide()
            self._tray_icon.deleteLater()
        self._tray_icon = None
        self._tray_menu = None

    def _install_windows_system_menu(self) -> None:
        """Add Info and Settings above Close in Windows' title-bar system menu."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            user32.GetSystemMenu.argtypes = [wintypes.HWND, wintypes.BOOL]
            user32.GetSystemMenu.restype = wintypes.HMENU
            user32.GetMenuItemCount.argtypes = [wintypes.HMENU]
            user32.GetMenuItemCount.restype = ctypes.c_int
            user32.InsertMenuW.argtypes = [
                wintypes.HMENU,
                wintypes.UINT,
                wintypes.UINT,
                ctypes.c_size_t,
                wintypes.LPCWSTR,
            ]
            user32.InsertMenuW.restype = wintypes.BOOL
            user32.DrawMenuBar.argtypes = [wintypes.HWND]
            user32.DrawMenuBar.restype = wintypes.BOOL

            hwnd = wintypes.HWND(int(self.winId()))
            menu = user32.GetSystemMenu(hwnd, False)
            if not menu:
                return
            count = user32.GetMenuItemCount(menu)
            # Standard order ends in: Maximize, separator, Close. Insert both
            # entries just before that separator, matching the symbol menu.
            position = max(0, count - 2)
            mf_byposition = 0x00000400
            mf_string = 0x00000000
            user32.InsertMenuW(
                menu,
                position,
                mf_byposition | mf_string,
                self._WIN_SETTINGS_COMMAND_ID,
                "Settings…",
            )
            user32.InsertMenuW(
                menu,
                position,
                mf_byposition | mf_string,
                self._WIN_ABOUT_COMMAND_ID,
                "About AutoTuner",
            )
            user32.DrawMenuBar(hwnd)
        except (AttributeError, OSError, TypeError, ValueError):
            # The normal toolbar Settings button remains available.
            pass

    def nativeEvent(self, event_type, message):  # noqa: N802
        """Handle the custom Settings command in the Windows system menu."""
        if sys.platform == "win32":
            try:
                from ctypes import wintypes

                msg = wintypes.MSG.from_address(int(message))
                if msg.message == 0x0112:  # WM_SYSCOMMAND
                    command = int(msg.wParam)
                    if command == self._WIN_SETTINGS_COMMAND_ID:
                        QTimer.singleShot(0, self._open_application_settings)
                        return True, 0
                    if command == self._WIN_ABOUT_COMMAND_ID:
                        QTimer.singleShot(0, self._show_about)
                        return True, 0
            except (AttributeError, TypeError, ValueError):
                pass
        # Calling QWidget.nativeEvent() after inspecting the MSG crashes in
        # PyQt 6.11 on Windows; False delegates normal processing to Qt.
        return False, 0

    # ------------------------------------------------------------------
    # Window geometry persistence
    # ------------------------------------------------------------------
    def _restore_window_geometry(self) -> None:
        """Re-apply the last QMainWindow geometry+state if persisted.

        Qt's saveGeometry/saveState produce opaque QByteArrays. We
        store them as base64 strings in autotuner_settings.json. If
        decoding or restoring fails for any reason (corrupt JSON,
        Qt version mismatch, screen layout no longer valid) we just
        keep the hard-coded default — no crash, no warning.
        """
        b64_geom = app_settings.get_window_geometry()
        if b64_geom:
            try:
                raw = base64.b64decode(b64_geom)
                self.restoreGeometry(QByteArray(raw))
            except (ValueError, TypeError, OSError):
                pass
        b64_state = app_settings.get_window_state()
        if b64_state:
            try:
                raw = base64.b64decode(b64_state)
                self.restoreState(QByteArray(raw))
            except (ValueError, TypeError, OSError):
                pass

    def _persist_window_geometry(self) -> None:
        """Snapshot the current window layout into settings JSON.

        Called from closeEvent. Errors here are non-fatal — losing
        the persisted layout is annoying but not a reason to refuse
        to quit.
        """
        try:
            geom_bytes = self.saveGeometry().data() or b""
            app_settings.set_window_geometry(
                base64.b64encode(geom_bytes).decode("ascii")
            )
            state_bytes = self.saveState().data() or b""
            app_settings.set_window_state(base64.b64encode(state_bytes).decode("ascii"))
        except Exception as exc:  # pragma: no cover - defensive
            self._log(f"[Warning] Could not save window layout: {exc}")
        # Inner pane arrangement — saved separately because QMainWindow
        # saveState() does not round-trip plain central-widget splitters.
        self._persist_splitter_states()

    def _persist_splitter_states(self) -> None:
        """Save each named QSplitter's handle positions.

        Stored per object name so the inner layout (model-list vs config
        width, and the log-panel height) is restored independently of the
        outer window size.
        """
        for sp in getattr(self, "_splitters", []):
            try:
                name = sp.objectName()
                if not name:
                    continue
                raw = sp.saveState().data() or b""
                app_settings.set_splitter_state(
                    name, base64.b64encode(raw).decode("ascii")
                )
            except Exception:  # pragma: no cover - defensive
                continue

    def _restore_splitter_states(self) -> None:
        """Re-apply persisted handle positions to each named QSplitter.

        Falls back silently to the hard-coded setSizes() defaults when no
        blob exists or restoreState() rejects it (e.g. a pane count change
        between versions).
        """
        for sp in getattr(self, "_splitters", []):
            try:
                name = sp.objectName()
                if not name:
                    continue
                b64 = app_settings.get_splitter_state(name)
                if not b64:
                    continue
                raw = base64.b64decode(b64)
                sp.restoreState(QByteArray(raw))
            except (ValueError, TypeError, OSError):
                continue

    # ------------------------------------------------------------------
    def _persist_base_port(self) -> None:
        """Save the current Base port field to settings (on edit / launch).

        Invalid input is ignored — the field keeps its text but nothing is
        stored, so the previous valid value is restored on the next restart.
        Also updates ``self._base_port`` so an immediate launch uses the
        just-typed value even before a relaunch.
        """
        try:
            port = int(self._port_edit.text().strip())
        except ValueError:
            return
        self._base_port = port
        app_settings.set_base_port(port)

    def _apply_mono_font(self, w: QTextEdit) -> None:
        w.setFont(self._theme_manager.mono_font(self._font_size))

    def _change_font(self, delta: int) -> None:
        """A+/A- handler — scale the WHOLE UI, not just two text panels.

        Until v3.1 the font buttons only resized self._config_preview
        and self._log_panel, which left the toolbar / model list /
        Expert panel labels stuck at whatever Qt's default was. Going
        through QApplication.setFont scales every widget that hasn't
        been explicitly assigned its own font — including future widgets
        added after this call — and we re-apply the monospace font to
        the two text panels afterwards so they keep their Consolas /
        monospace styling at the new size.
        """
        new_size = max(7, min(22, self._font_size + delta))
        if new_size == self._font_size:
            return
        self._font_size = new_size

        app = QApplication.instance()
        if app is not None:
            # QApplication.instance() returns QCoreApplication | None per stubs,
            # but at runtime it IS a QApplication which has font()/setFont().
            qapp = cast("QApplication", app)
            f = qapp.font()
            f.setPointSize(self._font_size)
            qapp.setFont(f)
        # The two monospace text panels need an explicit refresh: they
        # have their own QFont (Consolas / Monospace style hint), which
        # overrides the app-wide font, so QApplication.setFont alone
        # would skip them.
        for w in (self._config_preview, self._log_panel):
            wf = w.font()
            wf.setPointSize(self._font_size)
            w.setFont(wf)

        try:
            app_settings.set_font_size(self._font_size)
        except Exception as exc:  # pragma: no cover - defensive
            self._log(f"[Warning] Could not save font size: {exc}")

    # ------------------------------------------------------------------
    # Fork-container helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _llama_binary_subpaths() -> Tuple[str, ...]:
        # Mirrors auto_tuner._SERVER_SUBPATHS — native binaries only on
        # Linux/macOS so a shared Windows build tree cannot be auto-launched.
        bases = (
            "build/bin/Release/",
            "build/bin/Debug/",
            "build/bin/",
            "build/",
            "",
        )
        suffixes = (".exe",) if os.name == "nt" else ("",)
        out: List[str] = []
        for base in bases:
            for suffix in suffixes:
                sub = f"{base}llama-server{suffix}"
                if sub not in out:
                    out.append(sub)
        return tuple(out)

    @staticmethod
    def _is_runnable_binary(path: Path) -> bool:
        try:
            if not path.is_file():
                return False
        except OSError:
            return False
        if os.name == "nt":
            # Shared dual-boot build folders can hold a Linux ELF
            # "llama-server" next to "llama-server.exe" — only accept
            # Windows-executable suffixes (mirrors auto_tuner._is_runnable_binary).
            return path.suffix.lower() in (".exe", ".bat", ".cmd", ".com")
        if path.suffix.lower() == ".exe":
            return False
        return os.access(path, os.X_OK)

    @classmethod
    def _is_llama_build_dir(cls, path: Path) -> bool:
        return path.is_dir() and any(
            cls._is_runnable_binary(path / sub) for sub in cls._llama_binary_subpaths()
        )

    @staticmethod
    def _looks_like_llama_dir_name(name: str) -> bool:
        return bool(
            re.search(
                r"(?:(?:^|[-_.])llama(?:[-_.]|$)|llama\.cpp)",
                name,
                re.IGNORECASE,
            )
        )

    @classmethod
    def _expand_fork_container(cls, path: Path) -> List[Tuple[str, Path]]:
        """List all llama.cpp build directories inside `path`.

        Returns a list of (display_name, fork_path) pairs for every
        immediate child whose name contains "llama.cpp" and which has
        a built llama-server binary. Empty list when `path` is not a
        container (e.g. it IS a single build folder).

        The subpath list is intentionally kept in sync with
        ``auto_tuner._SERVER_SUBPATHS`` — both cmake build layouts
        (``build/bin/[Release/]llama-server[.exe]``) and prebuilt
        binary drops (``llama-server[.exe]`` at the folder root) are
        recognised.
        """
        result: List[Tuple[str, Path]] = []
        try:
            for child in sorted(path.iterdir(), key=lambda c: c.name.lower()):
                if not child.is_dir():
                    continue
                if not cls._looks_like_llama_dir_name(child.name):
                    continue
                has_binary = cls._is_llama_build_dir(child)
                if has_binary:
                    result.append((child.name, child))
                else:
                    # Debug aid: surface WHY a matching dir was skipped.
                    try:
                        from auto_tuner import debug_cat

                        debug_cat(
                            "llama_cpp",
                            f"fork-skip (GUI container): {child.name} matched "
                            "the name pattern but has no llama-server binary",
                        )
                    except Exception:
                        pass
        except (OSError, PermissionError):
            pass
        return result

    def _active_llama_roots(self) -> List[Path]:
        roots: List[Path] = []
        seen: set[str] = set()
        for path, enabled in self._fork_roots:
            if not enabled:
                continue
            try:
                rp = Path(path).expanduser().resolve(strict=False)
            except (OSError, RuntimeError):
                continue
            key = os.path.normcase(str(rp))
            if key in seen or not rp.is_dir():
                continue
            seen.add(key)
            roots.append(rp)
        return roots

    def _scan_llama_roots(self, roots: List[Path]) -> List[Tuple[str, Path]]:
        forks: List[Tuple[str, Path]] = []
        seen: set[str] = set()

        def add(name: str, path: Path) -> None:
            try:
                rp = path.resolve(strict=False)
            except (OSError, RuntimeError):
                return
            key = os.path.normcase(str(rp))
            if key in seen:
                return
            seen.add(key)
            forks.append((name, rp))

        for root in roots:
            if self._looks_like_llama_dir_name(root.name) and self._is_llama_build_dir(
                root
            ):
                add(root.name, root)
            for name, fork_path in self._expand_fork_container(root):
                add(name, fork_path)

        from auto_tuner import _fork_name_sort_key

        forks.sort(
            key=lambda item: (
                item[0].lower() != "llama.cpp",
                *_fork_name_sort_key(item[0]),
            )
        )
        return forks

    def _populate_fork_combo(
        self,
        forks: List[Tuple[str, Path]],
        preferred: Optional[Path] = None,
    ) -> None:
        self._fork_combo.blockSignals(True)
        self._fork_combo.clear()
        if not forks:
            self._fork_combo.addItem("not found", userData=None)
            self._fork_path = None
            self._refresh_fork_combo_width()
            self._fork_combo.blockSignals(False)
            return

        from auto_tuner import _default_fork_index

        for name, path in forks:
            self._fork_combo.addItem(name, userData=path)
        # A saved build that no longer exists (replaced by a newer build)
        # falls back to the newest build of its family, not to the first
        # listed family (the Ternary/Bonsai '2b_' fork sorts first).
        selected_idx = _default_fork_index(forks, preferred)
        self._fork_combo.setCurrentIndex(selected_idx)
        self._fork_path = forks[selected_idx][1]
        self._fork_combo.blockSignals(False)
        self._apply_fork(selected_idx)
        self._refresh_control_runtimes()

    def _active_llama_binary(self) -> Optional[str]:
        """Resolve the selected fork's server for backend-aware detection."""
        try:
            if self._fork_path is not None:
                from auto_tuner import _server_binary_in_fork

                direct = _server_binary_in_fork(self._fork_path)
                if direct:
                    return direct
            _, resolve_server, _ = _get_fork_tools()
            binary = resolve_server("llama-server")
            return binary or None
        except Exception:
            return None

    def _performance_runtime_options(self) -> List[_PerformanceRuntimeOption]:
        """Return every discovered runnable build, with the toolbar build first."""
        try:
            from auto_tuner import _fork_backend, _server_binary_in_fork
        except Exception:
            return []

        candidates: List[Tuple[str, Path]] = []
        for index in range(self._fork_combo.count()):
            path = self._fork_combo.itemData(index)
            if path is not None:
                candidates.append((self._fork_combo.itemText(index), Path(path)))
        candidates.extend(self._forks)
        seen: set[str] = set()
        options: List[_PerformanceRuntimeOption] = []
        active_path = self._fork_path
        try:
            active_resolved = (
                active_path.resolve(strict=False) if active_path is not None else None
            )
        except (OSError, RuntimeError):
            active_resolved = active_path
        for name, root in candidates:
            binary = _server_binary_in_fork(root)
            if not binary:
                continue
            try:
                binary_key = os.path.normcase(str(Path(binary).resolve(strict=False)))
                root_resolved = root.resolve(strict=False)
            except (OSError, RuntimeError):
                binary_key = os.path.normcase(str(binary))
                root_resolved = root
            if binary_key in seen:
                continue
            seen.add(binary_key)
            backend_hint = _fork_backend(name) or _fork_backend(root.name) or ""
            if not backend_hint:
                identity = f"_{name}_{root.name}_".lower()
                for candidate in ("cpu", "cuda", "metal", "sycl"):
                    if f"_{candidate}_" in identity or f"-{candidate}-" in identity:
                        backend_hint = candidate
                        break
            is_active = bool(
                active_resolved is not None and root_resolved == active_resolved
            )
            options.append(
                _PerformanceRuntimeOption(
                    display_name=name or root.name,
                    binary=str(binary),
                    root=root_resolved,
                    backend_hint=app_settings.normalise_performance_backend(
                        backend_hint
                    ),
                    active=is_active,
                )
            )

        if not options:
            active_binary = self._active_llama_binary()
            if active_binary:
                options.append(
                    _PerformanceRuntimeOption(
                        display_name=Path(active_binary).parent.name or "active build",
                        binary=active_binary,
                        root=active_path,
                        active=True,
                    )
                )
        backend_rank = {
            backend: index for index, backend in enumerate(_RUNTIME_BACKEND_ORDER)
        }
        options.sort(
            key=lambda option: (
                not option.active,
                backend_rank.get(option.backend_hint, 99),
                option.display_name.casefold(),
            )
        )
        return options

    def _start_hardware_detection(self) -> None:
        # Hardware detection (spawns PowerShell on Windows) → background thread
        # so it never blocks the UI and never flashes a window.
        # Use signal/slot pattern instead of QTimer.singleShot from bg thread
        # to avoid potential PyQt6 deadlocks when COM is involved.
        self._log("Detecting system hardware…")
        self._hw_detect_worker = _HwDetectWorker(
            timeout=30.0,
            llama_binary=self._active_llama_binary(),
        )
        self._hw_detect_thread = QThread(self)
        self._hw_detect_worker.moveToThread(self._hw_detect_thread)
        self._hw_detect_thread.started.connect(self._hw_detect_worker.run)
        self._hw_detect_worker.finished.connect(self._hw_detect_done)
        self._hw_detect_worker.finished.connect(self._hw_detect_thread.quit)
        self._hw_detect_thread.finished.connect(self._hw_detect_thread.deleteLater)
        self._hw_detect_thread.start()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def _startup_load(self) -> None:
        # Load profiles and discover forks first (fast, no subprocess) —
        # then kick off hardware detection in a daemon thread so the window
        # is already fully visible before any PowerShell calls happen.
        self._profiles = load_profiles(self.settings_path)
        n = len(self._profiles)
        self._log(
            f"Loaded {n} profile(s) from {self.settings_path}"
            if n
            else f"[Warning] No profiles found in {self.settings_path}"
        )

        # Explicitly configured llama paths are authoritative and much cheaper
        # to inspect than the broad fallback discovery across drive ancestors.
        # Do not perform both scans only to discard the first result.
        if not self._fork_roots:
            try:
                discover, _, _ = _get_fork_tools()
                self._forks = discover()
            except Exception as exc:
                self._log(f"[Warning] Fork discovery failed: {exc}")
                self._forks = []

        if self._fork_roots:
            active_roots = self._active_llama_roots()
            forks = self._scan_llama_roots(active_roots)
            self._forks = forks
            preferred = app_settings.get_fork_path()
            self._populate_fork_combo(forks, preferred)
            if forks:
                self._log(
                    f"[Fork] Loaded {len(forks)} build(s) from "
                    f"{len(active_roots)} active llama path(s)."
                )
            else:
                self._log("[Warning] No llama.cpp builds found in active llama paths.")
            # Model discovery and hardware probing are independent. Starting
            # both now removes the former 30-second serial wait on slow WMI/
            # driver probes while keeping all widget updates signal-driven.
            self._start_scan()
            self._start_hardware_detection()
            return

        # ── Resolve persisted fork state ────────────────────────────
        # The container path (the parent folder the user picked via
        # "📂 Fork") is the authoritative restore target — it lets us
        # show ALL sibling builds again. The active fork path is just
        # the last selection within that container, used to restore
        # the combo's current index.
        persisted_container = app_settings.get_fork_container_path()
        persisted_active = app_settings.get_fork_path()
        env_fork = os.environ.get("LLAMA_CPP_DIR", "")

        # If no container was ever explicitly stored but a manual fork
        # path is, peek at its parent: if that parent itself contains
        # multiple llama.cpp builds, treat it as a container — this
        # migrates older settings files where only `fork_path` existed.
        if persisted_container is None and persisted_active is not None:
            cand_parent = persisted_active.parent
            if cand_parent and cand_parent.is_dir():
                if self._expand_fork_container(cand_parent):
                    persisted_container = cand_parent
                    self._log(
                        f"[Fork] Migrating: treating {cand_parent} "
                        "as fork container (siblings found)."
                    )

        manual_path: Optional[Path] = None
        manual_source = ""  # "container" | "settings" | "env" | ""
        if persisted_container is not None:
            manual_path = persisted_container.resolve()
            manual_source = "container"
            self._log(f"[Fork] Loaded persisted container: {manual_path}")
        elif persisted_active is not None and persisted_active.is_dir():
            manual_path = persisted_active.resolve()
            manual_source = "settings"
            self._log(f"[Fork] Loaded persisted path: {manual_path}")
        elif env_fork and Path(env_fork).is_dir():
            manual_path = Path(env_fork).resolve()
            manual_source = "env"

        # Detect whether `manual_path` itself is a container with several
        # llama.cpp builds inside (e.g. C:\LAB\ai-local).
        container_children: List[Tuple[str, Path]] = []
        if manual_path is not None:
            container_children = self._expand_fork_container(manual_path)
        env_contains_forks = bool(container_children)

        self._fork_combo.blockSignals(True)
        self._fork_combo.clear()

        # If the persisted manual path matches one of the auto-discovered
        # forks, show it under its real name instead of as "📁 custom".
        # Avoids the cosmetic regression where every restart looked like
        # the path had been forgotten when it was actually loaded fine.
        matched_idx = -1
        if manual_path and self._forks and not env_contains_forks:
            for i, (_, p) in enumerate(self._forks):
                try:
                    if p.resolve() == manual_path:
                        matched_idx = i
                        break
                except OSError:
                    continue

        if matched_idx >= 0 and manual_path is not None:
            # Persisted path IS one of the discovered forks — restore by name.
            for name, path in self._forks:
                self._fork_combo.addItem(name, userData=path)
            self._fork_combo.setCurrentIndex(matched_idx)
            self._fork_path = self._forks[matched_idx][1]
            src_label = (
                "persisted settings" if manual_source == "settings" else "LLAMA_CPP_DIR"
            )
            self._log(
                f"[Fork] Restored from {src_label}: "
                f"{self._forks[matched_idx][0]}  →  {manual_path}"
            )
            self._apply_fork(matched_idx)
        elif env_contains_forks and manual_path is not None:
            # Container with multiple llama.cpp builds — this is the
            # "remember the parent folder" case. Show every sibling.
            self._fork_container = manual_path
            self._log(
                f"[Fork] Container '{manual_path.name}' "
                f"contains {len(container_children)} fork(s):"
            )
            for name, fork_path in container_children:
                self._log(f"  - {name} → {fork_path}")
                self._fork_combo.addItem(name, userData=fork_path)
            os.environ["LLAMA_CPP_DIR"] = str(manual_path)
            # Restore previously active selection inside the container,
            # if persisted_active points at one of these children; a
            # replaced build falls back to the newest of its family.
            from auto_tuner import _default_fork_index

            initial_idx = _default_fork_index(container_children, persisted_active)
            self._fork_combo.setCurrentIndex(initial_idx)
            self._fork_path = container_children[initial_idx][1]
            self._apply_fork(initial_idx)
        elif manual_path:
            # Truly custom path outside the auto-discover scope and not
            # a container — single-build manual fork. Label it by its
            # directory name so the user can recognise their selection.
            label = f"📁 {manual_path.name}"
            self._fork_combo.addItem(label, userData=manual_path)
            self._fork_path = manual_path
            self._fork_combo.setCurrentIndex(0)
            src_label = (
                "persisted settings" if manual_source == "settings" else "LLAMA_CPP_DIR"
            )
            self._log(f"[Fork] Using manual path from {src_label}: {manual_path}")
        elif self._forks:
            # No manual choice — auto-discovered forks; default to the newest
            # mainline build rather than the first listed family.
            from auto_tuner import _default_fork_index

            for name, path in self._forks:
                self._fork_combo.addItem(name, userData=path)
            default_idx = _default_fork_index(self._forks)
            self._fork_combo.setCurrentIndex(default_idx)
            self._fork_path = self._forks[default_idx][1]
            self._log(
                f"Found {len(self._forks)} fork(s). "
                f"Using: {self._forks[default_idx][0]}"
            )
            self._apply_fork(default_idx)
        else:
            self._fork_combo.addItem("not found", userData=None)
            self._fork_path = None
            self._log("[Warning] No llama.cpp forks found. Set LLAMA_CPP_DIR.")
        self._fork_combo.blockSignals(False)
        self._refresh_fork_combo_width()

        self._start_scan()
        self._start_hardware_detection()

    # ------------------------------------------------------------------
    # Fork selection
    # ------------------------------------------------------------------
    def _refresh_fork_combo_width(self) -> None:
        """Keep the selected fork name fully readable in the toolbar combo."""
        text = self._fork_combo.currentText().strip()
        text_width = (
            self._fork_combo.fontMetrics().horizontalAdvance(text) if text else 0
        )
        width = max(
            self._FORK_COMBO_MIN_WIDTH,
            text_width + self._FORK_COMBO_TEXT_PADDING,
        )
        self._fork_combo.setMinimumWidth(width)
        self._fork_combo.updateGeometry()

        technical = self._FORK_TOOLTIP_TECHNICAL
        if text:
            technical += f"\nActive build: {text}"
        path = self._fork_combo.currentData()
        if path is not None:
            technical += f"\nResolved path: {path}"
        self._fork_combo.setToolTip(
            _setting_tooltip(self._FORK_TOOLTIP_SUMMARY, technical)
        )

    def _refresh_control_runtimes(self) -> None:
        """Republish the selectable builds after the toolbar list changed."""
        if getattr(self, "_control_api", None) is not None:
            self._refresh_control_api_catalogue()

    def _on_fork_changed(self, index: int) -> None:
        self._fork_manual_override = True
        self._apply_fork(index)
        self._refresh_control_runtimes()
        # Persist the active build choice without touching the
        # container — switching combos within a container should NOT
        # collapse the container to a single fork.
        path: Optional[Path] = self._fork_combo.itemData(index)
        if path is not None:
            try:
                app_settings.set_fork_path(path)
            except Exception as exc:
                self._log(f"[Warning] Could not save fork path: {exc}")

    def _apply_fork(self, index: int) -> None:
        self._refresh_fork_combo_width()
        path: Optional[Path] = self._fork_combo.itemData(index)
        if path is not None:
            previous = self._fork_path
            self._fork_path = path
            os.environ["LLAMA_CPP_DIR"] = str(path)
            self._log(f"[Fork] → {path.name}")
            try:
                changed = previous is None or previous.resolve(
                    strict=False
                ) != path.resolve(strict=False)
            except (OSError, RuntimeError):
                changed = previous != path
            if changed and getattr(self, "_system", None) is not None:
                # Backend identity and measured-profile validity must follow the
                # exact toolbar build rather than the previously selected path.
                QTimer.singleShot(0, self._sysinfo_async)

    # ------------------------------------------------------------------
    # Performance target selection
    # ------------------------------------------------------------------
    def _on_perf_changed(self, index: int) -> None:
        """User picked a new performance target — persist + refresh view.

        Only the *config text* is recomputed; launch-option selections must
        NOT be touched here. Performance target affects
        VRAM placement and KV-cache decisions, never feature selection.
        """
        name = self._perf_combo.itemText(index).strip()
        try:
            app_settings.set_performance_target(name)
            current_entry = getattr(self, "_current_entry", None)
            if current_entry is not None:
                app_settings.set_model_performance_target(
                    current_entry.path, name, self._current_performance_backend()
                )
        except Exception as exc:
            self._log(f"[Warning] Could not save performance target: {exc}")
        self._log(f"[Perf] → {name}")
        self._refresh_setting_profile_selector()
        # Recompute the displayed config in-place, leaving every launch option
        # alone — `_update_config_text` reads the current state into the preview.
        entry = getattr(self, "_current_entry", None)
        if entry is not None and self._system is not None:
            try:
                profile = match_profile(
                    entry.name, self._profiles, getattr(entry, "architecture", "")
                )
                self._update_config_text(entry, profile)
            except Exception as exc:
                self._log(f"[Warning] Config refresh failed: {exc}")

    # ------------------------------------------------------------------
    # Mode (chat / coding) selection
    # ------------------------------------------------------------------
    def _current_mode(self) -> str:
        """Return the active sampling mode ("chat" / "coding")."""
        if not hasattr(self, "_mode_combo"):
            return "chat"
        m = self._mode_combo.currentText().strip().lower()
        return m if m in ("chat", "coding") else "chat"

    def _on_mode_changed(self, index: int) -> None:
        """User flipped chat ↔ coding — persist + refresh preview only.

        This does NOT touch checkboxes; only the config text and the
        persisted setting are updated.
        """
        name = self._mode_combo.itemText(index).strip()
        try:
            app_settings.set_mode(name)
        except Exception as exc:
            self._log(f"[Warning] Could not save mode: {exc}")
        self._log(f"[Mode] → {name}")
        entry = getattr(self, "_current_entry", None)
        if entry is not None and self._system is not None:
            try:
                profile = match_profile(
                    entry.name, self._profiles, getattr(entry, "architecture", "")
                )
                self._update_config_text(entry, profile)
            except Exception as exc:
                self._log(f"[Warning] Config refresh failed: {exc}")

    # ------------------------------------------------------------------
    # GPU pin (forced_gpu) selection
    # ------------------------------------------------------------------
    @staticmethod
    def _gpu_short_label(name: str) -> str:
        """Derive a short, stable pin token from a full driver name.

        The token is what we persist via app_settings.set_forced_gpu() and
        what compute_config(force_gpu=...) matches case-insensitively as a
        substring of the card name. We want something distinctive yet stable
        across driver-string changes, mirroring the CLI convention
        (`--gpu 9070`, `--gpu R9700`).

        Strategy: prefer a model-number-like token (contains a digit, e.g.
        "R9700", "9070") from the tail of the name; otherwise fall back to
        the last word, and finally to the whole (stripped) name.
        """
        words = name.split()
        for w in reversed(words):
            cleaned = w.strip("()[]")
            if cleaned and any(ch.isdigit() for ch in cleaned):
                return cleaned
        if words:
            return words[-1]
        return name.strip()

    def _populate_gpu_combo(self, s: SystemInfo) -> None:
        """(Re)fill the GPU pin dropdown from detected cards.

        Called whenever fresh hardware info arrives. Preserves the user's
        current selection by token, falling back to the persisted forced_gpu
        and finally to "Auto". Signals are blocked so repopulation never
        triggers a spurious persist/refresh.
        """
        combo = getattr(self, "_gpu_combo", None)
        if combo is None:
            return

        # Remember what is selected right now (token) so we can restore it.
        prev_token = combo.currentData()
        if prev_token is None:
            prev_token = app_settings.get_forced_gpu()

        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Auto", None)
        seen: set[str] = set()
        for g in s.gpus:
            token = self._gpu_short_label(g.name)
            # Guard against two cards collapsing to the same token.
            if token.lower() in seen:
                token = g.name
            seen.add(token.lower())
            combo.addItem(f"{token}  ({g.total_vram_gb:.0f} GB)", token)

        # Restore selection: match persisted/previous token as a substring,
        # case-insensitively, exactly like compute_config does.
        target_idx = 0  # Auto
        if prev_token:
            needle = prev_token.strip().lower()
            for i in range(1, combo.count()):
                data = combo.itemData(i)
                if isinstance(data, str) and needle in data.lower():
                    target_idx = i
                    break
        combo.setCurrentIndex(target_idx)
        combo.blockSignals(False)

    def _on_gpu_changed(self, index: int) -> None:
        """User picked a GPU pin — persist + refresh the preview.

        Like the perf/mode handlers, this only recomputes the *config text*;
        launch-option dropdowns and checkboxes are never touched. The
        persisted forced_gpu is read by both launch paths via
        app_settings.get_forced_gpu().
        """
        token = self._gpu_combo.itemData(index)  # None for "Auto"
        try:
            app_settings.set_forced_gpu(token)
        except Exception as exc:
            self._log(f"[Warning] Could not save GPU pin: {exc}")
        self._log(f"[GPU] pin → {token or 'Auto'}")
        entry = getattr(self, "_current_entry", None)
        if entry is not None and self._system is not None:
            try:
                profile = match_profile(
                    entry.name, self._profiles, getattr(entry, "architecture", "")
                )
                self._update_config_text(entry, profile)
            except Exception as exc:
                self._log(f"[Warning] Config refresh failed: {exc}")

    def _current_performance_target_name(self) -> str:
        """Return the active one of the four persisted performance targets."""
        if hasattr(self, "_perf_combo"):
            name = self._perf_combo.currentText().strip().lower()
            if name in PERFORMANCE_TARGETS:
                return name
        return DEFAULT_TARGET_NAME

    def _current_performance_backend(self) -> str:
        runtime_binary = self._active_llama_binary()
        if not runtime_binary or self._system is None:
            return ""
        build_name = self._fork_path.name if self._fork_path is not None else ""
        return _benchmark_backend_key(runtime_binary, self._system, build_name)

    def _current_drafter_profile_key(self) -> str:
        combo = getattr(self, "_cb_draft", None)
        if combo is None or combo.currentIndex() < 0:
            return app_settings.NO_DRAFTER_PROFILE_KEY
        return _drafter_profile_key(combo.currentData())

    def _current_setting_profile_slot(self) -> str:
        combo = getattr(self, "_setting_profile_combo", None)
        if combo is None or combo.currentIndex() < 0:
            return app_settings.PROFILE_AUTO
        slot = str(combo.currentData() or app_settings.PROFILE_AUTO)
        return slot if slot in app_settings.PROFILE_SLOTS else app_settings.PROFILE_AUTO

    def _refresh_setting_profile_selector(self) -> None:
        combo = getattr(self, "_setting_profile_combo", None)
        if combo is None:
            return
        entry = self._current_entry
        if entry is None:
            self._setting_profile_row.setEnabled(False)
            return
        target = self._current_performance_target_name()
        drafter_key = self._current_drafter_profile_key()
        bank = app_settings.get_setting_profile_bank(entry.name, entry.path, target)
        names = bank.get("names") if isinstance(bank.get("names"), dict) else {}
        backend = self._current_performance_backend()
        selected = app_settings.get_selected_setting_profile(
            entry.name, entry.path, target, drafter_key, backend
        )

        self._setting_profile_refreshing = True
        combo.blockSignals(True)
        try:
            model = cast(QStandardItemModel, combo.model())
            for index in range(combo.count()):
                slot = str(combo.itemData(index) or app_settings.PROFILE_AUTO)
                if slot in app_settings.CUSTOM_PROFILE_SLOTS:
                    combo.setItemText(
                        index, str(names.get(slot, combo.itemText(index)))
                    )
                item = model.item(index)
                if item is not None:
                    available = not app_settings.is_perform_profile_slot(
                        slot
                    ) or app_settings.has_setting_profile_snapshot(
                        entry.name,
                        entry.path,
                        target,
                        slot,
                        drafter_key,
                    )
                    item.setEnabled(available)
            selected_index = combo.findData(selected)
            combo.setCurrentIndex(max(0, selected_index))
        finally:
            combo.blockSignals(False)
            self._setting_profile_refreshing = False
        current_slot = self._current_setting_profile_slot()
        self._btn_rename_setting_profile.setEnabled(
            current_slot in app_settings.CUSTOM_PROFILE_SLOTS
        )
        self._setting_profile_row.setEnabled(True)

    def _on_setting_profile_changed(self, _index: int) -> None:
        if self._setting_profile_refreshing or self._current_entry is None:
            return
        if self._config_stack.currentIndex() == 1:
            self._expert_panel.flush_pending_save()
        entry = self._current_entry
        target = self._current_performance_target_name()
        drafter_key = self._current_drafter_profile_key()
        slot = self._current_setting_profile_slot()
        if not app_settings.set_selected_setting_profile(
            entry.name,
            entry.path,
            target,
            slot,
            drafter_key,
            self._current_performance_backend(),
        ):
            self._refresh_setting_profile_selector()
            return
        self._btn_rename_setting_profile.setEnabled(
            slot in app_settings.CUSTOM_PROFILE_SLOTS
        )
        self._log(
            f"[Profile] {entry.name} [{target}] {drafter_key} → "
            f"{self._setting_profile_combo.currentText()}."
        )
        profile = match_profile(
            entry.name, self._profiles, getattr(entry, "architecture", "")
        )
        if self._config_stack.currentIndex() == 1 and self._system is not None:
            self._load_expert_panel(entry, profile)
        else:
            self._refresh_config_preview()

    def _rename_setting_profile(self) -> None:
        entry = self._current_entry
        slot = self._current_setting_profile_slot()
        if entry is None or slot not in app_settings.CUSTOM_PROFILE_SLOTS:
            return
        current_name = self._setting_profile_combo.currentText()
        name, accepted = QInputDialog.getText(
            self,
            "Rename Custom profile",
            "Profile name:",
            text=current_name,
        )
        if not accepted or not name.strip():
            return
        if not app_settings.rename_custom_setting_profile(
            entry.name,
            entry.path,
            self._current_performance_target_name(),
            slot,
            name,
        ):
            QMessageBox.warning(
                self, "Rename profile", "The Custom profile name could not be saved."
            )
            return
        self._refresh_setting_profile_selector()

    def _resolve_perf_target_for_profile(
        self, profile: ModelProfile, performance_target: Optional[str] = None
    ):
        """Combine an explicit/GUI choice with the profile recommendation."""
        gui_choice = performance_target or self._current_performance_target_name()
        return resolve_performance_target(
            cli_choice=gui_choice,
            profile_choice=getattr(profile, "performance_target", "") or None,
        )

    def _hw_detect_done(self, s: Optional[SystemInfo], err: str = "") -> None:
        """Callback from hardware detection worker thread (via signal/slot)."""
        if s is not None:
            self._system = s
            self._update_sysinfo_labels(s)
            self._log(
                f"Hardware detected ({s.total_ram_gb:.0f}GB RAM, "
                f"{s.total_vram_gb:.0f}GB VRAM, {len(s.gpus)} GPU(s))."
            )
        else:
            self._log(f"[Warning] Hardware detection failed: {err}")
            # Model discovery already runs independently; selection remains
            # visible even if no system snapshot is available.
        pending = self._current_entry
        if pending is not None and self._system is not None:
            self._show_config(pending)

    def _browse_fork_folder(self) -> None:
        """Open the multi-path llama.cpp builds manager."""
        initial = self._fork_roots[:]
        if not initial:
            if self._fork_container is not None:
                initial = [(self._fork_container, True)]
            elif self._fork_path is not None:
                initial = [(self._fork_path, True)]
            else:
                env_fork = os.environ.get("LLAMA_CPP_DIR", "")
                if env_fork:
                    initial = [(Path(env_fork).expanduser(), True)]
        dlg = _PathListDialog(
            self,
            _tr("Manage llama.cpp build paths"),
            initial,
            _tr("Select a llama.cpp build or container folder"),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._fork_roots = dlg.paths()
        try:
            app_settings.set_llama_build_paths(self._fork_roots)
            self._log(
                f"[Fork] Saved {len(self._fork_roots)} llama path(s), "
                f"{len(self._active_llama_roots())} active."
            )
        except Exception as exc:
            self._log(f"[Warning] Could not save llama build paths: {exc}")
        self._reload_llama_builds()

    def _reload_llama_builds(self) -> None:
        active_roots = self._active_llama_roots()
        forks = self._scan_llama_roots(active_roots)
        self._forks = forks
        preferred = self._fork_path or app_settings.get_fork_path()
        self._populate_fork_combo(forks, preferred)
        if forks:
            self._log(
                f"[Fork] Loaded {len(forks)} build(s) from "
                f"{len(active_roots)} active llama path(s)."
            )
            if self._fork_path is not None:
                try:
                    app_settings.set_fork_path(self._fork_path)
                except Exception as exc:
                    self._log(f"[Warning] Could not save fork path: {exc}")
        else:
            self._log("[Warning] No llama.cpp builds found in active llama paths.")

    def _set_manual_fork_path(self, path: Path) -> None:
        r"""Manuellen Fork-Pfad setzen und UI aktualisieren.

        If `path` is a *container* — i.e. its immediate children include
        multiple llama.cpp builds — every sibling is shown in the combo
        and the container itself is persisted via
        ``fork_container_path``. Restarts then re-expand the same set
        of builds instead of dropping the user back to a single child.
        """
        if not path.is_dir():
            QMessageBox.warning(
                self,
                _tr("Invalid folder"),
                f"{_tr('The folder does not exist:')}\n{path}",
            )
            return

        path = path.resolve()
        child_forks = self._expand_fork_container(path)

        self._fork_path = path
        self._log(f"[Fork] Pfad: {path}")

        self._fork_combo.blockSignals(True)
        self._fork_combo.clear()

        if child_forks:
            # Container with multiple builds — persist as container so
            # the next restart still shows every sibling.
            self._fork_container = path
            self._log(f"[Fork] '{path.name}' enthält {len(child_forks)} Fork(s):")
            for name, fork_path in child_forks:
                self._log(f"  - {name} → {fork_path}")
                self._fork_combo.addItem(name, userData=fork_path)
            self._fork_combo.setCurrentIndex(0)
            os.environ["LLAMA_CPP_DIR"] = str(path)
            try:
                app_settings.set_fork_container_path(path)
                # Active selection within the container — the first build.
                app_settings.set_fork_path(child_forks[0][1])
                self._log(f"[Fork] Saved container: {path}")
            except Exception as exc:
                self._log(f"[Warning] Could not save fork container: {exc}")
        else:
            # Single build — clear any previous container so we don't keep
            # advertising one that no longer holds multiple forks.
            self._fork_container = None
            try:
                app_settings.clear_fork_container_path()
            except Exception as exc:
                self._log(f"[Warning] Could not clear fork container: {exc}")
            self._fork_combo.addItem(f"📁 {path.name}", userData=path)
            self._fork_combo.setCurrentIndex(0)
            os.environ["LLAMA_CPP_DIR"] = str(path)
            try:
                app_settings.set_fork_path(path)
                self._log(f"[Fork] Saved as default: {path}")
            except Exception as exc:
                self._log(f"[Warning] Could not save fork path: {exc}")

        self._fork_combo.blockSignals(False)
        self._apply_fork(0)
        self._refresh_control_runtimes()

    # ------------------------------------------------------------------
    # Background model scan
    # ------------------------------------------------------------------
    def _active_model_paths(self) -> List[Path]:
        """Return existing enabled model roots from the multi-folder list."""
        roots: List[Path] = []
        seen: set[str] = set()
        for path, enabled in self.model_paths:
            if not enabled:
                continue
            try:
                rp = Path(path).expanduser().resolve(strict=False)
            except (OSError, RuntimeError):
                continue
            key = os.path.normcase(str(rp))
            if key in seen or not rp.is_dir():
                continue
            seen.add(key)
            roots.append(rp)
        return roots

    def _models_label(self, paths: List[Path]) -> str:
        if not paths:
            return "no active model paths"
        if len(paths) == 1:
            return str(paths[0])
        return f"{len(paths)} model paths"

    def _start_scan(self) -> None:
        try:
            if self._scan_thread is not None and self._scan_thread.isRunning():
                return
        except RuntimeError:
            self._scan_thread = None

        self._btn_refresh.setEnabled(False)
        self._btn_launch.setEnabled(False)
        self._model_list.clear()
        self._model_tree.clear()
        roots = self._active_model_paths()
        self._last_scan_roots = roots
        label = self._models_label(roots)
        self._status.showMessage(f"Scanning {label} …")
        self._log(f"Scanning model folders: {label}")
        for root in roots:
            self._log(f"  - {root}")

        if not roots:
            configured = "\n".join(
                f"  - {p} {'(deaktiviert)' if not en else ''}"
                for p, en in self.model_paths
            )
            msg = (
                "Keine aktiven Model-Pfade verfügbar.\n\n"
                "Aktiviere oder füge Pfade über '📂 Models folder' hinzu."
            )
            if configured:
                msg += "\n\nKonfiguriert:\n" + configured
            self._config_preview.setPlainText(msg)
            self._status.showMessage("No active model folders.")
            self._btn_refresh.setEnabled(True)
            return

        worker = _ScanWorker(roots)
        thread = QThread(self)
        self._scan_worker = worker
        self._scan_thread = thread
        # Bind to locals so static checkers (Pylance) can see these are
        # definitely-not-None for the signal wiring below — the attributes
        # are typed Optional[...] because they're cleared on teardown.
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_scan_done)
        worker.error.connect(self._on_scan_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _on_scan_done(self, entries: List[ModelEntry]) -> None:
        self._all_entries = entries
        self._refresh_control_api_catalogue()
        self._btn_refresh.setEnabled(True)
        if not entries:
            roots = getattr(self, "_last_scan_roots", [])
            root_lines = "\n".join(f"  {p}" for p in roots)
            self._config_preview.setPlainText(
                "No *.gguf files found in active folders:\n" + root_lines
            )
            self._status.showMessage("No models found.")
            self._log("No models found in active folders.")
            return
        self._populate_list(entries)
        self._enable_launch_when_ocr_idle()
        roots = getattr(self, "_last_scan_roots", [])
        stats = getattr(self._scan_worker, "stats", {})
        elapsed = float(stats.get("elapsed_s", 0.0) or 0.0)
        hits = int(stats.get("hits", 0) or 0)
        misses = int(stats.get("misses", 0) or 0)
        workers = int(stats.get("workers", 1) or 1)
        self._status.showMessage(
            f"{len(entries)} model(s) loaded from {len(roots)} folder(s) "
            f"in {elapsed:.1f}s."
        )
        self._log(
            f"Found {len(entries)} model(s) from {len(roots)} active folder(s) "
            f"in {elapsed:.2f}s (metadata cache {hits} hit(s), {misses} miss(es), "
            f"up to {workers} worker(s))."
        )

    def _on_scan_error(self, msg: str) -> None:
        self._btn_refresh.setEnabled(True)
        self._log(f"[Error] Scan failed: {msg}")
        self._status.showMessage(f"Scan error: {msg}")

    # ------------------------------------------------------------------
    # GitHub updater
    # ------------------------------------------------------------------
    def _start_update(self) -> None:
        try:
            if self._update_thread is not None and self._update_thread.isRunning():
                return
        except RuntimeError:
            self._update_thread = None

        if getattr(sys, "frozen", False):
            # Compiled build (PyInstaller onefile): the source-ZIP/git updater
            # is meaningless — there are no .py files to replace. Use the
            # binary-swap worker instead (downloads the OS-correct release
            # asset + a swap shim).
            self._start_binary_update()
            return

        reply = QMessageBox.question(
            self,
            _tr("AutoTuner update"),
            f"{_tr('Check GitHub for source updates and install them?')} (v{VERSION})\n\n"
            + _tr(
                "Git clones use git pull --ff-only on the current branch; downloaded "
                "folders receive the current GitHub source ZIP. This source updater "
                "does not download release assets."
            )
            + "\n\n"
            + _tr(
                "autotuner_settings.json is backed up first and restored afterwards."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._btn_update.setEnabled(False)
        self._status.showMessage(f"Checking GitHub for source updates (v{VERSION}) …")
        self._log(f"[Update] Running source v{VERSION}; checking GitHub branch …")

        worker = _UpdateWorker(Path(__file__).resolve().parent)
        thread = QThread(self)
        self._update_worker = worker
        self._update_thread = thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._log)
        worker.finished.connect(self._on_update_done)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.finished.connect(lambda: self._clear_update_references(thread))
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _start_binary_update(self) -> None:
        """Frozen-build update path: OS-aware release-asset swap."""
        reply = QMessageBox.question(
            self,
            _tr("AutoTuner update"),
            f"{_tr('Check GitHub for a newer AutoTuner version?')}\n\n"
            f"{_tr('Running:')} v{VERSION} ({platform.system()}). "
            + _tr(
                "A newer release downloads the matching binary (.exe / Linux / macOS) "
                "and swaps it in after a restart. Your settings "
                "(autotuner_settings.json) are kept."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._btn_update.setEnabled(False)
        self._status.showMessage("Checking GitHub for a newer release …")
        self._log(
            f"[Update] Running v{VERSION} on {platform.system()}; checking GitHub …"
        )

        worker = _BinaryUpdateWorker()
        thread = QThread(self)
        self._update_worker = worker
        self._update_thread = thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._log)
        worker.finished.connect(self._on_binary_update_done)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.finished.connect(lambda: self._clear_update_references(thread))
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _clear_update_references(self, finished_thread: QThread) -> None:
        """Drop stale update QObject/thread references after a clean finish."""
        if self._update_thread is finished_thread:
            self._update_thread = None
            self._update_worker = None

    def _on_binary_update_done(self, ok: bool, msg: str, needs_restart: bool) -> None:
        self._btn_update.setEnabled(True)
        self._log(f"[Update] {msg}")
        if not ok:
            self._status.showMessage("Update failed.")
            QMessageBox.warning(self, "AutoTuner update failed", msg)
            return
        self._status.showMessage(msg)
        if needs_restart:
            QMessageBox.information(
                self,
                _tr("AutoTuner update"),
                msg
                + "\n\n"
                + _tr(
                    "Click OK: AutoTuner closes now and restarts automatically with "
                    "the new version."
                ),
            )
            # Stop child servers so VRAM is freed before the swap/relaunch.
            try:
                self._stop_all_servers()
            except Exception:
                pass
            QApplication.quit()
        else:
            QMessageBox.information(self, "AutoTuner update", msg)

    def _on_update_done(self, ok: bool, msg: str) -> None:
        self._btn_update.setEnabled(True)
        if ok:
            self._status.showMessage(msg)
            self._log(f"[Update] {msg}")
            QMessageBox.information(self, "AutoTuner update", msg)
        else:
            self._status.showMessage("Update failed.")
            self._log(f"[Update] ERROR: {msg}")
            QMessageBox.warning(self, "AutoTuner update failed", msg)

    @staticmethod
    def _entry_from_model_item(
        item: Optional[QListWidgetItem | QTreeWidgetItem],
    ) -> Optional[ModelEntry]:
        if item is None:
            return None
        if isinstance(item, QTreeWidgetItem):
            return item.data(0, Qt.ItemDataRole.UserRole)
        return item.data(Qt.ItemDataRole.UserRole)

    def _active_model_view(self) -> QListWidget | QTreeWidget:
        return self._model_tree if self._model_view_mode == "tree" else self._model_list

    def _selected_model_path(self) -> Optional[Path]:
        entry = self._entry_from_model_item(self._active_model_view().currentItem())
        if entry is not None:
            return entry.path
        return self._current_entry.path if self._current_entry is not None else None

    def _set_model_view(
        self, mode: str, *, persist: bool = True, repopulate: bool = True
    ) -> None:
        """Switch between flat and folder views without rescanning models."""
        normalized = mode if mode in ("list", "tree") else "list"
        selected_path = self._selected_model_path() if repopulate else None
        self._model_view_mode = normalized
        self._model_view_stack.setCurrentIndex(1 if normalized == "tree" else 0)
        self._btn_list_view.setChecked(normalized == "list")
        self._btn_tree_view.setChecked(normalized == "tree")
        if persist:
            app_settings.set_model_view_mode(normalized)
        if selected_path is not None:
            self._select_model_path(selected_path, self._active_model_view())

    def _remember_tree_expansion(self, item: QTreeWidgetItem, expanded: bool) -> None:
        key = item.data(0, _TREE_PATH_ROLE)
        if not isinstance(key, str) or not key:
            return
        changed = False
        if expanded and key in self._tree_collapsed_paths:
            self._tree_collapsed_paths.remove(key)
            changed = True
        elif not expanded and key not in self._tree_collapsed_paths:
            self._tree_collapsed_paths.add(key)
            changed = True
        if changed:
            app_settings.set_model_tree_collapsed_paths(self._tree_collapsed_paths)

    def _on_tree_expansion_changed(self, item: QTreeWidgetItem, expanded: bool) -> None:
        """Remember expansion and distinguish native arrow clicks from labels."""
        self._remember_tree_expansion(item, expanded)
        if self._tree_manual_toggle:
            return
        self._tree_native_toggle_item = item

        def clear_native_marker() -> None:
            if self._tree_native_toggle_item is item:
                self._tree_native_toggle_item = None

        # itemClicked follows the native branch-indicator toggle in the same
        # event turn. Clear afterward so keyboard expansion never suppresses a
        # later label click.
        QTimer.singleShot(0, clear_native_marker)

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        """Toggle a folder when its row/name is clicked, not only its arrow."""
        if self._entry_from_model_item(item) is not None:
            return
        key = item.data(0, _TREE_PATH_ROLE)
        if not isinstance(key, str) or not key:
            return
        if self._tree_native_toggle_item is item:
            # Qt already toggled because the user clicked the disclosure arrow.
            self._tree_native_toggle_item = None
            return
        self._tree_manual_toggle = True
        try:
            item.setExpanded(not item.isExpanded())
        finally:
            self._tree_manual_toggle = False

    def _add_tree_model_item(
        self, parent: Optional[QTreeWidgetItem], entry: ModelEntry
    ) -> QTreeWidgetItem:
        item = (
            QTreeWidgetItem(parent, [_model_display_text(entry)])
            if parent is not None
            else QTreeWidgetItem(self._model_tree, [_model_display_text(entry)])
        )
        favorite = app_settings.favorite_model_key(entry.path) in self._favorite_models
        item.setData(0, Qt.ItemDataRole.UserRole, entry)
        item.setData(0, _FAVORITE_ROLE, favorite)
        item.setToolTip(
            0, _model_tooltip(entry, favorite, self._language_manager.translate)
        )
        return item

    def _populate_tree(
        self, entries: List[ModelEntry], selected_path: Optional[Path], filtering: bool
    ) -> Optional[QTreeWidgetItem]:
        selected_item: Optional[QTreeWidgetItem] = None
        favorites = [
            entry
            for entry in entries
            if app_settings.favorite_model_key(entry.path) in self._favorite_models
        ]
        favorite_root = QTreeWidgetItem(
            self._model_tree,
            [f"{self._language_manager.translate('★ Favourites')} ({len(favorites)})"],
        )
        favorite_root.setData(0, _TREE_PATH_ROLE, "favorites")
        favorite_font = favorite_root.font(0)
        favorite_font.setBold(True)
        favorite_root.setFont(0, favorite_font)
        favorite_root.setToolTip(
            0,
            self._language_manager.translate(
                "Favourite models stay directly reachable here regardless of their "
                "folder location."
            ),
        )
        for entry in _sort_model_entries(favorites, self._favorite_models):
            item = self._add_tree_model_item(favorite_root, entry)
            if selected_path is not None and entry.path == selected_path:
                selected_item = item
        favorite_root.setExpanded(
            filtering or "favorites" not in self._tree_collapsed_paths
        )

        roots = getattr(self, "_last_scan_roots", self._active_model_paths())
        folder_items: Dict[Tuple[str, ...], QTreeWidgetItem] = {}
        ordered = sorted(
            entries,
            key=lambda entry: (
                tuple(part.casefold() for part in _model_folder_parts(entry, roots)),
                entry.name.casefold(),
            ),
        )
        for entry in ordered:
            parent: Optional[QTreeWidgetItem] = None
            cumulative: List[str] = []
            for folder in _model_folder_parts(entry, roots):
                cumulative.append(folder)
                folder_key = tuple(cumulative)
                folder_item = folder_items.get(folder_key)
                if folder_item is None:
                    folder_item = (
                        QTreeWidgetItem(parent, [folder])
                        if parent is not None
                        else QTreeWidgetItem(self._model_tree, [folder])
                    )
                    state_key = "folder:" + "\x1f".join(folder_key)
                    folder_item.setData(0, _TREE_PATH_ROLE, state_key)
                    folder_item.setIcon(
                        0, self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
                    )
                    folder_item.setToolTip(0, str(entry.path.parent))
                    folder_item.setExpanded(
                        filtering or state_key not in self._tree_collapsed_paths
                    )
                    folder_items[folder_key] = folder_item
                parent = folder_item
            item = self._add_tree_model_item(parent, entry)
            if selected_item is None and selected_path is not None:
                if entry.path == selected_path:
                    selected_item = item
        return selected_item

    def _select_model_path(self, path: Path, view: QListWidget | QTreeWidget) -> None:
        if isinstance(view, QListWidget):
            for row in range(view.count()):
                item = view.item(row)
                entry = self._entry_from_model_item(item)
                if entry is not None and entry.path == path:
                    view.setCurrentItem(item)
                    return
            return

        def visit(parent: QTreeWidgetItem) -> Optional[QTreeWidgetItem]:
            entry = self._entry_from_model_item(parent)
            if entry is not None and entry.path == path:
                return parent
            for child_index in range(parent.childCount()):
                found = visit(parent.child(child_index))
                if found is not None:
                    return found
            return None

        for top_index in range(view.topLevelItemCount()):
            found = visit(view.topLevelItem(top_index))
            if found is not None:
                view.setCurrentItem(found)
                return

    def _populate_list(self, entries: List[ModelEntry]) -> None:
        selected_path = self._selected_model_path()
        filtering = bool(self._search.text().strip())
        self._model_list.blockSignals(True)
        self._model_tree.blockSignals(True)
        try:
            self._model_list.clear()
            selected_list_item: Optional[QListWidgetItem] = None
            for entry in _sort_model_entries(entries, self._favorite_models):
                item = QListWidgetItem(_model_display_text(entry))
                favorite = (
                    app_settings.favorite_model_key(entry.path) in self._favorite_models
                )
                item.setData(Qt.ItemDataRole.UserRole, entry)
                item.setData(_FAVORITE_ROLE, favorite)
                item.setToolTip(
                    _model_tooltip(entry, favorite, self._language_manager.translate)
                )
                self._model_list.addItem(item)
                if selected_path is not None and entry.path == selected_path:
                    selected_list_item = item
            if selected_list_item is not None:
                self._model_list.setCurrentItem(selected_list_item)

            self._model_tree.clear()
            selected_tree_item = self._populate_tree(entries, selected_path, filtering)
            if selected_tree_item is not None:
                self._model_tree.setCurrentItem(selected_tree_item)
        finally:
            self._model_list.blockSignals(False)
            self._model_tree.blockSignals(False)

    def _set_model_favorite(self, entry: ModelEntry, favorite: bool) -> None:
        """Persist a star click, then safely rebuild after the delegate event."""
        key = app_settings.favorite_model_key(entry.path)
        if favorite:
            self._favorite_models.add(key)
        else:
            self._favorite_models.discard(key)
        app_settings.set_model_favorite(entry.path, favorite)
        if self._favorite_refresh_pending:
            return
        self._favorite_refresh_pending = True
        QTimer.singleShot(0, self._refresh_after_favorite_change)

    def _refresh_after_favorite_change(self) -> None:
        """Rebuild views only after Qt has finished the star-click event."""
        self._favorite_refresh_pending = False
        self._apply_filter(self._search.text())

    def _apply_filter(self, text: str) -> None:
        q = text.strip().casefold()
        self._populate_list(
            self._all_entries
            if not q
            else [
                entry
                for entry in self._all_entries
                if q in entry.name.casefold() or q in str(entry.path.parent).casefold()
            ]
        )

    def _show_model_context_menu(
        self, position: QPoint, view: QListWidget | QTreeWidget
    ) -> None:
        """Offer model-specific actions for the item under the pointer."""
        item = view.itemAt(position)
        entry = self._entry_from_model_item(item)
        if item is None or entry is None:
            return

        # Make the right-clicked model the active one as users expect, while
        # keeping the menu action tied to this exact item.
        view.setCurrentItem(item)
        menu = QMenu(view)
        tr = self._language_manager.translate
        open_folder = menu.addAction(tr("📂 Open GGUF folder"))
        menu.addSeparator()
        copy_expert = menu.addAction(tr("📋 Copy Expert settings"))
        paste_expert = menu.addAction(tr("📌 Paste Expert settings"))
        clipboard = self._expert_settings_clipboard
        paste_expert.setEnabled(clipboard is not None)
        if clipboard is not None:
            source_name = str(clipboard.get("source_model", "model"))
            source_target = str(clipboard.get("performance_target", ""))
            paste_expert.setToolTip(
                f"{source_name} [{source_target}] → {entry.name} [{source_target}]"
            )

        viewport = view.viewport()
        if viewport is None:  # defensive for incomplete Qt teardown states
            return
        chosen = menu.exec(viewport.mapToGlobal(position))
        if chosen is copy_expert:
            self._copy_expert_settings(entry)
            return
        if chosen is paste_expert:
            self._paste_expert_settings(entry)
            return
        if chosen is not open_folder:
            return

        folder = entry.path.parent
        if not folder.is_dir() or not _open_local_folder(folder):
            QMessageBox.warning(
                self,
                _tr("Could not open folder"),
                f"{_tr('The GGUF folder could not be opened:')}\n{folder}",
            )

    def _copy_expert_settings(self, entry: ModelEntry) -> None:
        """Copy one model's active target-scoped Expert snapshot in memory."""
        if self._current_entry is None or self._current_entry.path != entry.path:
            self._show_config(entry)
        target_name = self._current_performance_target_name()
        if self._config_stack.currentIndex() == 1:
            self._expert_panel.flush_pending_save()
        profile_slot = self._current_setting_profile_slot()
        drafter_key = self._current_drafter_profile_key()
        snapshot = app_settings.get_setting_profile_snapshot(
            entry.name, entry.path, target_name, profile_slot, drafter_key
        )
        if snapshot is None and self._system is not None:
            profile = match_profile(
                entry.name,
                self._profiles,
                getattr(entry, "architecture", ""),
            )
            self._load_expert_panel(entry, profile)
            snapshot = self._expert_panel._make_snapshot()
            snapshot["source"] = "copied-auto-expert-settings"
        if not isinstance(snapshot, dict) or not isinstance(
            snapshot.get("values"), dict
        ):
            QMessageBox.information(
                self,
                _tr("Copy Expert settings"),
                _tr(
                    "No copyable Expert settings exist for this model yet. Wait for "
                    "hardware detection to finish or open Expert settings first."
                ),
            )
            return
        self._expert_settings_clipboard = {
            "snapshot": copy.deepcopy(snapshot),
            "source_model": entry.name,
            "source_path": str(entry.path),
            "performance_target": target_name,
            "profile_slot": profile_slot,
            "drafter_key": drafter_key,
        }
        self._status.showMessage(
            f"Expert Settings kopiert: {entry.name} [{target_name}]", 6000
        )
        self._log(
            f"[Expert] Copied {entry.name} [{target_name}] settings to the "
            "internal clipboard."
        )

    def _paste_expert_settings(self, entry: ModelEntry) -> None:
        """Paste the copied snapshot into the same target on another model."""
        clipboard = self._expert_settings_clipboard
        if not isinstance(clipboard, dict):
            return
        snapshot = clipboard.get("snapshot")
        if not isinstance(snapshot, dict) or not isinstance(
            snapshot.get("values"), dict
        ):
            self._expert_settings_clipboard = None
            QMessageBox.warning(
                self,
                "Expert Settings einfügen",
                "Die kopierten Expert Settings sind ungültig und wurden verworfen.",
            )
            return
        target_name = str(clipboard.get("performance_target", ""))
        if target_name not in PERFORMANCE_TARGETS:
            target_name = self._current_performance_target_name()
        if self._current_entry is None or self._current_entry.path != entry.path:
            self._show_config(entry)
        if self._config_stack.currentIndex() == 1:
            self._expert_panel.flush_pending_save()

        pasted = copy.deepcopy(snapshot)
        pasted["saved_at"] = datetime.now().isoformat(timespec="seconds")
        pasted["source"] = "copied-expert-settings"
        pasted["copied_from"] = {
            "model": str(clipboard.get("source_model", "")),
            "performance_target": target_name,
        }
        destination_drafter = self._current_drafter_profile_key()
        destination_slot = self._current_setting_profile_slot()
        if destination_slot not in app_settings.CUSTOM_PROFILE_SLOTS:
            destination_slot = next(
                (
                    slot
                    for slot in app_settings.CUSTOM_PROFILE_SLOTS
                    if not app_settings.has_setting_profile_snapshot(
                        entry.name,
                        entry.path,
                        target_name,
                        slot,
                        destination_drafter,
                    )
                ),
                app_settings.CUSTOM_PROFILE_SLOTS[0],
            )
        try:
            saved = app_settings.set_setting_profile_snapshot(
                entry.name,
                entry.path,
                target_name,
                destination_slot,
                pasted,
                destination_drafter,
                select=True,
            )
            if not saved:
                raise OSError("profile settings write failed")
        except Exception as exc:
            QMessageBox.warning(
                self,
                _tr("Paste Expert settings"),
                f"{_tr('The Expert settings could not be saved:')}\n{exc}",
            )
            return

        self._refresh_setting_profile_selector()
        target_index = self._perf_combo.findText(target_name)
        if target_index >= 0 and self._perf_combo.currentIndex() != target_index:
            self._perf_combo.setCurrentIndex(target_index)
        elif self._config_stack.currentIndex() == 1 and self._system is not None:
            profile = match_profile(
                entry.name,
                self._profiles,
                getattr(entry, "architecture", ""),
            )
            self._load_expert_panel(entry, profile)
        else:
            self._refresh_config_preview()
        source_name = str(clipboard.get("source_model", "model"))
        self._status.showMessage(
            f"Expert Settings eingefügt: {source_name} → {entry.name} [{target_name}]",
            7000,
        )
        self._log(
            f"[Expert] Pasted {source_name} [{target_name}] settings into "
            f"{entry.name}; source snapshot remains unchanged."
        )

    def _browse_models(self) -> None:
        dlg = _PathListDialog(
            self,
            _tr("Manage model paths"),
            self.model_paths,
            _tr("Select a model folder"),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.model_paths = dlg.paths()
        first = next((p for p, enabled in self.model_paths if enabled), None)
        if first is None and self.model_paths:
            first = self.model_paths[0][0]
        if first is not None:
            self.models_path = first
        try:
            app_settings.set_model_paths(self.model_paths)
            self._log(
                f"[Models] Saved {len(self.model_paths)} path(s), "
                f"{len(self._active_model_paths())} active."
            )
        except Exception as exc:
            self._log(f"[Warning] Could not save model paths: {exc}")
        self._start_scan()

    # ------------------------------------------------------------------
    # Config preview + options (single-click)
    # ------------------------------------------------------------------
    def _on_selection_changed(
        self,
        current: Optional[QListWidgetItem | QTreeWidgetItem],
        _prev: Optional[QListWidgetItem | QTreeWidgetItem],
    ) -> None:
        entry = self._entry_from_model_item(current)
        if entry is not None:
            self._show_config(entry)

    def _on_model_activated(self, item: QListWidgetItem | QTreeWidgetItem) -> None:
        entry = self._entry_from_model_item(item)
        if entry is not None and is_ocr_model(entry):
            self._open_ocr_workflow()

    def _show_config(self, entry: ModelEntry) -> None:
        """Called on model selection — updates checkboxes, auto-selects fork, refreshes preview.

        Gracefully handles the case when hardware detection has not yet
        completed (self._system is None).  The user will see a placeholder
        message and the config will update automatically once detection
        finishes.
        """
        if self._system is None:
            # Retain the requested entry so the initial hardware worker can
            # finish the preview automatically. Previously this early return
            # forgot the click when scanning completed before WMI/driver probes.
            self._current_entry = entry
            self._btn_diagnose.setEnabled(True)
            self._config_preview.setPlainText(
                _tr("Hardware detection is still running…")
                + "\n\n"
                + _tr("Please wait until the system information has been loaded.")
                + "\n"
                + _tr("The configuration is updated automatically.")
            )
            return
        # Switching models drops the Expert state — the panel's pins were
        # for the *previous* model. Flush a pending edit while _current_entry
        # still identifies that previous model, otherwise the delayed signal
        # could save its snapshot onto the newly selected/right-clicked model.
        if self._config_stack.currentIndex() == 1:
            if (
                self._current_entry is not None
                and self._current_entry.path != entry.path
            ):
                self._expert_panel.flush_pending_save()
            self._config_stack.setCurrentIndex(0)
            self._btn_expert_row.setVisible(True)
        self._current_entry = entry
        preferred_target = app_settings.get_model_performance_target(
            entry.path, self._current_performance_backend()
        )
        if preferred_target:
            preferred_index = self._perf_combo.findText(preferred_target)
            if (
                preferred_index >= 0
                and self._perf_combo.currentIndex() != preferred_index
            ):
                self._perf_combo.blockSignals(True)
                self._perf_combo.setCurrentIndex(preferred_index)
                self._perf_combo.blockSignals(False)
                app_settings.set_performance_target(preferred_target)
                self._log(
                    f"[Perf] Restored {entry.name}'s preferred/fastest mode: "
                    f"{preferred_target}"
                )
        self._current_draft = _find_draft_model(entry, self._all_entries)
        self._btn_diagnose.setEnabled(True)
        self._update_checkboxes(entry)
        self._refresh_setting_profile_selector()
        profile = match_profile(
            entry.name, self._profiles, getattr(entry, "architecture", "")
        )
        self._auto_select_fork(profile)
        self._update_config_text(entry, profile)
        # _start_scan() deliberately disables Launch. Model selection is the
        # state transition that must restore it once no exclusive workflow is
        # active; otherwise the button remains grey for the whole session.
        self._enable_launch_when_ocr_idle()

    def _update_checkboxes(self, entry: ModelEntry) -> None:
        """Refresh dropdown selections and the remaining checkbox states.

        The mmproj and draft dropdowns are authoritative: selecting a file or
        embedded MTP enables it, while their leading ``none`` entries disable
        it. Legacy ``vision``/``draft`` checkbox overrides are used only when
        no dropdown choice has been stored yet, preserving existing settings.
        """
        # Pull persisted overrides first so a fresh app launch already
        # honours last session's choices. The in-memory cache wins if
        # both exist, since the user may have toggled mid-session.
        persisted = app_settings.get_model_overrides(entry.name)
        cached = self._option_overrides.get(entry.name, {})
        ov = {**persisted, **cached}

        # Resolve both dropdowns before any dependent controls. The chosen
        # entries themselves now replace the former Vision and Draft checkboxes.
        self._populate_draft_combo(entry, ov)
        self._populate_mmproj_combo(entry, ov)

        # This option is meaningful only while a projector is selected.
        mmproj_cpu_state = ov.get("mmproj_cpu", False)
        mmproj_cpu_enabled = self._vision_enabled()
        self._chk_mmproj_cpu.blockSignals(True)
        self._chk_mmproj_cpu.setEnabled(mmproj_cpu_enabled)
        self._chk_mmproj_cpu.setChecked(mmproj_cpu_enabled and mmproj_cpu_state)
        self._chk_mmproj_cpu.blockSignals(False)

        # ── Thinking / Reasoning ────────────────────────────────────
        # Read the chat template from GGUF metadata (the authoritative source);
        # fall back to a conservative filename heuristic when the template is
        # missing. This fixes the Qwen3-Coder false-positive: the old heuristic
        # matched any "qwen3" filename, but Qwen3-Coder has no <think> tokens
        # and llama-server logs "reasoning 0".
        has_thinking = entry.supports_thinking
        thinking_state = ov["thinking"] if "thinking" in ov else has_thinking
        self._chk_thinking.blockSignals(True)
        self._chk_thinking.setEnabled(has_thinking)
        self._chk_thinking.setChecked(has_thinking and thinking_state)
        self._chk_thinking.blockSignals(False)

        # ── n-gram (ngram-mod) ──────────────────────────────────────
        # Always enabled: ngram-mod needs no draft model and works on any
        # GGUF, so it must never be greyed out (the whole point — "ngram
        # should always be available"). v5.2.4 defaults it on for every model;
        # an existing explicit per-model off choice remains authoritative.
        ngram_state = ov["ngram"] if "ngram" in ov else True
        self._chk_ngram.blockSignals(True)
        self._chk_ngram.setEnabled(True)
        self._chk_ngram.setChecked(ngram_state)
        self._chk_ngram.blockSignals(False)

        # ── Prompt caching (host RAM, -cram) ────────────────────────
        # Prompt caching is available for every model. build_command enables it
        # with Vision only on b10045+ and safely emits --cache-ram 0 for older
        # or unprobeable binaries.
        pc_state = ov["prompt_cache"] if "prompt_cache" in ov else True
        self._chk_prompt_cache.blockSignals(True)
        self._chk_prompt_cache.setEnabled(True)
        self._chk_prompt_cache.setChecked(pc_state)
        self._chk_prompt_cache.setText("Prompt caching (host RAM, -cram)")
        self._chk_prompt_cache.blockSignals(False)
        self._sp_prompt_cache_mib.setEnabled(pc_state)

        ocr_selected = is_ocr_model(entry)
        self._btn_ocr.setVisible(ocr_selected)
        self._btn_ocr.setEnabled(
            ocr_selected and self._vision_enabled() and self._ocr_thread is None
        )

    def _vision_enabled(self) -> bool:
        """Return whether the mmproj dropdown currently selects a projector."""
        return bool(self._cb_mmproj.currentData()) and bool(
            self._current_entry is not None and self._current_entry.mmproj is not None
        )

    def _draft_enabled(self) -> bool:
        """Return whether the draft dropdown selects external or embedded MTP."""
        return bool(self._cb_draft.currentData())

    def _populate_mmproj_combo(self, entry: ModelEntry, ov: dict) -> None:
        """Fill the always-on mmproj dropdown from ``entry.folder_mmprojs``.

        Lists every projector in the model's folder plus a leading
        "— no mmproj —" entry. Projectors the scanner considers incompatible
        with this model are prefixed with a warning marker but remain
        selectable. The remembered per-model selection wins; otherwise the
        scanner's auto pick (``entry.mmproj``) is preselected. The resolved
        choice is written back onto ``entry.mmproj`` so launch + preview agree
        (None when "no mmproj" is selected).
        """
        WARN = "⚠ "
        NONE_LABEL = "— no mmproj —"
        folder = list(getattr(entry, "folder_mmprojs", []) or [])
        auto = entry.mmproj  # scanner's best pick (may be None)

        self._cb_mmproj.blockSignals(True)
        self._cb_mmproj.clear()
        # Index 0 is always the explicit "none" choice (userData empty string).
        self._cb_mmproj.addItem(NONE_LABEL, userData="")

        remembered = app_settings.get_mmproj_selection(entry.name)
        chosen_idx = 0  # default to "none"
        # Explicit dropdown choice wins. With no stored dropdown choice, a
        # legacy unchecked Vision override migrates to the leading none entry.
        deliberate_none = remembered == app_settings.MMPROJ_NONE_SENTINEL or (
            remembered is None and ov.get("vision") is False
        )
        for c in folder:
            compatible = is_mmproj_compatible(entry.path, c)
            try:
                size_label = f"{c.stat().st_size / (1024**3):.1f} GB"
            except OSError:
                size_label = "size unavailable"
            label = f"{c.name}   ({size_label})"
            if c == auto:
                label += "   (auto)"
            if not compatible:
                label = WARN + label
            self._cb_mmproj.addItem(label, userData=str(c))
            idx = self._cb_mmproj.count() - 1
            if remembered and not deliberate_none and c.name == remembered:
                chosen_idx = idx
        # No remembered choice → preselect the scanner's auto pick if present.
        if not remembered and not deliberate_none and auto is not None:
            for i in range(1, self._cb_mmproj.count()):
                if self._cb_mmproj.itemData(i) == str(auto):
                    chosen_idx = i
                    break

        self._cb_mmproj.setCurrentIndex(chosen_idx)
        # Apply the resolved choice to the entry so launch uses it.
        sel = self._cb_mmproj.itemData(chosen_idx)
        entry.mmproj = Path(sel) if sel else None
        self._mmproj_row.setVisible(True)
        self._cb_mmproj.blockSignals(False)

    def _populate_draft_combo(self, entry: ModelEntry, ov: dict) -> None:
        """Fill the always-on draft dropdown from ``entry.folder_drafts``.

        Mirrors :meth:`_populate_mmproj_combo`. Lists a leading
        "— no draft —" entry, an "MTP (embedded in GGUF)" entry when the model
        carries an embedded MTP head, then every draft GGUF in the folder
        (DFlash/EAGLE-3/MTP heads labelled explicitly; incompatible ones
        flagged with '⚠'). Resolves and applies the
        per-model selection onto ``self._current_draft`` (a draft ModelEntry
        with metadata, or None). The remembered choice wins; otherwise the
        scanner's auto pick (``entry.draft``) is preselected, falling back to
        embedded-MTP when present.
        """
        WARN = "⚠ "
        NONE_LABEL = "— no draft —"
        MTP_LABEL = "MTP (embedded in GGUF)"
        MTP_DATA = app_settings.DRAFT_EMBEDDED_SENTINEL
        folder = list(getattr(entry, "folder_drafts", []) or [])
        auto = entry.draft  # scanner's best external pick (Path or None)
        has_embedded = entry.has_embedded_mtp

        self._cb_draft.blockSignals(True)
        self._cb_draft.clear()
        self._cb_draft.addItem(NONE_LABEL, userData="")
        if has_embedded:
            self._cb_draft.addItem(MTP_LABEL, userData=MTP_DATA)

        remembered = app_settings.get_draft_selection(entry.name)
        # Default selection priority: remembered → external auto → embedded.
        # A legacy unchecked Draft override migrates to none only when no
        # authoritative dropdown selection exists yet.
        deliberate_none = remembered == app_settings.DRAFT_NONE_SENTINEL or (
            remembered is None and ov.get("draft") is False
        )
        chosen_idx = 0
        if remembered == app_settings.DRAFT_EMBEDDED_SENTINEL and has_embedded:
            chosen_idx = 1
        for c in folder:
            md = read_gguf_metadata(c)
            compatible = is_draft_compatible(entry.path, c, md)
            arch = str(md.get("general.architecture", "") or "").lower().strip()
            kind = ""
            is_dspark = arch == "dspark" or (
                arch == "dflash"
                and (
                    md.get("__dspark_scan__") == "found"
                    or bool(re.search(r"(?:^|[-_.])dspark(?:[-_.]|$)", c.name, re.I))
                )
            )
            if is_dspark:
                kind = "  [DSpark]"
            elif arch == "dflash":
                kind = "  [DFlash]"
            elif arch == "eagle3":
                kind = "  [EAGLE-3]"
            elif arch.endswith("-assistant") or arch.endswith("_assistant"):
                kind = "  [MTP]"
            try:
                size_label = f"{c.stat().st_size / (1024**3):.1f} GB"
            except OSError:
                size_label = "size unavailable"
            label = f"{c.name}{kind}   ({size_label})"
            if auto is not None and c == auto:
                label += "  (auto)"
            if not compatible:
                label = WARN + label
            self._cb_draft.addItem(label, userData=str(c))
            idx = self._cb_draft.count() - 1
            if remembered and remembered not in (
                "",
                app_settings.DRAFT_NONE_SENTINEL,
                app_settings.DRAFT_EMBEDDED_SENTINEL,
            ):
                if c.name == remembered:
                    chosen_idx = idx
        # No remembered choice → prefer external auto pick, else embedded MTP.
        # A profile with ``auto_draft: false`` lists the paired drafter but
        # leaves it unselected (MiniCPM-V 4.7 + DSpark measured slower).
        profile_auto_draft = getattr(
            match_profile(
                entry.name, self._profiles, getattr(entry, "architecture", "")
            ),
            "auto_draft",
            True,
        )
        if not remembered and not deliberate_none:
            if auto is not None and not profile_auto_draft:
                pass
            elif auto is not None:
                for i in range(self._cb_draft.count()):
                    if self._cb_draft.itemData(i) == str(auto):
                        chosen_idx = i
                        break
            elif has_embedded:
                chosen_idx = 1  # the MTP entry

        self._cb_draft.setCurrentIndex(chosen_idx)
        self._draft_row.setVisible(True)
        self._cb_draft.blockSignals(False)
        # Resolve the selection into _current_draft (and update the checkbox
        # text / state) via the shared applier.
        self._apply_draft_selection(entry, self._cb_draft.itemData(chosen_idx))

    def _apply_draft_selection(self, entry: ModelEntry, data: object) -> None:
        """Resolve a draft-combo userData value onto ``self._current_draft``.

        ``data`` is "" (none), the embedded-MTP sentinel, or a draft file
        path string. For an external path we build a draft ModelEntry WITH
        metadata so ``is_standalone_drafter`` resolves (drives draft-mtp).
        Embedded MTP carries no separate file, so ``_current_draft`` is None
        and the embedded path inside the GGUF is used at launch.
        """
        s = "" if data is None else str(data)
        if s and s != app_settings.DRAFT_EMBEDDED_SENTINEL:
            self._current_draft = _make_draft_entry(Path(s), entry.group)
        else:
            self._current_draft = None

    def _auto_select_fork(self, profile: ModelProfile) -> None:
        """Select a required fork family without losing the active backend.

        Profile hints stay backend-neutral. When both optimized siblings exist,
        switching from a HIP mainline build chooses the HIP fork and switching
        from Vulkan stays on Vulkan. An explicitly backend-qualified hint wins.
        """
        if not profile.server_binary:
            return

        from auto_tuner import _fork_backend, _fork_family, _fork_newest_first_key

        first = Path(profile.server_binary).parts[0]
        if not first.endswith(".cpp"):
            first = first + ".cpp"
        required_family = _fork_family(first)
        required_backend = _fork_backend(first)
        current_text = self._fork_combo.currentText()
        current_backend = _fork_backend(current_text)

        matching: List[int] = []
        for i in range(self._fork_combo.count()):
            item = self._fork_combo.itemText(i)
            if _fork_family(item) != required_family:
                continue
            item_backend = _fork_backend(item)
            if required_backend and item_backend not in (required_backend, None):
                continue
            matching.append(i)

        if self._fork_manual_override:
            if not matching:
                self._log(
                    f"[Fork] Profile requires '{first}' but it's not available. "
                    f"Keeping manual selection: {current_text}"
                )
            return

        current_family_matches = _fork_family(current_text) == required_family
        current_backend_matches = not required_backend or current_backend in (
            required_backend,
            None,
        )
        if current_family_matches and current_backend_matches:
            return
        if not matching:
            return

        preferred_backend = required_backend or current_backend
        # Older pinned trees stay installed next to a new pin: prefer the
        # active backend, then the newest build of the required family.
        matching.sort(
            key=lambda index: _fork_newest_first_key(
                self._fork_combo.itemText(index),
                preferred_backend=preferred_backend,
            )
        )
        chosen = matching[0]
        if self._fork_combo.currentIndex() != chosen:
            self._fork_combo.blockSignals(True)
            self._fork_combo.setCurrentIndex(chosen)
            self._fork_combo.blockSignals(False)
            self._apply_fork(chosen)
            self._log(f"[Fork] Auto-selected: {self._fork_combo.itemText(chosen)}")

    # ------------------------------------------------------------------
    # Per-option toggle slots
    #
    # Each slot:
    #   1. records the override against the currently-selected model
    #      (in-memory + persisted JSON), so the choice survives both
    #      a model switch and an app restart, and
    #   2. recomputes the config preview to reflect the new option set.
    #
    # The override is keyed by `entry.name` (GGUF filename stem). We
    # only persist when there's actually a current model — slot calls
    # during programmatic checkbox setup are guarded by blockSignals.
    # ------------------------------------------------------------------
    def _record_override(self, key: str, checked: bool) -> None:
        entry = self._current_entry
        if entry is None:
            return
        cur = self._option_overrides.setdefault(entry.name, {})
        cur[key] = bool(checked)
        try:
            app_settings.set_model_override(entry.name, key, bool(checked))
        except Exception as exc:
            self._log(f"[Warning] Could not save {key} override: {exc}")

    def _on_mmproj_cpu_toggled(self, checked: bool) -> None:
        self._record_override("mmproj_cpu", checked)
        self._refresh_config_preview()

    def _on_thinking_toggled(self, checked: bool) -> None:
        self._record_override("thinking", checked)
        self._refresh_config_preview()

    def _on_ngram_toggled(self, checked: bool) -> None:
        # n-gram is independent of the model (no draft file needed), so just
        # persist the setting and refresh the preview.
        self._record_override("ngram", checked)
        self._refresh_config_preview()

    def _on_prompt_cache_toggled(self, checked: bool) -> None:
        # Persist the per-model prompt-cache choice. build_command applies the
        # conservative version gate when Vision is active.
        self._record_override("prompt_cache", checked)
        self._sp_prompt_cache_mib.setEnabled(checked)
        self._refresh_config_preview()

    def _on_prompt_cache_limit_changed(self, value: int) -> None:
        app_settings.set_prompt_cache_ram_mib(value)
        self._refresh_config_preview()

    def _on_mmproj_changed(self, index: int) -> None:
        """User picked a different vision projector from the dropdown.

        Updates the current model's ``mmproj`` to the chosen file (or None for
        the "— no mmproj —" entry), remembers the choice per model, refreshes
        dependent controls, and recomputes projector memory in the preview.
        """
        if self._current_entry is None or index < 0:
            return
        path_str = self._cb_mmproj.itemData(index)
        chosen = Path(path_str) if path_str else None
        self._current_entry.mmproj = chosen
        try:
            # A real filename is remembered; selecting "— no mmproj —" records
            # the explicit none-sentinel so re-population doesn't re-apply the
            # scanner's auto pick over the user's deliberate choice.
            app_settings.set_mmproj_selection(
                self._current_entry.name,
                chosen.name if chosen else app_settings.MMPROJ_NONE_SENTINEL,
            )
        except Exception as exc:
            self._log(f"[Warning] Could not save mmproj selection: {exc}")
        # mmproj presence changes CPU placement, OCR availability, and the
        # image-input capability advertised to external clients.
        self._update_checkboxes(self._current_entry)
        self._refresh_config_preview()
        self._refresh_control_api_catalogue()

    def _on_draft_combo_changed(self, index: int) -> None:
        """User picked a different draft head from the dropdown.

        Resolves the selection onto ``self._current_draft`` (None for
        "— no draft —", a metadata-bearing draft ModelEntry for a file, None
        for embedded MTP since it lives inside the main GGUF), remembers the
        choice per model, refreshes dependent controls, and updates the preview.
        """
        if self._current_entry is None or index < 0:
            return
        data = self._cb_draft.itemData(index)
        s = "" if data is None else str(data)
        # Persist all three states distinctly. In particular embedded MTP must
        # not collapse to the same sentinel as "no draft" during repopulation.
        try:
            if not s:
                selection = app_settings.DRAFT_NONE_SENTINEL
            elif s == app_settings.DRAFT_EMBEDDED_SENTINEL:
                selection = app_settings.DRAFT_EMBEDDED_SENTINEL
            else:
                selection = Path(s).name
            app_settings.set_draft_selection(self._current_entry.name, selection)
        except Exception as exc:
            self._log(f"[Warning] Could not save draft selection: {exc}")
        # Warn (don't block) when an incompatible draft was chosen.
        if s and s != app_settings.DRAFT_EMBEDDED_SENTINEL:
            chosen = Path(s)
            if not is_draft_compatible(
                self._current_entry.path, chosen, read_gguf_metadata(chosen)
            ):
                self._log(
                    f"[Warning] Draft '{chosen.name}' looks incompatible with "
                    f"'{self._current_entry.name}' (different model/architecture). "
                    "Launching anyway — speculative decoding may fail or be slow."
                )
        self._apply_draft_selection(self._current_entry, data)
        # Re-run dependent controls; the dropdown itself is the enable state.
        self._update_checkboxes(self._current_entry)
        self._refresh_setting_profile_selector()
        self._refresh_config_preview()

    def _refresh_config_preview(self) -> None:
        """Checkbox changed → recompute context/memory with new options."""
        if self._current_entry is not None and self._system is not None:
            profile = match_profile(
                self._current_entry.name,
                self._profiles,
                getattr(self._current_entry, "architecture", ""),
            )
            self._update_config_text(self._current_entry, profile)

    def _build_auto_config(
        self,
        entry: ModelEntry,
        profile: ModelProfile,
        force_overrides: Optional[dict] = None,
        performance_target: Optional[str] = None,
    ) -> Optional[TunedConfig]:
        """Helper: rebuild a TunedConfig for the given model with the
        current checkbox states. Returns None when system info is missing.

        Centralised so both the preview path and the Expert panel's
        recompute callback share the same vision/draft handling.
        """
        if self._system is None:
            return None

        use_vision = self._vision_enabled()
        use_draft = self._draft_enabled()
        no_mmproj_offload = (
            self._chk_mmproj_cpu.isChecked() and self._chk_mmproj_cpu.isEnabled()
        )
        prompt_cache_ram_mib = (
            self._sp_prompt_cache_mib.value()
            if self._chk_prompt_cache.isChecked() and self._chk_prompt_cache.isEnabled()
            else 0
        )

        entry_for_cfg = copy.copy(entry)
        if not use_vision:
            entry_for_cfg.mmproj = None

        # Build the kwargs dict carefully — only forward keys whose
        # values the caller actually pinned. Sending None for an unset
        # force_* parameter is fine (compute_config handles it), but
        # being explicit makes the call site easier to read in logs.
        kwargs = dict(force_overrides or {})
        self._config_error = ""

        try:
            return compute_config(
                model=entry_for_cfg,
                system=self._system,
                profile=profile,
                draft_model=self._current_draft if use_draft else None,
                force_mlock=False,
                perf_target=self._resolve_perf_target_for_profile(
                    profile, performance_target
                ),
                mode=self._current_mode(),
                no_mmproj_offload=no_mmproj_offload,
                prompt_cache_ram_mib=prompt_cache_ram_mib,
                gpu_priorities=app_settings.get_gpu_priorities(),
                force_gpu=app_settings.get_forced_gpu(),
                **kwargs,
            )
        except Exception as exc:
            self._config_error = str(exc)
            self._log(f"[Warning] compute_config failed: {exc}")
            return None

    def _measured_snapshot_matches_environment(
        self, snapshot: dict, baseline: TunedConfig
    ) -> bool:
        """Reject measured winners from another binary/backend/search schema."""
        source = str(snapshot.get("source", "") or "").lower()
        if not source.startswith("measured-"):
            return True
        stored = snapshot.get("benchmark_environment")
        if not isinstance(stored, dict) or self._system is None:
            return False
        runtime_binary = self._active_llama_binary()
        if not runtime_binary:
            return False
        expected = _benchmark_environment_fingerprint(
            runtime_binary,
            probe_binary_build_number(runtime_binary),
            self._system,
            baseline,
        )
        return stored == expected

    def _effective_config(
        self,
        entry: ModelEntry,
        profile: ModelProfile,
        performance_target: Optional[str] = None,
    ) -> Optional[TunedConfig]:
        """The config actually shown in the preview and used at launch.

        Honours a saved Expert override per model; otherwise the
        AutoTuner's auto-tuned default. This is what makes a hand-tuned
        Expert setup "stick" for a model (the low-VRAM use case): once
        saved, the override is applied automatically, like the persisted
        launch-option selections and checkbox overrides.

        * Auto-mode overrides are re-derived through ``compute_config``
          with the saved pins so they ADAPT to the current VRAM /
          launch-option state, then the saved non-cascading values are
          stamped back on.
        * Manual-mode overrides are applied as a frozen config (the user
          owns the exact values); the launch-path VRAM fit-check still
          gates them, and Reset reverts to Auto.
        """
        target_name = performance_target or self._current_performance_target_name()
        base = self._build_auto_config(entry, profile, performance_target=target_name)
        if base is None:
            return None
        slot = self._current_setting_profile_slot()
        drafter_key = self._current_drafter_profile_key()
        override = app_settings.get_setting_profile_snapshot(
            entry.name, entry.path, target_name, slot, drafter_key
        )
        if not override:
            return base
        if not self._measured_snapshot_matches_environment(override, base):
            self._log(
                f"[Performance] Ignoring stale measured profile for {entry.name} "
                f"[{target_name}]; binary/backend/search environment changed."
            )
            return base
        vals = override.get("values") or {}
        try:
            if override.get("mode") == "manual" and vals:
                return expert_cfg_from_values(base, vals)
            # Auto mode: re-cascade from the saved pins (adapts to the
            # live VRAM / checkbox state), then overlay the saved
            # non-cascading widget values (threads / batch / flags / …).
            pins = _expert_cascading_pins(override)
            cascaded = (
                self._build_auto_config(
                    entry, profile, pins, performance_target=target_name
                )
                or base
            )
            if vals:
                cascaded = apply_expert_values(cascaded, vals)
            return cascaded
        except Exception as exc:
            self._log(
                f"[Warning] Saved Expert override for {entry.name} invalid "
                f"({exc}); falling back to Auto."
            )
            return base

    def _load_expert_panel(self, entry: ModelEntry, profile: ModelProfile) -> None:
        """Bind the Expert panel to the current model + apply any saved
        Expert override. Used when entering Expert mode and when a
        checkbox toggles while the panel is already open."""
        assert self._system is not None  # callers guard / assert this first
        target_name = self._current_performance_target_name()
        cfg = self._build_auto_config(entry, profile, performance_target=target_name)
        if cfg is None:
            return
        self._expert_loaded_performance_target = target_name
        self._expert_loaded_profile_slot = self._current_setting_profile_slot()
        self._expert_loaded_drafter_key = self._current_drafter_profile_key()
        self._expert_loaded_performance_backend = self._current_performance_backend()
        self._expert_panel.configure_for_model(
            cfg=cfg,
            system=self._system,
            native_ctx=entry.native_context,
            profile_max=profile.max_context,
            recompute_cb=lambda overrides: self._build_auto_config(
                entry, profile, overrides, performance_target=target_name
            ),
        )
        override = app_settings.get_setting_profile_snapshot(
            entry.name,
            entry.path,
            target_name,
            self._expert_loaded_profile_slot,
            self._expert_loaded_drafter_key,
        )
        if override:
            if self._measured_snapshot_matches_environment(override, cfg):
                self._expert_panel.restore_from_snapshot(override)
            else:
                self._log(
                    f"[Performance] Measured Expert profile for {entry.name} "
                    f"[{target_name}] needs revalidation on this runtime."
                )

    def _update_config_text(self, entry: ModelEntry, profile: ModelProfile) -> None:
        """Recompute the effective config (Auto or saved Expert override)
        using the current checkbox states and refresh the preview."""
        assert self._system is not None
        eff = self._effective_config(entry, profile)
        if eff is None:
            self._config_preview.setPlainText(
                f"Model: {entry.name}\nNo safe automatic configuration.\n"
                + (
                    getattr(self, "_config_error", "")
                    or "Insufficient memory or hardware information."
                )
            )
            return
        self._render_cfg_to_preview(entry, profile, eff)
        # When Expert mode is open, keep the panel in sync with checkbox /
        # hardware changes — but first flush any in-flight edit so it is
        # not lost when we repaint from the (now current) override.
        if self._config_stack.currentIndex() == 1:
            self._expert_panel.flush_pending_save()
            self._load_expert_panel(entry, profile)

    def _render_cfg_to_preview(
        self,
        entry: ModelEntry,
        profile: ModelProfile,
        cfg: TunedConfig,
    ) -> None:
        """Format ``cfg`` into the read-only preview QTextEdit."""
        assert self._system is not None
        use_vision = self._vision_enabled() and not cfg.memory_disable_vision
        use_draft = self._draft_enabled() and not cfg.memory_disable_draft
        use_ngram = (
            self._chk_ngram.isChecked()
            and self._chk_ngram.isEnabled()
            and not cfg.memory_disable_draft
        )
        use_prompt_cache = (
            self._chk_prompt_cache.isChecked()
            and self._chk_prompt_cache.isEnabled()
            and cfg.prompt_cache_ram_mib != 0
        )

        W = 64
        bar = "─" * W
        lines = [bar]
        lines.append(f"Model   : {entry.name}")
        lines.append(
            f"Settings: {self._setting_profile_combo.currentText()} "
            f"[{self._current_performance_target_name()}]"
        )
        lines.append(
            f"Profile : {profile.display_name}"
            + (f"  ({profile.source_file})" if profile.source_file else "")
        )
        # Profile explanations follow the interface language; English (UK)
        # and finally the YAML text remain the fallbacks.
        profile_notes = self._language_manager.profile_notes(profile).strip()
        if profile_notes:
            wrapped = textwrap.wrap(
                profile_notes,
                width=W - 10,
                break_long_words=True,
                break_on_hyphens=False,
            )
            for i, chunk in enumerate(wrapped):
                prefix = "Notes   : " if i == 0 else "          "
                lines.append(f"{prefix}{chunk}")
        if entry.mmproj:
            vis = "✓" if use_vision else "✗"
            placement = "RAM" if cfg.no_mmproj_offload else "VRAM"
            lines.append(f"Vision  : {entry.mmproj.name}  [{vis}, {placement}]")
        if self._current_draft:
            drf = "✓" if use_draft else "✗"
            lines.append(f"Draft   : {self._current_draft.name}  [{drf}]")
        if use_ngram:
            lines.append("n-gram  : ngram-mod (self-speculative)  [✓]")

        cache_limit = cfg.prompt_cache_ram_mib
        cache_label = "unlimited" if cache_limit == -1 else f"{cache_limit} MiB"
        lines.append(
            f"Prompt$ : host-RAM cache (-cram)  [{'✓' if use_prompt_cache else '✗'}] "
            f"[{cache_label}]" + (" (Vision requires b10045+)" if use_vision else "")
        )
        if profile.server_binary:
            lines.append(f"Requires: {profile.server_binary}")
        lines.append(bar)

        if cfg.full_offload:
            placement = f"GPU full offload  ({entry.n_layers or '?'} layers)"
        elif cfg.is_moe and cfg.n_cpu_moe:
            placement = (
                f"MoE hybrid — {cfg.n_cpu_moe} CPU expert layer(s) "
                f"of {entry.n_layers or '?'} total"
            )
        elif cfg.ngl > 0:
            placement = f"Hybrid — {cfg.ngl}/{entry.n_layers or '?'} layers GPU + CPU"
        else:
            placement = "CPU only"

        # KV-quant line annotated with the strategy (symmetric /
        # asymmetric / turbo / manual) so the user sees at a glance
        # what the AutoTuner actually applied.
        kv_line = f"KV cache quant  : K={cfg.cache_k}  V={cfg.cache_v}"
        if cfg.kv_quant_strategy and cfg.kv_quant_strategy != "symmetric":
            kv_line += f"  [{cfg.kv_quant_strategy}]"
        if _is_turbo_kv_type(cfg.cache_k) or _is_turbo_kv_type(cfg.cache_v):
            kv_line += "  [TurboQuant fork required]"

        lines += [
            f"Placement       : {placement}",
            f"Perf target     : {cfg.performance_target}",
            f"Mode            : {self._current_mode()}",
            f"Context         : {cfg.ctx:,} tokens",
            kv_line,
            f"Threads         : {cfg.threads}  (batch: {cfg.batch_threads})",
            f"Batch / ubatch  : {cfg.batch} / {cfg.ubatch}",
            f"Parallel slots   : {cfg.n_parallel}"
            + (" (manual)" if getattr(cfg, "n_parallel_forced", False) else "")
            + "  (--parallel / -np)",
            f"HTTP diagnostics: metrics={'on' if cfg.metrics_enabled else 'off'}  "
            f"slots={'on' if cfg.slots_api_enabled else 'off'}",
            f"Flash attention : {'on' if cfg.flash_attn else 'off'}",
        ]
        load_mode = effective_load_mode(cfg)
        if load_mode is not None:
            lines.append(f"load mode       : {load_mode}")
        if cfg.no_kv_offload:
            # LOW-VRAM lever (low_vram perf-target): the KV cache lives in
            # system RAM, attention compute runs on CPU. Surface it so the
            # user understands why context is huge but generation is slower.
            lines.append(
                "KV in RAM       : on (--no-kv-offload)  [slower gen, max context]"
            )
        if cfg.rope_scaling:
            lines.append(f"RoPE scaling    : on (factor {cfg.rope_scale_factor:.1f}×)")
        s = cfg.sampling
        lines.append(
            f"Sampling        : temp={s.get('temperature')}  "
            f"top_k={s.get('top_k')}  top_p={s.get('top_p')}  "
            f"min_p={s.get('min_p')}  rep={s.get('repeat_penalty')}"
        )

        # ── Memory estimate (with vision / draft / KV breakdown) ────
        # The old version only printed `Model GPU` for the main weights,
        # which made vision/draft toggles look counter-intuitive (the
        # main number went down while total GPU usage went up). We now
        # show every component plus a `Total GPU` row so the user sees
        # exactly what fits where.
        total_gpu = (
            cfg.estimated_model_vram_gb
            + cfg.vision_vram_gb
            + cfg.draft_vram_gb
            + cfg.kv_vram_gb
            + cfg.recurrent_state_vram_gb
            + cfg.runtime_vram_overhead_gb
            + float(getattr(cfg, "compute_vram_gb", 0.0) or 0.0)
            + cfg.batch_vram_overhead_gb
        )
        mapped_resident_gb = float(
            getattr(cfg, "mapped_model_resident_gb", cfg.mapped_model_ram_gb) or 0.0
        )
        total_cpu = (
            cfg.estimated_model_ram_gb
            + mapped_resident_gb
            + cfg.vision_ram_gb
            + cfg.kv_ram_gb
            + cfg.recurrent_state_ram_gb
            + cfg.runtime_ram_overhead_gb
            + float(getattr(cfg, "compute_ram_gb", 0.0) or 0.0)
            + cfg.prompt_cache_ram_gb
        )
        lines += [bar, "Memory estimate (with current options):"]
        lines.append(
            f"  Model GPU : ~{cfg.estimated_model_vram_gb:5.1f} GB"
            f"   (free VRAM: {self._system.free_vram_gb:.1f} GB)"
        )
        if cfg.vision_vram_gb > 0.05:
            lines.append(f"  Vision GPU: ~{cfg.vision_vram_gb:5.1f} GB")
        if cfg.vision_ram_gb > 0.05:
            lines.append(f"  Vision RAM: ~{cfg.vision_ram_gb:5.1f} GB")
        if cfg.draft_vram_gb > 0.05:
            lines.append(f"  Draft GPU : ~{cfg.draft_vram_gb:5.1f} GB")
        # KV split: show both parts when hybrid; otherwise the single number.
        if cfg.kv_ram_gb > 0.05:
            lines.append(
                f"  KV cache  : ~{cfg.estimated_kv_gb:5.1f} GB"
                f"   (VRAM {cfg.kv_vram_gb:.1f} + RAM {cfg.kv_ram_gb:.1f})"
            )
        else:
            lines.append(f"  KV cache  : ~{cfg.estimated_kv_gb:5.1f} GB")
        recurrent_total = cfg.recurrent_state_vram_gb + cfg.recurrent_state_ram_gb
        if recurrent_total > 0.005:
            lines.append(
                f"  Recurrent : ~{recurrent_total:5.2f} GB"
                f" (GPU {cfg.recurrent_state_vram_gb:.2f} / "
                f"RAM {cfg.recurrent_state_ram_gb:.2f})"
            )
        if cfg.runtime_vram_overhead_gb > 0.05:
            lines.append(f"  Runtime GPU: ~{cfg.runtime_vram_overhead_gb:5.1f} GB")
        if cfg.runtime_ram_overhead_gb > 0.05:
            lines.append(f"  Runtime RAM: ~{cfg.runtime_ram_overhead_gb:5.1f} GB")
        if cfg.batch_vram_overhead_gb > 0.05:
            lines.append(f"  Batch GPU : ~{cfg.batch_vram_overhead_gb:5.1f} GB")
        compute_vram_gb = float(getattr(cfg, "compute_vram_gb", 0.0) or 0.0)
        compute_ram_gb = float(getattr(cfg, "compute_ram_gb", 0.0) or 0.0)
        if compute_vram_gb > 0.05:
            lines.append(f"  Compute GPU: ~{compute_vram_gb:5.1f} GB")
        if compute_ram_gb > 0.05:
            lines.append(f"  Compute RAM: ~{compute_ram_gb:5.1f} GB")
        lines.append(f"  Model CPU : ~{cfg.estimated_model_ram_gb:5.1f} GB")
        if cfg.mapped_model_ram_gb > 0.05:
            lines.append(
                f"  Lazy mmap : ~{cfg.mapped_model_ram_gb:5.1f} GB file-backed"
                f"   (active budget {mapped_resident_gb:.1f} GB)"
            )
        if cfg.unified_memory:
            lines.append(
                f"  Unified total: ~{total_gpu + total_cpu:5.1f} GB"
                f"   of {min(self._system.free_ram_gb, self._system.free_vram_gb):.1f} "
                "GB accelerator-addressable"
            )
            lines.append("  (GPU and CPU allocations share this one physical pool.)")
        else:
            lines.append(
                f"  Total GPU : ~{total_gpu:5.1f} GB"
                f"   of {self._system.free_vram_gb:.1f} GB free"
            )
            lines.append(f"              (free RAM: {self._system.free_ram_gb:.1f} GB)")
        if cfg.prompt_cache_ram_gb > 0.05:
            lines.append(f"  Prompt RAM: ~{cfg.prompt_cache_ram_gb:5.1f} GB")
        if total_cpu > cfg.estimated_model_ram_gb + 0.05:
            lines.append(f"  Total CPU : ~{total_cpu:5.1f} GB")
        if cfg.warning:
            lines.append(f"  ⚠ {cfg.warning}")
        # Discoverability nudge for the LOW-VRAM escape hatch. On a small
        # GPU where the current tier can only squeeze out a sub-agentic
        # context (<32k) while the box has plenty of RAM, point the user at
        # the low_vram Performance preset — it moves the KV cache into
        # system RAM and typically unlocks an order of magnitude more
        # context. Only shown when the user is NOT already on low_vram, so
        # it never nags once they've opted in.
        if (
            not cfg.no_kv_offload
            and self._system is not None
            and self._system.total_vram_gb <= 16.5
            and self._system.free_ram_gb >= 16
            and cfg.ctx < max(32768, min(int(entry.native_context or 32768), 131072))
        ):
            lines.append(
                "  💡 Spare RAM available: Performance → low_vram moves the KV "
                "cache into RAM for larger context (slower n_decode). The "
                "Performance test stores a separate measured profile for it."
            )
        lines.append(bar)

        self._config_preview.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # Expert mode entry / exit
    # ------------------------------------------------------------------
    def _enter_expert_mode(self) -> None:
        """Swap the read-only preview for the editable Expert panel.

        Restores the saved Expert override for the current model if one
        exists (so a hand-tuned setup is right where the user left it);
        otherwise starts from the AutoTuner's auto-tuned default.
        """
        if self._current_entry is None or self._system is None:
            QMessageBox.information(
                self,
                "No model selected",
                "Select a model first — the Expert panel needs a current "
                "configuration to start from.",
            )
            return
        entry = self._current_entry
        profile = match_profile(
            entry.name,
            self._profiles,
            getattr(entry, "architecture", ""),
        )
        if self._build_auto_config(entry, profile) is None:
            return
        self._load_expert_panel(entry, profile)
        self._config_stack.setCurrentIndex(1)
        # Hide the Expert button (it's now "covered" by the panel — the
        # Auto/Manual/Reset toggles inside the panel take its place at
        # the top of the same area).
        self._btn_expert_row.setVisible(False)
        slot = self._current_setting_profile_slot()
        override = app_settings.get_setting_profile_snapshot(
            entry.name,
            entry.path,
            self._current_performance_target_name(),
            slot,
            self._current_drafter_profile_key(),
        )
        if override:
            self._log(
                f"[Expert] Entered {self._setting_profile_combo.currentText()} — "
                f"restored {override.get('mode', 'auto')} settings for {entry.name}."
            )
        else:
            self._log(
                f"[Expert] Entered {self._setting_profile_combo.currentText()} "
                "from Auto defaults."
            )

    def _exit_expert_mode(self) -> None:
        """Return to the read-only preview view."""
        # Flush a pending debounced save so the on-disk override matches
        # what is on screen (covers the edit-then-immediately-close case).
        self._expert_panel.flush_pending_save()
        self._config_stack.setCurrentIndex(0)
        self._btn_expert_row.setVisible(True)
        # Re-render the preview from the panel's current cfg so the
        # user's last Expert tweaks remain visible until they pick a
        # different model.
        cfg = self._expert_panel.current_config()
        if cfg is not None and self._current_entry is not None:
            profile = match_profile(
                self._current_entry.name,
                self._profiles,
                getattr(self._current_entry, "architecture", ""),
            )
            self._render_cfg_to_preview(self._current_entry, profile, cfg)
        self._log("[Expert] Returned to preview.")

    def _on_expert_cfg_changed(self, cfg: TunedConfig) -> None:
        """Slot: Expert panel finished a cascade. Mirror to preview footer.

        We do NOT swap the stacked widget back here — the user is still
        editing. We just refresh the on-disk preview text so the next
        time they exit, it reflects their state.
        """
        if self._current_entry is not None:
            profile = match_profile(
                self._current_entry.name,
                self._profiles,
                getattr(self._current_entry, "architecture", ""),
            )
            self._render_cfg_to_preview(self._current_entry, profile, cfg)

    def _on_expert_mode_changed(self, mode: str) -> None:
        self._log(f"[Expert] Mode → {mode}.")

    def _on_expert_state_changed(self, snapshot: dict) -> None:
        """Autosave only to Custom slots; Auto and Perform remain immutable."""
        entry = self._current_entry
        if entry is None or not isinstance(snapshot, dict):
            return
        target_name = self._expert_loaded_performance_target
        drafter_key = self._expert_loaded_drafter_key
        slot = self._expert_loaded_profile_slot
        try:
            if slot not in app_settings.CUSTOM_PROFILE_SLOTS:
                slot = next(
                    (
                        candidate
                        for candidate in app_settings.CUSTOM_PROFILE_SLOTS
                        if not app_settings.has_setting_profile_snapshot(
                            entry.name,
                            entry.path,
                            target_name,
                            candidate,
                            drafter_key,
                        )
                    ),
                    "",
                )
                if not slot:
                    self._status.showMessage(
                        "All four Custom profiles contain settings. Select the "
                        "Custom profile you want to edit first.",
                        9000,
                    )
                    self._log(
                        "[Profile] Auto/Perform edit was not saved because all "
                        "Custom slots are occupied."
                    )
                    return
                snapshot = copy.deepcopy(snapshot)
                snapshot["source"] = f"forked-from-{self._expert_loaded_profile_slot}"
                self._expert_loaded_profile_slot = slot
            saved = app_settings.set_setting_profile_snapshot(
                entry.name,
                entry.path,
                target_name,
                slot,
                snapshot,
                drafter_key,
                select=True,
                selection_backend=self._expert_loaded_performance_backend,
            )
            if not saved:
                raise OSError("profile settings write failed")
            QTimer.singleShot(0, self._refresh_setting_profile_selector)
        except Exception as exc:
            self._log(f"[Warning] Could not save Expert settings: {exc}")

    def _on_expert_reset(self) -> None:
        """Reset the current Custom variant; Auto/Perform reload unchanged."""
        entry = self._current_entry
        if entry is None or self._system is None:
            return
        profile = match_profile(
            entry.name,
            self._profiles,
            getattr(entry, "architecture", ""),
        )
        self._expert_panel._save_timer.stop()
        slot = self._expert_loaded_profile_slot
        try:
            if slot in app_settings.CUSTOM_PROFILE_SLOTS:
                app_settings.clear_custom_setting_profile(
                    entry.name,
                    entry.path,
                    self._expert_loaded_performance_target,
                    slot,
                    self._expert_loaded_drafter_key,
                    self._expert_loaded_performance_backend,
                )
                self._expert_loaded_profile_slot = app_settings.PROFILE_AUTO
            else:
                app_settings.set_selected_setting_profile(
                    entry.name,
                    entry.path,
                    self._expert_loaded_performance_target,
                    slot,
                    self._expert_loaded_drafter_key,
                    self._expert_loaded_performance_backend,
                )
        except Exception as exc:
            self._log(f"[Warning] Could not reset profile settings: {exc}")
        self._refresh_setting_profile_selector()
        self._load_expert_panel(entry, profile)
        eff = self._effective_config(entry, profile)
        if eff is not None:
            self._render_cfg_to_preview(entry, profile, eff)
        self._log(f"[Expert] Reset {self._setting_profile_combo.currentText()}.")

    # ------------------------------------------------------------------
    # Diagnostic report
    # ------------------------------------------------------------------
    def _show_diagnostic_report(self) -> None:
        """Open a modal dialog showing the metadata diagnostic for the
        currently selected model.

        Reuses the same ``diagnostics`` module the CLI ``--diagnose``
        path uses, so the output is identical and there's no second
        place to maintain.
        """
        if self._current_entry is None:
            QMessageBox.information(
                self,
                "No model selected",
                "Select a model first — the diagnostic report needs a "
                "model to analyse.",
            )
            return

        # Import lazily so the GUI module does not pay the cost on
        # startup, and so missing diagnostics.py degrades to a
        # graceful error message rather than refusing to launch.
        try:
            from diagnostics import format_diagnostic_report
        except ImportError as exc:  # pragma: no cover — defensive
            QMessageBox.warning(
                self,
                "Diagnostics module missing",
                f"Could not load diagnostics.py:\n{exc}",
            )
            return

        report = format_diagnostic_report(self._current_entry)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Diagnose — {self._current_entry.name}")
        dlg.resize(720, 560)
        layout = QVBoxLayout(dlg)

        view = QTextEdit()
        view.setReadOnly(True)
        view.setPlainText(report)
        self._apply_mono_font(view)
        layout.addWidget(view, 1)

        hint = QLabel(
            "Support report creates a redacted, shareable system/model summary. "
            "Full metadata scans every GGUF header (including drafters/projectors) "
            "only after confirmation and never reads tensor payloads."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        support_button = QPushButton("Save support report")
        metadata_button = QPushButton("Scan all model metadata…")
        bb.addButton(support_button, QDialogButtonBox.ButtonRole.ActionRole)
        bb.addButton(metadata_button, QDialogButtonBox.ButtonRole.ActionRole)
        support_button.clicked.connect(self._save_support_report)

        def start_metadata_export() -> None:
            if self._start_metadata_diagnostic_export():
                metadata_button.setEnabled(False)
                metadata_button.setText("Metadata scan running…")

        metadata_button.clicked.connect(start_metadata_export)
        bb.rejected.connect(dlg.reject)
        bb.accepted.connect(dlg.accept)
        close_btn: QPushButton | None = bb.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(dlg.accept)
        layout.addWidget(bb)

        dlg.exec()
        self._log(f"[Diagnose] Inspected metadata for {self._current_entry.name}")

    def _save_support_report(self) -> None:
        try:
            from diagnostics import write_support_report

            output = write_support_report(
                self._all_entries,
                system=self._system,
                forks=self._forks,
                active_fork=self._fork_path,
                model_roots=self._active_model_paths(),
                app_log_path=self._app_log_path,
                debug_enabled=self._debug_mode,
            )
        except Exception as exc:
            QMessageBox.warning(
                self, "Support report failed", f"Could not create the report:\n{exc}"
            )
            self._log(f"[Diagnose] Support report failed: {exc}")
            return
        self._log(f"[Diagnose] Redacted support report: {output}")
        answer = QMessageBox.question(
            self,
            "Support report saved",
            f"Saved a redacted report with {len(self._all_entries)} model(s):\n"
            f"{output}\n\nPrompts, credentials, raw settings, and server output are excluded. "
            "Open it now?",
            QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Close,
            QMessageBox.StandardButton.Open,
        )
        if answer == QMessageBox.StandardButton.Open:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output)))

    def _start_metadata_diagnostic_export(self) -> bool:
        try:
            if (
                self._diagnostic_thread is not None
                and self._diagnostic_thread.isRunning()
            ):
                QMessageBox.information(
                    self, "Metadata scan", "A full metadata scan is already running."
                )
                return False
        except RuntimeError:
            self._diagnostic_thread = None
            self._diagnostic_worker = None

        roots = self._active_model_paths()
        if not roots:
            QMessageBox.information(
                self, "Metadata scan", "No active model folders are configured."
            )
            return False
        answer = QMessageBox.question(
            self,
            "Scan every GGUF header?",
            f"Scan all GGUF files below {len(roots)} active model folder(s) and "
            "write a detailed Markdown inventory?\n\nThis reads metadata and tensor "
            "names only—not model weights or prompts. GGUF metadata may contain "
            "converter provenance strings, so review the file before sharing it. "
            "Unchanged files use the local metadata cache, but a large first scan can "
            "still take several minutes.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False

        reports = app_settings.app_data_dir() / "reports"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output = reports / f"AutoTuner-model-metadata-{stamp}.md"
        worker = _MetadataDiagnosticWorker(roots, output)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_metadata_diagnostic_progress)
        worker.finished.connect(self._on_metadata_diagnostic_finished)
        worker.error.connect(self._on_metadata_diagnostic_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(
            lambda: self._clear_metadata_diagnostic_references(thread)
        )
        thread.finished.connect(thread.deleteLater)
        self._diagnostic_worker = worker
        self._diagnostic_thread = thread
        self._status.showMessage("Scanning all GGUF metadata…")
        self._log(
            f"[Diagnose] Confirmed full metadata scan across {len(roots)} root(s)."
        )
        thread.start()
        return True

    def _on_metadata_diagnostic_progress(
        self, completed: int, total: int, name: str
    ) -> None:
        self._status.showMessage(f"Metadata scan {completed}/{max(1, total)}: {name}")

    def _on_metadata_diagnostic_finished(self, path: str, count: int) -> None:
        self._status.showMessage(f"Metadata report complete — {count} GGUF file(s)")
        self._log(f"[Diagnose] Full metadata report ({count} files): {path}")
        answer = QMessageBox.question(
            self,
            "Metadata report complete",
            f"Scanned {count} GGUF file(s) and saved:\n{path}\n\nOpen it now?",
            QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Close,
            QMessageBox.StandardButton.Open,
        )
        if answer == QMessageBox.StandardButton.Open:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _on_metadata_diagnostic_error(self, message: str) -> None:
        self._status.showMessage("Metadata scan failed")
        self._log(f"[Diagnose] Full metadata scan failed: {message}")
        QMessageBox.warning(self, "Metadata scan failed", message)

    def _clear_metadata_diagnostic_references(self, thread: QThread) -> None:
        if self._diagnostic_thread is thread:
            self._diagnostic_thread = None
            self._diagnostic_worker = None

    # ------------------------------------------------------------------
    # Deterministic real-model performance tuning
    # ------------------------------------------------------------------
    def _show_performance_analysis(self) -> None:
        entry = self._current_entry
        if entry is None:
            QMessageBox.information(
                self,
                "Select a model",
                "Select a model first. In-app analysis is intentionally scoped to "
                "that model; the generated HTML report remains global.",
            )
            return
        all_records = app_settings.list_performance_run_results()
        report_path: Optional[Path] = None
        try:
            # Export remains global even though the in-app view is filtered.
            report_path = write_performance_report(all_records)
            self._log(f"[Performance] Detailed HTML report: {report_path}")
        except Exception as exc:
            self._log(f"[Warning] Could not write performance HTML report: {exc}")
        selected_records = _performance_records_for_model(all_records, entry)
        dialog = _PerformanceAnalysisDialog(
            selected_records,
            self,
            html_path=report_path,
            selected_model_name=entry.name,
        )
        dialog.exec()

    @staticmethod
    def _is_benchmarkable_entry(entry: ModelEntry, profile: ModelProfile) -> bool:
        runner = str(profile.runner or "llama-server")
        extra = {str(arg).strip().lower() for arg in profile.extra_args}
        return bool(
            runner in ("", "llama-server")
            and not entry.is_diffusion
            and not entry.is_standalone_drafter
            and not is_ocr_model(entry)
            and "--embeddings" not in extra
            and "--embedding" not in extra
        )

    def _benchmark_model_options(
        self,
        entry: ModelEntry,
        *,
        tune_mtp: bool,
        drafter_selection: Optional[str] = None,
    ) -> Dict[str, object]:
        """Resolve persisted launch options without mutating another model's UI."""
        model = copy.copy(entry)
        overrides = app_settings.get_model_overrides(entry.name)

        remembered_mmproj = app_settings.get_mmproj_selection(entry.name)
        if remembered_mmproj == app_settings.MMPROJ_NONE_SENTINEL or (
            remembered_mmproj is None and overrides.get("vision") is False
        ):
            model.mmproj = None
        elif remembered_mmproj:
            model.mmproj = next(
                (
                    path
                    for path in list(getattr(entry, "folder_mmprojs", []) or [])
                    if path.name == remembered_mmproj
                ),
                model.mmproj,
            )

        remembered_draft = (
            drafter_selection
            if drafter_selection is not None
            else app_settings.get_draft_selection(entry.name)
        )
        draft_model: Optional[ModelEntry] = None
        enable_speculative = False
        external_path: Optional[Path] = None
        if remembered_draft and remembered_draft not in (
            app_settings.DRAFT_NONE_SENTINEL,
            app_settings.DRAFT_EMBEDDED_SENTINEL,
        ):
            external_path = next(
                (
                    path
                    for path in list(getattr(entry, "folder_drafts", []) or [])
                    if path.name == remembered_draft
                ),
                None,
            )
        elif remembered_draft is None and entry.draft is not None:
            external_path = entry.draft

        deliberate_draft_off = remembered_draft == app_settings.DRAFT_NONE_SENTINEL or (
            remembered_draft is None and overrides.get("draft") is False
        )
        embedded_selected = (
            remembered_draft == app_settings.DRAFT_EMBEDDED_SENTINEL
            or (
                remembered_draft is None
                and external_path is None
                and entry.has_embedded_mtp
                and not deliberate_draft_off
            )
        )
        if tune_mtp:
            # The explicit option asks to include an available MTP/drafter even
            # when normal launch selection is "none"; no compatible head means
            # the option is simply skipped for this model.
            if entry.has_embedded_mtp:
                embedded_selected = True
                deliberate_draft_off = False
            elif external_path is None and entry.draft is not None:
                external_path = entry.draft
                deliberate_draft_off = False

        if not deliberate_draft_off and external_path is not None:
            draft_model = _make_draft_entry(external_path, entry.group)
            enable_speculative = draft_model is not None
        elif not deliberate_draft_off and embedded_selected and entry.has_embedded_mtp:
            enable_speculative = True

        prompt_cache = overrides.get("prompt_cache", True)
        drafter_key = (
            _drafter_profile_key(draft_model)
            if draft_model is not None
            else (
                app_settings.EMBEDDED_DRAFTER_PROFILE_KEY
                if enable_speculative and embedded_selected
                else app_settings.NO_DRAFTER_PROFILE_KEY
            )
        )
        return {
            "model": model,
            "draft_model": draft_model,
            "drafter_key": drafter_key,
            "drafter_label": (
                draft_model.path.name
                if draft_model is not None
                else "Embedded MTP"
                if enable_speculative
                else "No drafter"
            ),
            "enable_speculative": enable_speculative,
            "use_thinking": bool(
                entry.supports_thinking
                and overrides.get("thinking", entry.supports_thinking)
            ),
            "enable_ngram": bool(overrides.get("ngram", True)),
            "enable_prompt_cache": bool(prompt_cache),
            "prompt_cache_ram_mib": (
                app_settings.get_prompt_cache_ram_mib() if prompt_cache else 0
            ),
            "no_mmproj_offload": bool(
                model.mmproj is not None and overrides.get("mmproj_cpu", False)
            ),
        }

    def _benchmark_option_variants(
        self,
        entry: ModelEntry,
        *,
        tune_mtp: bool,
        include_external_drafters: bool,
    ) -> List[Dict[str, object]]:
        """Return current options or every compatible embedded/external head."""
        if not tune_mtp or not include_external_drafters:
            return [self._benchmark_model_options(entry, tune_mtp=tune_mtp)]

        selections: List[str] = []
        if entry.has_embedded_mtp:
            selections.append(app_settings.DRAFT_EMBEDDED_SENTINEL)
        draft_paths = list(getattr(entry, "folder_drafts", []) or [])
        if entry.draft is not None and entry.draft not in draft_paths:
            draft_paths.append(entry.draft)
        for path in draft_paths:
            try:
                metadata = read_gguf_metadata(path)
            except Exception:
                metadata = {}
            if is_draft_compatible(entry.path, path, metadata):
                selections.append(path.name)

        variants: List[Dict[str, object]] = []
        seen: set[str] = set()
        for selection in selections:
            options = self._benchmark_model_options(
                entry, tune_mtp=True, drafter_selection=selection
            )
            key = str(options["drafter_key"])
            if not bool(options["enable_speculative"]) or key in seen:
                continue
            seen.add(key)
            variants.append(options)
        if variants:
            return variants
        return [self._benchmark_model_options(entry, tune_mtp=tune_mtp)]

    def _resolve_selected_benchmark_runtime(
        self,
        option: _PerformanceRuntimeOption,
        profile: ModelProfile,
    ) -> Tuple[Optional[str], str]:
        """Resolve one selected build without mutating global LLAMA_CPP_DIR."""
        binary = str(option.binary or "").strip()
        if not binary or not self._is_runnable_binary(Path(binary)):
            return None, f"{option.display_name}: llama-server is not runnable."
        if not profile.server_binary:
            return binary, ""

        try:
            from auto_tuner import _fork_backend, _fork_family

            required_name = Path(profile.server_binary).parts[0]
            selected_name = (
                option.root.name if option.root is not None else option.display_name
            )
            if _fork_family(required_name) != _fork_family(selected_name):
                return (
                    None,
                    f"{option.display_name}: model profile requires the "
                    f"{required_name} build family.",
                )
            required_backend = _fork_backend(required_name)
            selected_backend = _fork_backend(selected_name) or option.backend_hint
            if (
                required_backend
                and selected_backend
                and required_backend != selected_backend
            ):
                return (
                    None,
                    f"{option.display_name}: model profile requires "
                    f"{required_backend.upper()}.",
                )
        except Exception as exc:
            return (
                None,
                f"{option.display_name}: build compatibility check failed: {exc}",
            )
        return binary, ""

    def _benchmark_compute_config(
        self,
        entry: ModelEntry,
        profile: ModelProfile,
        system: SystemInfo,
        performance_target: str,
        options: Dict[str, object],
        overrides: Optional[dict] = None,
    ) -> TunedConfig:
        return compute_config(
            model=cast(ModelEntry, options["model"]),
            system=system,
            profile=profile,
            draft_model=cast(Optional[ModelEntry], options["draft_model"])
            if bool(options["enable_speculative"])
            else None,
            force_mlock=False,
            perf_target=PERFORMANCE_TARGETS[performance_target],
            mode=self._current_mode(),
            no_mmproj_offload=bool(options["no_mmproj_offload"]),
            prompt_cache_ram_mib=cast(int, options["prompt_cache_ram_mib"]),
            gpu_priorities=app_settings.get_gpu_priorities(),
            force_gpu=app_settings.get_forced_gpu(),
            **dict(overrides or {}),
        )

    def _benchmark_config_for_target(
        self,
        entry: ModelEntry,
        profile: ModelProfile,
        system: SystemInfo,
        performance_target: str,
        options: Dict[str, object],
        *,
        desired_context: int,
        enable_yarn: bool,
        real_validation: bool,
        benchmark_backend: str = "",
        runtime_binary: str = "",
        ignore_saved_profile: bool = False,
    ) -> Tuple[TunedConfig, int, bool]:
        """Build a safe plan, then optionally let the private server prove more."""
        auto = self._benchmark_compute_config(
            entry, profile, system, performance_target, options
        )
        drafter_key = str(
            options.get("drafter_key", app_settings.NO_DRAFTER_PROFILE_KEY)
        )
        seed_slot = app_settings.get_selected_setting_profile(
            entry.name,
            entry.path,
            performance_target,
            drafter_key,
            benchmark_backend,
        )
        saved = (
            None
            if ignore_saved_profile
            else app_settings.get_setting_profile_snapshot(
                entry.name,
                entry.path,
                performance_target,
                seed_slot,
                drafter_key,
            )
        )
        if saved and str(saved.get("source", "") or "").lower().startswith("measured-"):
            expected_environment = (
                _benchmark_environment_fingerprint(
                    runtime_binary,
                    probe_binary_build_number(runtime_binary),
                    system,
                    auto,
                )
                if runtime_binary
                else None
            )
            if (
                expected_environment is None
                or saved.get("benchmark_environment") != expected_environment
            ):
                saved = None
        effective = auto
        if saved:
            values = saved.get("values") or {}
            if saved.get("mode") == "manual" and values:
                effective = expert_cfg_from_values(auto, values)
            else:
                saved_pins = _expert_cascading_pins(saved)
                effective = self._benchmark_compute_config(
                    entry, profile, system, performance_target, options, saved_pins
                )
                if values:
                    effective = apply_expert_values(effective, values)

        requested = max(0, int(desired_context))
        if requested <= 0:
            if enable_yarn and profile.rope_scale_max_ctx > max(
                0, entry.native_context
            ):
                requested = int(profile.rope_scale_max_ctx)
            else:
                requested = int(effective.ctx)

        native_limit = max(0, int(entry.native_context or profile.max_context or 0))
        rope_limit = max(native_limit, int(profile.rope_scale_max_ctx or 0))
        model_limit = rope_limit if enable_yarn else native_limit
        if model_limit > 0:
            requested = min(requested, model_limit)

        pins: Dict[str, object] = {
            "user_ctx": requested,
            "force_cache_k": effective.cache_k,
            "force_cache_v": effective.cache_v,
            "force_flash_attn": effective.flash_attn,
            "force_n_parallel": 1,
            "force_draft_n_max": int(effective.draft_n_max or 0) or None,
            "force_rope_scale": bool(enable_yarn),
        }
        if effective.is_moe and effective.n_cpu_moe is not None:
            pins["force_n_cpu_moe"] = effective.n_cpu_moe
        else:
            pins["force_ngl"] = effective.ngl
        safe = self._benchmark_compute_config(
            entry,
            profile,
            system,
            performance_target,
            options,
            {key: value for key, value in pins.items() if value is not None},
        )

        # Preserve non-cascading values from the mode-specific profile. The
        # benchmark will deliberately vary only threads/batches/draft depth.
        for attr in (
            "threads",
            "batch_threads",
            "batch",
            "ubatch",
            "load_mode",
            "mlock",
            "no_mmap",
            "numa",
            "metrics_enabled",
            "slots_api_enabled",
        ):
            setattr(safe, attr, copy.deepcopy(getattr(effective, attr)))
        safe.sampling = copy.deepcopy(effective.sampling)
        safe.extra_cli_flags = list(effective.extra_cli_flags or [])
        safe.n_parallel = 1
        safe.n_parallel_forced = True

        safe_context = int(safe.ctx)
        try_exact = bool(
            real_validation
            and desired_context > 0
            and requested > safe_context
            and (model_limit <= 0 or requested <= model_limit)
        )
        if try_exact:
            # Static headroom is intentionally conservative and cannot know
            # every backend/ubatch allocation. Only the isolated benchmark gets
            # this exact trial; normal auto launch remains on the safe planner.
            safe.ctx = requested
            safe.no_context_shift = requested >= 32768 or safe.full_offload
            if enable_yarn and native_limit > 0 and requested > native_limit:
                safe.rope_scaling = True
                requested_factor = (requested + native_limit - 1) // native_limit
                safe.rope_scale_factor = max(
                    1.0,
                    min(
                        float(profile.rope_scale_factor or 4.0), float(requested_factor)
                    ),
                )
            safe.warning = (
                f"Real validation trial at {requested:,}; static safe estimate was "
                f"{safe_context:,}."
            )
        return safe, safe_context, try_exact

    @staticmethod
    def _completed_benchmark_matches(
        record: Optional[dict],
        entry: ModelEntry,
        baseline: TunedConfig,
        benchmark_type: str,
        prompt_fraction: float,
        *,
        runtime_binary: str,
        system: SystemInfo,
    ) -> bool:
        if not isinstance(record, dict) or not isinstance(
            record.get("candidates"), list
        ):
            return False
        try:
            stat = entry.path.stat()
            stored_size = int(record.get("model_size", 0) or 0)
            stored_mtime = int(record.get("model_mtime_ns", 0) or 0)
            if stored_size > 0 and stored_size != stat.st_size:
                return False
            if stored_mtime > 0 and stored_mtime != stat.st_mtime_ns:
                return False
        except (OSError, TypeError, ValueError):
            return False
        try:
            if int(record.get("desired_context", 0) or 0) != int(baseline.ctx):
                return False
            stored_fraction = float(record.get("prompt_context_fraction", 0.0))
        except (TypeError, ValueError):
            return False
        if abs(stored_fraction - float(prompt_fraction)) > 1e-9:
            return False
        if str(record.get("benchmark_type", "")) != benchmark_type:
            return False
        expected_environment = _benchmark_environment_fingerprint(
            runtime_binary,
            probe_binary_build_number(runtime_binary),
            system,
            baseline,
        )
        return record.get("environment_fingerprint") == expected_environment

    def _shortlist_for_full_benchmark(
        self,
        entry: ModelEntry,
        performance_target: str,
        drafter_key: str,
        baseline: TunedConfig,
        *,
        runtime_binary: str,
        system: SystemInfo,
        benchmark_backend: str = "",
    ) -> List[BenchmarkCandidate]:
        short_record = app_settings.get_performance_tuning_result(
            entry.path,
            performance_target,
            "fast",
            drafter_key,
            benchmark_backend,
        )
        if not isinstance(short_record, dict):
            return []
        expected_environment = _benchmark_environment_fingerprint(
            runtime_binary,
            probe_binary_build_number(runtime_binary),
            system,
            baseline,
        )
        if short_record.get("environment_fingerprint") != expected_environment:
            return []
        try:
            if int(short_record.get("desired_context", 0) or 0) != int(baseline.ctx):
                return []
            stat = entry.path.stat()
            if int(short_record.get("model_size", 0) or 0) not in (0, stat.st_size):
                return []
            if int(short_record.get("model_mtime_ns", 0) or 0) not in (
                0,
                stat.st_mtime_ns,
            ):
                return []
        except (OSError, TypeError, ValueError):
            return []
        return list(
            shortlist_candidates_from_record(
                short_record,
                baseline_candidate(baseline),
                maximum=6,
            )
        )

    def _update_benchmark_button(self, profile: Optional[ModelProfile] = None) -> None:
        button = getattr(self, "_btn_benchmark", None)
        if button is None:
            return
        entry = self._current_entry
        eligible = bool(
            entry is not None
            and self._system is not None
            and self._benchmark_thread is None
            and not self._benchmark_locked_states
            and not self._servers
        )
        if eligible and entry is not None:
            profile = profile or match_profile(
                entry.name,
                self._profiles,
                getattr(entry, "architecture", ""),
            )
            eligible = self._is_benchmarkable_entry(entry, profile)
        button.setEnabled(eligible)

    def _set_benchmark_controls_locked(self, locked: bool) -> None:
        controls: List[QWidget] = [
            self._model_list,
            self._model_tree,
            self._btn_list_view,
            self._btn_tree_view,
            self._btn_models_folder,
            self._btn_refresh,
            self._btn_update,
            self._btn_settings,
            self._fork_combo,
            self._btn_fork_folder,
            self._perf_combo,
            self._mode_combo,
            self._gpu_combo,
            self._chk_mmproj_cpu,
            self._chk_ngram,
            self._chk_prompt_cache,
            self._sp_prompt_cache_mib,
            self._chk_thinking,
            self._cb_mmproj,
            self._cb_draft,
            self._setting_profile_combo,
            self._btn_rename_setting_profile,
            self._btn_expert,
            self._btn_diagnose,
            self._btn_benchmark,
            self._btn_performance_analysis,
            self._btn_launch,
            self._expert_panel,
            self._host_edit,
            self._port_edit,
            self._port_offset_combo,
        ]
        if locked:
            if self._benchmark_locked_states:
                return
            self._benchmark_locked_states = {
                widget: widget.isEnabled() for widget in controls
            }
            for widget in controls:
                widget.setEnabled(False)
            return
        states = self._benchmark_locked_states
        self._benchmark_locked_states = {}
        for widget, enabled in states.items():
            try:
                widget.setEnabled(enabled)
            except RuntimeError:
                pass
        # Do not trust a stale pre-benchmark widget snapshot for the primary
        # action. Keep Launch disabled until the worker thread really exits.
        self._enable_launch_when_ocr_idle()

    @staticmethod
    def _benchmark_snapshot(
        cfg: TunedConfig, *, validated_exact_context: bool = False
    ) -> dict:
        """Translate a measured winner into the mode-scoped Expert schema."""
        extras_in = list(cfg.extra_cli_flags or [])
        reasoning, think_budget, leftovers = ExpertPanel._parse_reasoning_from_extras(
            extras_in
        )
        modeled = {
            "--jinja",
            "--verbose",
            "--metrics",
            "--slots",
            "--reasoning-preserve",
        }
        values = {
            "ctx": int(cfg.ctx),
            "cache_k": str(cfg.cache_k),
            "cache_v": str(cfg.cache_v),
            "ngl": int(cfg.ngl),
            "n_cpu_moe": int(cfg.n_cpu_moe or 0),
            "threads": int(cfg.threads),
            "batch_threads": int(cfg.batch_threads),
            "batch": int(cfg.batch),
            "ubatch": int(cfg.ubatch),
            "flash_attn": bool(cfg.flash_attn),
            "load_mode": effective_load_mode(cfg) or "auto",
            "jinja": "--jinja" in extras_in,
            "verbose": "--verbose" in extras_in,
            "metrics_enabled": bool(cfg.metrics_enabled),
            "slots_api_enabled": bool(cfg.slots_api_enabled),
            "numa": cfg.numa or "off",
            "rope_scaling": bool(cfg.rope_scaling),
            "rope_factor": float(cfg.rope_scale_factor or 1.0),
            "temperature": float(cfg.sampling.get("temperature", 0.7)),
            "top_k": int(cfg.sampling.get("top_k", 40)),
            "top_p": float(cfg.sampling.get("top_p", 0.9)),
            "min_p": float(cfg.sampling.get("min_p", 0.05)),
            "repeat_penalty": float(cfg.sampling.get("repeat_penalty", 1.05)),
            "presence_penalty": float(cfg.sampling.get("presence_penalty", 0.0)),
            "reasoning": reasoning,
            "think_budget": int(think_budget),
            "reasoning_preserve": "--reasoning-preserve" in extras_in,
            "parallel_enabled": True,
            "parallel_count": 1,
            "draft_n_max": int(cfg.draft_n_max or 0),
            "extras": " ".join(flag for flag in leftovers if flag not in modeled),
        }
        return {
            # A real server can prove a context above the conservative static
            # ceiling. Store that exact winner as manual so a later auto
            # recascade cannot silently clamp the validated value back down.
            "mode": "manual" if validated_exact_context else "auto",
            "pins": {"user_ctx": int(cfg.ctx), "force_n_parallel": 1},
            "values": values,
            "source": (
                "measured-real-context-validation"
                if validated_exact_context
                else "measured-performance-test"
            ),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _start_performance_tuning(self) -> None:
        self._prune_dead_servers()
        entry = self._current_entry
        if entry is None or self._system is None:
            QMessageBox.information(
                self,
                "Performance test",
                "Select a model and wait for hardware detection.",
            )
            return
        if self._benchmark_thread is not None:
            QMessageBox.information(
                self, "Performance test", "A performance test is already running."
            )
            return
        if self._servers:
            QMessageBox.warning(
                self,
                "Stop running servers first",
                "Performance measurements need uncontaminated RAM, VRAM, and GPU "
                "time. Stop all AutoTuner servers before starting the test.",
            )
            return

        try:
            hw_thread = getattr(self, "_hw_detect_thread", None)
            hardware_busy = self._sysinfo_busy or (
                hw_thread is not None and hw_thread.isRunning()
            )
        except RuntimeError:
            hardware_busy = self._sysinfo_busy
        if hardware_busy:
            QMessageBox.information(
                self,
                "Hardware refresh in progress",
                "Wait for the current hardware refresh to finish, then start the "
                "performance test. This avoids overlapping GPU probes.",
            )
            return

        profile = match_profile(
            entry.name, self._profiles, getattr(entry, "architecture", "")
        )
        self._update_benchmark_button(profile)
        if not self._btn_benchmark.isEnabled():
            QMessageBox.information(
                self,
                "Model is not benchmarkable",
                "The automatic performance test supports normal llama-server "
                "text/chat models. Diffusion, OCR-only, standalone drafters, and "
                "embedding runners use different workloads and are excluded.",
            )
            return

        preview = self._effective_config(entry, profile)
        if preview is None:
            QMessageBox.warning(
                self, "Performance test", "AutoTuner could not build a baseline."
            )
            return
        upper_context = min(
            10_000_000,
            max(
                int(preview.ctx),
                int(entry.native_context or 0),
                int(profile.max_context or 0),
                int(profile.rope_scale_max_ctx or 0),
                512,
            ),
        )
        benchmarkable_count = sum(
            1
            for candidate in self._all_entries
            if self._is_benchmarkable_entry(
                candidate,
                match_profile(
                    candidate.name,
                    self._profiles,
                    getattr(candidate, "architecture", ""),
                ),
            )
        )
        runtime_options = self._performance_runtime_options()
        if not runtime_options:
            QMessageBox.warning(
                self,
                "No llama build found",
                "No runnable llama-server build is available for the performance test.",
            )
            return
        setup = _PerformanceTuneSetupDialog(
            entry.name,
            int(preview.ctx),
            upper_context,
            benchmarkable_count,
            self,
            runtime_options=runtime_options,
        )
        if setup.exec() != QDialog.DialogCode.Accepted:
            return

        selected_targets = setup.selected_targets()
        source_entries = (
            list(self._all_entries) if setup.all_models.isChecked() else [entry]
        )
        source_entries = [
            candidate
            for candidate in source_entries
            if self._is_benchmarkable_entry(
                candidate,
                match_profile(
                    candidate.name,
                    self._profiles,
                    getattr(candidate, "architecture", ""),
                ),
            )
        ]
        desired_context = setup.desired_context()
        benchmark_type = setup.benchmark_type()
        prompt_context_fraction = setup.prompt_context_fraction()
        tune_mtp = setup.tune_mtp.isChecked()
        include_external_drafters = setup.test_external_drafters.isChecked()
        enable_yarn = setup.enable_yarn.isChecked()
        real_validation = setup.real_validation.isChecked()
        all_models_selected = setup.all_models.isChecked()
        rerun_completed = setup.rerun_all_models.isChecked()
        try_best_only = setup.try_best_settings.isChecked()
        selected_runtime_options = setup.selected_runtime_options()

        jobs: List[BenchmarkSuiteJob] = []
        skipped: List[str] = []
        planning_notes: List[str] = []
        system_cache: Dict[str, SystemInfo] = {}
        runtime_resolution_cache: Dict[Tuple[str, str], Tuple[Optional[str], str]] = {}
        draft_runtime_cache: Dict[Tuple[str, str], Optional[str]] = {}
        planning_started = time.monotonic()
        self._status.showMessage(
            f"Preparing performance suite for {len(source_entries)} model(s)…"
        )
        QApplication.processEvents()

        from auto_tuner import _find_compatible_draft_server

        active_runtime = self._active_llama_binary()
        active_runtime_key = _runtime_identity(active_runtime) if active_runtime else ""

        for candidate in source_entries:
            candidate_profile = match_profile(
                candidate.name,
                self._profiles,
                getattr(candidate, "architecture", ""),
            )
            option_variants = self._benchmark_option_variants(
                candidate,
                tune_mtp=tune_mtp,
                include_external_drafters=include_external_drafters,
            )
            runtime_lanes: List[
                Tuple[_PerformanceRuntimeOption, str, str, str, SystemInfo]
            ] = []
            for runtime_option in selected_runtime_options:
                resolve_key = (
                    runtime_option.binary,
                    str(candidate_profile.server_binary or ""),
                )
                resolved = runtime_resolution_cache.get(resolve_key)
                if resolved is None:
                    resolved = self._resolve_selected_benchmark_runtime(
                        runtime_option, candidate_profile
                    )
                    runtime_resolution_cache[resolve_key] = resolved
                runtime_binary, runtime_error = resolved
                if not runtime_binary:
                    skipped.append(f"{candidate.name}: {runtime_error}")
                    continue
                build_allowed, build_message, _build = check_profile_build(
                    candidate_profile, runtime_binary
                )
                if build_message:
                    self._log(f"[Build compatibility] {build_message}")
                if not build_allowed:
                    skipped.append(f"{candidate.name}: {build_message}")
                    continue

                runtime_key = _runtime_identity(runtime_binary)
                try:
                    exact_system = system_cache.get(runtime_key)
                    if exact_system is None:
                        if (
                            self._system is not None
                            and active_runtime_key
                            and runtime_key == active_runtime_key
                        ):
                            exact_system = copy.deepcopy(self._system)
                            planning_notes.append(
                                f"{candidate.name}: reused startup hardware snapshot"
                            )
                        else:
                            exact_system = detect_system(runtime_binary)
                        system_cache[runtime_key] = exact_system
                except Exception as exc:
                    skipped.append(
                        f"{candidate.name} [{runtime_option.display_name}]: "
                        f"hardware probe failed ({exc})"
                    )
                    continue

                backend = _benchmark_backend_key(
                    runtime_binary,
                    exact_system,
                    runtime_option.display_name,
                )
                runtime_lanes.append(
                    (
                        runtime_option,
                        runtime_binary,
                        runtime_key,
                        backend,
                        exact_system,
                    )
                )
                if candidate.path == entry.path and runtime_key == active_runtime_key:
                    self._update_sysinfo_labels(exact_system)
                if (
                    runtime_option.backend_hint
                    and runtime_option.backend_hint != backend
                ):
                    planning_notes.append(
                        f"{runtime_option.display_name}: folder suggests "
                        f"{runtime_option.backend_hint}, device probe reports {backend}"
                    )

            for (
                runtime_option,
                selected_binary,
                selected_runtime_key,
                backend,
                exact_system,
            ) in runtime_lanes:
                runtime_label = (
                    f"{app_settings.performance_backend_label(backend)} · "
                    f"{runtime_option.display_name}"
                )
                for options in option_variants:
                    use_draft = bool(options["enable_speculative"])
                    drafter_key = str(options["drafter_key"])
                    drafter_label = str(options["drafter_label"])
                    runtime_binary = selected_binary
                    runtime_key = selected_runtime_key
                    try:
                        if gemma_draft_needs_ik_fork(
                            candidate.name, use_draft, runtime_binary
                        ):
                            raise BenchmarkFailure(
                                "selected build is too old for Gemma-4 --spec-type; "
                                "select a current mainline or compatible ik build"
                            )
                        compatibility_key = (drafter_key, runtime_key)
                        if compatibility_key in draft_runtime_cache:
                            compatible_draft_binary = draft_runtime_cache[
                                compatibility_key
                            ]
                        else:
                            selected_forks = (
                                [(runtime_option.display_name, runtime_option.root)]
                                if runtime_option.root is not None
                                else []
                            )
                            compatible_draft_binary = _find_compatible_draft_server(
                                cast(Optional[ModelEntry], options["draft_model"]),
                                runtime_binary,
                                discovered_forks=selected_forks,
                            )
                            draft_runtime_cache[compatibility_key] = (
                                compatible_draft_binary
                            )
                        if compatible_draft_binary is None:
                            raise BenchmarkFailure(
                                "selected build cannot load this draft model"
                            )
                        runtime_binary = compatible_draft_binary
                        runtime_key = _runtime_identity(runtime_binary)
                        draft_allowed, draft_message, _draft_build = (
                            check_draft_model_build(
                                cast(Optional[ModelEntry], options["draft_model"]),
                                runtime_binary,
                                target=candidate,
                            )
                        )
                        if draft_message:
                            self._log(f"[Draft compatibility] {draft_message}")
                        if not draft_allowed:
                            raise BenchmarkFailure(draft_message)
                    except Exception as exc:
                        skipped.append(
                            f"{candidate.name} [{runtime_label}, {drafter_label}]: {exc}"
                        )
                        continue

                    for target_name in selected_targets:
                        try:
                            baseline, safe_context, exact_trial = (
                                self._benchmark_config_for_target(
                                    candidate,
                                    candidate_profile,
                                    exact_system,
                                    target_name,
                                    options,
                                    desired_context=desired_context,
                                    enable_yarn=enable_yarn,
                                    real_validation=real_validation,
                                    benchmark_backend=backend,
                                    runtime_binary=runtime_binary,
                                    ignore_saved_profile=rerun_completed,
                                )
                            )
                        except Exception as exc:
                            skipped.append(
                                f"{candidate.name} [{target_name}, {runtime_label}, "
                                f"{drafter_label}]: {exc}"
                            )
                            continue

                        if all_models_selected and not rerun_completed:
                            completed_record = (
                                app_settings.get_performance_tuning_result(
                                    candidate.path,
                                    target_name,
                                    benchmark_type,
                                    drafter_key,
                                    backend,
                                )
                            )
                            if self._completed_benchmark_matches(
                                completed_record,
                                candidate,
                                baseline,
                                benchmark_type,
                                prompt_context_fraction,
                                runtime_binary=runtime_binary,
                                system=exact_system,
                            ):
                                skipped.append(
                                    f"{candidate.name} [{target_name}, {runtime_label}, "
                                    f"{drafter_label}]: already completed"
                                )
                                continue

                        candidate_plan: List[BenchmarkCandidate] = []
                        if try_best_only and benchmark_type != "fast":
                            candidate_plan = self._shortlist_for_full_benchmark(
                                candidate,
                                target_name,
                                drafter_key,
                                baseline,
                                runtime_binary=runtime_binary,
                                system=exact_system,
                                benchmark_backend=backend,
                            )
                            if candidate_plan:
                                planning_notes.append(
                                    f"{candidate.name} [{target_name}, {runtime_label}, "
                                    f"{drafter_label}]: remeasuring "
                                    f"{len(candidate_plan)} stable Quick finalists"
                                )
                            else:
                                planning_notes.append(
                                    f"{candidate.name} [{target_name}, {runtime_label}, "
                                    f"{drafter_label}]: Quick evidence missing/noisy; "
                                    "full search retained"
                                )

                        limits = _benchmark_limits_for_workload(
                            benchmark_type, prompt_context_fraction
                        )
                        runner = BenchmarkRunner(
                            model=cast(ModelEntry, options["model"]),
                            profile=candidate_profile,
                            base_config=baseline,
                            runtime_binary=runtime_binary,
                            physical_cores=exact_system.cpu_cores_physical,
                            logical_cores=exact_system.cpu_cores_logical,
                            draft_model=cast(
                                Optional[ModelEntry], options["draft_model"]
                            ),
                            use_thinking=bool(options["use_thinking"]),
                            enable_speculative=use_draft,
                            enable_ngram=bool(options["enable_ngram"]),
                            tune_draft_n_max=bool(tune_mtp and use_draft),
                            enable_prompt_cache=bool(options["enable_prompt_cache"]),
                            prompt_cache_ram_mib=cast(
                                int, options["prompt_cache_ram_mib"]
                            ),
                            limits=limits,
                            candidate_plan=candidate_plan,
                        )
                        model_key = app_settings.favorite_model_key(candidate.path)
                        key = (
                            f"{model_key}::{runtime_key}::{backend}::"
                            f"{target_name}::{drafter_key}"
                        )
                        jobs.append(
                            BenchmarkSuiteJob(
                                key=key,
                                label=(
                                    f"{candidate.name} [{target_name}] · {runtime_label} "
                                    f"· {drafter_label} · ctx {baseline.ctx:,}"
                                ),
                                performance_target=target_name,
                                runner=runner,
                                metadata={
                                    "system": exact_system,
                                    "model_key": model_key,
                                    "model_path": str(
                                        candidate.path.resolve(strict=False)
                                    ),
                                    "runtime_key": runtime_key,
                                    "runtime_label": runtime_label,
                                    "benchmark_backend": backend,
                                    "safe_context": safe_context,
                                    "exact_trial": exact_trial,
                                    "requested_context": desired_context,
                                    "benchmark_type": benchmark_type,
                                    "prompt_context_fraction": prompt_context_fraction,
                                    "drafter_key": drafter_key,
                                    "drafter_label": drafter_label,
                                    "try_best_only": bool(candidate_plan),
                                },
                            )
                        )

        planning_elapsed = time.monotonic() - planning_started
        self._log(
            f"[Performance] Prepared {len(jobs)} job(s) in {planning_elapsed:.2f}s; "
            f"resolved {len(runtime_resolution_cache)} runtime family/families and "
            f"probed {len(system_cache)} hardware backend(s)."
        )

        if not jobs:
            completed_only = bool(skipped) and all(
                message.endswith("already completed") for message in skipped
            )
            detail = "\n".join(skipped[:12]) or "No eligible model/mode profiles."
            if completed_only:
                QMessageBox.information(
                    self,
                    "Performance results are up to date",
                    f"All {len(skipped)} matching model/mode/drafter run(s) are "
                    "already completed. Nothing will be re-tested while Rerun is "
                    f"off.\n\n{detail}",
                )
            else:
                QMessageBox.warning(self, "No performance tests prepared", detail)
            return

        exact_trials = sum(bool(job.metadata.get("exact_trial")) for job in jobs)
        candidate_runs = sum(job.runner._total_runs for job in jobs)
        context_label = (
            "each model's safe auto maximum"
            if desired_context == 0
            else f"requested {desired_context:,} tokens"
        )
        warning_lines = ""
        if exact_trials:
            warning_lines += (
                f"\n\n{exact_trials} profile(s) will first try the exact requested "
                "context above the static safe estimate. An OOM only fails that "
                "profile."
            )
        if rerun_completed:
            warning_lines += (
                "\n\nRERUN RESET: after you confirm, every old Quick, Standard, "
                "and Custom run plus every measured Perform backend profile for "
                "the selected models and modes is deleted before the first server "
                "starts. Custom user profiles remain. If interrupted, only newly "
                "completed checkpoints will be visible."
            )
        if skipped:
            warning_lines += (
                f"\n\n{len(skipped)} completed/ineligible item(s) were skipped."
            )
        if planning_notes:
            warning_lines += (
                f"\n\nTry-only-best used a stable shortlist for "
                f"{sum('remeasuring' in note for note in planning_notes)} item(s); "
                "other items fall back to the full search."
            )
        if benchmark_type == "fast":
            test_label = (
                "Quick pass (≤3.125% context, 16,384-token cap, 128 decode tokens, "
                "no overall runtime cutoff)"
            )
        elif benchmark_type == "quick":
            test_label = "Standard test (12.5% context, 65,536-token prompt cap)"
        else:
            test_label = (
                f"Custom test ({prompt_context_fraction * 100:.2f}% context, "
                "no 65,536-token cap)"
            )
        confirmation = QMessageBox.question(
            self,
            "Start real performance test?",
            f"Prepared {len(jobs)} independent model/mode profile(s) at "
            f"{context_label}, with up to {candidate_runs} fresh server launches.\n\n"
            f"{test_label} measures a deterministic prompt plus the stated n_decode window and "
            "stores separate settings for every performance mode. Runs are "
            "sequential and may take hours for all models. Hardware clocks, "
            f"voltages, and power limits are never changed.{warning_lines}\n\nStart now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if confirmation != QMessageBox.StandardButton.Yes:
            return

        reset_count = 0
        if rerun_completed:
            reset_ok, reset_count = app_settings.clear_performance_campaign_data(
                [
                    (candidate.path, candidate.size_bytes, candidate.name)
                    for candidate in source_entries
                ],
                selected_targets,
            )
            if not reset_ok:
                QMessageBox.critical(
                    self,
                    "Rerun reset failed",
                    "Old performance data could not be cleared atomically. No "
                    "benchmark server was started and the previous data remains.",
                )
                return
            self._log(
                f"[Performance] Rerun reset committed before execution: removed "
                f"{reset_count} old record/profile item(s)."
            )

        suite = BenchmarkSuiteRunner(jobs)
        planned_model_count = len(
            {str(job.metadata.get("model_key", "")) for job in jobs}
        )
        worker = _PerformanceTuneWorker(
            suite,
            checkpoint_callback=self._save_performance_job_outcome,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        summary = (
            f"{test_label}: testing {planned_model_count} model(s) across "
            f"{len(selected_runtime_options)} selected llama build(s) in "
            f"{len(jobs)} model/mode/backend/drafter run(s)."
        )
        distinct_models = {str(job.metadata.get("model_key", "")) for job in jobs}
        dialog = _PerformanceTuneDialog(
            summary,
            len(jobs),
            self,
            allow_model_stop=(
                setup.all_models.isChecked() and len(distinct_models) > 1
            ),
        )
        dialog.cancel_requested.connect(self._cancel_performance_tuning)
        dialog.stop_after_model_requested.connect(
            self._stop_performance_tuning_after_model
        )
        dialog.stop_after_mode_requested.connect(
            self._stop_performance_tuning_after_mode
        )
        thread.started.connect(worker.run)
        worker.progress.connect(dialog.update_progress)
        worker.progress.connect(self._on_performance_tuning_progress)
        worker.checkpointed.connect(self._on_performance_tuning_checkpointed)
        worker.finished.connect(self._on_performance_tuning_finished)
        worker.failed.connect(self._on_performance_tuning_failed)
        worker.cancelled.connect(self._on_performance_tuning_cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(self._on_performance_tuning_thread_finished)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        first = jobs[0]
        self._benchmark_checkpoints = {}
        self._benchmark_rerun_reset = bool(rerun_completed)
        self._benchmark_thread = thread
        self._benchmark_worker = worker
        self._benchmark_dialog = dialog
        # Retain the legacy single-job fields for cleanup/tests; completion uses
        # the suite job metadata so every model/mode is saved independently.
        self._benchmark_base_config = copy.copy(first.runner.base_config)
        self._benchmark_entry = copy.copy(first.runner.model)
        self._benchmark_system = cast(SystemInfo, first.metadata["system"])
        self._set_benchmark_controls_locked(True)
        self._status.showMessage(
            f"Performance test running — {len(jobs)} model/mode profile(s)"
        )
        self._log(
            f"[Performance] Starting realistic suite with {len(jobs)} profiles, "
            f"targets={selected_targets}, desired_ctx={desired_context or 'auto'}, "
            f"test={benchmark_type} ({prompt_context_fraction:.2%} context), "
            f"YaRN={enable_yarn}, MTP-sweep={tune_mtp}, "
            f"external-drafters={include_external_drafters}, "
            f"try-best={try_best_only}, all-models={all_models_selected}, "
            f"builds={[option.display_name for option in selected_runtime_options]}, "
            f"rerun-completed={rerun_completed}, reset-items={reset_count}."
        )
        for note in planning_notes[:12]:
            self._log(f"[Performance] Plan: {note}")
        for message in skipped[:12]:
            self._log(f"[Performance] Skipped: {message}")
        dialog.show()
        thread.start()

    def _cancel_performance_tuning(self) -> None:
        worker = self._benchmark_worker
        if worker is not None:
            self._status.showMessage("Cancelling performance test…")
            worker.cancel()

    def _stop_performance_tuning_after_model(self) -> None:
        worker = self._benchmark_worker
        if worker is not None:
            self._status.showMessage(
                "Performance test will stop after the current model; completed modes are saved."
            )
            worker.stop_after_model()

    def _stop_performance_tuning_after_mode(self) -> None:
        worker = self._benchmark_worker
        if worker is not None:
            self._status.showMessage(
                "Performance test will stop after the active mode; its result will be saved."
            )
            worker.stop_after_performance_mode()

    def _save_performance_job_outcome(self, value: object) -> dict:
        """Synchronously persist one completed model/mode benchmark job.

        This method deliberately performs no Qt/widget operations: the worker
        thread calls it before starting the next job, making the atomic settings
        write the durable checkpoint boundary. Its returned payload is emitted
        to the GUI thread for logging, summaries, and fastest-mode selection.
        """
        if not isinstance(value, BenchmarkSuiteJobResult):
            return {
                "key": "",
                "saved": False,
                "error": "benchmark returned an invalid job result",
            }
        outcome = value
        job = outcome.job
        payload: dict = {
            "key": job.key,
            "job": job,
            "saved": False,
            "error": "",
        }
        if not outcome.valid or outcome.result is None:
            entry = job.runner.model
            benchmark_type = str(job.metadata.get("benchmark_type", "quick"))
            if benchmark_type not in ("fast", "quick", "custom"):
                benchmark_type = "custom"
            drafter_key = str(
                job.metadata.get("drafter_key", app_settings.NO_DRAFTER_PROFILE_KEY)
            )
            benchmark_backend = app_settings.normalise_performance_backend(
                str(job.metadata.get("benchmark_backend", ""))
            )
            runtime_key = str(job.metadata.get("runtime_key", "") or "")
            try:
                stat = entry.path.stat()
                model_size = stat.st_size
                model_mtime_ns = stat.st_mtime_ns
            except OSError:
                model_size = 0
                model_mtime_ns = 0
            error = outcome.error or "performance mode failed"
            base = job.runner.base_config
            failure_candidate = {
                "id": "job-failure",
                "label": "Benchmark job failed",
                "settings": baseline_candidate(base).settings(),
                "prompt_tps": None,
                "generation_tps": None,
                "overall_tps": None,
                "inference_s": None,
                "draft_tokens": None,
                "draft_tokens_accepted": None,
                "draft_acceptance": None,
                "score": None,
                "samples": [],
                "confirmations": 0,
                "error": error,
                "log_tail": [line for line in error.splitlines()[-20:] if line],
            }
            failure_record = {
                "schema": BENCHMARK_RECORD_SCHEMA,
                "search_schema": BENCHMARK_SEARCH_SCHEMA,
                "status": "failed",
                "error": error,
                "reason": error,
                "model_name": entry.name,
                "model_path": str(entry.path.resolve(strict=False)),
                "model_size": model_size,
                "model_mtime_ns": model_mtime_ns,
                "desired_context": int(
                    job.metadata.get("safe_context", base.ctx) or base.ctx
                ),
                "runtime_binary": job.runner.runtime_binary,
                "runtime_build": probe_binary_build_number(job.runner.runtime_binary),
                "performance_target": job.performance_target,
                "benchmark_type": benchmark_type,
                "benchmark_backend": benchmark_backend,
                "runtime_key": runtime_key,
                "runtime_label": str(job.metadata.get("runtime_label", "")),
                "drafter_key": drafter_key,
                "drafter_label": str(job.metadata.get("drafter_label", "No drafter")),
                "winner_id": "",
                "candidates": [failure_candidate],
            }
            try:
                saved = app_settings.save_performance_failure_result(
                    entry.path,
                    failure_record,
                    job.performance_target,
                    benchmark_type,
                    drafter_key,
                    benchmark_backend,
                )
            except Exception as exc:
                payload["error"] = f"settings checkpoint failed: {exc}"
                return payload
            if not saved:
                payload["error"] = "settings save failed"
                return payload
            payload.update(
                {
                    "saved": True,
                    "failed": True,
                    "error": error,
                    "entry": entry,
                    "model_path": entry.path,
                    "model_name": entry.name,
                    "performance_target": job.performance_target,
                    "benchmark_backend": benchmark_backend,
                    "runtime_key": runtime_key,
                    "runtime_label": str(job.metadata.get("runtime_label", "")),
                }
            )
            return payload

        measured = outcome.result
        entry = job.runner.model
        base = job.runner.base_config
        try:
            system = cast(SystemInfo, job.metadata["system"])
            winning_cfg = measured.winning_config(base)
            snapshot = self._benchmark_snapshot(
                winning_cfg,
                validated_exact_context=bool(job.metadata.get("exact_trial")),
            )
            environment_fingerprint = _benchmark_environment_fingerprint(
                measured.runtime_binary,
                measured.runtime_build,
                system,
                base,
            )
            snapshot["benchmark_environment"] = copy.deepcopy(environment_fingerprint)
            try:
                stat = entry.path.stat()
                record = measured.to_record(
                    model_path=str(entry.path.resolve(strict=False)),
                    model_size=stat.st_size,
                    model_mtime_ns=stat.st_mtime_ns,
                )
            except OSError:
                record = measured.to_record(
                    model_path=str(entry.path.resolve(strict=False)),
                    model_size=0,
                    model_mtime_ns=0,
                )
            benchmark_type = str(job.metadata.get("benchmark_type", "quick"))
            if benchmark_type not in ("fast", "quick", "custom"):
                benchmark_type = "custom"
            if benchmark_type == "fast":
                snapshot["source"] = "measured-quick-pass"
                snapshot["confidence"] = "provisional"
            else:
                snapshot["confidence"] = "validated"
            prompt_fraction = float(
                job.metadata.get(
                    "prompt_context_fraction",
                    job.runner.limits.prompt_context_fraction,
                )
            )
            drafter_key = str(
                job.metadata.get("drafter_key", app_settings.NO_DRAFTER_PROFILE_KEY)
            )
            benchmark_backend = app_settings.normalise_performance_backend(
                str(job.metadata.get("benchmark_backend", ""))
            )
            runtime_key = str(job.metadata.get("runtime_key", "") or "")
            if benchmark_backend:
                snapshot["benchmark_backend"] = benchmark_backend
            record["model_name"] = entry.name
            record["performance_target"] = job.performance_target
            record["environment_fingerprint"] = environment_fingerprint
            record["benchmark_type"] = benchmark_type
            record["benchmark_backend"] = benchmark_backend
            record["runtime_key"] = runtime_key
            record["runtime_label"] = str(job.metadata.get("runtime_label", ""))
            record["drafter_key"] = drafter_key
            record["drafter_label"] = str(
                job.metadata.get("drafter_label", "No drafter")
            )
            record["search_strategy"] = (
                "quick-pass"
                if benchmark_type == "fast"
                else "quick-shortlist-validation"
                if bool(job.metadata.get("try_best_only"))
                else "full-staged-search"
            )
            record["profile_confidence"] = (
                "provisional" if benchmark_type == "fast" else "validated"
            )
            record["prompt_context_fraction"] = prompt_fraction
            record["prompt_token_cap"] = job.runner.limits.max_prompt_tokens
            record["generated_token_target"] = int(job.runner.limits.generated_tokens)
            record["static_safe_context"] = int(
                job.metadata.get("safe_context", measured.desired_context) or 0
            )
            record["real_context_validation"] = bool(job.metadata.get("exact_trial"))
            record["quality_frozen"] = {
                "cache_k": winning_cfg.cache_k,
                "cache_v": winning_cfg.cache_v,
                "flash_attn": winning_cfg.flash_attn,
                "ngl": winning_cfg.ngl,
                "n_cpu_moe": winning_cfg.n_cpu_moe,
                "tensor_split": winning_cfg.tensor_split,
                "main_gpu": winning_cfg.main_gpu,
                "no_kv_offload": winning_cfg.no_kv_offload,
                "rope_scaling": winning_cfg.rope_scaling,
                "rope_factor": winning_cfg.rope_scale_factor,
            }
            record["hardware"] = {
                "os": system.os_name,
                "cpu": system.cpu_name,
                "physical_cores": system.cpu_cores_physical,
                "logical_cores": system.cpu_cores_logical,
                "total_ram_gb": system.total_ram_gb,
                "gpus": [
                    {
                        "name": gpu.name,
                        "vram_mb": gpu.total_vram_mb,
                        "runtime_backend": gpu.runtime_backend,
                        "runtime_device": gpu.runtime_device,
                    }
                    for gpu in system.gpus
                ],
            }
            winner_samples = measured.winner.samples
            record["workload_signature"] = {
                "desired_context": int(measured.desired_context),
                "benchmark_type": benchmark_type,
                "benchmark_backend": benchmark_backend,
                "runtime_key": runtime_key,
                "drafter_key": drafter_key,
                "prompt_context_fraction": prompt_fraction,
                "prompt_token_cap": job.runner.limits.max_prompt_tokens,
                "generated_token_target": int(job.runner.limits.generated_tokens),
                "prompt_tokens": sorted(
                    {int(sample.prompt_tokens) for sample in winner_samples}
                ),
                "generated_tokens": sorted(
                    {int(sample.generated_tokens) for sample in winner_samples}
                ),
            }
            saved = app_settings.save_performance_tuning_result(
                entry.name,
                entry.path,
                record,
                snapshot,
                job.performance_target,
                benchmark_type,
                drafter_key,
                benchmark_backend,
            )
        except Exception as exc:
            payload["error"] = f"settings checkpoint failed: {exc}"
            return payload

        if not saved:
            payload["error"] = "settings save failed"
            return payload

        winner = measured.winner
        payload.update(
            {
                "saved": True,
                "measured": measured,
                "entry": entry,
                "winner": winner,
                "model_key": app_settings.favorite_model_key(entry.path),
                "model_path": entry.path,
                "model_name": entry.name,
                "performance_target": job.performance_target,
                "benchmark_backend": benchmark_backend,
                "runtime_key": runtime_key,
                "runtime_label": str(job.metadata.get("runtime_label", "")),
                "prompt_tps": winner.prompt_tps,
                "decode_tps": winner.generation_tps,
                "overall_tps": winner.overall_tps,
                "desired_context": measured.desired_context,
                "workload_signature": copy.deepcopy(
                    record.get("workload_signature", {})
                ),
                "improvement": (measured.score(winner) - 1.0) * 100.0,
            }
        )
        return payload

    def _on_performance_tuning_checkpointed(self, value: object) -> None:
        if not isinstance(value, dict):
            return
        key = str(value.get("key", ""))
        if key:
            self._benchmark_checkpoints[key] = value
        if not value.get("saved"):
            error = str(value.get("error", "checkpoint failed"))
            self._log(f"[Performance] Checkpoint failed for {key or 'job'}: {error}")
            return
        if value.get("failed"):
            self._log(
                f"[Performance] Failure checkpoint saved for {key or 'job'}: "
                f"{value.get('error', 'performance mode failed')}"
            )
            return
        self._log(
            f"[Performance] Checkpoint saved {value.get('model_path')} "
            f"[{value.get('performance_target')}, "
            f"{value.get('runtime_label') or value.get('benchmark_backend')}]: "
            f"PP={float(value.get('prompt_tps', 0.0)):.2f}, "
            f"n_decode={float(value.get('decode_tps', 0.0)):.2f}, "
            f"end_to_end={float(value.get('overall_tps', 0.0)):.2f}."
        )

    def _on_performance_tuning_progress(
        self, completed: int, total: int, message: str
    ) -> None:
        self._status.showMessage(
            f"Performance test {min(completed + 1, total)}/{total}: {message}"
        )
        self._log(f"[Performance] {message}")

    def _finish_performance_tuning_ui(self) -> None:
        dialog = self._benchmark_dialog
        self._benchmark_dialog = None
        if dialog is not None:
            dialog.mark_finished()
            dialog.close()
            dialog.deleteLater()
        self._set_benchmark_controls_locked(False)

    def _on_performance_tuning_finished(self, result: object) -> None:
        if not isinstance(result, BenchmarkSuiteResult):
            self._on_performance_tuning_failed("benchmark returned an invalid result")
            return

        saved_rows: List[str] = []
        failed_rows: List[str] = []
        # model/backend key -> independently checkpointed mode measurements.
        # Automatic fastest-mode selection never compares HIP, Vulkan, CPU, or
        # another runtime and is allowed only for identical workload signatures.
        mode_measurements: Dict[str, List[dict]] = {}
        measured_model_paths: Dict[str, Tuple[Path, str, str]] = {}
        detailed: Optional[Tuple[BenchmarkSuiteJob, BenchmarkResult]] = None
        refresh_current = False
        for outcome in result.jobs:
            job = outcome.job
            checkpoint = self._benchmark_checkpoints.get(job.key)
            if checkpoint is None:
                # Direct unit/integration callers may bypass the worker signal;
                # retain the same durable save contract as the live workflow.
                checkpoint = self._save_performance_job_outcome(outcome)
                if job.key:
                    self._benchmark_checkpoints[job.key] = checkpoint

            if not checkpoint.get("saved") or checkpoint.get("failed"):
                safe_context = int(job.metadata.get("safe_context", 0) or 0)
                suffix = (
                    f" (static safe estimate: {safe_context:,})"
                    if safe_context > 0 and bool(job.metadata.get("exact_trial"))
                    else ""
                )
                error = str(
                    checkpoint.get("error")
                    or outcome.error
                    or "performance mode failed"
                )
                failed_rows.append(f"{job.label}: {error}{suffix}")
                self._log(f"[Performance] Failed {job.label}: {error}{suffix}")
                continue

            measured = cast(BenchmarkResult, checkpoint["measured"])
            entry = cast(ModelEntry, checkpoint["entry"])
            winner = measured.winner
            improvement = float(checkpoint.get("improvement", 0.0))
            benchmark_backend = app_settings.normalise_performance_backend(
                str(checkpoint.get("benchmark_backend", ""))
            )
            runtime_label = str(
                checkpoint.get("runtime_label")
                or app_settings.performance_backend_label(benchmark_backend)
            )
            saved_rows.append(
                f"{entry.name} [{job.performance_target}, {runtime_label}] "
                f"ctx {measured.desired_context:,}: "
                f"PP {winner.prompt_tps:.1f}, n_decode {winner.generation_tps:.1f} tok/s, "
                f"end-to-end {winner.overall_tps:.1f} tok/s ({improvement:+.1f}%)"
            )
            model_key = str(checkpoint["model_key"])
            runtime_key = str(checkpoint.get("runtime_key", "") or "legacy")
            measurement_key = (
                f"{model_key}::{benchmark_backend or 'other'}::{runtime_key}"
            )
            measured_model_paths[measurement_key] = (
                cast(Path, checkpoint["model_path"]),
                benchmark_backend,
                runtime_key,
            )
            mode_measurements.setdefault(measurement_key, []).append(
                {
                    "target": job.performance_target,
                    "model_name": entry.name,
                    "benchmark_backend": benchmark_backend,
                    "runtime_label": runtime_label,
                    "prompt_tps": winner.prompt_tps,
                    "decode_tps": winner.generation_tps,
                    "overall_tps": winner.overall_tps,
                    "context": measured.desired_context,
                    "workload_signature": copy.deepcopy(
                        checkpoint.get("workload_signature", {})
                    ),
                }
            )
            self._log(
                f"[Performance] Saved {entry.path} [{job.performance_target}, "
                f"{runtime_label}]: "
                f"threads={winner.candidate.threads}, "
                f"batch_threads={winner.candidate.batch_threads}, "
                f"batch={winner.candidate.batch}, ubatch={winner.candidate.ubatch}, "
                f"draft_n_max={winner.candidate.draft_n_max}, "
                f"PP={winner.prompt_tps:.2f}, n_decode={winner.generation_tps:.2f}, "
                f"end_to_end_score={measured.score(winner):.3f}."
            )
            if detailed is None:
                detailed = (job, measured)
            if (
                self._current_entry is not None
                and self._current_entry.path == entry.path
                and self._current_performance_target_name() == job.performance_target
                and self._current_performance_backend() == benchmark_backend
            ):
                refresh_current = True

        self._finish_performance_tuning_ui()

        fastest_by_model: Dict[str, dict] = {}
        incomparable_models: set[str] = set()
        for model_key, rows in mode_measurements.items():
            signatures = {
                json.dumps(row.get("workload_signature"), sort_keys=True)
                for row in rows
                if isinstance(row.get("workload_signature"), dict)
                and row.get("workload_signature")
            }
            # One mode cannot establish a cross-mode winner. Missing/different
            # signatures likewise make raw end-to-end TPS incomparable.
            if len(rows) < 2 or len(signatures) != 1:
                incomparable_models.add(model_key)
                continue
            fastest_by_model[model_key] = max(
                rows, key=lambda row: float(row.get("overall_tps", 0.0))
            )

        backend_runtime_lanes: Dict[Tuple[str, str], set[str]] = {}
        for model_path, backend, runtime_key in measured_model_paths.values():
            identity = app_settings.favorite_model_key(model_path)
            backend_runtime_lanes.setdefault((identity, backend), set()).add(
                runtime_key
            )
        active_runtime = self._active_llama_binary()
        active_runtime_key = _runtime_identity(active_runtime) if active_runtime else ""
        for measurement_key, fastest_row in fastest_by_model.items():
            model_identity = measured_model_paths.get(measurement_key)
            if model_identity is not None:
                model_path, benchmark_backend, runtime_key = model_identity
                identity = app_settings.favorite_model_key(model_path)
                sibling_runtimes = backend_runtime_lanes.get(
                    (identity, benchmark_backend), set()
                )
                # Backend-level launch preferences cannot represent two builds
                # of the same backend. Persist an unambiguous lane, or the
                # currently active build; retain all other build results only
                # as analysis evidence.
                if len(sibling_runtimes) == 1 or runtime_key == active_runtime_key:
                    app_settings.set_model_performance_target(
                        model_path,
                        str(fastest_row["target"]),
                        benchmark_backend,
                    )

        selected_fastest: Optional[str] = None
        if self._current_entry is not None:
            current_key = app_settings.favorite_model_key(self._current_entry.path)
            current_runtime = self._active_llama_binary()
            current_runtime_key = (
                _runtime_identity(current_runtime) if current_runtime else "legacy"
            )
            current_measurement_key = (
                f"{current_key}::{self._current_performance_backend() or 'other'}"
                f"::{current_runtime_key}"
            )
            current_fastest = fastest_by_model.get(current_measurement_key)
            current_incomparable_key = current_measurement_key
            if current_fastest is None:
                # Backward-compatible unit/imported records may predate the
                # backend field. Use one unambiguous legacy/other lane, but
                # never substitute a concrete HIP/Vulkan/CPU sibling.
                legacy_keys = [
                    key
                    for key, row in fastest_by_model.items()
                    if key.startswith(f"{current_key}::")
                    and str(row.get("benchmark_backend", "")) in ("", "other")
                ]
                if len(legacy_keys) == 1:
                    current_incomparable_key = legacy_keys[0]
                    current_fastest = fastest_by_model[legacy_keys[0]]
            if current_fastest is not None:
                selected_fastest = str(current_fastest["target"])
                index = self._perf_combo.findText(selected_fastest)
                if index >= 0:
                    if self._perf_combo.currentIndex() != index:
                        # The normal signal path persists and refreshes the
                        # newly selected mode-specific Expert profile.
                        self._perf_combo.setCurrentIndex(index)
                    else:
                        app_settings.set_performance_target(selected_fastest)
                        refresh_current = True
                self._log(
                    f"[Performance] Fastest measured mode for "
                    f"{self._current_entry.name}: {selected_fastest} "
                    f"({float(current_fastest['overall_tps']):.2f} end-to-end tok/s); "
                    "selected automatically."
                )
            elif current_incomparable_key in incomparable_models:
                self._log(
                    f"[Performance] No automatic fastest-mode selection for "
                    f"{self._current_entry.name}: measured workloads differed or "
                    "only one mode completed."
                )
        if refresh_current:
            self._refresh_setting_profile_selector()
            self._refresh_config_preview()

        if not saved_rows:
            self._status.showMessage(
                "Performance test finished without a saved profile."
            )
            detail = "\n".join(failed_rows[:12]) or "Every profile failed."
            self._show_long_message(
                "Performance test failed", detail, QMessageBox.Icon.Warning
            )
            return

        status = (
            f"Performance test stopped — saved {len(saved_rows)} profile(s)"
            if result.stopped_early
            else f"Performance test complete — saved {len(saved_rows)} profile(s)"
        )
        if selected_fastest:
            status += f"; fastest mode selected: {selected_fastest}"
        self._status.showMessage(status)
        if len(saved_rows) == 1 and detailed is not None:
            job, measured = detailed
            winner = measured.winner
            baseline = measured.baseline
            improvement = (measured.score(winner) - 1.0) * 100.0
            safe_context = int(job.metadata.get("safe_context", 0) or 0)
            validation = ""
            if bool(job.metadata.get("exact_trial")):
                validation = (
                    f"\nReal context validation succeeded at {measured.desired_context:,} "
                    f"tokens (static safe estimate: {safe_context:,}).\n"
                )
            selection_line = (
                f"Fastest comparable measured mode: {job.performance_target} "
                "(selected automatically)."
                if selected_fastest == job.performance_target
                else "Only one comparable mode completed; no automatic mode selection."
            )
            message = (
                f"Saved {app_settings.setting_profile_label(app_settings.performance_profile_slot(str(job.metadata.get('benchmark_backend', ''))))} "
                f"for {job.performance_target} at {measured.desired_context:,} "
                f"context.\n{validation}\n"
                f"Winner: {winner.candidate.label}\n"
                f"Threads: {winner.candidate.threads} / batch threads: "
                f"{winner.candidate.batch_threads}\n"
                f"Batch / ubatch: {winner.candidate.batch} / {winner.candidate.ubatch}\n"
                f"MTP/draft n-max: {winner.candidate.draft_n_max or 'profile auto'}\n\n"
                f"Prompt processing: {baseline.prompt_tps:.1f} → "
                f"{winner.prompt_tps:.1f} tok/s\n"
                f"n_decode: {baseline.generation_tps:.1f} → "
                f"{winner.generation_tps:.1f} tok/s\n"
                f"Measured end-to-end workload: {winner.overall_tps:.1f} tok/s "
                f"({improvement:+.1f}%)\n\n"
                f"Decision: {measured.reason}\n"
                f"{selection_line}\n"
                + (f"{result.stop_reason}\n" if result.stopped_early else "")
                + "This mode-specific profile remains editable under Expert settings."
            )
        else:
            lines = [
                f"Saved {len(saved_rows)} independent performance profile(s).",
                *([result.stop_reason] if result.stopped_early else []),
                "Speeds below are the measured winning settings for every mode;",
                "★ marks a fastest mode only when workloads are identical.",
            ]
            shown_modes = 0
            mode_order = {name: index for index, name in enumerate(list_target_names())}
            for model_key in sorted(mode_measurements, key=str.casefold):
                rows = sorted(
                    mode_measurements[model_key],
                    key=lambda row: mode_order.get(str(row.get("target", "")), 999),
                )
                if not rows:
                    continue
                lines.extend(
                    [
                        "",
                        f"{rows[0].get('model_name', model_key)} · "
                        f"{rows[0].get('runtime_label') or rows[0].get('benchmark_backend')}",
                    ]
                )
                fastest = fastest_by_model.get(model_key)
                fastest_target = str(fastest.get("target", "")) if fastest else ""
                for row in rows:
                    if shown_modes >= 20:
                        break
                    target = str(row.get("target", ""))
                    marker = "★" if fastest_target and target == fastest_target else " "
                    lines.append(
                        f"{marker} {target}: PP {float(row.get('prompt_tps', 0.0)):.1f}, "
                        f"n_decode {float(row.get('decode_tps', 0.0)):.1f}, "
                        f"end-to-end {float(row.get('overall_tps', 0.0)):.1f} tok/s "
                        f"(ctx {int(row.get('context', 0)):,})"
                    )
                    shown_modes += 1
                if fastest_target:
                    lines.append(f"  Fastest comparable: {fastest_target}")
                else:
                    lines.append(
                        "  No auto-selection: workloads differ or only one mode completed."
                    )
                if shown_modes >= 20:
                    break
            total_modes = sum(len(rows) for rows in mode_measurements.values())
            if total_modes > shown_modes:
                lines.append(
                    f"\n… and {total_modes - shown_modes} more mode result(s)."
                )
            if selected_fastest:
                lines.append(
                    f"\nSelected model now uses the fastest measured mode: "
                    f"{selected_fastest}."
                )
            if failed_rows:
                lines.extend(
                    ["", f"Failed/skipped during measurement: {len(failed_rows)}"]
                )
                lines.extend(failed_rows[:6])
            message = "\n".join(lines)
        self._show_long_message("Performance test complete", message)

    def _show_long_message(
        self,
        title: str,
        message: str,
        icon: QMessageBox.Icon = QMessageBox.Icon.Information,
    ) -> QDialog:
        """Show a multi-line report in a scrollable dialog that fits the screen.

        ``QMessageBox`` grows with its text and, for a multi-model performance
        summary, becomes taller than the display so the OK button ends up
        off-screen. This dialog keeps the text selectable, scrolls when needed
        and never exceeds roughly three quarters of the available screen.
        """
        dialog = _LongMessageDialog(title, message, icon, self)
        dialog.exec()
        return dialog

    def _on_performance_tuning_failed(self, message: str) -> None:
        self._finish_performance_tuning_ui()
        saved_count = sum(
            bool(item.get("saved")) and not bool(item.get("failed"))
            for item in self._benchmark_checkpoints.values()
        )
        suffix = (
            f" {saved_count} completed profile(s) were already checkpointed."
            if saved_count
            else ""
        )
        self._status.showMessage(f"Performance test failed.{suffix}")
        self._log(f"[Performance] Failed: {message}.{suffix}")
        QMessageBox.warning(self, "Performance test failed", message + suffix)

    def _on_performance_tuning_cancelled(self) -> None:
        self._finish_performance_tuning_ui()
        saved_count = sum(
            bool(item.get("saved")) and not bool(item.get("failed"))
            for item in self._benchmark_checkpoints.values()
        )
        if saved_count:
            message = (
                f"Performance test cancelled — {saved_count} completed profile(s) "
                "were already saved; only the active incomplete run was discarded."
            )
        else:
            message = (
                "Performance test cancelled after the rerun reset; no old measured "
                "run data was restored."
                if bool(getattr(self, "_benchmark_rerun_reset", False))
                else "Performance test cancelled; previous settings preserved."
            )
        self._status.showMessage(message)
        self._log(f"[Performance] {message}")

    def _on_performance_tuning_thread_finished(self) -> None:
        self._benchmark_thread = None
        self._benchmark_worker = None
        self._benchmark_base_config = None
        self._benchmark_entry = None
        self._benchmark_system = None
        self._benchmark_rerun_reset = False
        # This is the authoritative unlock after success, failure, or cancel.
        self._enable_launch_when_ocr_idle()

    # ------------------------------------------------------------------
    # System info — non-blocking (daemon thread → signal/slot)
    # ------------------------------------------------------------------
    def _sysinfo_async(self) -> None:
        if self._sysinfo_busy or self._benchmark_thread is not None:
            return
        # Do NOT start a concurrent detect_system() while the initial
        # _HwDetectWorker QThread is still running.  On new RDNA5 hardware the
        # WMI / PowerShell calls inside detect_system() can take longer than the
        # 6-second timer interval, and two simultaneous calls to
        # pythoncom.CoInitialize() + WMI queries reliably crash the GUI.
        try:
            hw_thread = getattr(self, "_hw_detect_thread", None)
            if hw_thread is not None and hw_thread.isRunning():
                return
        except RuntimeError:
            pass  # QThread was already deleted via deleteLater — safe to continue
        self._sysinfo_busy = True
        threading.Thread(target=self._sysinfo_bg, daemon=True).start()

    def _sysinfo_bg(self) -> None:
        """Background thread for hardware detection (runs every 6 seconds).

        IMPORTANT: never touches Qt widgets directly. The original code
        called `self._update_sysinfo_labels(s)` and `self._log(...)`
        from this thread, which crashed the app sporadically (Qt is
        thread-affine — widgets must only be touched from the GUI
        thread). We now emit signals; their slots run on the GUI thread.
        """
        import time

        try:
            start = time.monotonic()
            s = detect_system(self._active_llama_binary())
            elapsed = time.monotonic() - start
            self._sysinfo_ready.emit(s)
            self._bg_log.emit(f"[SysInfo] Refreshed ({elapsed:.1f}s)")
        except Exception as exc:
            self._bg_log.emit(f"[Warning] Sysinfo detection failed: {exc}")
        finally:
            self._sysinfo_busy = False

    def _update_sysinfo_labels(self, s: SystemInfo) -> None:
        """Update system info labels in the UI bar.

        Always updates self._system to ensure model selection and config
        preview work even if hardware detection happened after startup.
        """
        self._system = s

        # Memory display. Unified-memory systems get one capacity, not a
        # misleading RAM + VRAM double report.
        if s.has_unified_memory:
            self._vram_lbl.setText(
                f"Unified: {min(s.free_ram_gb, s.free_vram_gb):.1f} / "
                f"{s.total_ram_gb:.1f} GB accelerator-addressable"
            )
            self._ram_lbl.setText("RAM/VRAM: shared physical pool")
        elif s.total_vram_gb > 0:
            self._vram_lbl.setText(
                f"VRAM: {s.free_vram_gb:.1f} / {s.total_vram_gb:.1f} GB free"
            )
            self._ram_lbl.setText(
                f"RAM: {s.free_ram_gb:.1f} / {s.total_ram_gb:.1f} GB free"
            )
        else:
            self._vram_lbl.setText("VRAM: no GPU")
            self._ram_lbl.setText(
                f"RAM: {s.free_ram_gb:.1f} / {s.total_ram_gb:.1f} GB free"
            )

        # CPU-Anzeige
        if s.cpu_name:
            display_cpu = (
                (s.cpu_name[:40] + "...") if len(s.cpu_name) > 40 else s.cpu_name
            )
            self._cpu_lbl.setText(f"CPU: {display_cpu}")

        # GPU-Anzeige mit Utilization
        if s.gpus:
            gpu_parts = []
            for g in s.gpus:
                util = f"{g.gpu_util_percent:.0f}%" if g.gpu_util_percent > 0 else "—"
                display_name = (g.name[:25] + "...") if len(g.name) > 25 else g.name
                gpu_parts.append(f"{display_name} ({util})")
            txt = "GPU: " + ", ".join(gpu_parts)
            # Ignorierte GPUs (iGPU etc.) auch zeigen — Transparenz darüber, was
            # erkannt aber bewusst nicht für Inference verwendet wird.
            if s.ignored_gpus:
                ign_parts = []
                for g in s.ignored_gpus:
                    size = (
                        f"{g.total_vram_gb:.1f} GB"
                        if g.total_vram_mb > 0
                        else "VRAM unknown"
                    )
                    display_ign_name = (
                        (g.name[:20] + "...") if len(g.name) > 20 else g.name
                    )
                    ign_parts.append(f"{display_ign_name} ({size}, ignored)")
                txt += "  ·  " + ", ".join(ign_parts)
            self._gpu_lbl.setText(txt)
        else:
            self._gpu_lbl.setText("GPU: none")

        self._sysbar.refresh_layout()

        self._log(
            f"[SysInfo] CPU={s.cpu_name}, VRAM={s.free_vram_gb:.1f}/{s.total_vram_gb:.1f}GB, RAM={s.free_ram_gb:.1f}/{s.total_ram_gb:.1f}GB, GPU={[g.name for g in s.gpus]}"
        )

        # Keep the GPU pin dropdown in sync with the detected cards.
        self._populate_gpu_combo(s)
        self._update_benchmark_button()

    # ------------------------------------------------------------------
    # Binary resolution
    # ------------------------------------------------------------------
    def _resolve_binary(
        self, profile: ModelProfile, use_draft: bool, model_name: str
    ) -> str:
        # Gemma 4 + external drafter runs in MAINLINE since PR #23398
        # (--spec-type draft-mtp for the gemma4-assistant head; plain sibling
        # drafters are auto-detected from -md). The old unconditional
        # redirect to ik_llama.cpp silently overrode the fork dropdown and
        # broke Gemma drafting on current mainline builds (b9190+) — the
        # ik build's --help has no --spec-type, so the prune stripped
        # "draft-mtp" and the assistant head aborted at load. Now the
        # SELECTED fork is used whenever it advertises --spec-type; only a
        # genuinely old build still falls back to ik_llama.cpp.
        try:
            _, resolve, _ = _get_fork_tools()
        except Exception:
            return "llama-server"
        spec = profile.server_binary or "llama-server"
        resolved = resolve(spec)
        if not profile.server_binary and gemma_draft_needs_ik_fork(
            model_name, use_draft, resolved
        ):
            legacy = resolve("ik_llama.cpp/llama-server")
            self._log(
                "[Binary] Ausgewählter Build kennt kein --spec-type "
                "(pre-b9190) — Gemma-4-Drafter braucht dort ik_llama.cpp: "
                f"→ {legacy}"
            )
            return legacy
        self._log(f"[Binary] {spec!r} → {resolved}")
        return resolved

    # ------------------------------------------------------------------
    # Multi-server helpers
    # ------------------------------------------------------------------
    def _prune_dead_servers(self) -> None:
        """Drop entries whose process has exited from the registry.

        Keeping this tidy is what makes the port assignment "reset":
        _next_free_port() scans from the requested base port and can reuse a
        stopped/crashed server's port once it has been pruned from _servers.
        """
        live = []
        dead = []
        for server in self._servers:
            proc = server.get("proc")
            if proc is not None and proc.is_running():
                live.append(server)
            else:
                dead.append(server)
        self._servers = live
        for server in dead:
            code = server.get("proc").returncode() if server.get("proc") else None
            self._fail_control_record(
                server,
                f"llama-server exited while loading the requested model (code {code}).",
            )
            if server is self._ocr_server_record:
                if server.get("ocr_started") and self._ocr_worker is not None:
                    self._ocr_worker.cancel()
                else:
                    self._fail_pending_ocr(
                        server,
                        f"llama-server exited while loading the OCR model (code {code}).",
                    )

    def _requested_start_port(self, base_port: int, offset: int) -> int:
        """Return the user's requested first port before collision probing.

        Deliberately do *not* add ``len(self._servers)`` here. A manually
        entered port should stay literal; _next_free_port() is responsible for
        moving to the next port only when that exact port is already occupied.
        """
        return base_port + offset

    def _next_free_port(self, host: str, base: int) -> int:
        """Return the lowest base+N not used by a live server or another app.

        Walks base, base+1, base+2… skipping ports already claimed by one
        of our running servers AND ports an unrelated process is listening
        on (so we never collide with something outside the AutoTuner).
        """
        import socket

        used = {int(s.get("port", -1)) for s in self._servers}

        def _port_busy(p: int) -> bool:
            if p in used:
                return True
            # Probe: can we bind? If not, something else holds it.
            probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
                # Windows SO_REUSEADDR binds next to an ACTIVE listener and
                # would report a taken port as free; exclusive use fails
                # with WSAEADDRINUSE as intended.
                exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
                if os.name == "nt" and exclusive is not None:
                    sk.setsockopt(socket.SOL_SOCKET, exclusive, 1)
                else:
                    sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sk.bind((probe_host, p))
                    return False
                except OSError:
                    return True

        port = base
        # Cap the search so a misconfigured host can't loop forever.
        for _ in range(64):
            if not _port_busy(port):
                return port
            port += 1
        return base  # give up gracefully — caller still tries

    def _choose_gpu_for_launch(
        self,
        cfg: TunedConfig,
        entry: ModelEntry,
        runtime_binary: Optional[str] = None,
    ) -> Tuple[Optional[GPUInfo], Optional[str]]:
        """Pick which GPU a new server should target, given live VRAM use.

        Returns ``(gpu_or_None, refusal_message_or_None)``.

        Re-detects hardware so the free-VRAM figures reflect models that
        earlier launches already loaded (the OS reports the real residency,
        not our estimate). Then:
          * estimates this model's GPU footprint (weights on GPU + KV in
            VRAM + vision + draft),
          * picks the GPU with the most free VRAM that can still hold it,
          * if none can, returns a human-readable refusal so the caller can
            stop and tell the user instead of piling onto a full card.

        On single-GPU / CPU-only systems it returns ``(None, None)`` — the
        existing tensor-split / env logic in compute_config already handles
        placement and there is nothing to balance.
        """
        # Re-detect so "free" reflects already-loaded servers.
        try:
            fresh = detect_system(runtime_binary or self._active_llama_binary())
            if fresh is not None and fresh.gpus:
                self._system = fresh
        except Exception as exc:
            self._log(
                f"[Balance] Live GPU re-detect failed ({exc}); using cached info."
            )

        sysinfo = self._system
        if sysinfo is None or not sysinfo.gpus:
            return None, None

        # Footprint this model wants on a GPU. For MoE/hybrid the experts on
        # CPU don't count; model_vram already excludes them. KV that lives in
        # VRAM + vision + draft are all GPU-resident.
        footprint_gb = (
            float(cfg.estimated_model_vram_gb)
            + float(cfg.kv_vram_gb)
            + float(cfg.vision_vram_gb)
            + float(cfg.draft_vram_gb)
            + float(cfg.runtime_vram_overhead_gb)
            + float(getattr(cfg, "compute_vram_gb", 0.0) or 0.0)
            + float(cfg.batch_vram_overhead_gb)
            + float(cfg.recurrent_state_vram_gb)
        )
        # A little breathing room so we don't fill a card to the last MB.
        SAFETY_GB = 1.0
        need = footprint_gb + SAFETY_GB

        # Single GPU: let the AutoTuner's own placement decide. compute_config
        # has ALREADY guaranteed the footprint fits within free VRAM (minus its
        # mode-specific safety band), so a flat +1 GB refusal here only
        # double-counts headroom and — worse — flips on low-VRAM cards (e.g. an
        # 8 GB 3060 Ti): throughput/balanced pack more expert layers onto the
        # GPU than safe does, producing a larger footprint that trips the flat
        # 1 GB margin even though the tuner planned a perfectly runnable
        # hybrid config. The paradoxical result was that the most aggressive
        # modes got blocked hardest on exactly the small GPUs that need them.
        #
        # We therefore only hard-refuse in the ONE case that would actually
        # crash llama-server: a FULLY-offloaded model whose GPU footprint
        # exceeds free VRAM (→ OOM, no CPU fallback). For any hybrid/offloaded
        # config (MoE with CPU-resident experts via --n-cpu-moe, or a dense
        # model with partial -ngl) the server spills the overflow to CPU and
        # runs fine — that graceful fallback IS the whole point of the tuner's
        # placement pass, so refusing it here would defeat it.
        if len(sysinfo.gpus) == 1:
            g = sysinfo.gpus[0]
            full_off = bool(getattr(cfg, "full_offload", False))
            if full_off and g.free_vram_gb < need:
                # `need` = model weights + KV-in-VRAM + the conservative
                # +SAFETY_GB compute-buffer margin. A full_off model whose
                # actual GPU footprint (weights + VRAM-resident KV) fits but
                # whose +margin tips over free VRAM still RUNS: the compute
                # buffer is small and compute_config already reserved
                # FULL_OFF_HEADROOM_GB for the full-offload decision. This
                # was the gemma-4-12b / low-VRAM-card report: an 8 GB box
                # with 7.4 GB free refused a 6.5–7.2 GB model (weights fit,
                # but weights + 1 GB margin did not). Only hard-refuse when
                # the model's real footprint exceeds free VRAM — the one
                # case with no CPU fallback (full_off pins every layer to
                # GPU, so an overflow OOMs instead of spilling).
                hard_footprint = (
                    float(cfg.estimated_model_vram_gb)
                    + float(cfg.kv_vram_gb)
                    + float(cfg.vision_vram_gb)
                    + float(cfg.draft_vram_gb)
                    + float(cfg.runtime_vram_overhead_gb)
                    + float(getattr(cfg, "compute_vram_gb", 0.0) or 0.0)
                    + float(cfg.batch_vram_overhead_gb)
                    + float(cfg.recurrent_state_vram_gb)
                )
                if g.free_vram_gb >= hard_footprint:
                    self._log(
                        f"[Balance] {g.name}: full-offload footprint "
                        f"{hard_footprint:.1f} GB fits in "
                        f"{g.free_vram_gb:.1f} GB free (compute-buffer "
                        f"margin {SAFETY_GB:.1f} GB tight) — proceeding."
                    )
                    return None, None
                return None, (
                    f"Not enough free VRAM on {g.name}: needs ≈{need:.1f} GB "
                    f"(model {footprint_gb:.1f} fully on GPU + "
                    f"{SAFETY_GB:.0f} GB headroom), only "
                    f"{g.free_vram_gb:.1f} GB free.\n\n"
                    "Stop a running server to free memory, or pick a smaller "
                    "model / lower context, or lower the context in Expert "
                    "mode so some layers spill to CPU."
                )
            return None, None

        # Multi-GPU: choose the emptiest card that can hold the footprint.
        # Sort by free VRAM descending; the first that fits wins.
        ranked = sorted(sysinfo.gpus, key=lambda g: g.free_vram_mb, reverse=True)
        for g in ranked:
            if g.free_vram_gb >= need:
                self._log(
                    f"[Balance] Targeting {g.name} "
                    f"({g.free_vram_gb:.1f} GB free ≥ {need:.1f} GB needed)."
                )
                return g, None

        # Nothing fits on a single card. Report the fullest picture so the
        # user understands why — this is the "tell me when it's full" case.
        usage = "\n".join(
            f"  • {g.name}: {g.free_vram_gb:.1f} / {g.total_vram_gb:.1f} GB free"
            for g in sysinfo.gpus
        )
        return None, (
            f"No GPU has enough free VRAM for this model.\n"
            f"Needs ≈{need:.1f} GB on one card "
            f"(model {footprint_gb:.1f} + {SAFETY_GB:.0f} GB headroom).\n\n"
            f"Current GPU usage:\n{usage}\n\n"
            "Stop one of the running servers to free memory, or choose a "
            "smaller model / lower context. (Splitting one model across both "
            "cards is handled automatically by the AutoTuner, but a second "
            "concurrent model still needs room on a single card.)"
        )

    def _pin_cfg_to_gpu(self, cfg: TunedConfig, gpu: GPUInfo) -> None:
        """Force one server onto a GPU with the selected backend's selector."""
        device_index = gpu.hip_index
        name = gpu.name
        if device_index is None:
            self._log(
                f"[Balance] {name} has no resolved runtime device index; "
                "cannot hard-pin safely."
            )
            return

        selectors, remapped = _visibility_env_for_gpus([gpu], [int(device_index)])
        if not remapped or not selectors:
            self._log(
                f"[Balance] {name} backend {gpu.runtime_backend or 'unknown'} "
                "has no safe visibility remap; keeping computed placement."
            )
            return

        cfg.env_overrides = dict(cfg.env_overrides or {})
        # Remove stale selectors from the previous auto-primary before applying
        # the one selector namespace understood by the exact backend.
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "GGML_VK_VISIBLE_DEVICES",
            "ONEAPI_DEVICE_SELECTOR",
        ):
            cfg.env_overrides.pop(key, None)
        cfg.env_overrides.update(selectors)
        cfg.main_gpu = 0  # selected device is remapped to visible index zero
        cfg.tensor_split = None
        self._last_pinned_gpu = name
        rendered = ", ".join(f"{key}={value}" for key, value in selectors.items())
        self._log(f"[Balance] Pinned to {name} ({rendered}).")

    # ------------------------------------------------------------------
    # OCR workflow
    # ------------------------------------------------------------------
    def _set_ocr_controls_locked(self, locked: bool) -> None:
        controls: List[QWidget] = [
            self._model_list,
            self._model_tree,
            self._btn_list_view,
            self._btn_tree_view,
            self._btn_models_folder,
            self._btn_refresh,
            self._btn_update,
            self._btn_settings,
            self._fork_combo,
            self._btn_fork_folder,
            self._perf_combo,
            self._mode_combo,
            self._gpu_combo,
            self._chk_mmproj_cpu,
            self._chk_ngram,
            self._chk_prompt_cache,
            self._sp_prompt_cache_mib,
            self._chk_thinking,
            self._cb_mmproj,
            self._cb_draft,
            self._setting_profile_combo,
            self._btn_rename_setting_profile,
            self._btn_expert,
            self._btn_launch,
            self._host_edit,
            self._port_edit,
            self._port_offset_combo,
        ]
        if locked:
            if self._ocr_locked_states:
                return
            self._ocr_locked_states = {
                widget: widget.isEnabled() for widget in controls
            }
            for widget in controls:
                widget.setEnabled(False)
            return
        states = self._ocr_locked_states
        self._ocr_locked_states = {}
        for widget, enabled in states.items():
            try:
                widget.setEnabled(enabled)
            except RuntimeError:
                pass

    def _enable_launch_when_ocr_idle(self) -> None:
        active = bool(
            self._ocr_thread is not None
            or self._ocr_server_record is not None
            or self._ocr_locked_states
            or self._benchmark_thread is not None
            or self._benchmark_locked_states
        )
        # Preserve the pre-v5.2.2 launcher contract: after a successful model
        # scan the primary action is available, and _launch_server() itself
        # reports a missing selection/system if needed. Requiring
        # _current_entry here caused Launch to stay disabled because scan
        # population intentionally blocks selection signals.
        self._btn_launch.setEnabled(not active)
        self._update_benchmark_button()

    def _open_ocr_workflow(self) -> None:
        entry = self._current_entry
        if entry is None or not is_ocr_model(entry):
            QMessageBox.information(self, "OCR", "Select an OCR/document model first.")
            return
        if self._ocr_thread is not None or self._ocr_server_record is not None:
            QMessageBox.information(
                self, "OCR", "An OCR job is already starting or running."
            )
            return
        if not self._vision_enabled():
            QMessageBox.warning(
                self,
                "OCR projector required",
                "OCR requires the matching mmproj and the Vision option. "
                "Select the projector and enable Vision first.",
            )
            return

        projector_warning = ocr_projector_warning(entry)
        if projector_warning:
            answer = QMessageBox.warning(
                self,
                "Legacy Unlimited-OCR projector",
                projector_warning + "\n\nContinue with the reduced tile limit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        dialog = _OcrSetupDialog(entry, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        options = dialog.options()
        profile = match_profile(
            entry.name, self._profiles, getattr(entry, "architecture", "")
        )
        preset = ocr_model_preset(entry)
        binary = self._resolve_binary(profile, False, entry.name)
        build = probe_binary_build_number(binary)
        required_build = max(
            int(getattr(profile, "min_llama_build", 0) or 0),
            preset.min_llama_build,
        )
        if required_build and build is not None and build < required_build:
            QMessageBox.critical(
                self,
                "llama.cpp build too old",
                f"{preset.label} requires llama.cpp b{required_build}+ for the "
                f"validated OCR path. The selected binary reports b{build}.\n\n"
                f"Select llama.cpp b{required_build} or newer and try again.",
            )
            return
        if required_build and build is None:
            answer = QMessageBox.warning(
                self,
                "Could not verify llama.cpp build",
                f"AutoTuner could not read the selected binary's build number. "
                f"{preset.label} is validated on b{required_build}+.\n\n"
                "Continue with this unverified binary?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        runner = OcrJobRunner(
            "",
            _clean_model_name(entry.name),
            options,
            model_name=entry.name,
            llama_build=build,
            progress=None,
        )
        worker = _OcrPrepareWorker(runner)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_ocr_progress)
        worker.prepared.connect(self._on_ocr_prepared)
        worker.failed.connect(self._on_ocr_prepare_failed)
        worker.prepared.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._on_ocr_prepare_thread_finished)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._ocr_worker = worker
        self._ocr_thread = thread
        self._ocr_prepared_runner = None
        self._ocr_prepare_error = ""
        self._ocr_progress_dialog = _OcrProgressDialog(self)
        self._ocr_progress_dialog.cancel_requested.connect(self._cancel_ocr_job)
        self._ocr_progress_dialog.update_progress(
            "prepare", 0, 0, "Preparing documents before loading the OCR model…"
        )
        self._ocr_progress_dialog.show()
        self._btn_ocr.setEnabled(False)
        self._set_ocr_controls_locked(True)
        self._log("[OCR] Preparing inputs before llama-server claims model memory.")
        thread.start()

    def _on_ocr_prepared(self, value: object) -> None:
        self._ocr_prepared_runner = cast(OcrJobRunner, value)
        job_dir, total = self._ocr_prepared_runner.prepare()
        self._log(f"[OCR] Prepared {total} page(s) in {job_dir}.")

    def _on_ocr_prepare_failed(self, message: str) -> None:
        self._ocr_prepare_error = message

    def _on_ocr_prepare_thread_finished(self) -> None:
        worker = self._ocr_worker
        cancelled = bool(worker and worker.runner.cancel_event.is_set())
        self._ocr_worker = None
        self._ocr_thread = None
        if cancelled:
            if self._ocr_prepared_runner is not None:
                self._ocr_prepared_runner.finalize_prepared(
                    "cancelled", "OCR cancelled before server launch"
                )
            self._ocr_prepared_runner = None
            self._finish_ocr_ui()
            self._set_ocr_controls_locked(False)
            if self._current_entry is not None:
                self._update_checkboxes(self._current_entry)
            self._status.showMessage("OCR cancelled.")
            return
        if self._ocr_prepare_error or self._ocr_prepared_runner is None:
            message = self._ocr_prepare_error or "OCR input preparation failed."
            self._ocr_prepared_runner = None
            self._finish_ocr_ui()
            self._set_ocr_controls_locked(False)
            if self._current_entry is not None:
                self._update_checkboxes(self._current_entry)
            self._log(f"[OCR] Preparation failed: {message}")
            QMessageBox.critical(self, "OCR preparation failed", message)
            return

        # _launch_server reads checkbox enabled states. Temporarily restore
        # the validated controls for this synchronous call, then lock the
        # exact launched configuration until the OCR job ends.
        self._set_ocr_controls_locked(False)
        record = self._launch_server()
        if record is None:
            if self._ocr_prepared_runner is not None:
                self._ocr_prepared_runner.finalize_prepared(
                    "failed", "llama-server could not be launched"
                )
            self._ocr_prepared_runner = None
            self._finish_ocr_ui()
            if self._current_entry is not None:
                self._update_checkboxes(self._current_entry)
            return
        self._set_ocr_controls_locked(True)
        record["ocr_prepared_runner"] = self._ocr_prepared_runner
        record["ocr_deadline"] = time.monotonic() + 300.0
        record["ocr_started"] = False
        self._ocr_prepared_runner = None
        self._ocr_server_record = record
        if self._ocr_progress_dialog is not None:
            self._ocr_progress_dialog.update_progress(
                "server", 0, 0, "Loading the OCR model in llama-server…"
            )
        self._log(
            f"[OCR] Waiting for the OCR server on port {record.get('port')} "
            "before page inference starts."
        )

    def _start_pending_ocr(self, record: dict) -> None:
        if record is not self._ocr_server_record or record.get("ocr_started"):
            return
        runner = record.pop("ocr_prepared_runner", None)
        if not isinstance(runner, OcrJobRunner):
            return
        record["ocr_started"] = True
        runner.configure_server(
            str(record.get("client_base_url") or record.get("base_url") or ""),
            str(record.get("alias") or record.get("model") or "ocr-model"),
            model_name=str(record.get("model") or "OCR model"),
            llama_build=record.get("llama_build"),
            server_command=record.get("command") or [],
        )
        worker = _OcrWorker(runner)
        thread = QThread(self)
        worker.moveToThread(thread)
        runner.progress = worker.progress.emit
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_ocr_progress)
        worker.finished.connect(self._on_ocr_finished)
        worker.failed.connect(self._on_ocr_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._on_ocr_thread_finished)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._ocr_worker = worker
        self._ocr_thread = thread
        if self._ocr_progress_dialog is not None:
            self._ocr_progress_dialog.update_progress(
                "ocr", 0, 0, "OCR model ready. Processing prepared pages…"
            )
        self._log("[OCR] Server ready; document pipeline started.")
        thread.start()

    def _on_ocr_progress(
        self, stage: str, current: int, total: int, message: str
    ) -> None:
        if self._ocr_progress_dialog is not None:
            self._ocr_progress_dialog.update_progress(stage, current, total, message)
        counter = f" [{current}/{total}]" if total else ""
        self._status.showMessage(f"OCR {stage}{counter}: {message}")
        self._log(f"[OCR:{stage}]{counter} {message}")

    def _cancel_ocr_job(self) -> None:
        if self._ocr_worker is not None:
            self._ocr_worker.cancel()
            # If inference is already waiting on HTTP response headers, closing
            # the job-owned server immediately guarantees the socket unblocks.
            record = self._ocr_server_record
            if record is not None and record.get("ocr_started"):
                self._stop_specific_server(record)
            self._log("[OCR] Cancellation requested.")
            return
        # Still waiting for the model to load: there is no worker yet.
        record = self._ocr_server_record
        self._ocr_server_record = None
        if record is not None:
            pending_runner = record.pop("ocr_prepared_runner", None)
            if isinstance(pending_runner, OcrJobRunner):
                pending_runner.cancel()
                pending_runner.finalize_prepared(
                    "cancelled", "OCR cancelled while loading llama-server"
                )
            self._stop_specific_server(record)
        self._finish_ocr_ui()
        self._set_ocr_controls_locked(False)
        if self._current_entry is not None:
            self._update_checkboxes(self._current_entry)
        self._status.showMessage("OCR cancelled.")

    def _on_ocr_finished(self, value: object) -> None:
        result = cast(OcrJobResult, value)
        record = self._ocr_server_record
        stop_server = False
        # The pending options were popped when the worker started; read the
        # authoritative normalized options from its runner instead.
        if self._ocr_worker is not None:
            stop_server = self._ocr_worker.runner.options.stop_server_when_done
        self._ocr_server_record = None
        if stop_server and record is not None:
            self._stop_specific_server(record)
        self._finish_ocr_ui()
        if result.cancelled:
            self._status.showMessage("OCR cancelled.")
            self._log(f"[OCR] Cancelled. Manifest: {result.manifest_path}")
            return
        self._status.showMessage(
            f"OCR complete: {result.completed_pages}/{result.total_pages} pages"
        )
        self._log(
            f"[OCR] Complete: {result.completed_pages}/{result.total_pages} pages, "
            f"{result.failed_pages} failed. Output: {result.job_dir}"
        )
        box = QMessageBox(self)
        box.setWindowTitle("OCR complete")
        box.setIcon(
            QMessageBox.Icon.Warning
            if result.failed_pages
            else QMessageBox.Icon.Information
        )
        box.setText(
            f"Processed {result.completed_pages} of {result.total_pages} pages.\n"
            f"Failed pages: {result.failed_pages}\n\n"
            f"Output:\n{result.job_dir}"
        )
        open_button = box.addButton(
            "Open output folder", QMessageBox.ButtonRole.ActionRole
        )
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is open_button:
            _open_local_folder(result.job_dir)

    def _on_ocr_failed(self, message: str) -> None:
        record = self._ocr_server_record
        stop_server = True
        if self._ocr_worker is not None:
            stop_server = self._ocr_worker.runner.options.stop_server_when_done
        self._ocr_server_record = None
        if stop_server and record is not None:
            self._stop_specific_server(record)
        self._finish_ocr_ui()
        self._status.showMessage("OCR failed.")
        self._log(f"[OCR] Failed: {message}")
        QMessageBox.critical(self, "OCR failed", message)

    def _on_ocr_thread_finished(self) -> None:
        self._ocr_worker = None
        self._ocr_thread = None
        self._ocr_server_record = None
        self._set_ocr_controls_locked(False)
        if self._current_entry is not None:
            self._update_checkboxes(self._current_entry)

    def _finish_ocr_ui(self) -> None:
        dialog = self._ocr_progress_dialog
        self._ocr_progress_dialog = None
        if dialog is not None:
            dialog.mark_finished()
            dialog.close()
            dialog.deleteLater()

    def _fail_pending_ocr(self, record: dict, message: str) -> None:
        if record is not self._ocr_server_record or record.get("ocr_started"):
            return
        pending_runner = record.pop("ocr_prepared_runner", None)
        if isinstance(pending_runner, OcrJobRunner):
            pending_runner.cancel()
            pending_runner.finalize_prepared("failed", message)
        self._finish_ocr_ui()
        self._ocr_server_record = None
        self._set_ocr_controls_locked(False)
        if self._current_entry is not None:
            self._update_checkboxes(self._current_entry)
        self._log(f"[OCR] {message}")
        QMessageBox.critical(self, "OCR could not start", message)

    def _stop_specific_server(self, record: dict) -> None:
        if record not in self._servers:
            return
        index = self._server_combo.findData(record.get("id"))
        if index >= 0:
            self._server_combo.setCurrentIndex(index)
        self._stop_server()

    # ------------------------------------------------------------------
    # Server control
    # ------------------------------------------------------------------
    def _launch_server(self, *, interactive: bool = True) -> Optional[dict]:
        self._last_launch_error = ""

        def launch_warning(title: str, message: str) -> None:
            self._last_launch_error = message or title
            self._log(f"[Launch] {title}: {message}")
            if interactive:
                QMessageBox.warning(self, title, message)

        # Multi-server: we no longer refuse when one is already running.
        # Prune any that have exited so the port counter and VRAM picture
        # are current before we plan this launch.
        self._prune_dead_servers()

        if self._current_entry is None:
            launch_warning("No model selected", "Click a model in the list first.")
            return None

        if self._system is None:
            launch_warning(
                "System info unavailable",
                "Hardware detection has not completed yet. Please wait a moment and try again.",
            )
            return None

        use_vision = self._vision_enabled()
        use_draft = self._draft_enabled()
        use_thinking = self._chk_thinking.isChecked() and self._chk_thinking.isEnabled()
        use_ngram = self._chk_ngram.isChecked() and self._chk_ngram.isEnabled()
        use_prompt_cache = (
            self._chk_prompt_cache.isChecked() and self._chk_prompt_cache.isEnabled()
        )

        # Build a copy of entry so we can control mmproj inclusion
        entry = copy.copy(self._current_entry)
        if not use_vision:
            entry.mmproj = None

        profile = match_profile(
            entry.name, self._profiles, getattr(entry, "architecture", "")
        )

        # Resolve and probe the EXACT binary this profile will launch before
        # computing placement. Profiles can switch the selected fork to CUDA,
        # SYCL, OpenVINO, or a dedicated diffusion runner; tuning against the
        # toolbar's generic llama-server would reuse the wrong device table.
        is_diffusion_cli = profile.runner == "llama-diffusion-cli" or (
            entry.is_diffusion and profile.runner != "llama-diffusion-gemma-server"
        )
        try:
            if is_diffusion_cli or profile.runner == "llama-diffusion-gemma-server":
                _, _, resolve_diffusion = _get_fork_tools()
                arch = (entry.metadata or {}).get("general.architecture")
                request = (
                    profile.server_binary or "llama-diffusion-cli"
                    if is_diffusion_cli
                    else "llama-diffusion-gemma-server"
                )
                runtime_binary = resolve_diffusion(request, arch=arch)
            else:
                runtime_binary = self._resolve_binary(profile, use_draft, entry.name)
            fresh_system = detect_system(runtime_binary)
            self._update_sysinfo_labels(fresh_system)
            self._log(
                f"[SysInfo] Launch runtime probe: {runtime_binary} → "
                f"backends={fresh_system.runtime_backends or ('CPU',)}"
            )
        except Exception as exc:
            self._log(
                f"[Warning] Exact launch-runtime hardware probe failed ({exc}); "
                "using the latest cached system information."
            )
            if is_diffusion_cli:
                runtime_binary = profile.server_binary or "llama-diffusion-cli"
            elif profile.runner == "llama-diffusion-gemma-server":
                runtime_binary = "llama-diffusion-gemma-server"
            else:
                runtime_binary = self._resolve_binary(profile, use_draft, entry.name)

        draft_for_launch = self._current_draft if use_draft else None
        try:
            from auto_tuner import _find_compatible_draft_server

            compatible_draft_binary = _find_compatible_draft_server(
                draft_for_launch, runtime_binary
            )
        except Exception as exc:
            compatible_draft_binary = None
            self._log(f"[Draft compatibility] Build search failed: {exc}")
        if compatible_draft_binary is not None and os.path.normcase(
            compatible_draft_binary
        ) != os.path.normcase(runtime_binary):
            self._log(
                "[Draft compatibility] Selected stock build lacks DFlash2; "
                f"using {compatible_draft_binary} for this launch only."
            )
            runtime_binary = compatible_draft_binary
            try:
                fresh_system = detect_system(runtime_binary)
                self._update_sysinfo_labels(fresh_system)
            except Exception as exc:
                self._log(
                    f"[Warning] DFlash2 runtime hardware probe failed ({exc}); "
                    "using the latest cached system information."
                )

        draft_allowed, draft_message, _draft_build = check_draft_model_build(
            draft_for_launch,
            runtime_binary,
            target=entry,
        )
        if draft_message:
            self._log(f"[Draft compatibility] {draft_message}")
        if not draft_allowed:
            launch_warning(
                "Compatible llama.cpp draft runtime required",
                draft_message,
            )
            return None

        model_allowed, model_message, _model_build = check_model_build(
            entry,
            runtime_binary,
        )
        if model_message:
            self._log(f"[Model compatibility] {model_message}")
        if not model_allowed:
            launch_warning(
                "Compatible llama.cpp build required",
                model_message,
            )
            return None

        # Resolve the launch config:
        #   • Expert panel open → the user is editing live; flush any
        #     pending autosave so the on-disk override matches what we
        #     launch, then use the panel's current config.
        #   • Panel closed → the per-model saved Expert override if one
        #     exists (so a hand-tuned setup is applied automatically),
        #     otherwise the AutoTuner's auto-tuned default.
        expert_open = self._config_stack.currentIndex() == 1
        if expert_open:
            # Persist live widget state, then re-cascade it against the exact
            # runtime's freshly detected capacity/backend inventory.
            self._expert_panel.flush_pending_save()
        cfg: Optional[TunedConfig] = self._effective_config(entry, profile)
        if cfg is None:
            # launch_warning keeps API/headless launches non-modal and records
            # the planner's reason as the control API's launch error.
            launch_warning(
                "Cannot fit model",
                getattr(self, "_config_error", "")
                or "No safe automatic configuration.",
            )
            return None

        if cfg.adaptive_memory:
            if cfg.memory_disable_vision:
                entry = copy.copy(entry)
                entry.mmproj = None
                use_vision = False
            if cfg.memory_disable_draft:
                use_draft = use_ngram = False
                draft_for_launch = None
            use_prompt_cache = use_prompt_cache and cfg.prompt_cache_ram_mib != 0
            self._log("[Adaptive memory] " + "; ".join(cfg.memory_adjustments))

        # ── Diffusion routing ────────────────────────────────────────
        # llama-diffusion-gemma-server (PR #24427) is a REAL persistent
        # OpenAI HTTP server (/health, /v1/chat/completions, port) → run it
        # through the normal launch path below (port / health / registry),
        # just with the dedicated binary + command builder. Everything else
        # diffusion (mainline Dream/LLaDA/RND1) is single-shot CLI: no port,
        # no /health, no registry. The binary is found in the SELECTED fork
        # (the fork dropdown points LLAMA_CPP_DIR at it).
        if is_diffusion_cli:
            self._launch_diffusion(entry, cfg, profile)
            return

        # ── Load-balancing across GPUs for a 2nd/3rd concurrent model ──
        # When at least one server is already running, re-check live VRAM
        # and steer this model onto the emptier card — or refuse outright
        # if nothing has room. The first server (none running yet) keeps the
        # AutoTuner's own placement so single-model multi-GPU splits still
        # work as before.
        #
        # EXCEPTION — explicit user pin wins: when the GPU dropdown is set
        # to a specific card (forced_gpu), that choice is ABSOLUTE. The old
        # code let this balance pass re-pin the model onto whichever card
        # had the most free VRAM, silently overriding the user's explicit
        # selection (reported: model pinned to the RX 9070 XT loaded onto
        # the R9700 because a first server left the R9700 emptier). Now we
        # keep compute_config's pin (it already targets the forced card and
        # hides every other GPU), only warn when the fit looks tight, and
        # never move the model elsewhere.
        forced_token = app_settings.get_forced_gpu()
        forced_here = None
        if forced_token and self._system is not None:
            # Same OS-robust matcher compute_config uses for force_gpu, so
            # the launch path and the placement pass agree on which card a
            # token like "9070" / "R9700" means — also when the persisted
            # token came from the other OS's driver-name style.
            forced_here = match_gpu_by_token(forced_token, self._system.gpus)
            if forced_here is None:
                self._log(
                    f"[Balance] GPU-Pin '{forced_token}' matcht keine der "
                    "erkannten Karten — falle auf Auto-Placement zurück. "
                    "Pin im GPU-Dropdown neu setzen."
                )
        if forced_here is not None:
            need = (
                float(cfg.estimated_model_vram_gb)
                + float(cfg.kv_vram_gb)
                + float(cfg.vision_vram_gb)
                + float(cfg.draft_vram_gb)
                + float(cfg.runtime_vram_overhead_gb)
                + float(getattr(cfg, "compute_vram_gb", 0.0) or 0.0)
                + float(cfg.batch_vram_overhead_gb)
                + float(cfg.recurrent_state_vram_gb)
            )
            if forced_here.free_vram_gb < need:
                self._log(
                    f"[Balance] User pin → {forced_here.name} kept, but only "
                    f"{forced_here.free_vram_gb:.1f} GB free for ≈{need:.1f} GB "
                    "footprint — expect CPU spill or OOM."
                )
            else:
                self._log(
                    f"[Balance] User pin → {forced_here.name} "
                    f"({forced_here.free_vram_gb:.1f} GB free); "
                    "skipping auto-balance."
                )
        elif self._servers:
            chosen_gpu, refusal = self._choose_gpu_for_launch(
                cfg, entry, runtime_binary
            )
            if refusal is not None:
                self._log(f"[Balance] Launch refused — {refusal.splitlines()[0]}")
                launch_warning("Not enough free VRAM", refusal)
                return None
            if chosen_gpu is not None:
                self._pin_cfg_to_gpu(cfg, chosen_gpu)
        else:
            # First model: still verify it actually fits somewhere so the
            # user gets a clear message instead of an opaque server crash.
            _gpu, refusal = self._choose_gpu_for_launch(cfg, entry, runtime_binary)
            if refusal is not None:
                # For a single multi-GPU-splittable model the per-card check
                # can be over-strict, so only hard-refuse on single-GPU /
                # CPU systems; otherwise warn and let the split proceed.
                if self._system and len(self._system.gpus) <= 1:
                    self._log(f"[Balance] Launch refused — {refusal.splitlines()[0]}")
                    launch_warning("Not enough free VRAM", refusal)
                    return None
                self._log(
                    "[Balance] First model may not fit on a single card; "
                    "letting the AutoTuner split it across GPUs."
                )

        host = self._host_edit.text().strip() or "127.0.0.1"
        # Auto-assign the port by starting at the user-requested base+offset
        # and skipping only ports that are actually taken. This preserves
        # manual entries (for example 1235 stays 1235 even if another server
        # is already running on 8080), while the default 1234 still advances
        # to 1235/1236/... when those previous AutoTuner ports are occupied.
        try:
            base_port = int(self._port_edit.text().strip())
        except ValueError:
            base_port = self._base_port
        self._base_port = base_port
        # Persist the chosen base port + offset so they survive a restart
        # (same convention as fork_path / font_size). Only committed here, at
        # launch time, so a typo that is immediately corrected is not stored.
        app_settings.set_base_port(base_port)

        try:
            offset = int(self._port_offset_combo.currentText())
        except (ValueError, AttributeError):
            offset = 0
        app_settings.set_port_offset(offset)

        start_port = self._requested_start_port(base_port, offset)
        port = self._next_free_port(host, start_port)

        # Clean alias so RooCode/clients show a readable name, not the file path
        alias = _clean_model_name(entry.name)

        if profile.runner == "llama-diffusion-gemma-server":
            # DiffusionGemma HTTP server (PR #24427): persistent OpenAI-
            # compatible server with its own binary + flag set. Build the
            # command with the dedicated builder (the gemma-server's manual
            # arg parser does NOT understand llama-server-only flags like
            # --fit/--jinja/--spec-type). The binary is resolved in the
            # selected fork (the dropdown already pointed LLAMA_CPP_DIR
            # at it); the rest — port, /health, registry — is identical to
            # a normal server, so a queryable chat endpoint is exposed.
            from tuner import build_diffusion_server_command

            arch = (entry.metadata or {}).get("general.architecture")
            gemma_server_bin = runtime_binary
            self._log(
                f"[Diffusion-Server] binary: 'llama-diffusion-gemma-server' "
                f"→ {gemma_server_bin} (arch={arch!r})"
            )
            if not self._is_runnable_binary(
                Path(gemma_server_bin)
            ) and not shutil.which(gemma_server_bin):
                self._log(
                    f"[Diffusion-Server] Binary nicht gefunden: {gemma_server_bin}"
                )
                launch_warning(
                    _tr("llama-diffusion-gemma-server not found"),
                    _tr(
                        "DiffusionGemma needs llama-diffusion-gemma-server from a "
                        "DiffusionGemma-capable build (PR #24427)."
                    )
                    + "\n\n"
                    + _tr(
                        "Select the build that contains llama-diffusion-gemma-server "
                        "in the Fork dropdown (for example d_bXXXX_vulkan_llama.cpp / "
                        "d_bXXXX_hip_llama.cpp)."
                    ),
                )
                return None
            cmd = build_diffusion_server_command(
                model=entry,
                config=cfg,
                profile=profile,
                server_binary=gemma_server_bin,
                host=host,
                port=port,
                alias=alias,
            )
        else:
            server_binary = runtime_binary
            # Locking is safe with the new split load-mode implementation, but
            # older/unprobeable GPU builds may still hit the historic Vulkan
            # host-buffer assertion. Resolve the exact binary before the veto.
            if self._system is not None and veto_unsafe_mlock(
                cfg, self._system, binary=server_binary
            ):
                self._log(
                    "[Compat] Locking load mode disabled: the selected GPU build "
                    "is older than b10151 or its version could not be probed safely."
                )
            try:
                cmd = build_command(
                    model=entry,
                    config=cfg,
                    profile=profile,
                    draft_model=self._current_draft if use_draft else None,
                    server_binary=server_binary,
                    host=host,
                    port=port,
                    extra_args=["-a", alias],
                    use_thinking=use_thinking,
                    # The dropdown governs external drafts AND embedded MTP.
                    enable_speculative=use_draft,
                    enable_ngram=use_ngram,
                    enable_prompt_cache=use_prompt_cache,
                )
            except ValueError as exc:
                self._log(f"[Memory plan] {exc}")
                launch_warning("Incompatible settings", str(exc))
                return None

        build_allowed, build_message, _detected_build = check_profile_build(
            profile, cmd[0]
        )
        if build_message:
            self._log(f"[Compat] {build_message}")
        if not build_allowed:
            launch_warning(
                "llama.cpp build too old",
                build_message,
            )
            return None

        cmd, removed_args = prepare_command_for_binary(cmd)
        for adjustment in removed_args:
            self._log(
                "[Compat] Selected llama.cpp binary required an argument "
                f"adjustment/removal: {adjustment}"
            )

        if use_prompt_cache and use_vision and "--cache-ram" in cmd:
            cache_idx = cmd.index("--cache-ram")
            if cache_idx + 1 < len(cmd) and cmd[cache_idx + 1] == "0":
                self._log(
                    "[Prompt cache] Vision cache disabled: selected llama.cpp "
                    "build is older than b10045 or its version could not be "
                    "probed safely."
                )

        # Draft angefordert, aber kein -md im finalen Kommando → laut sagen
        # statt still ohne Drafter zu starten (das war der „Gemma-Drafter
        # geht nicht"-Fall: Vision + alter Build unterdrückte Path A stumm).
        if (
            use_draft
            and self._current_draft is not None
            and "-md" not in cmd
            and "--model-draft" not in cmd
        ):
            self._log(
                "[Draft] Externer Drafter ist NICHT aktiv: der gewählte Build "
                "kombiniert -md nicht mit Vision/--mmproj (pre-b9190) oder "
                "kennt das Argument nicht. Vision-Checkbox deaktivieren oder "
                "einen aktuellen mainline-Build (--spec-type) wählen."
            )

        self._log("\n" + "─" * 60)
        self._log(f"Starting: {_redacted_command(cmd)}")
        metrics_active = "--metrics" in cmd
        slots_active = "--slots" in cmd
        self._log(
            f"Options : vision={use_vision} draft={use_draft} thinking={use_thinking} "
            f"ngram={use_ngram} prompt_cache={use_prompt_cache} "
            f"metrics={metrics_active} slots_api={slots_active} "
            f"mode={self._current_mode()}"
        )
        self._log(
            f"Server  : #{len(self._servers) + 1}  requested port {start_port} "
            f"→ assigned port {port}  "
            f"({len(self._servers)} already running)"
        )
        if cfg.env_overrides:
            for k, v in cfg.env_overrides.items():
                self._log(f"Env     : {k}={v}")

        # _bg_log.emit is thread-safe (queued into the GUI thread) — the pump
        # thread streams every server line live into the log panel below.
        proc = _TerminalProcess(
            cmd, env_overrides=cfg.env_overrides, on_output=self._bg_log.emit
        )
        try:
            proc.start()
        except FileNotFoundError:
            self._log(f"[Error] Binary not found: {cmd[0]}")
            self._log("  → Check fork selection or set LLAMA_CPP_DIR / LLAMA_SERVER")
            self._last_launch_error = f"Binary not found: {cmd[0]}"
            return None
        except OSError as exc:
            self._log(f"[Error] Could not start binary: {cmd[0]} ({exc})")
            self._log(
                "  → Check that the selected llama.cpp build matches this OS/CPU "
                "and is executable."
            )
            launch_warning(
                _tr("llama-server could not start"),
                f"{cmd[0]}\n\n{exc}\n\n"
                + _tr(
                    "Check that the selected llama.cpp build matches this operating "
                    "system and is executable."
                ),
            )
            return None

        pid = proc.proc.pid if proc.proc else "?"
        base_url = f"http://{host}:{port}"
        self._log(f"[AutoTuner] Server started — PID: {pid}")
        if proc.log_path is not None:
            self._log(
                "[AutoTuner] Server output → live below + terminal, "
                f"log file: {proc.log_path}"
            )
        else:
            self._log("[AutoTuner] Server output → separate terminal window")
        self._log(f"[AutoTuner] Web UI → {base_url}")

        # Register the new server. `_server`/`_server_base_url` always point
        # at the MOST RECENT launch so the existing status/health code keeps
        # working unchanged; the registry holds every live instance.
        record = {
            "proc": proc,
            "id": self._next_server_id,
            "port": port,
            "base_url": base_url,
            "client_base_url": client_base_url(host, port),
            "ready": False,
            "model": entry.name,
            "alias": alias,
            "command": list(cmd),
            "llama_build": probe_binary_build_number(cmd[0]),
            "gpu": getattr(self, "_last_pinned_gpu", None),
            "vram_gb": (
                float(cfg.estimated_model_vram_gb)
                + float(cfg.kv_vram_gb)
                + float(cfg.vision_vram_gb)
                + float(cfg.draft_vram_gb)
                + float(cfg.runtime_vram_overhead_gb)
                + float(getattr(cfg, "compute_vram_gb", 0.0) or 0.0)
                + float(cfg.batch_vram_overhead_gb)
                + float(cfg.recurrent_state_vram_gb)
            ),
            "metrics_enabled": metrics_active,
            "slots_api_enabled": slots_active,
            "slots_summary": "",
            "slots_next_probe": 0.0,
            # Snapshot for the external control API's status document.
            "cfg": cfg,
            "entry": entry,
            "runtime_binary": str(cmd[0]) if cmd else "",
            "profile_name": profile.display_name,
            "profile_file": profile.source_file or "",
            "fork_label": (
                str(self._fork_combo.currentText()).strip()
                if hasattr(self, "_fork_combo")
                else ""
            ),
            "performance_target": self._current_performance_target_name(),
            "mode": self._current_mode(),
            "thinking": bool(use_thinking),
        }
        self._next_server_id += 1
        self._servers.append(record)
        self._server = proc
        self._server_base_url = base_url
        self._server_ready = False
        self._last_pinned_gpu = None

        # Stop is enabled whenever ≥1 server runs. OCR holds Launch disabled
        # until its worker and validated server handoff are fully finished.
        self._enable_launch_when_ocr_idle()
        self._btn_stop.setEnabled(True)
        self._btn_stop_all.setEnabled(True)
        self._refresh_server_combo()
        self._status.showMessage(
            f"Loading model — PID {pid} — {base_url}  "
            f"({len(self._servers)} server(s) running)"
        )
        return record

    def _launch_diffusion(
        self, entry: ModelEntry, cfg: TunedConfig, profile: ModelProfile
    ) -> None:
        """Run a diffusion text model via llama-diffusion-cli (single-shot).

        Unlike the server path this does not open a port, has no /health
        and is not added to the server registry — llama-diffusion-cli takes
        a prompt, denoises, prints the result and exits. The binary is
        resolved from the fork currently selected in the toolbar dropdown
        (which has already pointed LLAMA_CPP_DIR at it), so no path is
        hard-coded: build a new diffusion-capable fork, pick it in the
        dropdown, done.
        """
        from tuner import build_diffusion_command

        # Resolve llama-diffusion-cli. A profile's server_binary may still
        # name a specific fork (fork/inner form); otherwise we look for
        # llama-diffusion-cli inside the selected LLAMA_CPP_DIR fork.
        try:
            _, _, resolve_diffusion = _get_fork_tools()
        except Exception as exc:
            self._log(f"[Diffusion] Could not load resolver: {exc}")
            QMessageBox.warning(
                self,
                "Diffusion unavailable",
                f"Could not import the diffusion binary resolver:\n{exc}",
            )
            return

        request = profile.server_binary or "llama-diffusion-cli"
        # DiffusionGemma (PR #24427) ships its own llama-diffusion-gemma-cli;
        # pass the architecture so the resolver prefers it over the generic
        # llama-diffusion-cli (which is for mainline Dream/LLaDA/RND1).
        arch = (entry.metadata or {}).get("general.architecture")
        diffusion_bin = resolve_diffusion(request, arch=arch)
        self._log(f"[Diffusion] binary: {request!r} → {diffusion_bin} (arch={arch!r})")

        if not self._is_runnable_binary(Path(diffusion_bin)) and not shutil.which(
            diffusion_bin
        ):
            self._log(f"[Diffusion] Binary not found: {diffusion_bin}")
            QMessageBox.warning(
                self,
                "llama-diffusion-cli not found",
                "Diffusion models need llama-diffusion-cli, which is not in "
                "the selected build.\n\n"
                "Pick a diffusion-capable fork in the toolbar dropdown "
                "(the one containing llama-diffusion-cli), or set "
                "LLAMA_CPP_DIR to it.",
            )
            return

        # llama-diffusion-cli is single-shot — it needs a prompt. Ask for
        # one (multi-line). Cancel aborts the launch.
        prompt, ok = QInputDialog.getMultiLineText(
            self,
            "Diffusion prompt",
            f"Prompt for {_clean_model_name(entry.name)} "
            f"(llama-diffusion-cli runs once and prints the result):",
            "",
        )
        if not ok:
            self._log("[Diffusion] Launch cancelled (no prompt).")
            return
        if not prompt.strip():
            QMessageBox.information(
                self,
                "Empty prompt",
                "A diffusion run needs a non-empty prompt.",
            )
            return

        cmd = build_diffusion_command(
            model=entry,
            config=cfg,
            profile=profile,
            diffusion_binary=diffusion_bin,
            prompt=prompt,
        )

        cmd, removed_args = prepare_command_for_binary(cmd)
        for adjustment in removed_args:
            self._log(
                "[Compat] Selected llama.cpp binary required an argument "
                f"adjustment/removal: {adjustment}"
            )

        self._log("\n" + "─" * 60)
        self._log(f"Diffusion: {' '.join(cmd)}")
        if cfg.env_overrides:
            for k, v in cfg.env_overrides.items():
                self._log(f"Env     : {k}={v}")

        # Run in a terminal window like the server path, but do NOT register
        # it as a server (no port / health / switcher entry). It exits on
        # its own when generation finishes.
        proc = _TerminalProcess(
            cmd, env_overrides=cfg.env_overrides, on_output=self._bg_log.emit
        )
        try:
            proc.start()
        except FileNotFoundError:
            self._log(f"[Error] Binary not found: {cmd[0]}")
            self._log("  → Check fork selection or set LLAMA_CPP_DIR")
            return
        except OSError as exc:
            self._log(f"[Error] Could not start binary: {cmd[0]} ({exc})")
            QMessageBox.warning(
                self,
                _tr("Diffusion binary could not start"),
                f"{cmd[0]}\n\n{exc}\n\n"
                + _tr(
                    "Check that the selected llama.cpp build matches this operating "
                    "system and is executable."
                ),
            )
            return

        pid = proc.proc.pid if proc.proc else "?"
        self._log(f"[Diffusion] Started — PID: {pid}")
        if proc.log_path is not None:
            self._log(
                f"[Diffusion] Output → live below + terminal, log file: {proc.log_path}"
            )
        else:
            self._log("[Diffusion] Output → separate terminal window")
        self._log(
            "[Diffusion] Single-shot run; the process exits when generation completes."
        )
        self._status.showMessage(
            f"Diffusion generation running — PID {pid} "
            f"({_clean_model_name(entry.name)})"
        )

    def _stop_server(self) -> None:
        """Stop the server currently selected in the switcher dropdown.

        Falls back to the most-recently-launched server when the dropdown
        has no valid selection. Removes it from the registry so its port is
        reclaimed. Disables the Stop buttons only once the last server is
        gone.
        """
        self._prune_dead_servers()
        if not self._servers:
            self._server = None
            self._server_base_url = None
            self._server_ready = False
            self._btn_stop.setEnabled(False)
            self._btn_stop_all.setEnabled(False)
            self._enable_launch_when_ocr_idle()
            self._refresh_server_combo()
            return

        # Resolve the selected server by its stable id (stored in the combo's
        # item data). Fall back to the most recent if nothing is selected.
        target_id = self._server_combo.currentData()
        record = None
        if target_id is not None:
            for r in self._servers:
                if r.get("id") == target_id:
                    record = r
                    break
        if record is None:
            record = self._servers[-1]
        if record is self._ocr_server_record:
            if self._ocr_worker is not None:
                self._ocr_worker.cancel()
            elif not record.get("ocr_started"):
                self._fail_pending_ocr(
                    record, "The OCR server was stopped before it became ready."
                )
        self._fail_control_record(
            record,
            "The API-managed model server was stopped.",
            status=409,
            code="model_stopped",
        )
        self._servers.remove(record)

        srv = record.get("proc")
        self._log(
            f"[AutoTuner] Stopping server #{record.get('id')} on port "
            f"{record.get('port')} ({record.get('model')})…"
        )
        if srv is not None:
            srv.stop()  # sends signal + waits in daemon thread

        # Re-point the "current" server at whatever is still running (if any).
        if self._servers:
            top = self._servers[-1]
            self._server = top.get("proc")
            self._server_base_url = top.get("base_url")
            self._server_ready = bool(top.get("ready"))
            self._btn_stop.setEnabled(True)
            self._btn_stop_all.setEnabled(True)
            self._status.showMessage(
                f"Server stopped — {len(self._servers)} still running."
            )
        else:
            self._server = None
            self._server_base_url = None
            self._server_ready = False
            self._btn_stop.setEnabled(False)
            self._btn_stop_all.setEnabled(False)
            self._status.showMessage("Server stopped.")
        self._enable_launch_when_ocr_idle()
        self._refresh_server_combo()
        self._log("[AutoTuner] Stop signal sent.")

    def _stop_all_clicked(self) -> None:
        """User pressed “Stop all”: terminate every running server."""
        self._prune_dead_servers()
        n = len(self._servers)
        if n == 0:
            self._refresh_server_combo()
            return
        self._log(f"[AutoTuner] Stopping all {n} server(s)…")
        self._stop_all_servers()
        self._btn_stop.setEnabled(False)
        self._btn_stop_all.setEnabled(False)
        self._enable_launch_when_ocr_idle()
        self._refresh_server_combo()
        self._status.showMessage(f"Stopped all {n} server(s).")

    def _refresh_server_combo(self) -> None:
        """Repopulate the switcher dropdown from the live registry.

        Preserves the current selection (by server id) when possible.
        """
        combo = getattr(self, "_server_combo", None)
        if combo is None:
            return
        prev_id = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for r in self._servers:
            gpu = r.get("gpu")
            ready = "✓" if r.get("ready") else "…"
            label = (
                f"#{r.get('id')}  :{r.get('port')}  {ready}  "
                f"{_clean_model_name(str(r.get('model', '?')))}"
            )
            slots_summary = r.get("slots_summary")
            if slots_summary:
                label += f"  slots {slots_summary}"
            if gpu:
                label += f"  [{gpu}]"
            combo.addItem(label, r.get("id"))
        # Restore prior selection, else default to the most recent.
        if prev_id is not None:
            idx = combo.findData(prev_id)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            elif combo.count() > 0:
                combo.setCurrentIndex(combo.count() - 1)
        elif combo.count() > 0:
            combo.setCurrentIndex(combo.count() - 1)
        combo.blockSignals(False)
        self._update_benchmark_button()

    def _toggle_log_panel(self) -> None:
        """Fully retract or restore the bottom info panel in one click."""
        split = getattr(self, "_main_split", None)
        if split is None:
            return
        if self._btn_toggle_log.isChecked():
            # Restore: give the log panel a sensible share again.
            total = sum(split.sizes()) or 800
            split.setSizes([int(total * 0.7), int(total * 0.3)])
            self._btn_toggle_log.setText("▾ Log")
        else:
            # Fully collapse the log panel (size 0).
            total = sum(split.sizes()) or 800
            split.setSizes([total, 0])
            self._btn_toggle_log.setText("▸ Log")

    def _stop_all_servers(self) -> None:
        """Stop every running server (used on quit and by ‘Stop all’)."""
        if self._ocr_worker is not None:
            self._ocr_worker.cancel()
        elif self._ocr_server_record is not None:
            pending_runner = self._ocr_server_record.pop("ocr_prepared_runner", None)
            if isinstance(pending_runner, OcrJobRunner):
                pending_runner.cancel()
                pending_runner.finalize_prepared(
                    "cancelled", "OCR cancelled while stopping all servers"
                )
            self._ocr_server_record = None
            self._finish_ocr_ui()
            self._set_ocr_controls_locked(False)
        for record in self._servers:
            self._fail_control_record(
                record,
                "The API-managed model server was stopped.",
                status=409,
                code="model_stopped",
            )
            srv = record.get("proc")
            if srv is not None:
                try:
                    srv.stop()
                except Exception:
                    pass
        self._servers = []
        self._server = None
        self._server_base_url = None
        self._server_ready = False
        self._refresh_server_combo()

    # ------------------------------------------------------------------
    # Server crash detection
    # ------------------------------------------------------------------
    def _poll_server(self) -> None:
        if not self._servers:
            return

        # Detect any server that exited (crash or external close). Removing it
        # frees its port for reuse — this is what makes the counter reset when
        # a llama-server is terminated.
        still_live: List[dict] = []
        for record in self._servers:
            proc = record.get("proc")
            if proc is not None and proc.is_running():
                still_live.append(record)
            else:
                code = proc.returncode() if proc is not None else None
                self._log(
                    f"[AutoTuner] Server on port {record.get('port')} "
                    f"({record.get('model')}) exited (code {code})."
                )
                self._fail_pending_ocr(
                    record,
                    f"llama-server exited while loading the OCR model (code {code}).",
                )
                self._fail_control_record(
                    record,
                    f"llama-server exited while loading the requested model (code {code}).",
                )
        if len(still_live) != len(self._servers):
            self._servers = still_live
            if self._servers:
                top = self._servers[-1]
                self._server = top.get("proc")
                self._server_base_url = top.get("base_url")
                self._server_ready = bool(top.get("ready"))
                self._btn_stop.setEnabled(True)
                self._btn_stop_all.setEnabled(True)
                self._status.showMessage(f"{len(self._servers)} server(s) running.")
            else:
                self._server = None
                self._server_base_url = None
                self._server_ready = False
                self._btn_stop.setEnabled(False)
                self._btn_stop_all.setEnabled(False)
                self._status.showMessage("Server exited.")
            self._enable_launch_when_ocr_idle()
            self._refresh_server_combo()

        # Health-probe any not-yet-ready server so its status flips to Ready.
        for record in self._servers:
            if record.get("ready"):
                continue
            base_url = record.get("base_url")
            health_base_url = record.get("client_base_url") or base_url
            if not health_base_url:
                continue
            control_deadline = float(record.get("control_deadline", 0.0) or 0.0)
            if control_deadline and time.monotonic() >= control_deadline:
                self._fail_control_record(
                    record,
                    "llama-server did not become ready before the control API timeout.",
                    status=504,
                    code="switch_timeout",
                )
                QTimer.singleShot(0, lambda r=record: self._stop_specific_server(r))
                continue
            deadline = float(record.get("ocr_deadline", 0.0) or 0.0)
            if deadline and time.monotonic() >= deadline:
                self._fail_pending_ocr(
                    record,
                    "llama-server did not become ready for OCR within 300 seconds.",
                )
                QTimer.singleShot(0, lambda r=record: self._stop_specific_server(r))
                continue
            # The HTTP probe runs on the worker; _on_probe_result applies it.
            self._probe_worker.submit(
                {
                    "kind": "health",
                    "token": self._probe_token(record),
                    "url": str(health_base_url),
                    "verify_models": record is self._ocr_server_record,
                }
            )

        # If enabled, poll /slots at a lower cadence than the 500 ms process
        # liveness check. This keeps the server switcher useful for continuous
        # batching without hammering the local HTTP API.
        for record in self._servers:
            if record.get("ready") and record.get("slots_api_enabled"):
                self._poll_slots_endpoint(record)

    @staticmethod
    def _probe_token(record: dict) -> int:
        """Stable per-record identity for probe results (never a reused id())."""
        token = record.get("probe_token")
        if not isinstance(token, int):
            token = next(_PROBE_TOKENS)
            record["probe_token"] = token
        return token

    def _record_for_probe(self, result: dict) -> Optional[dict]:
        token = result.get("token")
        for record in self._servers:
            if record.get("probe_token") == token:
                return record
        return None

    def _on_probe_result(self, result: object) -> None:
        """Apply one worker probe result on the GUI thread."""
        if not isinstance(result, dict):
            return
        record = self._record_for_probe(result)
        if record is None:
            return  # the server exited or was stopped meanwhile
        if result.get("kind") == "slots":
            if self._apply_slots_probe(record, result):
                self._refresh_server_combo()
            return
        if record.get("ready"):
            return
        ready = bool(result.get("ready"))
        base_url = record.get("base_url")
        if ready and record is self._ocr_server_record:
            expected_alias = str(record.get("alias") or "")
            served_ids = result.get("served_ids")
            if served_ids is None:
                # /health can turn green a fraction before /v1/models is
                # queryable. Retry on the next poll instead of calling a
                # transient verification failure a hostile endpoint.
                return
            if not expected_alias or expected_alias not in served_ids:
                shown = ", ".join(served_ids) if served_ids else "none"
                self._fail_pending_ocr(
                    record,
                    "A different service answered on the OCR port "
                    f"(expected {expected_alias!r}; served models: {shown}).",
                )
                QTimer.singleShot(0, lambda r=record: self._stop_specific_server(r))
                return
        if not ready:
            return
        record["ready"] = True
        proc = record.get("proc")
        pid = proc.proc.pid if proc is not None and proc.proc else "?"
        self._log(
            f"[AutoTuner] Server ready (/health → 200) — port {record.get('port')}."
        )
        if record.get("slots_api_enabled"):
            self._log(
                f"[AutoTuner] /slots monitoring enabled — "
                f"{record.get('base_url')}/slots"
            )
        self._refresh_server_combo()  # flip the …→✓ marker in the list
        if record is self._servers[-1]:
            self._server_ready = True
            self._status.showMessage(
                f"Ready — PID {pid} — {base_url}  "
                f"({len(self._servers)} server(s) running)"
            )
        self._complete_control_record(record)
        self._start_pending_ocr(record)

    def _poll_slots_endpoint(self, record: dict) -> bool:
        """Queue a GET /slots probe for a ready server at a 2 s cadence.

        Returns True when a probe was queued. The result is applied by
        ``_apply_slots_probe`` when the worker reports back.
        """
        now = time.monotonic()
        if now < float(record.get("slots_next_probe", 0.0) or 0.0):
            return False
        record["slots_next_probe"] = now + 2.0
        base_url = record.get("client_base_url") or record.get("base_url")
        if not base_url:
            return False
        return self._probe_worker.submit(
            {"kind": "slots", "token": self._probe_token(record), "url": str(base_url)}
        )

    def _apply_slots_probe(self, record: dict, result: dict) -> bool:
        """Cache a compact /slots summary; True when the displayed text changed.

        The endpoint shape has varied between llama.cpp builds, so this accepts
        either a raw list or a dict containing a ``slots`` list and treats
        unknown slot fields as idle.
        """
        old = str(record.get("slots_summary", "") or "")
        if "error" in result:
            if not record.get("slots_error_logged"):
                self._log(
                    f"[AutoTuner] /slots not reachable on port {record.get('port')}: "
                    f"{result['error']}"
                )
                record["slots_error_logged"] = True
            return False
        payload = result.get("payload")
        slots = payload.get("slots") if isinstance(payload, dict) else payload
        if not isinstance(slots, list):
            return False

        busy_states = {"busy", "processing", "receiving", "generating", "decoding"}
        busy = 0
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            state = str(slot.get("state", slot.get("status", ""))).lower()
            if (
                bool(slot.get("is_processing"))
                or bool(slot.get("processing"))
                or state in busy_states
            ):
                busy += 1
        summary = f"{busy}/{len(slots)} busy"
        if summary != old:
            record["slots_summary"] = summary
            return True
        return False

    # ------------------------------------------------------------------
    # Log helper
    # ------------------------------------------------------------------
    def _set_internal_debug_mode(self, enabled: bool) -> None:
        self._debug_mode = bool(enabled)
        if self._debug_mode:
            os.environ["AUTOTUNER_DEBUG"] = "1"
        else:
            os.environ.pop("AUTOTUNER_DEBUG", None)
        try:
            module = sys.modules.get("auto_tuner")
            if module is None and self._debug_mode:
                import auto_tuner as module
            if module is not None:
                module.set_debug_sink(self._bg_log.emit)
                module.set_debug_mode(self._debug_mode)
        except Exception:
            pass
        if hasattr(self, "_log_panel"):
            self._log(
                f"[Debug] AutoTuner internal logging "
                f"{'enabled' if self._debug_mode else 'disabled'}."
            )

    def _log(self, msg: str) -> None:
        clean = msg.rstrip("\n")
        self._log_panel.append(clean)
        if self._app_log_path is not None:
            try:
                with self._app_log_path.open("a", encoding="utf-8") as handle:
                    stamp = datetime.now().isoformat(timespec="seconds")
                    handle.write(f"{stamp} {clean}\n")
            except OSError:
                self._app_log_path = None
        sb = self._log_panel.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------
    def _request_quit(self) -> None:
        """Exit explicitly, even when title-bar X is configured for the tray."""
        # A server-confirmation dialog needs a visible parent when Quit came
        # from the tray menu. Avoid flashing the window when no server exists.
        if not self.isVisible() and self._servers:
            self._restore_from_tray()
        self._force_quit = True
        if self.close():
            # close() accepted: the window is gone for good. Do not rely on
            # quitOnLastWindowClosed alone — a window that was already hidden
            # in the notification area is not "the last visible window" from
            # Qt's point of view, so without this the event loop kept running
            # and the process stayed behind after Quit from the tray menu.
            QApplication.quit()

    def _activate_from_other_instance(self) -> None:
        """Bring this window forward because another launch was redirected here."""
        self._log("A second AutoTuner start was redirected to this running instance.")
        if self.isHidden():
            self._restore_from_tray()
            return
        if self.isMinimized():
            self.setWindowState(
                (self.windowState() & ~Qt.WindowState.WindowMinimized)
                | Qt.WindowState.WindowActive
            )
        self.show()
        self.raise_()
        self.activateWindow()
        QApplication.alert(self)

    def closeEvent(self, a0: QCloseEvent | None) -> None:  # noqa: N802
        # An updater may be replacing files or installing dependencies. QThread.quit()
        # cannot interrupt its active worker safely, so leave the window alive until
        # completion rather than destroying a still-running thread.
        try:
            update_running = (
                self._update_thread is not None and self._update_thread.isRunning()
            )
        except RuntimeError:
            self._update_thread = None
            self._update_worker = None
            update_running = False
        if update_running:
            self._force_quit = False
            QMessageBox.information(
                self,
                "Update in progress",
                "AutoTuner is still updating. Please wait until the update finishes before closing the application.",
            )
            if a0 is not None:
                a0.ignore()
            return

        try:
            diagnostic_running = (
                self._diagnostic_thread is not None
                and self._diagnostic_thread.isRunning()
            )
        except RuntimeError:
            self._diagnostic_thread = None
            self._diagnostic_worker = None
            diagnostic_running = False
        if diagnostic_running:
            self._force_quit = False
            QMessageBox.information(
                self,
                "Metadata scan in progress",
                "AutoTuner is still writing the confirmed all-model metadata "
                "report. Please wait for it to finish before closing.",
            )
            if a0 is not None:
                a0.ignore()
            return

        # Snapshot the current window layout BEFORE any potential
        # "are you sure?" dialog, so even an Escape-out of that dialog
        # has saved state. The save itself never blocks the close.
        self._persist_window_geometry()

        # This preference applies only to an ordinary title-bar/system-menu
        # close. The dedicated Quit actions and signal handler set
        # ``_force_quit`` and continue through the normal shutdown path.
        if app_settings.get_minimize_on_close() and not self._force_quit:
            if self._hide_to_tray():
                if a0 is not None:
                    a0.ignore()
                return
            QMessageBox.warning(
                self,
                "Notification area unavailable",
                "This desktop does not currently provide a system tray, so "
                "AutoTuner cannot be hidden there. Disable the option in "
                "Settings or enable tray support in your desktop environment.",
            )
            if a0 is not None:
                a0.ignore()
            return

        # A benchmark owns a private llama-server and active HTTP connection.
        # Cancel it explicitly and wait for its finally block to stop the process;
        # destroying a live worker thread can leave the model resident in VRAM.
        benchmark_thread = self._benchmark_thread
        try:
            benchmark_running = (
                benchmark_thread is not None and benchmark_thread.isRunning()
            )
        except RuntimeError:
            benchmark_running = False
            self._benchmark_thread = None
            self._benchmark_worker = None
        if benchmark_running:
            reply = QMessageBox.question(
                self,
                "Performance test still running",
                "Cancel the active performance test and quit AutoTuner?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._force_quit = False
                if a0 is not None:
                    a0.ignore()
                return
            if self._benchmark_worker is not None:
                self._benchmark_worker.cancel()
            assert benchmark_thread is not None
            benchmark_thread.quit()
            if not benchmark_thread.wait(20000):
                QMessageBox.warning(
                    self,
                    "Performance test is still stopping",
                    "The benchmark server has not released its resources yet. "
                    "Please wait a moment and try Quit again.",
                )
                self._force_quit = False
                if a0 is not None:
                    a0.ignore()
                return
            self._benchmark_thread = None
            self._benchmark_worker = None
            self._benchmark_base_config = None
            self._benchmark_entry = None
            self._benchmark_system = None
            self._finish_performance_tuning_ui()

        # Never destroy a QThread while document conversion or an HTTP OCR
        # request is active. Give the user an explicit cancel-and-quit path,
        # then wait for the shared runner to close LibreOffice/HTTP cleanly.
        ocr_thread = self._ocr_thread
        try:
            ocr_running = ocr_thread is not None and ocr_thread.isRunning()
        except RuntimeError:
            ocr_running = False
            self._ocr_thread = None
            self._ocr_worker = None
        if ocr_running:
            reply = QMessageBox.question(
                self,
                "OCR still running",
                "Cancel the active OCR job and quit AutoTuner?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._force_quit = False
                if a0 is not None:
                    a0.ignore()
                return
            if self._ocr_worker is not None:
                self._ocr_worker.cancel()
            if self._ocr_server_record is not None:
                self._stop_specific_server(self._ocr_server_record)
            assert ocr_thread is not None
            ocr_thread.quit()
            if not ocr_thread.wait(15000):
                QMessageBox.warning(
                    self,
                    "OCR is still stopping",
                    "The OCR worker has not released its resources yet. "
                    "Please wait a moment and try Quit again.",
                )
                self._force_quit = False
                if a0 is not None:
                    a0.ignore()
                return
            self._ocr_thread = None
            self._ocr_worker = None
            self._finish_ocr_ui()
            self._set_ocr_controls_locked(False)

        # Stop periodic timers first so no new background work is started
        # while we're tearing down.  Both timers are children of self so Qt
        # would delete them anyway, but stopping them explicitly prevents a
        # slot from firing between now and the actual object deletion.
        try:
            self._sysinfo_timer.stop()
        except Exception:
            pass
        try:
            self._poll_timer.stop()
            self._probe_worker.stop()
        except Exception:
            pass

        # Guard against already-deleted QThread (deleteLater race)
        try:
            if self._scan_thread is not None and self._scan_thread.isRunning():
                self._scan_thread.quit()
                self._scan_thread.wait(2000)
        except RuntimeError:
            pass
        self._scan_thread = None

        # Clean up the initial hardware-detection thread.  If it is still
        # running (slow WMI / PowerShell on new RDNA5 hardware) we ask it to
        # stop gracefully and wait briefly.  Without this the worker emits
        # _hw_detect_done on a half-destroyed MainWindow which can segfault.
        try:
            hw_thread = getattr(self, "_hw_detect_thread", None)
            if hw_thread is not None and hw_thread.isRunning():
                hw_thread.quit()
                hw_thread.wait(3000)
        except RuntimeError:
            pass

        self._prune_dead_servers()
        if self._servers:
            n = len(self._servers)
            reply = QMessageBox.question(
                self,
                "Servers still running",
                f"Stop {n} running server(s) and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._force_quit = False
                # Closing was cancelled; restore the periodic work stopped at
                # the start of this method so the live window keeps updating.
                self._poll_timer.start(500)
                self._sysinfo_timer.start(6000)
                if a0 is not None:
                    a0.ignore()
                return

        # No later user decision can cancel shutdown. Stop accepting external
        # model switches before terminating the model processes they control.
        self._control_closing = True
        self._stop_control_api()
        if self._servers:
            self._stop_all_servers()

        self._destroy_tray_icon()
        try:
            module = sys.modules.get("auto_tuner")
            if module is not None:
                module.set_debug_sink(None)
        except Exception:
            pass
        _release_windows_native_icons()
        if a0 is not None:
            a0.accept()


# ---------------------------------------------------------------------------
def _run_model_tree_interaction_smoke(app: QApplication, settings_path: Path) -> None:
    """Exercise favorites, launch-state recovery, and folder persistence."""
    temp_state = tempfile.TemporaryDirectory(prefix="autotuner-model-tree-smoke-")
    root = Path(temp_state.name)
    entry = ModelEntry(
        path=root / "Alibaba" / "Qwen3.6" / "Qwen3.6-27B-Q4_K_M.gguf",
        name="Qwen3.6-27B-Q4_K_M",
        group="Alibaba/Qwen3.6",
        size_bytes=1 * 1024**3,
    )
    original_settings_file = app_settings._settings_file
    original_set_favorite = app_settings.set_model_favorite
    persisted: List[Tuple[Path, bool]] = []
    windows: List[MainWindow] = []
    try:
        # The smoke test must never read or modify a user's portable settings.
        app_settings._settings_file = lambda: root / "autotuner_settings.json"
        app_settings.set_model_favorite = lambda path, favorite: persisted.append(
            (path, favorite)
        )

        window = MainWindow(root, settings_path, start_background=False)
        windows.append(window)
        window._favorite_models = set()
        window._all_entries = [entry]
        window._last_scan_roots = [root]
        window._populate_list(window._all_entries)
        window._set_model_view("tree", persist=False)

        # Frozen regression for v5.2.2: scan setup disables Launch, and model
        # selection plus benchmark cleanup must deterministically restore it.
        window._system = SystemInfo(
            os_name="smoke",
            cpu_name="smoke-cpu",
            cpu_cores_physical=4,
            cpu_cores_logical=8,
            total_ram_gb=32.0,
            free_ram_gb=24.0,
        )
        window._btn_launch.setEnabled(False)
        window._show_config(entry)
        if not window._btn_launch.isEnabled():
            raise RuntimeError("model selection left Launch disabled")
        window._benchmark_thread = QThread(window)
        window._set_benchmark_controls_locked(True)
        window._benchmark_locked_states[window._btn_launch] = False
        window._set_benchmark_controls_locked(False)
        if window._btn_launch.isEnabled():
            raise RuntimeError("Launch unlocked before benchmark thread cleanup")
        window._on_performance_tuning_thread_finished()
        if not window._btn_launch.isEnabled():
            raise RuntimeError("benchmark cleanup left Launch disabled")

        favorite_root = window._model_tree.topLevelItem(0)
        vendor_folder = window._model_tree.topLevelItem(1)
        family_folder = vendor_folder.child(0)
        model_item = family_folder.child(0)
        if not (
            favorite_root.isExpanded()
            and vendor_folder.isExpanded()
            and family_folder.isExpanded()
        ):
            raise RuntimeError("model tree did not start fully expanded")

        index = window._model_tree.indexFromItem(model_item)
        option = QStyleOptionViewItem()
        option.rect = QRect(0, 0, 500, 26)
        event = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPointF(12, 13),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        handled = window._tree_favorite_delegate.editorEvent(
            event, window._model_tree.model(), option, index
        )
        key = app_settings.favorite_model_key(entry.path)
        if not handled or key not in window._favorite_models:
            raise RuntimeError("tree favorite click was not handled")
        if favorite_root.childCount() != 0 or not window._favorite_refresh_pending:
            raise RuntimeError("tree rebuilt synchronously inside the delegate event")

        app.processEvents()
        refreshed_favorites = window._model_tree.topLevelItem(0)
        if window._favorite_refresh_pending or refreshed_favorites.childCount() != 1:
            raise RuntimeError("deferred tree favorite refresh did not complete")
        if persisted != [(entry.path, True)]:
            raise RuntimeError("tree favorite state was not persisted exactly once")

        # Collapse through the same label-click production slot used by the UI,
        # then prove a view switch leaves the user's choice untouched.
        vendor_folder = window._model_tree.topLevelItem(1)
        collapsed_key = vendor_folder.data(0, _TREE_PATH_ROLE)
        window._on_tree_item_clicked(vendor_folder, 0)
        if vendor_folder.isExpanded():
            raise RuntimeError("folder label click did not collapse the branch")
        if app_settings.get_model_tree_collapsed_paths() != {collapsed_key}:
            raise RuntimeError("collapsed folder state was not persisted")
        window._set_model_view("list", persist=False)
        window._set_model_view("tree", persist=False)
        if vendor_folder.isExpanded():
            raise RuntimeError("view switch forgot the collapsed folder state")

        window.deleteLater()
        app.processEvents()
        windows.remove(window)

        # Recreate the real MainWindow to exercise the application-restart path.
        restored = MainWindow(root, settings_path, start_background=False)
        windows.append(restored)
        restored._favorite_models = set()
        restored._all_entries = [entry]
        restored._last_scan_roots = [root]
        restored._populate_list(restored._all_entries)
        restored._set_model_view("tree", persist=False)
        restored_vendor = restored._model_tree.topLevelItem(1)
        if restored_vendor.isExpanded():
            raise RuntimeError("application restart forgot the collapsed folder state")

        # Opening the branch is a manual state change too: it must clear the
        # persisted collapse so the next restart starts that branch open.
        restored._on_tree_item_clicked(restored_vendor, 0)
        if not restored_vendor.isExpanded():
            raise RuntimeError("folder label click did not expand the branch")
        if app_settings.get_model_tree_collapsed_paths():
            raise RuntimeError("expanded folder remained persisted as collapsed")

        restored.deleteLater()
        app.processEvents()
        windows.remove(restored)

        reopened = MainWindow(root, settings_path, start_background=False)
        windows.append(reopened)
        reopened._favorite_models = set()
        reopened._all_entries = [entry]
        reopened._last_scan_roots = [root]
        reopened._populate_list(reopened._all_entries)
        reopened._set_model_view("tree", persist=False)
        if not reopened._model_tree.topLevelItem(1).isExpanded():
            raise RuntimeError("application restart forgot the expanded folder state")
    finally:
        app_settings.set_model_favorite = original_set_favorite
        app_settings._settings_file = original_settings_file
        for window in windows:
            window.deleteLater()
        app.processEvents()
        temp_state.cleanup()


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    # Frozen noconsole build (PyInstaller --windowed): sys.stdout/stderr are
    # None, so a stray print()/traceback would crash. Redirect them to a
    # rotating log file in the persistent app-data dir so crashes stay
    # debuggable. (Source installs keep the terminal.)
    if getattr(sys, "frozen", False) and sys.stdout is None:
        try:
            log_dir = app_settings.app_data_dir()
            log_path = log_dir / "autotuner_console.log"
            # Keep the previous run as .1 (light rotation, text stays small).
            try:
                prev = log_dir / "autotuner_console.log.1"
                if log_path.exists():
                    log_path.replace(prev)
            except OSError:
                pass
            sys.stdout = log_path.open("a", encoding="utf-8", buffering=1)
            sys.stderr = sys.stdout
            print(
                f"\n=== AutoTuner v{VERSION} console log "
                f"{datetime.now().isoformat(timespec='seconds')} ===",
                flush=True,
            )
        except Exception:
            # Last resort: silence streams so the app still runs.
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
            sys.stderr = sys.stdout

    p = argparse.ArgumentParser(
        prog="qt_launcher", description="AutoTuner Qt GUI launcher"
    )
    p.add_argument("--models-path", default=str(_default_models_path()))
    p.add_argument("--settings-path", default=str(_default_settings_path()))
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--model-tree-smoke-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        startup_manager.AUTOSTART_ARGUMENT,
        dest="autostart",
        action="store_true",
        help="launched by the login entry; honours 'Start minimized after login'",
    )
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.model_tree_smoke_test:
        app = QApplication(sys.argv)
        _application_theme_manager(app, _bundled_resource("assets", "themes"))
        _run_model_tree_interaction_smoke(app, Path(args.settings_path))
        print(
            f"AutoTuner v{VERSION} model tree interaction smoke test OK",
            flush=True,
        )
        return

    if args.smoke_test:
        # Frozen-release CI entry point: exercises the PyInstaller bootloader,
        # imports, and bundled profile data without opening a GUI/event loop.
        profiles = load_profiles(Path(args.settings_path))
        if not profiles:
            raise RuntimeError("frozen smoke test found no bundled profiles")
        themes_dir = _bundled_resource("assets", "themes")
        if not themes_dir.is_dir():
            raise RuntimeError("frozen smoke test found no bundled themes")
        languages_dir = _bundled_resource("assets", "languages")
        if len(list(languages_dir.glob("*.json"))) < 8:
            raise RuntimeError("frozen smoke test found incomplete bundled languages")
        print(
            f"AutoTuner v{VERSION} frozen smoke test OK ({len(profiles)} profiles)",
            flush=True,
        )
        return

    # Hide the parent console on Windows when launched via python.exe. A stable
    # AppUserModelID also makes Windows use our icon for the running taskbar app
    # instead of grouping it under python.exe / the PyInstaller bootloader.
    if os.name == "nt":
        try:
            import ctypes

            set_app_id = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
            set_app_id.argtypes = [ctypes.c_wchar_p]
            set_app_id.restype = ctypes.c_long
            set_app_id("DaWasteh.AutoTuner")

            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("AutoTuner")

    # One AutoTuner per user and data folder. A second double-click on the
    # executable brings the running window back (also out of the notification
    # area) instead of starting a rival process that fights over the control
    # API port and the console log.
    guard = SingleInstanceGuard(instance_key(app_settings.app_data_dir()))
    if multiple_instances_allowed():
        print(
            f"[AutoTuner] {ALLOW_MULTIPLE_ENV} is set — single-instance guard disabled.",
            flush=True,
        )
    elif not guard.try_acquire():
        if args.autostart:
            # A login entry must never pop up an already running window.
            print(
                "[AutoTuner] AutoTuner is already running; login start ignored.",
                flush=True,
            )
            return
        if guard.notify_running_instance():
            print(
                "[AutoTuner] AutoTuner is already running — its window was "
                "brought to the front instead of starting a second instance.",
                flush=True,
            )
        else:
            print(
                "[AutoTuner] AutoTuner is already running but did not answer; "
                "not starting a second instance.",
                flush=True,
            )
        return

    manager = _application_theme_manager(app, _bundled_resource("assets", "themes"))
    selected_theme = app_settings.get_theme_id()
    applied_theme = manager.apply(app, selected_theme, app_settings.get_font_size())
    if selected_theme != applied_theme:
        app_settings.set_theme_id(applied_theme)
    for warning in manager.errors:
        print(f"[Warning] Theme ignored: {warning}")
    icon_path = _bundled_resource("assets", "AutoTuner.png")
    if icon_path.is_file():
        app_icon = QIcon(str(icon_path))
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)
    # Apply persisted font size to the WHOLE app before we build any
    # widgets — that way every QLabel / QPushButton / dropdown picks
    # up the user's chosen size on the very first paint instead of
    # flashing the Qt default and then resizing.
    try:
        base_font = app.font()
        base_font.setPointSize(app_settings.get_font_size())
        app.setFont(base_font)
    except Exception:
        pass

    window = MainWindow(
        models_path=Path(args.models_path),
        settings_path=Path(args.settings_path),
    )
    # Explicitly set it as well as QApplication's icon for consistent title-bar
    # and taskbar behavior across Windows, Linux window managers, and macOS.
    if not app.windowIcon().isNull():
        window.setWindowIcon(app.windowIcon())
    # A login entry written before v5.5.6 lacks --autostart; upgrade it for the
    # next login (only when it exactly matches this installation).
    if startup_manager.refresh_autostart_registration():
        window._log("[Autostart] Login entry updated with the --autostart marker.")
    _show_initial_window(window, args.autostart)
    guard.activate_requested.connect(window._activate_from_other_instance)
    app.aboutToQuit.connect(guard.release)

    # ── Ctrl+C / SIGTERM: stop the servers BEFORE the GUI dies ──────────
    # llama-server children are spawned with start_new_session (Unix) /
    # CREATE_NEW_CONSOLE (Windows), so a Ctrl+C in the terminal that
    # launched the GUI reaches ONLY the Python process. Without a handler,
    # Qt's event loop dies with a raw KeyboardInterrupt and the servers
    # keep running as orphans — on Ubuntu this showed up as "KeyboardInterrupt
    # printed, but the VRAM is still full". We install SIGINT/SIGTERM
    # handlers that shut every registered server down (SIGTERM to its
    # process group, escalating to SIGKILL in _TerminalProcess.stop) and
    # then quit the app cleanly.
    #
    # Python-level signal handlers only run between bytecodes; while Qt is
    # blocked inside its C++ event loop no Python bytecode executes, so the
    # handler would be delayed indefinitely. The idle QTimer below wakes
    # the interpreter a few times per second, giving pending signals a spot
    # to fire — the standard Qt-plus-Python-signals pattern.
    def _shutdown_on_signal(_signum: int, _frame: object) -> None:
        try:
            window._control_closing = True
            window._stop_control_api()
            window._stop_all_servers()
        except Exception:
            pass
        window._request_quit()

    for _sig_name in ("SIGINT", "SIGTERM"):
        _sig = getattr(signal, _sig_name, None)
        if _sig is not None:
            try:
                signal.signal(_sig, _shutdown_on_signal)
            except (ValueError, OSError):
                pass  # non-main thread / unsupported on this platform

    _signal_wakeup_timer = QTimer()
    _signal_wakeup_timer.timeout.connect(lambda: None)
    _signal_wakeup_timer.start(250)

    exit_code = app.exec()
    guard.release()
    # A stop issued from closeEvent escalates to kill after 10 s on a daemon
    # thread. Ending the process before that fires would orphan llama-server
    # (its CTRL_BREAK never reaches a separate Windows console).
    still_stopping = _TerminalProcess.wait_pending_stops(13.0)
    if still_stopping:
        print(
            f"[AutoTuner] {len(still_stopping)} server process(es) did not stop "
            "within the shutdown grace period.",
            flush=True,
        )
    _arm_exit_watchdog(exit_code)
    lingering = _drain_background_work(window)
    if lingering:
        # Interpreter teardown with a live QThread aborts ("QThread: Destroyed
        # while thread is still running") and a daemon thread that still
        # writes to stdout trips "could not acquire lock ... at interpreter
        # shutdown". Both end as a crash dialog or a wrong exit code instead
        # of a clean close, so end the process directly instead.
        print(
            "[AutoTuner] Ending the process with background work still active: "
            + ", ".join(lingering),
            flush=True,
        )
        _flush_std_streams()
        os._exit(exit_code)
    sys.exit(exit_code)


def _show_initial_window(window: MainWindow, autostart: bool) -> str:
    """Show the main window, or keep it out of the way for a login start.

    Returns ``"tray"``, ``"minimized"`` or ``"normal"``. A tray start needs
    "Hide on close" and a notification area; without them a minimized
    taskbar window keeps AutoTuner reachable.
    """
    minimized = autostart and app_settings.get_start_minimized_at_login()
    if minimized and app_settings.get_minimize_on_close():
        if window._start_in_notification_area():
            return "tray"
    if minimized:
        window.showMinimized()
    else:
        window.show()
    # The HWND is stable only after show(); native icon/system-menu calls made
    # earlier can target an unrealized handle. WM_SETICON is explicit because
    # Qt 6 + PyInstaller can otherwise show the icon visually while returning
    # an empty WM_GETICON handle to Windows shell integrations.
    window._finish_native_window_setup()
    return "minimized" if minimized else "normal"


def _drain_background_work(window: MainWindow, timeout_s: float = 10.0) -> List[str]:
    """Wait (bounded) for worker threads; return the names of any still alive.

    Every worker QThread is parented to the main window, so ``findChildren``
    sees the hardware probe, model scan, update, OCR, and benchmark threads
    even when their attribute was already cleared. Plain daemon threads
    (llama-server exit watchers, sysinfo refresh) cannot be joined safely
    here; they are only reported.
    """
    deadline = time.monotonic() + timeout_s
    lingering: List[str] = []
    try:
        threads = window.findChildren(QThread)
    except RuntimeError:
        threads = []
    for thread in threads:
        try:
            if not thread.isRunning():
                continue
            thread.quit()
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            if not thread.wait(remaining_ms):
                lingering.append(thread.objectName() or "QThread")
        except RuntimeError:
            continue
    for py_thread in threading.enumerate():
        if py_thread is threading.main_thread() or not py_thread.is_alive():
            continue
        if py_thread.name in ("AutoTunerExitWatchdog", "AutoTunerServerProbe"):
            continue
        lingering.append(py_thread.name)
    return lingering


def _flush_std_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.flush()
        except Exception:
            pass


def _arm_exit_watchdog(exit_code: int, timeout_s: float = 15.0) -> None:
    """Guarantee that the process ends once the Qt event loop has finished.

    Interpreter teardown after ``app.exec()`` normally takes milliseconds.
    Should a worker (metadata scan, hardware probe, a llama-server wait) still
    hold the interpreter at that point, the user would see a window-less
    AutoTuner lingering in Task Manager. The watchdog is a daemon thread, so
    it never delays a normal exit; it only ends a shutdown that hangs.
    """

    def _force_exit() -> None:
        time.sleep(timeout_s)
        try:
            print(
                f"[AutoTuner] Shutdown did not finish within {timeout_s:.0f}s "
                "— ending the process.",
                flush=True,
            )
        except Exception:
            pass
        os._exit(exit_code)

    threading.Thread(
        target=_force_exit, name="AutoTunerExitWatchdog", daemon=True
    ).start()


if __name__ == "__main__":
    main()
