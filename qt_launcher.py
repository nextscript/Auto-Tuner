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
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, cast, Tuple

from PyQt6.QtCore import Qt, QByteArray, QObject, QThread, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QCloseEvent, QFont
from PyQt6.QtWidgets import (
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
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from hardware import detect_system, SystemInfo
from scanner import (
    scan_models,
    group_entries,
    ModelEntry,
    read_gguf_metadata,
    is_mmproj_compatible,
    is_draft_compatible,
)
from settings_loader import load_profiles, match_profile, ModelProfile
from tuner import build_command, compute_config, TunedConfig
from performance_target import (
    PERFORMANCE_TARGETS,
    list_target_names,
    resolve_performance_target,
    DEFAULT_TARGET_NAME,
)
import app_settings


def _get_fork_tools():
    """Lazy import — never triggers auto_tuner.main()."""
    from auto_tuner import (
        _discover_llama_forks,
        _resolve_diffusion_binary,
        _resolve_server_binary,
    )

    return _discover_llama_forks, _resolve_server_binary, _resolve_diffusion_binary


def _default_settings_path() -> Path:
    return Path(__file__).resolve().parent / "settings"


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
    script_dir = Path(__file__).resolve().parent
    for c in (script_dir / "models", script_dir.parent / "models"):
        if c.exists():
            return c
    return script_dir / "models"


# ---------------------------------------------------------------------------
# Terminal process — spawns llama-server in its own visible terminal window


class _TerminalProcess:
    """Spawn llama-server in an independent terminal (CREATE_NEW_CONSOLE on
    Windows, start_new_session on Unix). No stdout pipe — the user sees the
    full server output in the separate window; our log panel shows status only.
    """

    def __init__(self, cmd: List[str], env_overrides: Optional[dict] = None) -> None:
        self.cmd = cmd
        self.env_overrides: dict = env_overrides or {}
        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        env = os.environ.copy()
        if self.env_overrides:
            env.update(self.env_overrides)
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
            self.proc = subprocess.Popen(self.cmd, creationflags=flags, env=env)
        else:
            self.proc = subprocess.Popen(self.cmd, start_new_session=True, env=env)

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def returncode(self) -> Optional[int]:
        return self.proc.returncode if self.proc is not None else None

    def stop(self) -> None:
        """Non-blocking signal + background wait."""
        if self.proc is None:
            return
        try:
            if os.name == "nt":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.kill(-self.proc.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

        # Capture in a local variable BEFORE clearing self.proc —
        # the daemon thread runs after self.proc is already None.
        _proc = self.proc
        self.proc = None

        def _wait() -> None:
            try:
                _proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    _proc.kill()
                except (ProcessLookupError, OSError):
                    pass

        threading.Thread(target=_wait, daemon=True).start()


# ---------------------------------------------------------------------------
# Hardware detection worker with global timeout


class _HwDetectWorker(QObject):
    """Runs detect_system() in a background thread with a global timeout."""

    finished = pyqtSignal(object, str)  # SystemInfo|None, error_msg

    def __init__(self, timeout: float = 30.0) -> None:
        super().__init__()
        self._timeout = timeout

    def run(self) -> None:
        result: list = [None, ""]  # [SystemInfo|None, error_str]

        def _detect() -> None:
            try:
                result[0] = detect_system()
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

    def __init__(self, root: Path) -> None:
        super().__init__()
        self._root = root

    def run(self) -> None:
        try:
            self.finished.emit(scan_models(self._root))
        except Exception as exc:
            self.error.emit(str(exc))


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
# batch_threads, batch, ubatch, flash_attn, mlock, no_mmap, jinja,
# verbose, numa, rope_scaling, rope_factor, temperature, top_k,
# top_p, min_p, repeat_penalty, presence_penalty, reasoning,
# think_budget, parallel_enabled, parallel_count, extras.


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
        _reasoning_flags_from_values(vals.get("reasoning", "auto"), vals.get("think_budget", -1))
    )
    free = (vals.get("extras") or "").strip()
    if free:
        extras.extend(free.split())
    return extras


def apply_expert_values(cfg: TunedConfig, vals: dict) -> TunedConfig:
    """Overlay the NON-cascading values onto ``cfg`` (in place + returned).

    Used both by the live panel (``_apply_noncascading``) and by the
    override-aware launch path: after compute_config runs with the saved
    auto-mode pins, the user's threads / batch / flags / sampling /
    reasoning choices are stamped back on so they survive the recompute.
    Cascading fields (ctx, KV quants, ngl, n_cpu_moe, rope) are left
    untouched — those belong to compute_config.
    """
    try:
        if vals.get("threads"):
            cfg.threads = int(vals["threads"]) or cfg.threads
        if vals.get("batch_threads"):
            cfg.batch_threads = int(vals["batch_threads"]) or cfg.batch_threads
        if vals.get("batch"):
            cfg.batch = int(vals["batch"]) or cfg.batch
        if vals.get("ubatch"):
            cfg.ubatch = int(vals["ubatch"]) or cfg.ubatch
        cfg.flash_attn = bool(vals.get("flash_attn", cfg.flash_attn))
        cfg.mlock = bool(vals.get("mlock", cfg.mlock))
        cfg.no_mmap = bool(vals.get("no_mmap", cfg.no_mmap))
        numa_choice = str(vals.get("numa", "off") or "off")
        cfg.numa = None if numa_choice == "off" else numa_choice
        cfg.sampling = _expert_sampling_from_values(vals)
        cfg.extra_cli_flags = _expert_extras_from_values(vals)
        # Parallel-slots override (--parallel / -np). When enabled, pin
        # the count and mark it forced so the panel can render the
        # checkbox state; when disabled, leave whatever compute_config
        # derived from the performance target and clear the flag.
        if vals.get("parallel_enabled"):
            try:
                cfg.n_parallel = max(1, int(vals.get("parallel_count", cfg.n_parallel) or cfg.n_parallel))
            except (TypeError, ValueError):
                pass
            cfg.n_parallel_forced = True
        else:
            cfg.n_parallel_forced = False
    except Exception:
        pass
    return cfg


def expert_cfg_from_values(base: TunedConfig, vals: dict) -> TunedConfig:
    """Build a frozen (manual) TunedConfig from ``base`` + a values snapshot.

    Every editable field is taken from ``vals``; unmodelled fields
    (tensor_split, main_gpu, env_overrides, VRAM estimates, …) are
    inherited from ``base`` so the result is still a complete config.
    This is the disk equivalent of ``ExpertPanel._build_manual_config``.
    """
    cfg = copy.copy(base)
    cfg.ctx = int(vals.get("ctx", base.ctx))
    cfg.cache_k = str(vals.get("cache_k", base.cache_k))
    cfg.cache_v = str(vals.get("cache_v", base.cache_v))
    cfg.ngl = int(vals.get("ngl", base.ngl))
    try:
        n_cpu = int(vals.get("n_cpu_moe", 0) or 0)
    except (TypeError, ValueError):
        n_cpu = 0
    cfg.n_cpu_moe = n_cpu if n_cpu > 0 else None
    cfg.rope_scaling = bool(vals.get("rope_scaling", base.rope_scaling))
    try:
        cfg.rope_scale_factor = float(vals.get("rope_factor", base.rope_scale_factor) or 1.0)
    except (TypeError, ValueError):
        cfg.rope_scale_factor = float(base.rope_scale_factor or 1.0)
    # Non-cascading overlay (threads / batch / flags / sampling / reasoning)
    apply_expert_values(cfg, vals)
    cfg.kv_quant_strategy = "manual"
    return cfg


# ---------------------------------------------------------------------------
# Expert panel — editable settings overlay
# ---------------------------------------------------------------------------


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

    # Upstream supports the first six; turbo3/turbo4 only resolve on the
    # TurboQuant forks (TheTom/turboquant_plus, AtomicBot, spiritbuun).
    # The combo accepts both either way — a fork that doesn't understand
    # turbo3 will refuse to start and surface a clear error.
    _KV_QUANT_OPTIONS = [
        "q4_0",
        "q4_1",
        "iq4_nl",
        "q5_0",
        "q5_1",
        "q8_0",
        "f16",
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
            "Auto-cascade: edit any setting and the others re-fit around it."
        )
        self._btn_auto.clicked.connect(lambda: self._set_mode("auto"))
        mode_row.addWidget(self._btn_auto)

        self._btn_manual = QPushButton("✎ Manual")
        self._btn_manual.setCheckable(True)
        self._btn_manual.setToolTip(
            "Full manual: settings stay exactly as you set them. No cascade."
        )
        self._btn_manual.clicked.connect(lambda: self._set_mode("manual"))
        mode_row.addWidget(self._btn_manual)

        # Reset — drops the saved Expert state for this model and reloads
        # the AutoTuner's automatically-best config. Sits next to
        # Auto/Manual so the "back to Auto" path is one click.
        self._btn_reset = QPushButton("⟲ Reset")
        self._btn_reset.setToolTip(
            "Forget the saved Expert settings for this model and reload\n"
 "the AutoTuner's automatically-best configuration."
        )
        self._btn_reset.clicked.connect(self.resetRequested.emit)
        mode_row.addWidget(self._btn_reset)

        # "✓ gespeichert" flash — confirms a per-model autosave just
        # landed. Hidden until the first real edit; never shown for
        # programmatic load / restore / reset (those bypass
        # `_emit_state_changed` via the `_populating` guard).
        self._saved_lbl = QLabel("✓ gespeichert")
        self._saved_lbl.setStyleSheet("color:#6c6;font-style:italic;")
        self._saved_lbl.setVisible(False)
        mode_row.addWidget(self._saved_lbl)

        mode_row.addStretch(1)

        self._btn_close = QPushButton("✕")
        self._btn_close.setFixedWidth(28)
        self._btn_close.setToolTip("Close Expert panel — return to read-only preview.")
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
            label_widget.setStyleSheet("color:#bbb;")
            grid.addWidget(label_widget, row, 0)
            grid.addWidget(widget, row, 1)
            if tip:
                widget.setToolTip(tip)
                label_widget.setToolTip(tip)
            row += 1

        def _section(title: str) -> None:
            nonlocal row
            section_label = QLabel(f"── {title} ──")
            section_label.setStyleSheet("color:#8be;padding-top:4px;")
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
            "Maximum context length. Auto mode: changing this re-picks "
            "KV quants and placement to fit.",
        )

        self._cb_cache_k = QComboBox()
        self._cb_cache_k.addItems(self._KV_QUANT_OPTIONS)
        self._cb_cache_k.currentTextChanged.connect(
            lambda _: self._on_edit("force_cache_k")
        )
        _add(
            "K-quant",
            self._cb_cache_k,
            "K-cache quantisation. Higher = better attention recall.",
        )

        self._cb_cache_v = QComboBox()
        self._cb_cache_v.addItems(self._KV_QUANT_OPTIONS)
        self._cb_cache_v.currentTextChanged.connect(
            lambda _: self._on_edit("force_cache_v")
        )
        _add(
            "V-quant",
            self._cb_cache_v,
            "V-cache quantisation. May be lower than K-quant (asymmetric FA).",
        )

        # Layer placement
        _section("Layer placement")
        self._sp_ngl = QSpinBox()
        self._sp_ngl.setRange(0, 999)
        self._sp_ngl.valueChanged.connect(lambda _: self._on_edit("force_ngl"))
        _add(
            "GPU layers (ngl)",
            self._sp_ngl,
            "Dense models: how many layers go on GPU. 999 = all. "
            "Ignored for MoE — use n_cpu_moe.",
        )

        self._sp_ncpumoe = QSpinBox()
        self._sp_ncpumoe.setRange(0, 999)
        self._sp_ncpumoe.valueChanged.connect(
            lambda _: self._on_edit("force_n_cpu_moe")
        )
        _add(
            "n_cpu_moe",
            self._sp_ncpumoe,
            "MoE only: how many expert layers run on CPU.",
        )

        # Threads & batching
        _section("Threads & batching")
        self._sp_threads = QSpinBox()
        self._sp_threads.setRange(1, 256)
        _add("threads", self._sp_threads, "-t  (compute threads)")

        self._sp_batch_threads = QSpinBox()
        self._sp_batch_threads.setRange(1, 256)
        _add("batch threads", self._sp_batch_threads, "-tb (batch threads)")

        self._sp_batch = QSpinBox()
        self._sp_batch.setRange(1, 16384)
        self._sp_batch.setSingleStep(64)
        _add("batch", self._sp_batch, "-b  (logical batch size)")

        self._sp_ubatch = QSpinBox()
        self._sp_ubatch.setRange(1, 16384)
        self._sp_ubatch.setSingleStep(64)
        _add("ubatch", self._sp_ubatch, "-ub (physical batch size)")

        # Parallelism (llama-server --parallel N, short: -np N)
        # Each slot gets its own KV-cache window, so Auto mode re-fits
        # the context length around the chosen slot count (same math the
        # performance target uses internally). Off = keep the
        # performance-target default (1/2/4 for safe/balanced/throughput).
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
        self._chk_parallel.toggled.connect(
            lambda _: self._on_edit("force_n_parallel")
        )
        _add(
            "",
            self._chk_parallel,
            "Run multiple concurrent inference slots (continuous batching). "
            "Useful for local subagent testing / parallel requests. Each "
            "slot gets its own KV window, so context shrinks to fit. "
            "Off = use the performance-target default.",
        )
        _add(
            "parallel slots",
            self._sp_parallel,
            "Number of concurrent inference slots (llama-server --parallel N). "
            "Default scales with your GPU: 3 when the largest GPU has "
            "≥24 GB free VRAM, otherwise 2.",
        )

        # Flags
        _section("Flags")
        self._chk_fa = QCheckBox("flash attention (-fa)")
        _add("", self._chk_fa, "Flash Attention — required for KV-quantisation.")

        self._chk_mlock = QCheckBox("--mlock")
        _add(
            "",
            self._chk_mlock,
            "Lock model in memory. Windows: needs SeLockMemoryPrivilege.",
        )

        self._chk_no_mmap = QCheckBox("--no-mmap")
        _add("", self._chk_no_mmap, "Load model fully into memory at startup.")

        self._chk_jinja = QCheckBox("--jinja")
        _add(
            "",
            self._chk_jinja,
            "Use the embedded chat template (separates <think> tags into reasoning_content).",
        )

        self._chk_verbose = QCheckBox("--verbose")
        _add("", self._chk_verbose, "Verbose llama-server logging.")

        self._cb_numa = QComboBox()
        self._cb_numa.addItems(self._NUMA_OPTIONS)
        _add("NUMA", self._cb_numa, "--numa policy (off = no flag).")

        self._chk_rope = QCheckBox("RoPE scaling (YaRN)")
        self._chk_rope.toggled.connect(lambda _: self._on_edit("force_rope_scale"))
        _add(
            "",
            self._chk_rope,
            "Force YaRN context extension on/off (overrides profile default).",
        )

        self._sp_rope_factor = QDoubleSpinBox()
        self._sp_rope_factor.setRange(1.0, 32.0)
        self._sp_rope_factor.setSingleStep(0.5)
        self._sp_rope_factor.setDecimals(1)
        _add("RoPE factor", self._sp_rope_factor, "YaRN scale factor (1.0 = native).")

        # Sampling
        _section("Sampling")
        self._sp_temp = QDoubleSpinBox()
        self._sp_temp.setRange(0.0, 5.0)
        self._sp_temp.setSingleStep(0.05)
        self._sp_temp.setDecimals(2)
        _add("temperature", self._sp_temp, "--temp")

        self._sp_top_k = QSpinBox()
        self._sp_top_k.setRange(0, 1000)
        _add("top_k", self._sp_top_k, "--top-k  (0 = disabled)")

        self._sp_top_p = QDoubleSpinBox()
        self._sp_top_p.setRange(0.0, 1.0)
        self._sp_top_p.setSingleStep(0.01)
        self._sp_top_p.setDecimals(3)
        _add("top_p", self._sp_top_p, "--top-p")

        self._sp_min_p = QDoubleSpinBox()
        self._sp_min_p.setRange(0.0, 1.0)
        self._sp_min_p.setSingleStep(0.01)
        self._sp_min_p.setDecimals(3)
        _add("min_p", self._sp_min_p, "--min-p")

        self._sp_rep = QDoubleSpinBox()
        self._sp_rep.setRange(0.5, 2.5)
        self._sp_rep.setSingleStep(0.01)
        self._sp_rep.setDecimals(3)
        _add("repeat_penalty", self._sp_rep, "--repeat-penalty")

        self._sp_presence = QDoubleSpinBox()
        self._sp_presence.setRange(-2.0, 2.0)
        self._sp_presence.setSingleStep(0.1)
        self._sp_presence.setDecimals(2)
        _add("presence_penalty", self._sp_presence, "--presence-penalty")

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
            "Reasoning effort passed to the chat template via "
            "--chat-template-kwargs (or --reasoning off for 'off'). "
            "'auto' emits no flag so the model template decides.",
        )

        self._sp_think_budget = QSpinBox()
        self._sp_think_budget.setRange(-1, 1_048_576)
        self._sp_think_budget.setSingleStep(256)
        self._sp_think_budget.setValue(-1)
        self._sp_think_budget.setGroupSeparatorShown(True)
        _add(
            "Think budget",
            self._sp_think_budget,
            "--reasoning-budget N. -1 = unlimited (no flag), 0 = stop "
            "thinking immediately, N>0 = token budget for the "
            "thinking phase.",
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
            "Appended verbatim to the llama-server command line.",
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
            self._sp_ctx, self._sp_ngl, self._sp_ncpumoe, self._sp_threads,
            self._sp_batch_threads, self._sp_batch, self._sp_ubatch,
            self._sp_rope_factor, self._sp_temp, self._sp_top_k, self._sp_top_p,
            self._sp_min_p, self._sp_rep, self._sp_presence, self._sp_think_budget,
            self._sp_parallel,
        ):
            sp.valueChanged.connect(self._schedule_save)
        for cb in (
            self._cb_cache_k, self._cb_cache_v, self._cb_numa, self._cb_reasoning,
        ):
            cb.currentTextChanged.connect(self._schedule_save)
        for chk in (
            self._chk_fa, self._chk_mlock, self._chk_no_mmap, self._chk_jinja,
            self._chk_verbose, self._chk_rope, self._chk_parallel,
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
                int(cfg.n_parallel)
                if parallel_on
                else self._suggested_parallel_count()
            )

            # Flags
            self._chk_fa.setChecked(cfg.flash_attn)
            self._chk_mlock.setChecked(cfg.mlock)
            self._chk_no_mmap.setChecked(cfg.no_mmap)
            self._chk_jinja.setChecked("--jinja" in (cfg.extra_cli_flags or []))
            self._chk_verbose.setChecked("--verbose" in (cfg.extra_cli_flags or []))
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

            # Reasoning + think-budget: parse them out of extra_cli_flags
            # so the dedicated dropdowns show the right state and the
            # free-form field below doesn't display the raw flags.
            extras_in = list(cfg.extra_cli_flags or [])
            reasoning_value, think_budget_value, leftover_extras = (
                self._parse_reasoning_from_extras(extras_in)
            )
            self._set_combo(self._cb_reasoning, reasoning_value)
            self._sp_think_budget.setValue(think_budget_value)

            # Extra CLI: filter out the flags we already model as
            # checkboxes / dedicated widgets so they don't appear twice.
            modeled = {"--jinja", "--verbose"}
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
        elif kind == "force_n_parallel":
            # Pinning the parallel-slot count makes Auto mode re-fit ctx
            # around N slots; unchecking releases the pin so the
            # performance-target default (1/2/4) takes over again.
            if self._chk_parallel.isChecked():
                self._user_pins["force_n_parallel"] = max(
                    1, self._sp_parallel.value()
                )
            else:
                self._user_pins["force_n_parallel"] = None
        elif kind == "force_rope_scale":
            self._user_pins["force_rope_scale"] = self._chk_rope.isChecked()

        self._recompute(force_overrides=dict(self._user_pins))

    def _recompute(self, force_overrides: dict) -> None:
        """Ask the parent to rebuild the config with these overrides."""
        if self._recompute_cb is None:
            return
        cfg = self._recompute_cb(force_overrides)
        if cfg is None:
            return
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
            biggest_free = max(
                (g.free_vram_gb for g in sysinfo.gpus), default=0.0
            )
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
            "mlock": self._chk_mlock.isChecked(),
            "no_mmap": self._chk_no_mmap.isChecked(),
            "jinja": self._chk_jinja.isChecked(),
            "verbose": self._chk_verbose.isChecked(),
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
            "parallel_enabled": self._chk_parallel.isChecked(),
            "parallel_count": self._sp_parallel.value(),
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
            self._user_pins = {
                k: v for k, v in (snap.get("pins") or {}).items() if v is not None
            }
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
        self._recompute(force_overrides={})


# ---------------------------------------------------------------------------
# Main window


class MainWindow(QMainWindow):
    # Signal carrying SystemInfo updates from the background sysinfo thread.
    # Qt widgets are NOT thread-safe — touching a QLabel from a daemon
    # thread produced sporadic random crashes ("GUI just closed itself").
    # Background work emits this signal; the slot runs on the GUI thread.
    _sysinfo_ready = pyqtSignal(object)  # SystemInfo
    _bg_log = pyqtSignal(str)  # log message from background thread

    def __init__(self, models_path: Path, settings_path: Path) -> None:
        super().__init__()
        self.setWindowTitle("AutoTuner Qt Launcher")
        # Hard-coded default size — only kicks in when no persisted
        # geometry exists (first launch on this machine, or the JSON
        # was wiped). `restoreGeometry` below replaces this when a
        # blob is on disk.
        self.resize(1320, 840)
        self._restore_window_geometry()

        self.models_path = models_path
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
        self._system: Optional[SystemInfo] = None
        self._profiles: List[ModelProfile] = []
        self._forks: List[Tuple[str, Path]] = []
        self._fork_path: Optional[Path] = None  # manueller Fork-Ordner

        # Currently selected model + its draft (set in _show_config)
        self._current_entry: Optional[ModelEntry] = None
        self._current_draft: Optional[ModelEntry] = None

        # Per-model override cache for the vision/draft/thinking checkboxes.
        # Populated when the user toggles a checkbox, so switching to a
        # different model and back preserves the manual choice for the
        # rest of the session. Persisted to JSON on every change so the
        # choice also survives an app restart.
        # Shape:  { "<model_name>": {"vision": bool, "draft": bool, "thinking": bool} }
        self._option_overrides: dict = {}

        # Track whether the user has manually overridden the fork selection
        self._fork_manual_override = False

        # Remember the *container* the user pointed at via "📂 Fork" so
        # restarts still show every sibling build. This stays distinct
        # from the currently active fork in `self._fork_path`.
        self._fork_container: Optional[Path] = None

        self._scan_thread: Optional[QThread] = None
        self._scan_worker: Optional[_ScanWorker] = None
        self._sysinfo_busy = False
        # Persisted font size — falls back to 10pt on first launch.
        self._font_size = app_settings.get_font_size()

        self._build_ui()
        # Wire background → GUI signals BEFORE the first scan kicks off,
        # so a fast hardware probe can't fire its result into a slot
        # that hasn't been connected yet (one of the crash patterns).
        self._sysinfo_ready.connect(self._update_sysinfo_labels)
        self._bg_log.connect(self._log)
        QTimer.singleShot(0, self._startup_load)

        # Server crash-detection (lightweight poll — no stdout read)
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_server)
        self._poll_timer.start(500)

        # Sysinfo refresh (non-blocking — daemon thread)
        self._sysinfo_timer = QTimer(self)
        self._sysinfo_timer.timeout.connect(self._sysinfo_async)
        self._sysinfo_timer.start(6000)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # ── Toolbar ────────────────────────────────────────────────────
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        self._path_label = QLabel()
        self._path_label.setStyleSheet("padding:0 6px;color:#aaa;")
        tb.addWidget(self._path_label)
        tb.addSeparator()

        for label, slot in (
            ("📂 Models folder", self._browse_models),
            ("🔄 Refresh", self._start_scan),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            if label.startswith("🔄"):
                self._btn_refresh = btn
            tb.addWidget(btn)

        tb.addSeparator()
        tb.addWidget(QLabel(" Fork:"))
        self._fork_combo = QComboBox()
        self._fork_combo.setMinimumWidth(140)
        self._fork_combo.setToolTip(
            "Default llama.cpp fork (auto-overridden by profile)"
        )
        self._fork_combo.currentIndexChanged.connect(self._on_fork_changed)
        tb.addWidget(self._fork_combo)

        self._fork_path_lbl = QLabel()
        self._fork_path_lbl.setStyleSheet("color:#aaa;font-size:9pt;")
        self._fork_path_lbl.setMaximumWidth(120)
        self._fork_path_lbl.setText("")
        tb.addWidget(self._fork_path_lbl)

        self._btn_fork_folder = QPushButton("📂")
        self._btn_fork_folder.setFixedWidth(28)
        self._btn_fork_folder.setToolTip("Manuellen Fork-Ordner auswählen")
        self._btn_fork_folder.clicked.connect(self._browse_fork_folder)
        tb.addWidget(self._btn_fork_folder)

        tb.addSeparator()
        tb.addWidget(QLabel(" Performance:"))
        self._perf_combo = QComboBox()
        self._perf_combo.setMinimumWidth(120)
        # Build tooltip from registry so a future 4th tier auto-appears.
        tip_lines = ["VRAM utilisation preset:"]
        for tname in list_target_names():
            t = PERFORMANCE_TARGETS[tname]
            tip_lines.append(f"  • {tname}: {t.description}")
        self._perf_combo.setToolTip("\n".join(tip_lines))
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
        tb.addWidget(QLabel(" Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.setMinimumWidth(90)
        self._mode_combo.setToolTip(
            "Sampling profile:\n"
            "  • chat   — conversational defaults (higher temperature,\n"
            "             more diverse output)\n"
            "  • coding — deterministic defaults from each model's\n"
            "             official coding/agentic-bench setup\n"
            "Profiles without a coding block fall back to chat values."
        )
        for m in ("chat", "coding"):
            self._mode_combo.addItem(m)
        persisted_mode = app_settings.get_mode() or "chat"
        idx = self._mode_combo.findText(persisted_mode)
        self._mode_combo.setCurrentIndex(max(0, idx))
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        tb.addWidget(self._mode_combo)

        # ── GPU pin (Auto / per-card) ──────────────────────────────────
        # Hard-pins the next launch to a single card. This is the click-path
        # equivalent of the CLI `--gpu <name>` flag and the `forced_gpu` key
        # in autotuner_settings.json — all three feed the same
        # app_settings.set_forced_gpu() / compute_config(force_gpu=...) path.
        # The card entries are filled in once hardware detection finishes
        # (see _populate_gpu_combo, called from _update_sysinfo_labels); until
        # then only "Auto" is offered.
        tb.addSeparator()
        tb.addWidget(QLabel(" GPU:"))
        self._gpu_combo = QComboBox()
        self._gpu_combo.setMinimumWidth(110)
        self._gpu_combo.setToolTip(
            "Pin the next server to a single GPU:\n"
            "  • Auto — free-VRAM-aware selection across all cards\n"
            "  • <card> — boot exclusively on that card and hide the\n"
            "             others (use for a 2nd server so it lands on the\n"
            "             still-empty card instead of the full one)\n"
            "Mirrors the CLI --gpu flag and the forced_gpu setting."
        )
        # Seed with Auto only; real cards are appended after detection.
        self._gpu_combo.addItem("Auto", None)
        self._gpu_combo.currentIndexChanged.connect(self._on_gpu_changed)
        tb.addWidget(self._gpu_combo)

        tb.addSeparator()
        tb.addWidget(QLabel(" Font:"))
        for delta, label in ((-1, "A−"), (+1, "A+")):
            b = QPushButton(label)
            b.setFixedWidth(36)
            d = delta
            b.clicked.connect(lambda _, d=d: self._change_font(d))
            tb.addWidget(b)

        # ── Sysinfo bar ────────────────────────────────────────────────
        sysbar = QWidget()
        sl = QHBoxLayout(sysbar)
        sl.setContentsMargins(6, 1, 6, 1)
        self._cpu_lbl = QLabel("CPU: —")
        self._vram_lbl = QLabel("VRAM: —")
        self._ram_lbl = QLabel("RAM: —")
        self._gpu_lbl = QLabel("GPU: —")
        for lbl in (self._cpu_lbl, self._vram_lbl, self._ram_lbl, self._gpu_lbl):
            lbl.setStyleSheet("color:#8be;padding:0 12px;")
            sl.addWidget(lbl)
        sl.addStretch()
        sysbar.setMaximumHeight(24)
        sysbar.setStyleSheet("background:#161625;")

        # ── Filter + model list ────────────────────────────────────────
        fr = QWidget()
        frl = QHBoxLayout(fr)
        frl.setContentsMargins(2, 2, 2, 2)
        frl.addWidget(QLabel("Filter:"))
        self._search = QLineEdit()
        self._search.setPlaceholderText("type to filter…")
        self._search.textChanged.connect(self._apply_filter)
        frl.addWidget(self._search)

        self._model_list = QListWidget()
        self._model_list.currentItemChanged.connect(self._on_selection_changed)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(2)
        ll.addWidget(fr)
        ll.addWidget(self._model_list)

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

        # ── Expert button row (sits between preview and Launch options) ─
        # In normal mode this row shows a "🔧 Expert" button plus a
        # "🔍 Diagnose" button. When Expert mode is active the Expert
        # button is replaced by an [Auto] [Manual] pair (the Expert
        # panel itself owns those toggle buttons — see ExpertPanel — but
        # we still mirror the state here in the bottom row for parallel
        # access). The Diagnose button stays visible in both modes.
        self._btn_expert = QPushButton("🔧 Expert settings")
        self._btn_expert.setToolTip(
            "Open the Expert panel to override AutoTuner decisions."
        )
        self._btn_expert.clicked.connect(self._enter_expert_mode)
        self._btn_diagnose = QPushButton("🔍 Diagnose")
        self._btn_diagnose.setToolTip(
            "Show the metadata diagnostic report for the selected model — "
            "KV size estimate, hybrid/MoE detection inputs, capacity "
            "estimates, and any warnings."
        )
        self._btn_diagnose.clicked.connect(self._show_diagnostic_report)
        self._btn_diagnose.setEnabled(False)  # disabled until a model is picked
        self._btn_expert_row = QWidget()
        bex = QHBoxLayout(self._btn_expert_row)
        bex.setContentsMargins(0, 0, 0, 0)
        bex.addStretch(1)
        bex.addWidget(self._btn_expert)
        bex.addWidget(self._btn_diagnose)
        bex.addStretch(1)

        # ── Launch options (checkboxes) ────────────────────────────────
        opts = QGroupBox("Launch options")
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
        self._cb_mmproj.setToolTip(
            "Vision projector to load. Lists every projector in the model's\n"
            "folder; '⚠' marks ones that don't match this model (you can still\n"
            "select them to experiment). Remembered per model."
        )
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
        self._cb_draft.setToolTip(
            "Draft model for speculative decoding. Lists every draft/assistant\n"
            "GGUF in the model's folder; '⚠' marks ones that don't match this\n"
            "model (selectable anyway for experimenting). Remembered per model."
        )
        self._cb_draft.currentIndexChanged.connect(self._on_draft_combo_changed)
        _draft_l.addWidget(self._cb_draft, 1)
        self._draft_row.setVisible(True)
        ol.addWidget(self._draft_row)

        self._chk_vision = QCheckBox("Vision (mmproj)")
        self._chk_draft = QCheckBox("Draft model (speculative decoding)")
        # NEW: Turbo KV-quant toggle. Sits between Draft and Thinking,
        # as requested. When on, the AutoTuner maps the chosen KV
        # quants to their TurboQuant equivalents (denser packing on
        # the TheTom/AtomicBot forks; harmless no-op on stock builds
        # because the mapping is identity for unknown labels).
        self._chk_turbo_kv = QCheckBox("Turbo KV-quant (TurboQuant forks)")
        # n-gram (ngram-mod) self-speculative decoding. Unlike Draft, this
        # needs no draft model and works on ANY GGUF (builds a rolling-hash
        # lookup table from the live context, ~16 MB). It is therefore always
        # available — never greyed out — and independent of the Draft toggle.
        self._chk_ngram = QCheckBox("n-gram speculative (ngram-mod)")
        self._chk_ngram.setToolTip(
            "Self-speculative decoding from the context. No draft model needed,\n"
            "works on any model. Best for code/text iteration, reasoning models\n"
            "that echo their scratchpad, and summarisation."
        )
        # Host-memory prompt caching (--cache-ram / -cram). Auto-ON for every
        # model that supports it (i.e. every NON-vision model — the feature
        # is incompatible with mtmd). Stays user-toggleable. When a vision
        # model is selected the box is disabled + unchecked, because
        # llama-server cannot cache prompts while the multimodal path is live.
        self._chk_prompt_cache = QCheckBox("Prompt caching (host RAM, -cram)")
        self._chk_prompt_cache.setToolTip(
            "Cache computed prompt prefixes in system RAM so repeated/similar\n"
            "prompts (long system prompts, RAG scaffolds, Roo-Code preambles)\n"
            "skip re-processing and hit first-token faster.\n"
            "Auto-enabled where supported; unavailable while Vision is active\n"
            "(llama-server cannot cache prompts under the multimodal path)."
        )
        self._chk_thinking = QCheckBox("Thinking / Reasoning")

        for chk in (
            self._chk_vision,
            self._chk_draft,
            self._chk_turbo_kv,
            self._chk_ngram,
            self._chk_prompt_cache,
            self._chk_thinking,
        ):
            chk.setEnabled(False)
            ol.addWidget(chk)

        # Checkbox toggles → persist the override AND refresh the
        # context / memory estimates. Each slot knows which option it owns.
        self._chk_vision.toggled.connect(self._on_vision_toggled)
        self._chk_draft.toggled.connect(self._on_draft_toggled)
        self._chk_turbo_kv.toggled.connect(self._on_turbo_toggled)
        self._chk_ngram.toggled.connect(self._on_ngram_toggled)
        self._chk_prompt_cache.toggled.connect(self._on_prompt_cache_toggled)
        self._chk_thinking.toggled.connect(self._on_thinking_toggled)

        opts.setMaximumHeight(220)

        right = QWidget()
        rl2 = QVBoxLayout(right)
        rl2.setContentsMargins(0, 0, 0, 0)
        rl2.setSpacing(4)
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
        bl.addWidget(self._host_edit)

        bl.addWidget(QLabel(" Base port:"))
        self._port_edit = QLineEdit(str(self._base_port))
        self._port_edit.setFixedWidth(60)
        self._port_edit.setToolTip(
            "Base port for the FIRST server. Each additional concurrent\n"
            "server gets the next free port (1234, 1235, 1236…). Stopping\n"
            "a server frees its port for reuse.\n"
            "Remembered across restarts."
        )
        # Persist on focus loss / Enter so the chosen port is remembered even
        # without launching (matches fork_path / font_size behaviour).
        self._port_edit.editingFinished.connect(self._persist_base_port)
        bl.addWidget(self._port_edit)

        bl.addWidget(QLabel(" Offset:"))
        self._port_offset_combo = QComboBox()
        self._port_offset_combo.setFixedWidth(60)
        self._port_offset_combo.setToolTip(
            "Manual offset added to the requested base port before collision checks."
        )
        for i in range(11):  # 0 to 10
            self._port_offset_combo.addItem(str(i))
        # Restore the persisted offset selection (clamped to the combo range).
        self._port_offset_combo.setCurrentIndex(
            max(0, min(self._port_offset_combo.count() - 1, app_settings.get_port_offset()))
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
        bl.addWidget(QLabel(" Server:"))
        self._server_combo = QComboBox()
        self._server_combo.setMinimumWidth(220)
        self._server_combo.setToolTip(
            "Select a running server. ‘Stop’ terminates the selected one."
        )
        bl.addWidget(self._server_combo)

        self._btn_toggle_log = QPushButton("▾ Log")
        self._btn_toggle_log.setFixedHeight(32)
        self._btn_toggle_log.setCheckable(True)
        self._btn_toggle_log.setChecked(True)
        self._btn_toggle_log.setToolTip("Show / fully retract the bottom info panel.")
        self._btn_toggle_log.clicked.connect(self._toggle_log_panel)
        bl.addWidget(self._btn_toggle_log)

        self._btn_launch = QPushButton("▶  Launch")
        self._btn_launch.setFixedHeight(32)
        self._btn_launch.setEnabled(False)
        self._btn_launch.setToolTip(
            "Launch the selected model. If a server is already running, the\n"
            "new model is placed on the emptier GPU and given the next port."
        )
        self._btn_launch.clicked.connect(self._launch_server)
        bl.addWidget(self._btn_launch)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setFixedHeight(32)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setToolTip("Stop the server selected in the dropdown.")
        self._btn_stop.clicked.connect(self._stop_server)
        bl.addWidget(self._btn_stop)

        self._btn_stop_all = QPushButton("■ Stop all")
        self._btn_stop_all.setFixedHeight(32)
        self._btn_stop_all.setEnabled(False)
        self._btn_stop_all.setToolTip("Stop every running llama-server.")
        self._btn_stop_all.clicked.connect(self._stop_all_clicked)
        bl.addWidget(self._btn_stop_all)

        self._btn_quit = QPushButton("Quit")
        self._btn_quit.setFixedHeight(32)
        self._btn_quit.clicked.connect(self.close)
        bl.addWidget(self._btn_quit)

        # ── Root ───────────────────────────────────────────────────────
        root = QWidget()
        root_l = QVBoxLayout(root)
        root_l.setContentsMargins(4, 0, 4, 0)
        root_l.setSpacing(0)
        root_l.addWidget(sysbar)
        root_l.addWidget(main_split, 1)
        root_l.addWidget(btn_row)
        self.setCentralWidget(root)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Starting…")

        # Re-apply the inner pane arrangement now that every splitter exists.
        self._restore_splitter_states()

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
        f = QFont("Consolas")
        f.setStyleHint(QFont.StyleHint.Monospace)
        f.setPointSize(self._font_size)
        w.setFont(f)

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
    def _expand_fork_container(path: Path) -> List[Tuple[str, Path]]:
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
        # Mirrors auto_tuner._SERVER_SUBPATHS — update both if paths change.
        _BINARY_SUBPATHS = (
            "build/bin/Release/llama-server.exe",
            "build/bin/Debug/llama-server.exe",
            "build/bin/llama-server.exe",
            "build/bin/llama-server",
            "build/llama-server",
            "llama-server.exe",  # prebuilt / release-zip drops
            "llama-server",
        )
        result: List[Tuple[str, Path]] = []
        try:
            for child in sorted(path.iterdir(), key=lambda c: c.name.lower()):
                if not child.is_dir():
                    continue
                if not re.search(
                    r"(?:(?:^|[-_.])llama(?:[-_.]|$)|llama\.cpp)",
                    child.name,
                    re.IGNORECASE,
                ):
                    continue
                has_binary = any((child / sub).is_file() for sub in _BINARY_SUBPATHS)
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

        try:
            discover, _, _ = _get_fork_tools()
            self._forks = discover()
        except Exception as exc:
            self._log(f"[Warning] Fork discovery failed: {exc}")
            self._forks = []

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
            self._fork_path_lbl.setText(manual_path.name)
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
            self._fork_path_lbl.setText(manual_path.name + " (📁)")
            # Restore previously active selection inside the container,
            # if persisted_active points at one of these children.
            initial_idx = 0
            if persisted_active is not None:
                try:
                    pa = persisted_active.resolve()
                    for i, (_n, p) in enumerate(container_children):
                        if p.resolve() == pa:
                            initial_idx = i
                            break
                except OSError:
                    pass
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
            self._fork_path_lbl.setText(manual_path.name)
            src_label = (
                "persisted settings" if manual_source == "settings" else "LLAMA_CPP_DIR"
            )
            self._log(f"[Fork] Using manual path from {src_label}: {manual_path}")
        elif self._forks:
            # No manual choice — auto-discovered forks.
            for name, path in self._forks:
                self._fork_combo.addItem(name, userData=path)
            self._fork_combo.setCurrentIndex(0)
            self._fork_path = self._forks[0][1] if self._forks else None
            self._log(f"Found {len(self._forks)} fork(s). Using: {self._forks[0][0]}")
            self._apply_fork(0)
        else:
            self._fork_combo.addItem("not found", userData=None)
            self._fork_path = None
            self._log("[Warning] No llama.cpp forks found. Set LLAMA_CPP_DIR.")
        self._fork_combo.blockSignals(False)

        # Hardware detection (spawns PowerShell on Windows) → background thread
        # so it never blocks the UI and never flashes a window.
        # Use signal/slot pattern instead of QTimer.singleShot from bg thread
        # to avoid potential PyQt6 deadlocks when COM is involved.
        self._log("Detecting system hardware…")
        self._hw_detect_worker = _HwDetectWorker(timeout=30.0)
        self._hw_detect_thread = QThread(self)
        self._hw_detect_worker.moveToThread(self._hw_detect_thread)
        self._hw_detect_thread.started.connect(self._hw_detect_worker.run)
        self._hw_detect_worker.finished.connect(self._hw_detect_done)
        self._hw_detect_worker.finished.connect(self._hw_detect_thread.quit)
        self._hw_detect_thread.finished.connect(self._hw_detect_thread.deleteLater)
        self._hw_detect_thread.start()

    # ------------------------------------------------------------------
    # Fork selection
    # ------------------------------------------------------------------
    def _on_fork_changed(self, index: int) -> None:
        self._fork_manual_override = True
        self._apply_fork(index)
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
        path: Optional[Path] = self._fork_combo.itemData(index)
        if path is not None:
            os.environ["LLAMA_CPP_DIR"] = str(path)
            self._log(f"[Fork] → {path.name}")

    # ------------------------------------------------------------------
    # Performance target selection
    # ------------------------------------------------------------------
    def _on_perf_changed(self, index: int) -> None:
        """User picked a new performance target — persist + refresh view.

        Only the *config text* is recomputed; the vision/draft/thinking
        checkboxes must NOT be touched here. Performance target affects
        VRAM placement and KV-cache decisions, never feature selection.
        """
        name = self._perf_combo.itemText(index).strip()
        try:
            app_settings.set_performance_target(name)
        except Exception as exc:
            self._log(f"[Warning] Could not save performance target: {exc}")
        self._log(f"[Perf] → {name}")
        # Recompute the displayed config in-place, leaving every
        # checkbox alone — `_update_config_text` reads the current
        # checkbox state and reflects it back into the preview.
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
        feature checkboxes (vision/draft/thinking) are never touched. The
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

    def _resolve_perf_target_for_profile(self, profile: ModelProfile):
        """Combine GUI choice with profile-level recommendation.

        GUI choice always wins; profile.performance_target is only used
        if the user hasn't picked anything (which currently never happens
        because the combo is initialised to "balanced", but stay robust).
        """
        gui_choice = (
            self._perf_combo.currentText().strip()
            if hasattr(self, "_perf_combo")
            else None
        )
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
            # Still allow model selection even without sysinfo
        self._start_scan()

    def _browse_fork_folder(self) -> None:
        """Manuellen Fork-Ordner auswählen (ähnlich wie Models folder)."""
        dialog = QFileDialog(self, "LLama.cpp Fork-Ordner auswählen")
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)

        # Vorgabepfad: aktueller Fork oder Workspace
        if self._fork_path is not None:
            dialog.setDirectory(str(self._fork_path))
        elif self._forks:
            dialog.setDirectory(str(self._forks[0][1]))

        if dialog.exec() == QFileDialog.DialogCode.Accepted:
            selected = dialog.selectedFiles()
            if selected:
                new_path = Path(selected[0])
                self._set_manual_fork_path(new_path)

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
                self, "Ungültiger Ordner", f"Der Ordner existiert nicht:\n{path}"
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
            self._fork_path_lbl.setText(path.name + " (📁)")
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
            self._fork_path_lbl.setText(path.name)
            os.environ["LLAMA_CPP_DIR"] = str(path)
            try:
                app_settings.set_fork_path(path)
                self._log(f"[Fork] Saved as default: {path}")
            except Exception as exc:
                self._log(f"[Warning] Could not save fork path: {exc}")

        self._fork_combo.blockSignals(False)
        self._apply_fork(0)

    # ------------------------------------------------------------------
    # Background model scan
    # ------------------------------------------------------------------
    def _start_scan(self) -> None:
        try:
            if self._scan_thread is not None and self._scan_thread.isRunning():
                return
        except RuntimeError:
            self._scan_thread = None

        self._path_label.setText(f"Models: {self.models_path}")
        self._btn_refresh.setEnabled(False)
        self._btn_launch.setEnabled(False)
        self._model_list.clear()
        self._status.showMessage(f"Scanning {self.models_path} …")
        self._log(f"Scanning: {self.models_path}")

        if not self.models_path.exists():
            msg = (
                f"Models folder not found:\n  {self.models_path}\n\n"
                "Use '📂 Models folder' to pick the right location,\n"
                "or set the AUTOTUNER_MODELS environment variable."
            )
            self._config_preview.setPlainText(msg)
            self._status.showMessage(f"Folder not found: {self.models_path}")
            self._btn_refresh.setEnabled(True)
            return

        worker = _ScanWorker(self.models_path)
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
        self._btn_refresh.setEnabled(True)
        if not entries:
            self._config_preview.setPlainText(
                f"No *.gguf files found in:\n  {self.models_path}"
            )
            self._status.showMessage("No models found.")
            self._log("No models found.")
            return
        self._populate_list(entries)
        self._btn_launch.setEnabled(True)
        self._status.showMessage(f"{len(entries)} model(s) loaded.")
        self._log(f"Found {len(entries)} model(s).")

    def _on_scan_error(self, msg: str) -> None:
        self._btn_refresh.setEnabled(True)
        self._log(f"[Error] Scan failed: {msg}")
        self._status.showMessage(f"Scan error: {msg}")

    def _populate_list(self, entries: List[ModelEntry]) -> None:
        self._model_list.clear()
        groups = group_entries(entries)
        for group_name in sorted(groups.keys()):
            for entry in sorted(groups[group_name], key=lambda e: e.name.lower()):
                marks = _capability_markers(entry)
                # Right-align the size so capabilities stay readable when
                # filenames vary in length.
                tail = f"  ({entry.size_gb:.1f} GB)"
                if marks:
                    item = QListWidgetItem(f"{entry.name}  {marks}{tail}")
                else:
                    item = QListWidgetItem(f"{entry.name}{tail}")
                item.setData(Qt.ItemDataRole.UserRole, entry)
                # Tooltip lists what each symbol means and which assets
                # are paired. Use explicit `is not None` checks instead of
                # the convenience `has_*` properties so Pylance/Mypy can
                # narrow Optional[Path] → Path on the next line.
                lines = [entry.name, ""]
                if entry.mmproj is not None:
                    lines.append(f"👁  Vision      {entry.mmproj.name}")
                if entry.draft is not None:
                    lines.append(f"⚡  Draft       {entry.draft.name}")
                if entry.supports_thinking:
                    lines.append("🧠  Thinking    chat template emits <think>")
                if entry.supports_tool_use:
                    lines.append("🛠  Tool use    chat template supports tool_calls")
                if len(lines) > 2:
                    item.setToolTip("\n".join(lines))
                self._model_list.addItem(item)

    def _apply_filter(self, text: str) -> None:
        q = text.strip().lower()
        self._populate_list(
            self._all_entries
            if not q
            else [e for e in self._all_entries if q in e.name.lower()]
        )

    def _browse_models(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select models folder", str(self.models_path)
        )
        if folder:
            self.models_path = Path(folder)
            try:
                app_settings.set_models_path(self.models_path)
                self._log(f"[Models] Saved as default: {self.models_path}")
            except Exception as exc:
                self._log(f"[Warning] Could not save models path: {exc}")
            self._start_scan()

    # ------------------------------------------------------------------
    # Config preview + options (single-click)
    # ------------------------------------------------------------------
    def _on_selection_changed(
        self,
        current: Optional[QListWidgetItem],
        _prev: Optional[QListWidgetItem],
    ) -> None:
        if current is None:
            return
        entry: ModelEntry = current.data(Qt.ItemDataRole.UserRole)
        if entry is None:
            return
        self._show_config(entry)

    def _show_config(self, entry: ModelEntry) -> None:
        """Called on model selection — updates checkboxes, auto-selects fork, refreshes preview.

        Gracefully handles the case when hardware detection has not yet
        completed (self._system is None).  The user will see a placeholder
        message and the config will update automatically once detection
        finishes.
        """
        if self._system is None:
            self._config_preview.setPlainText(
                "Hardware-Erkennung laeuft noch...\n\n"
                "Bitte warten Sie, bis die Systeminformationen geladen sind.\n"
                "Die Konfiguration wird automatisch aktualisiert."
            )
            return
        # Switching models drops the Expert state — the panel's pins were
        # for the *previous* model. Keep the user in the read-only preview
        # so they see the fresh AutoTuner output before re-entering Expert.
        if self._config_stack.currentIndex() == 1:
            self._config_stack.setCurrentIndex(0)
            self._btn_expert_row.setVisible(True)
        self._current_entry = entry
        self._current_draft = _find_draft_model(entry, self._all_entries)
        self._btn_diagnose.setEnabled(True)
        self._update_checkboxes(entry)
        profile = match_profile(
            entry.name, self._profiles, getattr(entry, "architecture", "")
        )
        self._auto_select_fork(profile)
        self._update_config_text(entry, profile)

    def _update_checkboxes(self, entry: ModelEntry) -> None:
        """Set checkbox enabled/checked states.

        Defaults reflect the model's capabilities (vision when an mmproj
        was paired, draft when an assistant sibling was found, thinking
        when the chat template advertises it). Once the user has
        manually toggled any of them for this model, that override wins
        — both within the session (in-memory cache) and across restarts
        (persisted to autotuner_settings.json).
        """
        # Pull persisted overrides first so a fresh app launch already
        # honours last session's choices. The in-memory cache wins if
        # both exist, since the user may have toggled mid-session.
        persisted = app_settings.get_model_overrides(entry.name)
        cached = self._option_overrides.get(entry.name, {})
        ov = {**persisted, **cached}

        # Resolve the draft dropdown FIRST. It sets self._current_draft from
        # the remembered/auto selection, which the Vision + Draft sections
        # below read (an active external draft blocks vision). Populating it
        # here keeps a single source of truth for the chosen drafter.
        self._populate_draft_combo(entry, ov)

        # ── Vision ──────────────────────────────────────────────────
        mmproj = entry.mmproj
        has_external_draft = self._current_draft is not None
        is_embedded_mtp = entry.has_embedded_mtp
        # Determine whether the external draft will actually be used.
        # Having an external draft file available but draft unchecked does
        # NOT block vision — only an active (checked) external draft does.
        # The override dict already contains "draft": False when the user
        # has unticked the Draft checkbox for this model.
        draft_override = ov.get("draft", None)
        draft_default = has_external_draft  # default: enabled when file exists
        draft_effectively_on = has_external_draft and (
            draft_override if draft_override is not None else draft_default
        )
        # External draft (-md) conflicts with --mmproj in llama.cpp: both
        # try to load a second model and the server aborts. Integrated MTP
        # lives inside the main GGUF — no second-model conflict, so vision
        # is safe (and Qwen3.6-MTP models require it to work correctly).
        vision_blocked = draft_effectively_on and not is_embedded_mtp
        has_vision = mmproj is not None and not vision_blocked
        # Default: enable vision when mmproj is present and not blocked.
        # For embedded-MTP models default to True (they need vision).
        default_vision = has_vision
        vision_state = ov["vision"] if "vision" in ov else default_vision
        self._chk_vision.blockSignals(True)
        self._chk_vision.setEnabled(has_vision)
        self._chk_vision.setChecked(has_vision and vision_state)
        if mmproj is not None and vision_blocked:
            self._chk_vision.setText(
                f"Vision  ({mmproj.name})  [blocked: external draft active]"
            )
        elif mmproj is not None:
            self._chk_vision.setText(f"Vision  ({mmproj.name})")
        else:
            self._chk_vision.setText("Vision (no mmproj found)")
        self._chk_vision.blockSignals(False)

        # ── Draft ───────────────────────────────────────────────────
        draft = self._current_draft
        has_draft = draft is not None or is_embedded_mtp
        draft_state = ov["draft"] if "draft" in ov else has_draft
        self._chk_draft.blockSignals(True)
        self._chk_draft.setEnabled(has_draft)
        self._chk_draft.setChecked(has_draft and draft_state)
        if draft is not None:
            self._chk_draft.setText(f"Draft   {draft.name}  ({draft.size_gb:.1f} GB)")
        elif is_embedded_mtp:
            self._chk_draft.setText("Draft   MTP (embedded in GGUF)")
        else:
            self._chk_draft.setText("Draft (no assistant model found)")
        self._chk_draft.blockSignals(False)

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

        # ── Turbo KV-quant ──────────────────────────────────────────
        # Always enabled — the AutoTuner cannot detect whether the
        # active fork is a TurboQuant build (binary inspection would
        # be expensive and unreliable). On stock builds the toggle is
        # a harmless no-op because _turbo_quant_for() returns the
        # input label when no mapping exists. State is NOT persisted
        # per-model (it's a fork-level capability flag).
        self._chk_turbo_kv.setEnabled(True)
        # Don't touch the checked state on model switch — Turbo is a
        # session-global preference.

        # ── n-gram (ngram-mod) ──────────────────────────────────────
        # Always enabled: ngram-mod needs no draft model and works on any
        # GGUF, so it must never be greyed out (the whole point — "ngram
        # should always be available"). Default off (opt-in, since it can
        # slightly regress throughput on non-repetitive generation), but the
        # per-model choice is remembered like vision/draft/thinking.
        ngram_state = ov["ngram"] if "ngram" in ov else False
        self._chk_ngram.blockSignals(True)
        self._chk_ngram.setEnabled(True)
        self._chk_ngram.setChecked(ngram_state)
        self._chk_ngram.blockSignals(False)

        # ── mmproj precision dropdown ───────────────────────────────
        # Populate from the candidate list the scanner attached to the
        # model. Only shown when there's a real choice (>= 2 projectors).
        # The remembered selection (per model) wins; otherwise the entry's
        # auto-picked `mmproj` is preselected. Selecting here updates
        # `entry.mmproj` so the launch + preview use the chosen file.
        self._populate_mmproj_combo(entry, ov)

        # ── Prompt caching (host RAM, -cram) ────────────────────────
        # Traditionally, prompt caching was considered incompatible with the
        # multimodal/mtmd path. We now allow the user to toggle it anyway;
        # if the specific llama-server build refuses it, the server will
        # report an error in its terminal window.
        pc_state = ov["prompt_cache"] if "prompt_cache" in ov else True
        self._chk_prompt_cache.blockSignals(True)
        self._chk_prompt_cache.setEnabled(True)
        self._chk_prompt_cache.setChecked(pc_state)
        self._chk_prompt_cache.setText("Prompt caching (host RAM, -cram)")
        self._chk_prompt_cache.blockSignals(False)

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
        # Explicit "none" choice → keep index 0, skip auto-pick preselection.
        deliberate_none = remembered == app_settings.MMPROJ_NONE_SENTINEL
        for c in folder:
            compatible = is_mmproj_compatible(entry.path, c)
            label = c.name
            if c == auto:
                label += "   (auto)"
            if not compatible:
                label = WARN + label
            self._cb_mmproj.addItem(label, userData=str(c))
            idx = self._cb_mmproj.count() - 1
            if remembered and not deliberate_none and c.name == remembered:
                chosen_idx = idx
        # No remembered choice → preselect the scanner's auto pick if present.
        if not remembered and auto is not None:
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
        (incompatible ones flagged with '⚠'). Resolves and applies the
        per-model selection onto ``self._current_draft`` (a draft ModelEntry
        with metadata, or None). The remembered choice wins; otherwise the
        scanner's auto pick (``entry.draft``) is preselected, falling back to
        embedded-MTP when present.
        """
        WARN = "⚠ "
        NONE_LABEL = "— no draft —"
        MTP_LABEL = "MTP (embedded in GGUF)"
        MTP_DATA = "<embedded-mtp>"
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
        chosen_idx = 0
        if remembered == app_settings.DRAFT_NONE_SENTINEL:
            chosen_idx = 0
        for c in folder:
            md = read_gguf_metadata(c)
            compatible = is_draft_compatible(entry.path, c, md)
            label = f"{c.name}   ({c.stat().st_size / (1024**3):.1f} GB)"
            if auto is not None and c == auto:
                label += "  (auto)"
            if not compatible:
                label = WARN + label
            self._cb_draft.addItem(label, userData=str(c))
            idx = self._cb_draft.count() - 1
            if remembered and remembered not in ("", app_settings.DRAFT_NONE_SENTINEL):
                if c.name == remembered:
                    chosen_idx = idx
        # No remembered choice → prefer external auto pick, else embedded MTP.
        if not remembered:
            if auto is not None:
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
        if s and s != "<embedded-mtp>":
            self._current_draft = _make_draft_entry(Path(s), entry.group)
        else:
            self._current_draft = None

    def _auto_select_fork(self, profile: ModelProfile) -> None:
        """Auto-select fork from combo based on profile requirement.

        If the user has manually selected a fork (via dropdown or folder browse),
        respect that choice and do NOT override it — unless the profile requires
        a specific fork that is not available.
        """
        # Respect manual user override — only auto-switch if profile demands it
        if self._fork_manual_override:
            # Check if profile requires a specific fork
            if profile.server_binary:
                first = Path(profile.server_binary).parts[0]
                if not first.endswith(".cpp"):
                    first = first + ".cpp"
                first_l = first.lower()
                found = False
                for i in range(self._fork_combo.count()):
                    item_l = self._fork_combo.itemText(i).lower()
                    if item_l == first_l or item_l.rstrip(".cpp") in first_l:
                        found = True
                        break
                if not found:
                    self._log(
                        f"[Fork] Profile requires '{first}' but it's not available. "
                        f"Keeping manual selection: {self._fork_combo.currentText()}"
                    )
                # Keep manual selection regardless
            return

        # No manual override — apply profile-based auto-selection
        if profile.server_binary:
            first = Path(profile.server_binary).parts[0]
            if not first.endswith(".cpp"):
                first = first + ".cpp"
            first_l = first.lower()
            for i in range(self._fork_combo.count()):
                item_l = self._fork_combo.itemText(i).lower()
                if item_l == first_l or item_l.rstrip(".cpp") in first_l:
                    if self._fork_combo.currentIndex() != i:
                        self._fork_combo.blockSignals(True)
                        self._fork_combo.setCurrentIndex(i)
                        self._fork_combo.blockSignals(False)
                        self._apply_fork(i)
                        self._log(
                            f"[Fork] Auto-selected: {self._fork_combo.itemText(i)}"
                        )
                    return
        else:
            # No specific fork required — keep current selection, don't reset
            pass

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

    def _on_vision_toggled(self, checked: bool) -> None:
        self._record_override("vision", checked)
        # Vision interacts with prompt caching (mtmd is incompatible with
        # -cram) — re-run the checkbox logic so the prompt-cache box flips
        # enabled/disabled to match before rebuilding the preview.
        if self._current_entry is not None:
            self._update_checkboxes(self._current_entry)
        self._refresh_config_preview()

    def _on_draft_toggled(self, checked: bool) -> None:
        self._record_override("draft", checked)
        # Toggling draft changes whether vision is blocked (external draft
        # conflicts with --mmproj). Re-evaluate the vision checkbox state
        # before rebuilding the config preview.
        if self._current_entry is not None:
            self._update_checkboxes(self._current_entry)
        self._refresh_config_preview()

    def _on_turbo_toggled(self, checked: bool) -> None:
        """Turbo KV-quant toggle. Not persisted per-model: it's a
        fork-level capability flag rather than a model preference, so
        flipping it just rebuilds the preview / current Expert config.
        """
        self._refresh_config_preview()
        if self._config_stack.currentIndex() == 1:
            # Already in Expert mode → re-cascade through the panel so
            # the K/V quant widgets update to show the turbo-mapped labels.
            self._expert_panel._recompute(
                force_overrides=dict(self._expert_panel._user_pins)
            )

    def _on_thinking_toggled(self, checked: bool) -> None:
        self._record_override("thinking", checked)
        self._refresh_config_preview()

    def _on_ngram_toggled(self, checked: bool) -> None:
        # n-gram is independent of the model (no draft file needed), so it has
        # no effect on the vision/draft interlock — just persist and re-preview.
        self._record_override("ngram", checked)
        self._refresh_config_preview()

    def _on_prompt_cache_toggled(self, checked: bool) -> None:
        # Persist the per-model prompt-cache choice. No interlock with other
        # options (the only constraint — vision incompatibility — is enforced
        # by disabling the box in _update_checkboxes), so just record + preview.
        self._record_override("prompt_cache", checked)
        self._refresh_config_preview()

    def _on_mmproj_changed(self, index: int) -> None:
        """User picked a different vision projector from the dropdown.

        Updates the current model's ``mmproj`` to the chosen file (or None for
        the "— no mmproj —" entry), remembers the choice per model, re-runs the
        checkbox interlock (vision availability depends on mmproj), and
        refreshes the preview (the projector size feeds the VRAM estimate).
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
        # mmproj presence changes whether Vision can be enabled → re-run the
        # interlock so the checkbox label/state matches the new selection.
        self._update_checkboxes(self._current_entry)
        self._refresh_config_preview()

    def _on_draft_combo_changed(self, index: int) -> None:
        """User picked a different draft head from the dropdown.

        Resolves the selection onto ``self._current_draft`` (None for
        "— no draft —", a metadata-bearing draft ModelEntry for a file, None
        for embedded MTP since it lives inside the main GGUF), remembers the
        choice per model, re-runs the checkbox interlock (an active external
        draft blocks vision), and refreshes the preview.
        """
        if self._current_entry is None or index < 0:
            return
        data = self._cb_draft.itemData(index)
        s = "" if data is None else str(data)
        # Persist: "" → deliberate "no draft" sentinel; embedded-MTP →
        # sentinel too (no file to remember, embedded is implied by the GGUF);
        # otherwise the chosen draft filename.
        try:
            if not s or s == "<embedded-mtp>":
                app_settings.set_draft_selection(
                    self._current_entry.name, app_settings.DRAFT_NONE_SENTINEL
                )
            else:
                app_settings.set_draft_selection(self._current_entry.name, Path(s).name)
        except Exception as exc:
            self._log(f"[Warning] Could not save draft selection: {exc}")
        # Warn (don't block) when an incompatible draft was chosen.
        if s and s != "<embedded-mtp>":
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
        # Keep the Draft checkbox in sync with the dropdown: choosing a real
        # draft (or embedded MTP) implies draft ON; choosing "none" implies
        # OFF. We record this as the per-model "draft" override so the
        # interlock + launch path agree with the dropdown.
        draft_on = bool(s) and s != ""  # "" is the none entry
        if s == "":
            draft_on = False
        self._record_override("draft", draft_on)
        # An active external draft conflicts with --mmproj → re-run interlock.
        self._update_checkboxes(self._current_entry)
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
    ) -> Optional[TunedConfig]:
        """Helper: rebuild a TunedConfig for the given model with the
        current checkbox states. Returns None when system info is missing.

        Centralised so both the preview path and the Expert panel's
        recompute callback share the same code path (and therefore the
        same handling of vision / draft / turbo_kv).
        """
        if self._system is None:
            return None

        use_vision = self._chk_vision.isChecked() and self._chk_vision.isEnabled()
        use_draft = self._chk_draft.isChecked() and self._chk_draft.isEnabled()
        turbo_kv = self._chk_turbo_kv.isChecked() and self._chk_turbo_kv.isEnabled()

        entry_for_cfg = copy.copy(entry)
        if not use_vision:
            entry_for_cfg.mmproj = None

        # Build the kwargs dict carefully — only forward keys whose
        # values the caller actually pinned. Sending None for an unset
        # force_* parameter is fine (compute_config handles it), but
        # being explicit makes the call site easier to read in logs.
        kwargs = dict(force_overrides or {})

        try:
            return compute_config(
                model=entry_for_cfg,
                system=self._system,
                profile=profile,
                draft_model=self._current_draft if use_draft else None,
                force_mlock=False,
                perf_target=self._resolve_perf_target_for_profile(profile),
                mode=self._current_mode(),
                turbo_kv=turbo_kv,
                gpu_priorities=app_settings.get_gpu_priorities(),
                force_gpu=app_settings.get_forced_gpu(),
                **kwargs,
            )
        except Exception as exc:
            self._log(f"[Warning] compute_config failed: {exc}")
            return None

    def _effective_config(
        self, entry: ModelEntry, profile: ModelProfile
    ) -> Optional[TunedConfig]:
        """The config actually shown in the preview and used at launch.

        Honours a saved Expert override per model; otherwise the
        AutoTuner's auto-tuned default. This is what makes a hand-tuned
        Expert setup "stick" for a model (the low-VRAM use case): once
        saved, the override is applied automatically, just like the
        vision/draft/thinking checkbox overrides.

        * Auto-mode overrides are re-derived through ``compute_config``
          with the saved pins so they ADAPT to the current VRAM /
          checkbox state, then the saved non-cascading values are
          stamped back on.
        * Manual-mode overrides are applied as a frozen config (the user
          owns the exact values); the launch-path VRAM fit-check still
          gates them, and Reset reverts to Auto.
        """
        base = self._build_auto_config(entry, profile)
        if base is None:
            return None
        override = app_settings.get_expert_override(entry.name)
        if not override:
            return base
        vals = override.get("values") or {}
        try:
            if override.get("mode") == "manual" and vals:
                return expert_cfg_from_values(base, vals)
            # Auto mode: re-cascade from the saved pins (adapts to the
            # live VRAM / checkbox state), then overlay the saved
            # non-cascading widget values (threads / batch / flags / …).
            pins = {
                k: v
                for k, v in (override.get("pins") or {}).items()
                if v is not None
            }
            cascaded = self._build_auto_config(entry, profile, pins) or base
            if vals:
                cascaded = apply_expert_values(cascaded, vals)
            return cascaded
        except Exception as exc:
            self._log(
                f"[Warning] Saved Expert override for {entry.name} invalid "
                f"({exc}); falling back to Auto."
            )
            return base

    def _load_expert_panel(
        self, entry: ModelEntry, profile: ModelProfile
    ) -> None:
        """Bind the Expert panel to the current model + apply any saved
        Expert override. Used when entering Expert mode and when a
        checkbox toggles while the panel is already open."""
        assert self._system is not None  # callers guard / assert this first
        cfg = self._build_auto_config(entry, profile)
        if cfg is None:
            return
        self._expert_panel.configure_for_model(
            cfg=cfg,
            system=self._system,
            native_ctx=entry.native_context,
            profile_max=profile.max_context,
            recompute_cb=lambda overrides: self._build_auto_config(
                entry, profile, overrides
            ),
        )
        override = app_settings.get_expert_override(entry.name)
        if override:
            self._expert_panel.restore_from_snapshot(override)

    def _update_config_text(self, entry: ModelEntry, profile: ModelProfile) -> None:
        """Recompute the effective config (Auto or saved Expert override)
        using the current checkbox states and refresh the preview."""
        assert self._system is not None
        eff = self._effective_config(entry, profile)
        if eff is None:
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
        use_vision = self._chk_vision.isChecked() and self._chk_vision.isEnabled()
        use_draft = self._chk_draft.isChecked() and self._chk_draft.isEnabled()
        turbo_kv = self._chk_turbo_kv.isChecked() and self._chk_turbo_kv.isEnabled()
        use_ngram = self._chk_ngram.isChecked() and self._chk_ngram.isEnabled()
        use_prompt_cache = (
            self._chk_prompt_cache.isChecked() and self._chk_prompt_cache.isEnabled()
        )

        W = 64
        bar = "─" * W
        lines = [bar]
        lines.append(f"Model   : {entry.name}")
        lines.append(
            f"Profile : {profile.display_name}"
            + (f"  ({profile.source_file})" if profile.source_file else "")
        )
        if profile.notes:
            for i in range(0, len(profile.notes.strip()), W - 10):
                prefix = "Notes   : " if i == 0 else "          "
                lines.append(f"{prefix}{profile.notes.strip()[i : i + W - 10]}")
        if entry.mmproj:
            vis = "✓" if use_vision else "✗"
            lines.append(f"Vision  : {entry.mmproj.name}  [{vis}]")
        if self._current_draft:
            drf = "✓" if use_draft else "✗"
            lines.append(f"Draft   : {self._current_draft.name}  [{drf}]")
        if use_ngram:
            lines.append("n-gram  : ngram-mod (self-speculative)  [✓]")

        lines.append(
            f"Prompt$ : host-RAM cache (-cram)  [{'✓' if use_prompt_cache else '✗'}]"
            + (" (may conflict with Vision)" if use_vision else "")
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
        elif turbo_kv:
            kv_line += "  [turbo]"

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
            f"Flash attention : {'on' if cfg.flash_attn else 'off'}",
        ]
        if cfg.mlock:
            lines.append("mlock           : on")
        if cfg.no_kv_offload:
            # LOW-VRAM lever (low_vram perf-target): the KV cache lives in
            # system RAM, attention compute runs on CPU. Surface it so the
            # user understands why context is huge but generation is slower.
            lines.append(
                "KV in RAM       : on (--no-kv-offload)  "
                "[slower gen, max context]"
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
        )
        total_cpu = cfg.estimated_model_ram_gb + cfg.kv_ram_gb
        lines += [bar, "Memory estimate (with current options):"]
        lines.append(
            f"  Model GPU : ~{cfg.estimated_model_vram_gb:5.1f} GB"
            f"   (free VRAM: {self._system.free_vram_gb:.1f} GB)"
        )
        if cfg.vision_vram_gb > 0.05:
            lines.append(f"  Vision GPU: ~{cfg.vision_vram_gb:5.1f} GB")
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
        lines.append(
            f"  Total GPU : ~{total_gpu:5.1f} GB"
            f"   of {self._system.free_vram_gb:.1f} GB free"
        )
        lines.append(
            f"  Model CPU : ~{cfg.estimated_model_ram_gb:5.1f} GB"
            f"   (free RAM:  {self._system.free_ram_gb:.1f} GB)"
        )
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
            and self._system.total_vram_gb <= 12
            and self._system.free_ram_gb >= 16
            and cfg.ctx < 32768
        ):
            lines.append(
                "  💡 Low VRAM? Try Performance → low_vram — moves the KV "
                "cache into RAM for far larger context (slower gen)."
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
        override = app_settings.get_expert_override(entry.name)
        if override:
            self._log(
                f"[Expert] Entered Expert mode — restored saved "
                f"{override.get('mode', 'auto')} settings for {entry.name}."
            )
        else:
            self._log("[Expert] Entered Expert mode (Auto).")

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
        """Persist the live Expert state for the current model (debounced).

        This is the autosave the low-VRAM workflow asked for: every edit
        in either mode lands on disk keyed by model name, so the setup
        is restored next time and applied at launch like the checkbox
        overrides.
        """
        entry = self._current_entry
        if entry is None or not isinstance(snapshot, dict):
            return
        try:
            app_settings.set_expert_override(entry.name, snapshot)
        except Exception as exc:
            self._log(f"[Warning] Could not save Expert settings: {exc}")

    def _on_expert_reset(self) -> None:
        """Reset button: forget the saved Expert override for the current
        model and reload the AutoTuner's automatically-best config."""
        entry = self._current_entry
        if entry is None or self._system is None:
            return
        profile = match_profile(
            entry.name,
            self._profiles,
            getattr(entry, "architecture", ""),
        )
        # Make sure a pending save for the old state cannot land AFTER
        # the clear (it would resurrect the override we just dropped).
        self._expert_panel._save_timer.stop()
        try:
            app_settings.clear_expert_override(entry.name)
        except Exception as exc:
            self._log(f"[Warning] Could not clear Expert settings: {exc}")
        # Reload pure Auto into the panel (no save) and refresh preview.
        self._expert_panel.reset_to_auto()
        eff = self._effective_config(entry, profile)
        if eff is not None:
            self._render_cfg_to_preview(entry, profile, eff)
        self._log(f"[Expert] Reset to Auto for {entry.name}.")

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

        # Single OK button — this is a read-only inspector, no actions.
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(dlg.reject)
        bb.accepted.connect(dlg.accept)
        # QDialogButtonBox.Close emits the `rejected` signal by default;
        # wire both so either path closes cleanly.
        close_btn: QPushButton | None = bb.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(dlg.accept)
        layout.addWidget(bb)

        dlg.exec()
        # Also mirror a short notice into the main log so the user has
        # a record that they consulted the diagnostic (helpful when
        # debugging support tickets later).
        self._log(f"[Diagnose] Inspected metadata for {self._current_entry.name}")

    # ------------------------------------------------------------------
    # System info — non-blocking (daemon thread → signal/slot)
    # ------------------------------------------------------------------
    def _sysinfo_async(self) -> None:
        if self._sysinfo_busy:
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
            s = detect_system()
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

        # VRAM-Anzeige
        if s.total_vram_gb > 0:
            self._vram_lbl.setText(
                f"VRAM: {s.free_vram_gb:.1f} / {s.total_vram_gb:.1f} GB free"
            )
        else:
            self._vram_lbl.setText("VRAM: keine GPU")

        # RAM-Anzeige
        self._ram_lbl.setText(
            f"RAM: {s.free_ram_gb:.1f} / {s.total_ram_gb:.1f} GB free"
        )

        # CPU-Anzeige
        if s.cpu_name:
            display_cpu = (s.cpu_name[:40] + '...') if len(s.cpu_name) > 40 else s.cpu_name
            self._cpu_lbl.setText(f"CPU: {display_cpu}")

        # GPU-Anzeige mit Utilization
        if s.gpus:
            gpu_parts = []
            for g in s.gpus:
                util = f"{g.gpu_util_percent:.0f}%" if g.gpu_util_percent > 0 else "—"
                display_name = (g.name[:25] + '...') if len(g.name) > 25 else g.name
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
                    display_ign_name = (g.name[:20] + '...') if len(g.name) > 20 else g.name
                    ign_parts.append(f"{display_ign_name} ({size}, ignored)")
                txt += "  ·  " + ", ".join(ign_parts)
            self._gpu_lbl.setText(txt)
        else:
            self._gpu_lbl.setText("GPU: keine")

        self._log(
            f"[SysInfo] CPU={s.cpu_name}, VRAM={s.free_vram_gb:.1f}/{s.total_vram_gb:.1f}GB, RAM={s.free_ram_gb:.1f}/{s.total_ram_gb:.1f}GB, GPU={[g.name for g in s.gpus]}"
        )

        # Keep the GPU pin dropdown in sync with the detected cards.
        self._populate_gpu_combo(s)

    # ------------------------------------------------------------------
    # Binary resolution
    # ------------------------------------------------------------------
    def _resolve_binary(
        self, profile: ModelProfile, use_draft: bool, model_name: str
    ) -> str:
        # ik_llama.cpp is only required for Gemma 4 with an *external* sibling drafter.
        # Integrated MTP (Qwen3.6-MTP) uses --spec-type draft-mtp and works in mainline b9190+.
        try:
            _, resolve, _ = _get_fork_tools()
        except Exception:
            return "llama-server"
        if (
            "gemma-4" in model_name.lower() or "gemma4" in model_name.lower()
        ) and use_draft:
            spec = profile.server_binary or "ik_llama.cpp/llama-server"
        elif profile.server_binary:
            spec = profile.server_binary
        else:
            spec = "llama-server"
        resolved = resolve(spec)
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
        for s in self._servers:
            proc = s.get("proc")
            if proc is not None and proc.is_running():
                live.append(s)
        self._servers = live

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
        self, cfg: TunedConfig, entry: ModelEntry
    ) -> Tuple[Optional[object], Optional[str]]:
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
            fresh = detect_system()
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

    def _pin_cfg_to_gpu(self, cfg: TunedConfig, gpu: object) -> None:
        """Force this server's tensors onto a specific GPU via env vars.

        Sets HIP_VISIBLE_DEVICES / GGML_VK_VISIBLE_DEVICES to the chosen
        card's device index so a second/third concurrent model lands on the
        emptier GPU instead of defaulting back onto the (full) primary.
        Clears any tensor_split the single-GPU pin would conflict with.
        """
        hip_index = getattr(gpu, "hip_index", None)
        name = getattr(gpu, "name", "?")
        if hip_index is None:
            self._log(
                f"[Balance] {name} has no resolved device index; cannot hard-pin "
                "— relying on tensor-split. Ensure vulkaninfo is reachable."
            )
            return
        vis = str(hip_index)
        cfg.env_overrides = dict(cfg.env_overrides or {})
        cfg.env_overrides["HIP_VISIBLE_DEVICES"] = vis
        cfg.env_overrides["GGML_VK_VISIBLE_DEVICES"] = vis
        # After remapping, the chosen card is the only visible device (idx 0).
        cfg.main_gpu = 0
        cfg.tensor_split = None
        self._last_pinned_gpu = name
        self._log(f"[Balance] Pinned to {name} (device {vis}).")

    # ------------------------------------------------------------------
    # Server control
    # ------------------------------------------------------------------
    def _launch_server(self) -> None:
        # Multi-server: we no longer refuse when one is already running.
        # Prune any that have exited so the port counter and VRAM picture
        # are current before we plan this launch.
        self._prune_dead_servers()

        if self._current_entry is None:
            QMessageBox.warning(
                self, "No model selected", "Click a model in the list first."
            )
            return

        if self._system is None:
            QMessageBox.warning(
                self,
                "System info unavailable",
                "Hardware detection has not completed yet. Please wait a moment and try again.",
            )
            return

        use_vision = self._chk_vision.isChecked() and self._chk_vision.isEnabled()
        use_draft = self._chk_draft.isChecked() and self._chk_draft.isEnabled()
        use_thinking = self._chk_thinking.isChecked() and self._chk_thinking.isEnabled()
        # turbo_kv is read by _build_auto_config/_effective_config straight
        # from the checkbox, so there's no local to forward here.
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

        # Resolve the launch config:
        #   • Expert panel open → the user is editing live; flush any
        #     pending autosave so the on-disk override matches what we
        #     launch, then use the panel's current config.
        #   • Panel closed → the per-model saved Expert override if one
        #     exists (so a hand-tuned setup is applied automatically),
        #     otherwise the AutoTuner's auto-tuned default.
        expert_open = self._config_stack.currentIndex() == 1
        cfg: Optional[TunedConfig] = None
        if expert_open:
            self._expert_panel.flush_pending_save()
            cfg = self._expert_panel.current_config()
            if cfg is None:
                self._log("[Warning] Expert panel had no config; falling back to auto.")
        if cfg is None:
            cfg = self._effective_config(entry, profile)
        # cfg is always non-None here: either the expert panel provided it
        # or compute_config just returned one.  The assert narrows the type
        # for static checkers (Pylance / mypy) that cannot prove this.
        assert cfg is not None

        # ── Diffusion routing ────────────────────────────────────────
        # llama-diffusion-gemma-server (PR #24427) is a REAL persistent
        # OpenAI HTTP server (/health, /v1/chat/completions, port) → run it
        # through the normal launch path below (port / health / registry),
        # just with the dedicated binary + command builder. Everything else
        # diffusion (mainline Dream/LLaDA/RND1) is single-shot CLI: no port,
        # no /health, no registry. The binary is found in the SELECTED fork
        # (the fork dropdown points LLAMA_CPP_DIR at it).
        if profile.runner == "llama-diffusion-cli" or (
            entry.is_diffusion
            and profile.runner != "llama-diffusion-gemma-server"
        ):
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
            needle = forced_token.strip().lower()
            forced_here = next(
                (g for g in self._system.gpus if needle in g.name.lower()), None
            )
        if forced_here is not None:
            need = (
                float(cfg.estimated_model_vram_gb)
                + float(cfg.kv_vram_gb)
                + float(cfg.vision_vram_gb)
                + float(cfg.draft_vram_gb)
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
            chosen_gpu, refusal = self._choose_gpu_for_launch(cfg, entry)
            if refusal is not None:
                self._log(f"[Balance] Launch refused — {refusal.splitlines()[0]}")
                QMessageBox.warning(self, "Not enough free VRAM", refusal)
                return
            if chosen_gpu is not None:
                self._pin_cfg_to_gpu(cfg, chosen_gpu)
        else:
            # First model: still verify it actually fits somewhere so the
            # user gets a clear message instead of an opaque server crash.
            _gpu, refusal = self._choose_gpu_for_launch(cfg, entry)
            if refusal is not None:
                # For a single multi-GPU-splittable model the per-card check
                # can be over-strict, so only hard-refuse on single-GPU /
                # CPU systems; otherwise warn and let the split proceed.
                if self._system and len(self._system.gpus) <= 1:
                    self._log(f"[Balance] Launch refused — {refusal.splitlines()[0]}")
                    QMessageBox.warning(self, "Not enough free VRAM", refusal)
                    return
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

            try:
                _, _, resolve_diff = _get_fork_tools()
            except Exception as exc:
                self._log(f"[Diffusion-Server] Resolver nicht ladbar: {exc}")
                QMessageBox.warning(
                    self,
                    "DiffusionGemma-Server nicht verfügbar",
                    f"Der llama-diffusion-gemma-server-Resolver konnte nicht "
                    f"geladen werden:\n{exc}",
                )
                return
            arch = (entry.metadata or {}).get("general.architecture")
            gemma_server_bin = resolve_diff(
                "llama-diffusion-gemma-server", arch=arch
            )
            self._log(
                f"[Diffusion-Server] binary: 'llama-diffusion-gemma-server' "
                f"→ {gemma_server_bin} (arch={arch!r})"
            )
            if not Path(gemma_server_bin).is_file() and not shutil.which(
                gemma_server_bin
            ):
                self._log(
                    f"[Diffusion-Server] Binary nicht gefunden: {gemma_server_bin}"
                )
                QMessageBox.warning(
                    self,
                    "llama-diffusion-gemma-server nicht gefunden",
                    "DiffusionGemma benötigt llama-diffusion-gemma-server aus "
                    "einem DiffusionGemma-Fähigen Build (PR #24427).\n\n"
                    "Wähle im Fork-Dropdown den Build, der "
                    "llama-diffusion-gemma-server enthält (z.B. "
                    "d_b9781_llama.cpp / d_bXXXX_hip_llama.cpp).",
                )
                return
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
            server_binary = self._resolve_binary(profile, use_draft, entry.name)
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
                # The Draft checkbox governs BOTH external draft (-md) and embedded
                # MTP. For an MTP model draft_model is None, so unchecking Draft must
                # also flip enable_speculative off to actually suppress the MTP path.
                enable_speculative=use_draft,
                enable_ngram=use_ngram,
                enable_prompt_cache=use_prompt_cache,
            )

        self._log("\n" + "─" * 60)
        self._log(f"Starting: {' '.join(cmd)}")
        self._log(
            f"Options : vision={use_vision} draft={use_draft} thinking={use_thinking} "
            f"ngram={use_ngram} prompt_cache={use_prompt_cache} "
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

        proc = _TerminalProcess(cmd, env_overrides=cfg.env_overrides)
        try:
            proc.start()
        except FileNotFoundError:
            self._log(f"[Error] Binary not found: {cmd[0]}")
            self._log("  → Check fork selection or set LLAMA_CPP_DIR / LLAMA_SERVER")
            return

        pid = proc.proc.pid if proc.proc else "?"
        base_url = f"http://{host}:{port}"
        self._log(f"[AutoTuner] Server started — PID: {pid}")
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
            "ready": False,
            "model": entry.name,
            "gpu": getattr(self, "_last_pinned_gpu", None),
            "vram_gb": float(cfg.estimated_model_vram_gb) + float(cfg.kv_vram_gb),
        }
        self._next_server_id += 1
        self._servers.append(record)
        self._server = proc
        self._server_base_url = base_url
        self._server_ready = False
        self._last_pinned_gpu = None

        # Stop is enabled whenever ≥1 server runs; Launch stays enabled so the
        # user can fire up another model on the next port.
        self._btn_launch.setEnabled(True)
        self._btn_stop.setEnabled(True)
        self._btn_stop_all.setEnabled(True)
        self._refresh_server_combo()
        self._status.showMessage(
            f"Loading model — PID {pid} — {base_url}  "
            f"({len(self._servers)} server(s) running)"
        )

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

        if not Path(diffusion_bin).is_file() and not shutil.which(diffusion_bin):
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

        self._log("\n" + "─" * 60)
        self._log(f"Diffusion: {' '.join(cmd)}")
        if cfg.env_overrides:
            for k, v in cfg.env_overrides.items():
                self._log(f"Env     : {k}={v}")

        # Run in a terminal window like the server path, but do NOT register
        # it as a server (no port / health / switcher entry). It exits on
        # its own when generation finishes.
        proc = _TerminalProcess(cmd, env_overrides=cfg.env_overrides)
        try:
            proc.start()
        except FileNotFoundError:
            self._log(f"[Error] Binary not found: {cmd[0]}")
            self._log("  → Check fork selection or set LLAMA_CPP_DIR")
            return

        pid = proc.proc.pid if proc.proc else "?"
        self._log(f"[Diffusion] Started — PID: {pid}")
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
            self._btn_launch.setEnabled(True)
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
        self._btn_launch.setEnabled(True)
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
        self._btn_launch.setEnabled(True)
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
        for record in self._servers:
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
            self._btn_launch.setEnabled(True)
            self._refresh_server_combo()

        # Health-probe any not-yet-ready server so its status flips to Ready.
        for record in self._servers:
            if record.get("ready"):
                continue
            base_url = record.get("base_url")
            if not base_url:
                continue
            try:
                import urllib.request

                with urllib.request.urlopen(f"{base_url}/health", timeout=0.3) as resp:
                    ready = resp.status == 200
            except Exception:
                ready = False
            if ready:
                record["ready"] = True
                proc = record.get("proc")
                pid = proc.proc.pid if proc is not None and proc.proc else "?"
                self._log(
                    f"[AutoTuner] Server ready (/health → 200) — "
                    f"port {record.get('port')}."
                )
                self._refresh_server_combo()  # flip the …→✓ marker in the list
                if record is self._servers[-1]:
                    self._server_ready = True
                    self._status.showMessage(
                        f"Ready — PID {pid} — {base_url}  "
                        f"({len(self._servers)} server(s) running)"
                    )

    # ------------------------------------------------------------------
    # Log helper
    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        self._log_panel.append(msg.rstrip("\n"))
        sb = self._log_panel.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------
    def closeEvent(self, a0: QCloseEvent | None) -> None:  # noqa: N802
        # Snapshot the current window layout BEFORE any potential
        # "are you sure?" dialog, so even an Escape-out of that dialog
        # has saved state. The save itself never blocks the close.
        self._persist_window_geometry()

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
                if a0 is not None:
                    a0.ignore()
                return
            self._stop_all_servers()

        if a0 is not None:
            a0.accept()


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    p = argparse.ArgumentParser(
        prog="qt_launcher", description="AutoTuner Qt GUI launcher"
    )
    p.add_argument("--models-path", default=str(_default_models_path()))
    p.add_argument("--settings-path", default=str(_default_settings_path()))
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    # Hide the parent console on Windows when launched via python.exe
    if os.name == "nt":
        try:
            import ctypes

            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("AutoTuner")
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
    window.show()

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
            window._stop_all_servers()
        except Exception:
            pass
        app.quit()

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

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
    