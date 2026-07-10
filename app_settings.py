"""Persistent app settings for the AutoTuner GUI.

Stores last-used paths so that a manually picked models folder or
llama.cpp fork is remembered across launches. JSON file lives next to
the script when writable (portable), otherwise in the user's home dir.

Public API:
    load_settings()        -> dict
    save_settings(dict)    -> bool
    get_models_path()      -> Optional[Path]
    set_models_path(Path)
    get_model_paths()      -> list[(Path, enabled)]
    set_model_paths(list[(Path, enabled)])
    get_fork_path()        -> Optional[Path]
    set_fork_path(Path)
    get_llama_build_paths() -> list[(Path, enabled)]
    set_llama_build_paths(list[(Path, enabled)])
    get_window_geometry()  -> Optional[str]   # base64 of QByteArray
    set_window_geometry(str)
    get_window_state()     -> Optional[str]   # base64 of QByteArray (toolbars/docks)
    set_window_state(str)
    get_splitter_state(name) -> Optional[str]  # base64 of a QSplitter's saveState()
    set_splitter_state(name, str)
    get_mmproj_selection(model_name) -> Optional[str]  # chosen projector filename
    set_mmproj_selection(model_name, filename)
    get_font_size()        -> Optional[int]
    set_font_size(int)
    get_minimize_on_close() -> bool
    set_minimize_on_close(bool)
    get_base_port()        -> int
    set_base_port(int)
    get_port_offset()      -> int
    set_port_offset(int)
    get_reasoning_effort(model_name) -> Optional[str]
    set_reasoning_effort(model_name, value)
    get_expert_override(model_name) -> Optional[dict]   # saved Expert-panel state
    set_expert_override(model_name, snapshot: dict)
    clear_expert_override(model_name)
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_FILENAME = "autotuner_settings.json"


def app_data_dir() -> Path:
    r"""Persistent, user-writable directory for settings / logs / update state.

    Two regimes:
      * **Source install** — the script directory (portable, same folder the
        user runs ``qt_launcher.py`` from).
      * **Frozen build** (PyInstaller onefile) — the directory that contains
        the compiled ``AutoTuner.exe`` / Linux binary. The bundled code runs
        from a throw-away ``_MEIPASS`` temp folder that is DELETED on exit, so
        any user state written there (``autotuner_settings.json``, logs, …)
        would silently vanish between launches. Routing all user-writable
        state through the EXE folder keeps it stable across runs.

    If the resolved directory is read-only (e.g. the EXE sits in
    ``C:\Program Files``), fall back to the user's home directory so the
    app never crashes trying to persist state. Works on Windows and Linux.
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent
    try:
        probe = base / ".autotuner_write_probe"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        return base
    except (OSError, PermissionError):
        return Path.home()


def _settings_file() -> Path:
    """Resolve the settings file location.

    Preference: a portable install (alongside the script when running from
    source, or next to the .exe when frozen). Fallback: the user's home
    directory if that location is read-only (e.g. Program Files).
    """
    return app_data_dir() / _FILENAME


def load_settings() -> Dict[str, Any]:
    """Load settings from disk; return {} on missing file or parse error."""
    f = _settings_file()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(data: Dict[str, Any]) -> bool:
    """Atomically save settings; return True on success, False otherwise.

    Writes to a temp file in the same directory then renames, so a
    crash mid-write never leaves a half-written settings file.
    """
    f = _settings_file()
    tmp = f.with_suffix(f.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, f)
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _update(key: str, value: Any) -> None:
    s = load_settings()
    s[key] = value
    save_settings(s)


# ---------------------------------------------------------------------------
# Convenience accessors

# OS-namespaced path keys. The settings JSON is portable and on dual-boot
# machines SHARED between the Windows and the Linux boot (it lives next to
# the script on the data partition). An absolute path saved on one OS is
# invalid on the other, so a single key ping-ponged: every boot the "other"
# OS lost its models/fork selection ("/run/media/…" read on Windows, "L:\…"
# read on Ubuntu) and overwrote the entry when the user re-picked it.
# Path-valued settings are therefore stored per-OS ("models_path.windows" /
# "models_path.linux"); the plain legacy key is still read as a fallback so
# existing files migrate seamlessly, and it is mirrored on write so an older
# AutoTuner version on the same OS keeps working.

_OS_KEY_SUFFIX = "windows" if os.name == "nt" else "linux"


def _os_path_key(key: str) -> str:
    return f"{key}.{_OS_KEY_SUFFIX}"


def _get_os_path(key: str) -> Optional[Path]:
    """Read a per-OS path setting (legacy plain key as fallback); must exist."""
    s = load_settings()
    for k in (_os_path_key(key), key):
        p = s.get(k)
        if p:
            pp = Path(p)
            if pp.exists():
                return pp
    return None


def _set_os_path(key: str, value: str) -> None:
    s = load_settings()
    s[_os_path_key(key)] = value
    s[key] = value  # legacy mirror for older AutoTuner versions
    save_settings(s)


def get_models_path() -> Optional[Path]:
    return _get_os_path("models_path")


def set_models_path(path: Path) -> None:
    _set_os_path("models_path", str(path.resolve()))


PathEnabled = Tuple[Path, bool]


def _read_path_list(key: str) -> List[PathEnabled]:
    """Read a persisted multi-folder list as ``[(Path, enabled), ...]``.

    The JSON schema is intentionally small and human-editable:
    ``[{"path": "...", "enabled": true}, ...]``. Invalid and duplicate
    paths are skipped; missing folders are kept so the GUI can still show and
    edit a removable stale entry.
    """
    s = load_settings()
    raw = s.get(_os_path_key(key))
    if not isinstance(raw, list) or not raw:
        # Legacy fallback: plain key written before per-OS namespacing (or by
        # an older version). Stale other-OS entries surface as editable
        # missing folders in the GUI, exactly like before.
        raw = s.get(key)
    if not isinstance(raw, list):
        return []
    out: List[PathEnabled] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, dict):
            p_raw = item.get("path")
            enabled = bool(item.get("enabled", True))
        else:
            p_raw = item
            enabled = True
        if not p_raw:
            continue
        try:
            p = Path(str(p_raw)).expanduser()
            # ``resolve(strict=False)`` normalises duplicates without requiring
            # the directory to still exist.
            rp = p.resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        key_text = os.path.normcase(str(rp))
        if key_text in seen:
            continue
        seen.add(key_text)
        out.append((rp, enabled))
    return out


def _write_path_list(key: str, paths: List[PathEnabled]) -> None:
    clean = []
    seen: set[str] = set()
    for path, enabled in paths:
        try:
            rp = Path(path).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        key_text = os.path.normcase(str(rp))
        if key_text in seen:
            continue
        seen.add(key_text)
        clean.append({"path": str(rp), "enabled": bool(enabled)})
    s = load_settings()
    s[_os_path_key(key)] = clean
    s[key] = clean  # legacy mirror for older AutoTuner versions
    save_settings(s)


def get_model_paths() -> List[PathEnabled]:
    """Return configured model folders with their enabled state.

    Empty means the multi-folder setting has never been written; callers should
    fall back to ``models_path`` / ``AUTOTUNER_MODELS`` / defaults and then save
    through ``set_model_paths`` once the user edits the list.
    """
    return _read_path_list("model_paths")


def set_model_paths(paths: List[PathEnabled]) -> None:
    _write_path_list("model_paths", paths)
    first_enabled = next((p for p, enabled in paths if enabled), None)
    if first_enabled is None and paths:
        first_enabled = paths[0][0]
    if first_enabled is not None:
        set_models_path(first_enabled)


def get_fork_path() -> Optional[Path]:
    return _get_os_path("fork_path")


def set_fork_path(path: Path) -> None:
    _set_os_path("fork_path", str(path.resolve()))


# ---------------------------------------------------------------------------
# Fork-container path
#
# When the user picks a folder via "📂 Fork", they often pick a *parent*
# directory that holds several llama.cpp builds (e.g. C:\LAB\ai-local with
# `1b_llama.cpp/`, `atq_llama.cpp/`, `ik_llama.cpp/` inside). We must
# remember that container — not just the currently-selected build — so
# the next launch still shows ALL siblings instead of forcing the user
# to re-navigate up one level.
#
# `fork_path` keeps tracking the *currently active* build for things
# like `LLAMA_CPP_DIR`; `fork_container_path` is the root the GUI
# expanded the combo from.


def get_fork_container_path() -> Optional[Path]:
    return _get_os_path("fork_container_path")


def set_fork_container_path(path: Path) -> None:
    _set_os_path("fork_container_path", str(path.resolve()))


def clear_fork_container_path() -> None:
    s = load_settings()
    changed = False
    for k in (_os_path_key("fork_container_path"), "fork_container_path"):
        if k in s:
            s.pop(k, None)
            changed = True
    if changed:
        save_settings(s)


def get_llama_build_paths() -> List[PathEnabled]:
    """Return configured llama.cpp build/container folders with enabled state."""
    return _read_path_list("llama_build_paths")


def set_llama_build_paths(paths: List[PathEnabled]) -> None:
    """Persist llama.cpp build/container folders.

    The selected active fork remains stored separately in ``fork_path`` because
    this list represents scan roots (containers or individual builds), not the
    combo-box selection.
    """
    _write_path_list("llama_build_paths", paths)


# ---------------------------------------------------------------------------
# Per-model option overrides (vision / draft / thinking)
#
# Once a user toggles vision/draft/thinking for a specific model they
# expect that choice to stick — across performance-target changes,
# across selecting a different model and coming back, and across app
# restarts. We persist a small dict keyed by `entry.name` (the GGUF
# filename stem, which is stable for a given file on disk).
#
# Schema:
#   "model_overrides": {
#       "Qwen3.5-30B-A3B-UD-Q4_K_XL": {
#           "vision":       true,
#           "draft":        false,
#           "thinking":     true,
#           "ngram":        false,
#           "prompt_cache": true
#       },
#       ...
#   }
#
# Absent keys mean "use the model's default capability detection" so
# turning a feature back on is just a matter of clearing the override.

_OVERRIDE_KEYS = ("vision", "draft", "thinking", "ngram", "prompt_cache")


def get_model_overrides(model_name: str) -> Dict[str, bool]:
    """Return the per-model checkbox overrides, or {} when nothing stored."""
    if not model_name:
        return {}
    overrides = load_settings().get("model_overrides") or {}
    raw = overrides.get(model_name) or {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, bool] = {}
    for k in _OVERRIDE_KEYS:
        if k in raw:
            out[k] = bool(raw[k])
    return out


def set_model_override(model_name: str, key: str, value: bool) -> None:
    """Persist a single (model, option) → bool override.

    `key` must be one of "vision", "draft", "thinking", "ngram",
    "prompt_cache"; anything else is silently ignored to keep the JSON
    file uncluttered.
    """
    if not model_name or key not in _OVERRIDE_KEYS:
        return
    s = load_settings()
    overrides = s.get("model_overrides")
    if not isinstance(overrides, dict):
        overrides = {}
    cur = overrides.get(model_name)
    if not isinstance(cur, dict):
        cur = {}
    cur[key] = bool(value)
    overrides[model_name] = cur
    s["model_overrides"] = overrides
    save_settings(s)


def clear_model_overrides(model_name: str) -> None:
    """Drop all stored overrides for a single model (e.g. on uninstall)."""
    if not model_name:
        return
    s = load_settings()
    overrides = s.get("model_overrides") or {}
    if isinstance(overrides, dict) and model_name in overrides:
        overrides.pop(model_name, None)
        s["model_overrides"] = overrides
        save_settings(s)


# ---------------------------------------------------------------------------
# Expert-panel state (per model)
#
# The Expert panel lets a user override the AutoTuner's decisions for a
# single model — context length, KV quants, layer placement, threads,
# sampling, flags, reasoning, … Before this existed the panel started
# from the auto defaults every time the user (re)opened it, so a low-VRAM
# user had to re-enter the same hand-tuned settings on every launch.
#
# We now persist the full panel state per model so it is restored the
# next time that model is selected — and applied at launch just like the
# checkbox overrides above, completing the "remembers everything" story.
#
# Schema (stored under "expert_overrides", keyed by model name):
#   "expert_overrides": {
#       "Qwen3.5-30B-A3B-UD-Q4_K_XL": {
#           "mode": "auto",            # "auto" | "manual"
#           "pins": {                   # auto-mode cascade pins
##               "user_ctx": 32768,
#               "force_cache_k": "q8_0"
#           },
#           "values": {                 # full widget snapshot (both modes)
#               "ctx": 32768, "cache_k": "q8_0", …
#           },
#           "saved_at": "2026-06-30T12:00:00"
#       }
#   }
#
# Reset (the new button next to Auto/Manual) simply clears the entry.

def get_expert_override(model_name: str) -> Optional[Dict[str, Any]]:
    """Return the saved Expert-panel snapshot for ``model_name``, or None.

    The dict (when present) always carries ``mode`` and ``values``; ``pins``
    and ``saved_at`` are optional. A structurally invalid entry is treated
    as missing so a corrupt JSON blob never crashes the GUI.
    """
    if not model_name:
        return None
    raw = load_settings().get("expert_overrides") or {}
    if not isinstance(raw, dict):
        return None
    snap = raw.get(model_name)
    if not isinstance(snap, dict):
        return None
    if "mode" not in snap or "values" not in snap:
        return None
    return snap


def set_expert_override(model_name: str, snapshot: Dict[str, Any]) -> None:
    """Persist the Expert-panel snapshot for ``model_name``.

    ``snapshot`` must contain at least ``mode`` and ``values``. ``pins``
    and ``saved_at`` are preserved when present. An empty/invalid snapshot
    is ignored rather than written, so a half-built state can never land
    on disk.
    """
    if not model_name:
        return
    if not isinstance(snapshot, dict) or "mode" not in snapshot or "values" not in snapshot:
        return
    s = load_settings()
    overrides = s.get("expert_overrides")
    if not isinstance(overrides, dict):
        overrides = {}
    overrides[model_name] = snapshot
    s["expert_overrides"] = overrides
    save_settings(s)


def clear_expert_override(model_name: str) -> None:
    """Drop the saved Expert-panel state for a single model (the Reset button)."""
    if not model_name:
        return
    s = load_settings()
    overrides = s.get("expert_overrides") or {}
    if isinstance(overrides, dict) and model_name in overrides:
        overrides.pop(model_name, None)
        s["expert_overrides"] = overrides
        save_settings(s)


def get_performance_target() -> Optional[str]:
    """Return the persisted GUI performance-target choice, or None.

    Empty string and unknown values are treated as None so the GUI
    falls back to whatever the active profile (or global default)
    recommends.
    """
    val = load_settings().get("performance_target")
    if not val:
        return None
    val = str(val).lower().strip()
    return val if val in ("safe", "balanced", "throughput") else None


def set_performance_target(name: str) -> None:
    """Persist the GUI performance-target choice. Empty string clears it."""
    name = (name or "").lower().strip()
    if name in ("safe", "balanced", "throughput", ""):
        _update("performance_target", name)


# ---------------------------------------------------------------------------
# Sampling mode (chat / coding)
#
# Each YAML profile (new format) carries two sampling sub-blocks:
#   sampling:
#     chat:   { temperature: 1.0, top_k: 64, ... }
#     coding: { temperature: 1.5, top_k: 64, ... }
#
# The active mode is a global GUI choice (not per-model) — most users
# stay in one mode for hours, switch to coding when they pair-program,
# switch back. Per-model overrides would only add UI clutter without
# matching the actual workflow.

_VALID_MODES = ("chat", "coding")


def get_mode() -> Optional[str]:
    """Return the persisted sampling mode ("chat" / "coding"), or None."""
    val = load_settings().get("mode")
    if not val:
        return None
    val = str(val).lower().strip()
    return val if val in _VALID_MODES else None


def set_mode(name: str) -> None:
    """Persist the GUI sampling-mode choice. Empty string clears it."""
    name = (name or "").lower().strip()
    if name in _VALID_MODES + ("",):
        _update("mode", name)


# ---------------------------------------------------------------------------
# Window geometry & state
#
# Qt's QMainWindow can hand us two opaque QByteArrays:
#   * saveGeometry()  → size, position, screen, maximize/fullscreen state
#   * saveState()     → toolbar/dock/splitter positions
#
# We persist them as base64 strings (the only safe round-trip for
# arbitrary bytes inside JSON). On restart the GUI passes the bytes
# back to restoreGeometry/restoreState; if anything is corrupted or
# from an incompatible Qt version, those calls just return False and
# the GUI falls back to the hard-coded default size.


def _get_b64(key: str) -> Optional[str]:
    val = load_settings().get(key)
    if not isinstance(val, str) or not val:
        return None
    # Defensive: ignore obviously-broken payloads so a corrupt JSON
    # never crashes the GUI launch path.
    try:
        base64.b64decode(val, validate=True)
    except (ValueError, TypeError):
        return None
    return val


def get_window_geometry() -> Optional[str]:
    """Return the persisted QMainWindow.saveGeometry() blob (base64)."""
    return _get_b64("window_geometry")


def set_window_geometry(b64_value: str) -> None:
    """Persist the base64-encoded saveGeometry() output."""
    if isinstance(b64_value, str):
        _update("window_geometry", b64_value)


def get_window_state() -> Optional[str]:
    """Return the persisted QMainWindow.saveState() blob (base64)."""
    return _get_b64("window_state")


def set_window_state(b64_value: str) -> None:
    """Persist the base64-encoded saveState() output."""
    if isinstance(b64_value, str):
        _update("window_state", b64_value)


# ---------------------------------------------------------------------------
# Inner splitter layout
#
# QMainWindow.saveState() only round-trips toolbars and dock widgets — it
# does NOT capture the position of plain QSplitter handles that live inside
# the central widget. The AutoTuner GUI arranges its panes with two named
# QSplitters (top horizontal: model-list | config, and the vertical split
# between that row and the log panel). To remember the *inner* arrangement
# (not just the outer window size), each splitter's own saveState() blob is
# stored here under a stable object name.
#
# Schema:
#   "splitter_states": { "top_split": "<b64>", "main_split": "<b64>", ... }


def get_splitter_state(name: str) -> Optional[str]:
    """Return the persisted QSplitter.saveState() blob (base64) for *name*."""
    if not name:
        return None
    bucket = load_settings().get("splitter_states")
    if not isinstance(bucket, dict):
        return None
    val = bucket.get(name)
    if not isinstance(val, str) or not val:
        return None
    try:
        base64.b64decode(val, validate=True)
    except (ValueError, TypeError):
        return None
    return val


def set_splitter_state(name: str, b64_value: str) -> None:
    """Persist the base64-encoded saveState() of the QSplitter *name*."""
    if not name or not isinstance(b64_value, str):
        return
    s = load_settings()
    bucket = s.get("splitter_states")
    if not isinstance(bucket, dict):
        bucket = {}
    bucket[name] = b64_value
    s["splitter_states"] = bucket
    save_settings(s)


# ---------------------------------------------------------------------------
# Per-model mmproj (vision projector) selection
#
# A model can ship several projector precisions side by side (bf16 / f16 /
# f32). The scanner auto-picks one, but the user may prefer a different
# precision. We remember the chosen projector *filename* per model so the
# choice sticks across restarts. Stored as the bare filename (not full
# path) because models move between drives; the GUI matches it back against
# the freshly-scanned candidate list and falls back to the auto pick when
# the remembered file is no longer present.
#
# Schema:
#   "mmproj_selection": { "<model_name>": "mmproj-…-f32.gguf", ... }

MMPROJ_NONE_SENTINEL = "<none>"


def get_mmproj_selection(model_name: str) -> Optional[str]:
    """Return the remembered mmproj filename for *model_name*.

    Returns the literal ``"<none>"`` sentinel when the user explicitly chose
    no projector, the chosen filename when one was picked, or ``None`` when
    there is no stored preference (caller uses the scanner's auto pick).
    """
    if not model_name:
        return None
    bucket = load_settings().get("mmproj_selection")
    if not isinstance(bucket, dict):
        return None
    val = bucket.get(model_name)
    return val if isinstance(val, str) and val else None


def set_mmproj_selection(model_name: str, filename: Optional[str]) -> None:
    """Persist (or clear) the chosen mmproj filename for *model_name*.

    Pass ``None`` / empty to drop the override (model falls back to the
    scanner's automatic best pick).
    """
    if not model_name:
        return
    s = load_settings()
    bucket = s.get("mmproj_selection")
    if not isinstance(bucket, dict):
        bucket = {}
    if not filename:
        bucket.pop(model_name, None)
    else:
        bucket[model_name] = str(filename)
    s["mmproj_selection"] = bucket
    save_settings(s)


# ---------------------------------------------------------------------------
# Per-model draft (speculative-decoding head) selection.
#
# Mirrors mmproj_selection. The GUI exposes an always-on dropdown listing
# every draft/assistant GGUF in the model's folder; the chosen filename is
# remembered here. A sentinel empty string is NOT used — absence of a key
# means "use the scanner's auto pick", and the explicit literal "<none>"
# means "the user deliberately chose no draft" (so we don't silently
# re-enable the auto draft on the next launch).
#
# Schema:
#   "draft_selection": { "<model_name>": "…-assistant-Q4_K_M.gguf" | "<none>" }

DRAFT_NONE_SENTINEL = "<none>"


def get_draft_selection(model_name: str) -> Optional[str]:
    """Return the remembered draft filename for *model_name*.

    Returns the literal ``"<none>"`` sentinel when the user explicitly chose
    no draft, the chosen filename when one was picked, or ``None`` when there
    is no stored preference (caller should use the scanner's auto pick).
    """
    if not model_name:
        return None
    bucket = load_settings().get("draft_selection")
    if not isinstance(bucket, dict):
        return None
    val = bucket.get(model_name)
    return val if isinstance(val, str) and val else None


def set_draft_selection(model_name: str, filename: Optional[str]) -> None:
    """Persist (or clear) the chosen draft filename for *model_name*.

    Pass the filename to remember a specific draft, ``"<none>"`` to record a
    deliberate "no draft" choice, or ``None`` / empty to drop the override
    entirely (model reverts to the scanner's automatic pick).
    """
    if not model_name:
        return
    s = load_settings()
    bucket = s.get("draft_selection")
    if not isinstance(bucket, dict):
        bucket = {}
    if not filename:
        bucket.pop(model_name, None)
    else:
        bucket[model_name] = str(filename)
    s["draft_selection"] = bucket
    save_settings(s)


# ---------------------------------------------------------------------------
# Global font size
#
# The A+/A- toolbar buttons should affect the whole UI, not just the
# config preview and the log panel. We persist the chosen point size
# so a user who picked size 14 keeps size 14 across restarts.

_FONT_SIZE_MIN = 7
_FONT_SIZE_MAX = 22
_FONT_SIZE_DEFAULT = 10


# ---------------------------------------------------------------------------
# Server base port + offset
#
# The "Base port" field in the launcher toolbar selects the port the FIRST
# llama-server binds to (subsequent concurrent servers get base+1, base+2…).
# Persisting it means a user who switched away from the 1234 default — e.g.
# to avoid clashing with another local service — does not have to re-enter it
# on every restart. The manual port offset (0..10) is persisted alongside so
# the whole port-selection state round-trips. Both fall back to the hardcoded
# defaults when nothing is stored yet.

_BASE_PORT_MIN = 1
_BASE_PORT_MAX = 65535
_BASE_PORT_DEFAULT = 1234

_PORT_OFFSET_MIN = 0
_PORT_OFFSET_MAX = 10
_PORT_OFFSET_DEFAULT = 0


def get_base_port() -> int:
    """Return the persisted server base port (default 1234, clamped to 1..65535)."""
    val = load_settings().get("base_port")
    try:
        n = int(val) if val is not None else _BASE_PORT_DEFAULT
    except (TypeError, ValueError):
        return _BASE_PORT_DEFAULT
    return max(_BASE_PORT_MIN, min(_BASE_PORT_MAX, n))


def set_base_port(port: int) -> None:
    """Persist the server base port (clamped to the valid range)."""
    try:
        n = int(port)
    except (TypeError, ValueError):
        return
    _update("base_port", max(_BASE_PORT_MIN, min(_BASE_PORT_MAX, n)))


def get_port_offset() -> int:
    """Return the persisted manual port offset (default 0, clamped to 0..10)."""
    val = load_settings().get("port_offset")
    try:
        n = int(val) if val is not None else _PORT_OFFSET_DEFAULT
    except (TypeError, ValueError):
        return _PORT_OFFSET_DEFAULT
    return max(_PORT_OFFSET_MIN, min(_PORT_OFFSET_MAX, n))


def set_port_offset(offset: int) -> None:
    """Persist the manual port offset (clamped to the valid range)."""
    try:
        n = int(offset)
    except (TypeError, ValueError):
        return
    _update("port_offset", max(_PORT_OFFSET_MIN, min(_PORT_OFFSET_MAX, n)))


def get_font_size() -> int:
    """Return the persisted GUI point size; clamped to a sane range."""
    val = load_settings().get("font_size")
    try:
        n = int(val) if val is not None else _FONT_SIZE_DEFAULT
    except (TypeError, ValueError):
        return _FONT_SIZE_DEFAULT
    return max(_FONT_SIZE_MIN, min(_FONT_SIZE_MAX, n))


def set_font_size(size: int) -> None:
    """Persist the GUI point size (clamped to the safe range)."""
    try:
        n = int(size)
    except (TypeError, ValueError):
        return
    n = max(_FONT_SIZE_MIN, min(_FONT_SIZE_MAX, n))
    _update("font_size", n)


# ---------------------------------------------------------------------------
# Application behaviour


def get_minimize_on_close() -> bool:
    """Return whether title-bar X should hide in the notification area.

    This is deliberately opt-in: missing settings and non-boolean legacy
    values both resolve to ``False``.
    """
    return load_settings().get("minimize_on_close") is True


def set_minimize_on_close(enabled: bool) -> None:
    """Persist the opt-in X-to-notification-area behaviour."""
    _update("minimize_on_close", bool(enabled))


# ---------------------------------------------------------------------------
# Reasoning effort (per model)
#
# Some models (gpt-oss, certain Nemotron / Qwen3.5+ variants) honour a
# ``reasoning_effort`` kwarg that controls how much the model "thinks"
# before answering. Llama-server passes the value through to the chat
# template via ``--chat-template-kwargs '{"reasoning_effort":"high"}'``.
#
# Officially recognised values across the ecosystem:
#   * "low" / "medium" / "high"  — gpt-oss + Qwen3.5+ canonical set
#   * "minimal"                  — some Qwen3.6 builds
#   * "auto"                     — sentinel meaning "no flag, let the
#                                   chat template / model decide"
#
# "extra high" is not standardised upstream but several recent Qwen3.6
# community builds accept it; we keep it as an option and let the user
# discover whether their build supports it.
#
# Storage: per-model, alongside vision/draft/thinking overrides.

_VALID_REASONING = ("auto", "off", "minimal", "low", "medium", "high", "extra_high")


def get_reasoning_effort(model_name: str) -> Optional[str]:
    """Return the persisted reasoning_effort for ``model_name`` or None."""
    if not model_name:
        return None
    val = (load_settings().get("reasoning_effort") or {}).get(model_name)
    if not isinstance(val, str):
        return None
    val = val.lower().strip()
    return val if val in _VALID_REASONING else None


def set_reasoning_effort(model_name: str, value: Optional[str]) -> None:
    """Persist (or clear) the reasoning_effort for ``model_name``.

    Pass ``None`` or an empty string to drop the override (model falls
    back to "auto" — i.e. no CLI flag at all).
    """
    if not model_name:
        return
    s = load_settings()
    bucket = s.get("reasoning_effort")
    if not isinstance(bucket, dict):
        bucket = {}
    if not value:
        bucket.pop(model_name, None)
    else:
        v = value.lower().strip()
        if v not in _VALID_REASONING:
            return
        bucket[model_name] = v
    s["reasoning_effort"] = bucket
    save_settings(s)


def settings_file_location() -> Path:
    """Where settings are (or would be) written. For diagnostic logging."""
    return _settings_file()


# ---------------------------------------------------------------------------
# GPU priority overrides
#
# The user can mark each GPU with a priority (≥1) via the "gpu_overrides"
# section of autotuner_settings.json:
#
#   "gpu_overrides": {
#       "AMD Radeon AI PRO R9700":   { "enabled": true, "priority": 2 },
#       "AMD Radeon RX 9070 XT":     { "enabled": true, "priority": 1 }
#   }
#
# Higher priority → that GPU is preferred as the primary compute device
# (main_gpu in --tensor-split / --main-gpu).  When two GPUs have the same
# VRAM size the priority breaks the tie.  When VRAM sizes differ (e.g.
# 32 GB vs 16 GB) the score = priority × vram_mb already gives the larger
# GPU a comfortable lead, so the user can rely on VRAM winning naturally
# unless they explicitly want to invert the preference.


def get_gpu_priorities() -> Dict[str, int]:
    """Return a mapping of GPU name → user-assigned priority for all GPUs
    that have a priority entry in gpu_overrides.  Missing keys default to 1.
    """
    overrides = load_settings().get("gpu_overrides") or {}
    if not isinstance(overrides, dict):
        return {}
    result: Dict[str, int] = {}
    for gpu_name, entry in overrides.items():
        if not isinstance(entry, dict):
            continue
        try:
            result[str(gpu_name)] = max(1, int(entry.get("priority", 1)))
        except (TypeError, ValueError):
            result[str(gpu_name)] = 1
    return result


def get_gpu_priority(gpu_name: str) -> int:
    """Return the user-assigned priority for *gpu_name* (default 1)."""
    if not gpu_name:
        return 1
    overrides = load_settings().get("gpu_overrides") or {}
    entry = overrides.get(gpu_name) if isinstance(overrides, dict) else None
    if not isinstance(entry, dict):
        return 1
    try:
        return max(1, int(entry.get("priority", 1)))
    except (TypeError, ValueError):
        return 1


def set_gpu_priority(gpu_name: str, priority: int) -> None:
    """Persist *priority* for *gpu_name* in gpu_overrides.

    Creates the entry if it doesn't exist; leaves other fields (e.g.
    ``enabled``) untouched.
    """
    if not gpu_name:
        return
    s = load_settings()
    overrides = s.get("gpu_overrides")
    if not isinstance(overrides, dict):
        overrides = {}
    entry = overrides.get(gpu_name)
    if not isinstance(entry, dict):
        entry = {}
    try:
        entry["priority"] = max(1, int(priority))
    except (TypeError, ValueError):
        entry["priority"] = 1
    overrides[gpu_name] = entry
    s["gpu_overrides"] = overrides
    save_settings(s)


# ---------------------------------------------------------------------------
# Forced GPU (hard pin for the next server launch)
#
# Stored as a top-level string under "forced_gpu". When set to a GPU name
# (or a distinctive substring of it, e.g. "R9700"), compute_config pins the
# server to that single card and hides the others — the manual "boot only on
# the GPU I choose" control used when launching a second server so it lands
# on the still-empty card instead of piling onto an already-full one. An
# empty string / missing key means "auto" (free-VRAM-aware selection).


def get_forced_gpu() -> Optional[str]:
    """Return the GPU name the next launch is pinned to, or None for auto."""
    val = load_settings().get("forced_gpu")
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def set_forced_gpu(gpu_name: Optional[str]) -> None:
    """Pin launches to *gpu_name* exclusively, or clear the pin when None/empty."""
    s = load_settings()
    if gpu_name and gpu_name.strip():
        s["forced_gpu"] = gpu_name.strip()
    else:
        s.pop("forced_gpu", None)
    save_settings(s)