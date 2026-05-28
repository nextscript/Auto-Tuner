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
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from hardware import detect_system, format_system, SystemInfo
from launcher import launch
from scanner import scan_models, group_entries, ModelEntry
from settings_loader import load_profiles, match_profile, ModelProfile
from tuner import build_command, compute_config, TunedConfig
from performance_target import (
    resolve_performance_target,
    list_target_names,
    describe_targets,
)
import app_settings

# ---------------------------------------------------------------------------
# Pretty-printing helpers

_BAR = "─" * 64
_DEBUG_MODE = False
_DEBUG_CATEGORIES: set[str] = set()


def _spawn_detached(cmd: List[str], env_overrides: Optional[dict] = None) -> int:
    """Start llama-server detached and return its PID without waiting.

    Used by ``--detach`` so a script/agent can launch several models in a
    row (each on its own ``--port``) and have them all serve concurrently.
    On Windows the child gets its own console window
    (``CREATE_NEW_CONSOLE``); on Unix its own session (``start_new_session``),
    so it keeps running after this process exits.
    """
    import subprocess as _sp

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    if os.name == "nt":
        flags = _sp.CREATE_NEW_CONSOLE | _sp.CREATE_NEW_PROCESS_GROUP
        proc = _sp.Popen(cmd, creationflags=flags, env=env)
    else:
        proc = _sp.Popen(cmd, start_new_session=True, env=env)
    return proc.pid


def _debug_print(*args, **kwargs) -> None:
    if _DEBUG_MODE:
        print("[DEBUG]", *args, **kwargs)


def enable_debug_category(category: str) -> None:
    _DEBUG_CATEGORIES.add(category)


def debug_cat(category: str, *args, **kwargs) -> None:
    if _DEBUG_MODE or category in _DEBUG_CATEGORIES:
        print(f"[DEBUG:{category.upper()}]", *args, **kwargs)


def _print_banner() -> None:
    print()
    print(_BAR)
    print("  AutoTuner for llama.cpp  —  interactive launcher")
    print(_BAR)


def _print_system(info: SystemInfo) -> None:
    print(format_system(info))


def _capability_markers(entry: ModelEntry) -> str:
    """Compact capability symbols mirroring the Qt GUI:
    👁 vision · ⚡ draft · 🧠 thinking · 🛠 tool-use.
    Returned with leading space so the menu still aligns when empty.
    """
    syms = []
    if entry.has_vision:
        syms.append("👁")
    if entry.has_draft:
        syms.append("⚡")
    if entry.supports_thinking:
        syms.append("🧠")
    if entry.supports_tool_use:
        syms.append("🛠")
    return ("  " + " ".join(syms)) if syms else "    "


def _print_menu(groups: dict) -> List[ModelEntry]:
    """Print grouped model menu and return a flat list in display order."""
    flat: List[ModelEntry] = []
    print("\nAvailable models:")
    print(_BAR)
    print("  Symbols: 👁 vision · ⚡ draft · 🧠 thinking · 🛠 tool-use")
    idx = 1
    for group_name in sorted(groups.keys()):
        entries = sorted(groups[group_name], key=lambda e: e.name.lower())
        if not entries:
            continue
        print(f"\n  [{group_name}]")
        for e in entries:
            size = f"{e.size_gb:>5.1f} GB"
            ctx = ""
            if e.native_context:
                ctx = f"  ({e.native_context // 1024}k native)"
            print(f"    {idx:>2}.{_capability_markers(e)} {e.name:<55} {size}{ctx}")
            flat.append(e)
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

    if cfg.full_offload:
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
    if cfg.mlock:
        print("  mlock           : on (model pinned in RAM/VRAM)")
    if cfg.no_mmap:
        print("  no-mmap         : on (prevent swapping)")
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
    print(
        f"    Model GPU : ~ {cfg.estimated_model_vram_gb:5.1f} GB   (free VRAM: {system.free_vram_gb:5.1f} GB)"
    )
    print(
        f"    Model CPU : ~ {cfg.estimated_model_ram_gb:5.1f} GB   (free RAM:  {system.free_ram_gb:5.1f} GB)"
    )
    print(f"    KV cache  : ~ {cfg.estimated_kv_gb:5.1f} GB")
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
) -> tuple[bool, bool, bool, bool, Optional[ModelEntry]]:
    """Interaktive Fragen-Kette nach Modellauswahl.

    Returns:
        (use_vision, use_draft, use_thinking, use_ngram, effective_draft)
    """
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
    if effective_draft is not None:
        use_draft = _confirm(
            f"Draft-Modell aktivieren? ({effective_draft.name})",
            default_yes=True,
        )
        if not use_draft:
            effective_draft = None

    # ── Thinking / Reasoning ────────────────────────────────────────
    # Read the chat template from GGUF metadata (the authoritative source);
    # fall back to filename heuristics only when no template is available.
    use_thinking = False
    has_thinking_arch = model.supports_thinking
    if has_thinking_arch:
        use_thinking = _confirm(
            "Thinking/Reasoning aktivieren? (<|think|> / <|reserved_special_token>)",
            default_yes=True,
        )

    # ── n-gram (ngram-mod) ──────────────────────────────────────────
    # Model-agnostic self-speculative decoding — always offered. --ngram
    # forces it on without prompting; otherwise ask, defaulting to off.
    if force_ngram:
        use_ngram = True
    else:
        use_ngram = _confirm(
            "n-gram (ngram-mod) self-speculative decoding aktivieren?",
            default_yes=False,
        )

    return use_vision, use_draft, use_thinking, use_ngram, effective_draft


def _pick_model(
    flat: List[ModelEntry], cli_query: Optional[str]
) -> tuple[Optional[ModelEntry], List[str]]:
    if cli_query:
        parts = cli_query.split()
        query_parts = []
        flags = []
        for p in parts:
            if p.startswith("--") or p.lower() in ("novision", "nodraft", "nothinking", "ngram"):
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
            if p.startswith("--") or p.lower() in ("novision", "nodraft", "nothinking", "ngram"):
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

_SERVER_SUBPATHS = [
    "build/bin/Release/llama-server.exe",
    "build/bin/Debug/llama-server.exe",
    "build/bin/llama-server.exe",
    "build/bin/llama-server",
    "build/llama-server",
    "llama-server.exe",
    "llama-server",
]


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
            for sibling in parent.parent.iterdir():
                if sibling.is_dir() and re.search(
                    r"llama", sibling.name, re.IGNORECASE
                ):
                    add(sibling)
        except (OSError, PermissionError):
            pass
        add(parent.parent / "BitNet")

    bases = [Path(__file__).resolve().parent, Path.cwd()]
    common_subs = (
        "llama.cpp",
        "1b_llama.cpp",
        "ik_llama.cpp",
        "tq_llama.cpp",
        "atq_llama.cpp",
        "BitNet",
        "ai-local/llama.cpp",
        "ai-local/1b_llama.cpp",
        "ai-local/ik_llama.cpp",
        "ai-local/tq_llama.cpp",
    )
    for base in bases:
        chain = [base, *list(base.parents)[:5]]
        for p in chain:
            for sub in common_subs:
                add(p / sub)
    return roots


def _resolve_server_binary(user_value: str) -> str:
    """Turn a user-provided server name/path into something runnable."""
    p = Path(user_value).expanduser()
    if p.is_absolute() and p.is_file():
        return str(p)

    has_sep = os.sep in user_value or (
        os.altsep is not None and os.altsep in user_value
    )

    if has_sep and not p.is_absolute():
        parts = list(p.parts)
        fork_name = parts[0].lower() if parts else ""
        inner = Path(*parts[1:]) if len(parts) > 1 else None

        if inner is not None and fork_name:
            for root in _candidate_search_roots():
                root_base = root.name.lower()
                if root_base.endswith(".cpp"):
                    root_base = root_base[:-4]
                if root_base.startswith(fork_name) or fork_name.startswith(root_base):
                    candidate = root / inner
                    if candidate.is_file():
                        _debug_print(f"Found candidate: {candidate}")
                        return str(candidate)
                    for sub in _SERVER_SUBPATHS:
                        candidate = root / sub
                        if candidate.is_file():
                            _debug_print(
                                f"Found candidate in fork subpath: {candidate}"
                            )
                            return str(candidate)
                        candidate_with_inner = (root / sub) / inner
                        if candidate_with_inner.is_file():
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
        if candidate.is_file():
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
            if candidate.is_file():
                _debug_print(f"Found candidate in subpaths: {candidate}")
                return str(candidate)

    _debug_print(f"Defaulting to user value: {user_value}")
    return user_value


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
            has_binary = any((rp / sub).is_file() for sub in _SERVER_SUBPATHS)
            if not has_binary:
                continue
            seen.add(rp)
            forks.append((name, rp))

    # Sort: bare "llama.cpp" first (case-insensitive), then other *llama.cpp* forks alphabetically
    def _fork_sort_key(item: Tuple[str, Path]) -> tuple:
        name = item[0]
        name_lower = name.lower()
        is_bare = name_lower == "llama.cpp"
        # Strip common prefixes like "1b_", "atq_" etc for better sorting
        stripped = re.sub(r"^[\w-]+(?=_llama\.cpp)", "", name_lower)
        return (not is_bare, stripped)

    forks.sort(key=_fork_sort_key)
    return forks


def _pick_fork(
    forks: List[Tuple[str, Path]],
) -> Optional[Path]:
    """Show the fork menu and return the chosen fork directory.

    Returns ``None`` only when no forks were discovered at all.
    Handles ``EOFError`` gracefully (non-TTY / CI context) by defaulting to
    the first (standard) fork.
    """
    if not forks:
        return None

    if len(forks) == 1:
        print(f"[AutoTuner] Found one llama.cpp fork: {forks[0][0]}")
        return forks[0][1]

    print("\n" + "═" * 64)
    print("  LLAMA.CPP FORK SELECTION")
    print("═" * 64)
    print("  Found the following llama.cpp forks:\n")
    for i, (name, path) in enumerate(forks, 1):
        print(f"  {i}. {name:<30} {path}")
    print()
    print("  Enter a number to select, or press Enter for the default.")
    print("═" * 64)

    try:
        raw = input(f"Select fork [1-{len(forks)}] (default 1): ").strip()
    except EOFError:
        raw = ""

    if not raw:
        selected = forks[0]
        print(f"[AutoTuner] Using fork: {selected[0]}")
        return selected[1]

    if raw.isdigit():
        n = int(raw)
        if 1 <= n <= len(forks):
            selected = forks[n - 1]
            print(f"[AutoTuner] Using fork: {selected[0]}")
            return selected[1]

    print(f"[AutoTuner] Invalid choice '{raw}' — using default: {forks[0][0]}")
    return forks[0][1]


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
    print(f"    Context window    : {ctx:,} tokens   ← set this in your client")
    print(_BAR)


# ---------------------------------------------------------------------------
# Argument parsing


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="auto_tuner",
        description="Interactive launcher for llama-server with auto-tuned "
        "config based on free RAM/VRAM.",
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
        help="Enable n-gram (ngram-mod) self-speculative decoding. Needs no "
        "draft model and works on any model; good for code/text iteration, "
        "reasoning models and summarisation.",
    )
    p.add_argument(
        "--force-mlock",
        action="store_true",
        help="Force --mlock / --no-mmap even for full-GPU-offload models "
        "(prevents VRAM paging when enough free VRAM is available)",
    )
    p.add_argument(
        "--performance-target",
        choices=list_target_names(),
        default=None,
        metavar="{safe,balanced,throughput}",
        help="VRAM utilisation preset. Overrides any 'performance_target:' "
        "in the YAML profile. Tiers:\n" + describe_targets(),
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="Open the Qt log-viewer window after the server starts "
        "(requires PyQt6; server stdout/stderr stream into the window).",
    )
    p.add_argument(
        "--detach",
        action="store_true",
        help="Spawn llama-server in its own session/console and return "
        "immediately instead of waiting. Lets a script or agent start "
        "several models back-to-back (each --port a different value) so "
        "an orchestrator + spawned subagents can all serve concurrently. "
        "Implies --yes.",
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
        "--",
        dest="passthrough",
        nargs=argparse.REMAINDER,
        help="Extra arguments after `--` are forwarded to llama-server",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# main


def main(argv: Optional[List[str]] = None) -> int:  # noqa: C901  (complex but intentional)
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if getattr(args, "detach", False):
        args.yes = True  # --detach is non-interactive by definition
    if args.llama_cpp_dir:
        os.environ["LLAMA_CPP_DIR"] = args.llama_cpp_dir

    _print_banner()

    # ── Debug / verbose mode selection ─────────────────────────────────────
    global _DEBUG_MODE
    print("\n" + "=" * 60)
    print("  DEBUG / VERBOSE MODE SELECTION")
    print("=" * 60)
    print("  1. Debugging OFF (standard)")
    print("  2. Debugging ON (alle Kategorien)")
    print("-" * 60)
    print("  Kategorie-Debugging (einzelne Bereiche):")
    print("  3. Hardware-Erkennung (GPU/RAM/CPU)")
    print("  4. Model-Scanning & Profil-Matching")
    print("  5. Server-Pfad-Suche (llama.cpp)")
    print("  6. Konfigurations-Berechnung (KV-Cache, Kontext)")
    print("-" * 60)

    try:
        debug_choice = input("Wahl [1-6] (default 1): ").strip()
    except EOFError:
        debug_choice = ""

    if debug_choice == "2":
        _DEBUG_MODE = True
        print("[AutoTuner] Globaler Debug-Modus aktiviert.")
    elif debug_choice == "3":
        enable_debug_category("hardware")
        print("[AutoTuner] Kategorie-Debugging: Hardware-Erkennung")
    elif debug_choice == "4":
        enable_debug_category("scanner")
        print("[AutoTuner] Kategorie-Debugging: Model-Scanning & Profile")
    elif debug_choice == "5":
        enable_debug_category("llama_cpp")
        print("[AutoTuner] Kategorie-Debugging: Server-Pfad-Suche")
    elif debug_choice == "6":
        enable_debug_category("config")
        print("[AutoTuner] Kategorie-Debugging: Konfigurations-Berechnung")
    else:
        print("[AutoTuner] Debugging deaktiviert.")
    print("=" * 60 + "\n")

    # ── llama.cpp fork discovery ────────────────────────────────────────────
    # Only show the fork menu when --server was NOT specified explicitly by the
    # user (i.e. it is still the default "llama-server" or from LLAMA_SERVER).
    # An explicit --server path overrides everything.
    user_specified_server = (
        "--server" in (argv or sys.argv[1:]) or "LLAMA_SERVER" in os.environ
    )

    discovered_forks: List[Tuple[str, Path]] = []
    selected_fork_path: Optional[Path] = None

    if not user_specified_server:
        discovered_forks = _discover_llama_forks()
        selected_fork_path = _pick_fork(discovered_forks)
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

    # ── System detection ────────────────────────────────────────────────────
    system = detect_system()
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
            system = detect_system()
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
            return last_exit_code if first_iteration else 0

        for flag in picked_flags:
            if flag == "novision":
                args.novision = True
            elif flag == "nodraft":
                args.nodraft = True
            elif flag == "nothinking":
                args.nothinking = True
            elif flag == "ngram":
                args.ngram = True

        if args.novision and model.mmproj is not None:
            print(
                f"[AutoTuner] Vision disabled per --novision "
                f"(ignoring {model.mmproj.name})"
            )
            model.mmproj = None

        profile = match_profile(model.name, profiles)

        # ── Fork / model mismatch check ─────────────────────────────────
        # Some models require a specific fork (e.g. bonsai → 1b_llama.cpp).
        # Warn the user if their selected fork differs and offer to switch.
        # This check is skipped when the user passed --server explicitly.
        if not user_specified_server and selected_fork_path is not None:
            required_fork = _required_fork_name(profile)
            if required_fork:
                selected_name = selected_fork_path.name.lower()
                req_lower = required_fork.lower()
                if selected_name != req_lower:
                    print(
                        f"\n[AutoTuner] ⚠  Profile '{profile.display_name}' "
                        f"requires: {required_fork}"
                    )
                    print(f"             You selected:  {selected_fork_path.name}")
                    # Look for the required fork among already-discovered forks
                    matching = [
                        (n, p) for n, p in discovered_forks if n.lower() == req_lower
                    ]
                    if matching:
                        switch = _confirm(
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
        if profile.draft_max > 0 and not args.nodraft and model.draft is not None:
            try:
                draft_size = model.draft.stat().st_size
                draft_model = ModelEntry(
                    path=model.draft,
                    name=model.draft.stem,
                    group=model.group,
                    size_bytes=draft_size,
                    mmproj=None,
                    draft=None,
                    metadata={},
                )
                print(f"[AutoTuner] Found draft model: {draft_model.name}")
            except OSError as exc:
                print(f"[AutoTuner] Draft sibling unreadable ({exc}); skipping.")

        # ── Interactive feature chain ────────────────────────────────────
        (use_vision, use_draft, use_thinking, use_ngram, effective_draft) = (
            _ask_interactive_features(
                model, draft_model, args.settings_path, force_ngram=args.ngram
            )
        )
        if not use_draft:
            effective_draft = None

        # ── Config computation ───────────────────────────────────────────
        # Resolve performance target: CLI > YAML profile > "balanced".
        perf_target = resolve_performance_target(
            cli_choice=getattr(args, "performance_target", None),
            profile_choice=getattr(profile, "performance_target", "") or None,
        )

        cfg = compute_config(
            model,
            system,
            profile,
            draft_model=effective_draft,
            user_ctx=args.ctx,
            force_mlock=getattr(args, "force_mlock", False),
            perf_target=perf_target,
            gpu_priorities=app_settings.get_gpu_priorities(),
        )

        print(f"\n  [mlock] decision: model={model.name}")
        print(
            f"         full_offload={cfg.full_offload}  "
            f"vram={cfg.estimated_model_vram_gb:.1f}GB  "
            f"ram={cfg.estimated_model_ram_gb:.1f}GB"
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

        _print_config(model, profile, cfg, system)

        # ── Binary resolution ────────────────────────────────────────────
        def resolve_specialized_binary(
            profile: ModelProfile,
            use_draft_flag: bool,
            model_name: str,
        ) -> str:
            """Choose the llama-server binary for this model.

            Priority:
              1. Gemma 4 WITH external draft  → ik_llama.cpp (external drafter
                 still requires the fork; integrated MTP now works in mainline b9190+)
              2. server_binary from YAML profile
              3. Fallback: whatever the user selected / args.server
            """
            # Gemma 4 needs ik_llama.cpp only when an external sibling drafter is active.
            # Integrated MTP (Qwen3.6-MTP filenames) uses --spec-type draft-mtp and
            # works in mainline llama.cpp b9190+ without any special fork.
            if "gemma-4" in model_name.lower() or "gemma4" in model_name.lower():
                if use_draft_flag:
                    return (
                        profile.server_binary
                        if profile.server_binary
                        else "ik_llama.cpp/llama-server"
                    )
                # Without draft, use whichever fork the user selected

            # Explicit server_binary in YAML always wins
            if profile.server_binary:
                return profile.server_binary

            # Default: let _resolve_server_binary find it in LLAMA_CPP_DIR
            return args.server

        raw_server = profile.server_binary or args.server
        effective_server = resolve_specialized_binary(profile, use_draft, model.name)
        server = _resolve_server_binary(effective_server)

        if server != raw_server:
            print(f"[AutoTuner] Found server binary: {server}")
        elif not Path(server).is_file() and not shutil.which(server):
            print(f"[AutoTuner] Warning: server binary '{server}' not found.")
            print("  Pass --server /path/to/llama-server, set LLAMA_SERVER, or")
            print("  set LLAMA_CPP_DIR to your llama.cpp checkout.")

        # ── Build command ────────────────────────────────────────────────
        extra = args.passthrough or []
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
            # --nodraft disables all draft-based speculative decoding, including
            # embedded MTP (which has no external file and so isn't covered by
            # effective_draft=None alone). n-gram is independent (--ngram).
            enable_speculative=not args.nodraft,
            enable_ngram=use_ngram,
        )

        if args.dry_run:
            print("[AutoTuner] --dry-run — not starting the server.")
            print("Command:")
            print("  " + " ".join(cmd))
            _print_client_settings(args.host, args.port, cfg.ctx, model)
            return 0

        # ── Detached spawn ────────────────────────────────────────────────
        # Start the server in its own session/console and return at once, so
        # a script or agent can launch several models in a row (one per port)
        # and have them all serve concurrently. No menu loop, no waiting.
        if args.detach:
            _print_client_settings(args.host, args.port, cfg.ctx, model)
            pid = _spawn_detached(cmd, env_overrides=cfg.env_overrides)
            print(
                f"\n[AutoTuner] Detached llama-server — PID {pid} — "
                f"http://{args.host}:{args.port}"
            )
            print("[AutoTuner] Server keeps running in its own window/session.")
            return 0

        try:
            launch_now = args.yes or _confirm("Launch llama-server now?")
        except KeyboardInterrupt:
            print("\n[AutoTuner] Aborted by user.")
            return 0

        if not launch_now:
            print("[AutoTuner] Launch skipped — back to the model menu.")
            first_iteration = False
            continue

        _print_client_settings(args.host, args.port, cfg.ctx, model)
        print(
            f"\n[AutoTuner] Web UI will be available at "
            f"http://{args.host}:{args.port}\n"
        )

        # ── Launch (terminal or GUI) ─────────────────────────────────────
        if args.gui:
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
                srv.start()
                app = QApplication(sys.argv)
                window = LogViewerWindow(srv)
                window.show()
                sys.exit(app.exec())
        else:
            last_exit_code = launch(cmd, env_overrides=cfg.env_overrides)

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
