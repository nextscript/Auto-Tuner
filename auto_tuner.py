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
from tuner import (
    build_command,
    build_diffusion_command,
    build_diffusion_server_command,
    compute_config,
    gemma_draft_needs_ik_fork,
    prepare_command_for_binary,
    TunedConfig,
)
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
            if p.startswith("--") or p.lower() in (
                "novision",
                "nodraft",
                "nothinking",
                "ngram",
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
    "llama-diffusion-cli",       # mainline Dream/LLaDA/RND1
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
            for sibling in parent.parent.iterdir():
                if sibling.is_dir() and re.search(
                    r"llama", sibling.name, re.IGNORECASE
                ):
                    add(sibling)
        except (OSError, PermissionError):
            pass
        # If env_dir is a CONTAINER (a parent holding several *_llama.cpp
        # builds, e.g. H:/LAB/ai-local), enumerate its direct children so a
        # profile hint like '2b_llama/llama-server' resolves to a versioned
        # child dir like '2b_b8840_llama.cpp'. Without this the resolver only
        # sees the container, not the forks inside it.
        try:
            for child in parent.iterdir():
                if child.is_dir() and re.search(
                    r"llama", child.name, re.IGNORECASE
                ):
                    add(child)
        except (OSError, PermissionError):
            pass
        add(parent.parent / "BitNet")

    bases = [Path(__file__).resolve().parent, Path.cwd()]
    # Fork hint list — the on-disk names produced by the build scripts
    # follow '{prefix}_b{NUM}_llama.cpp' (versioned) or '{prefix}_llama.cpp'
    # (unversioned). The versioned form is NOT listed here because it varies
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


# Fork dir naming convention: '{prefix}_llama.cpp', '{prefix}_b{NUM}_llama.cpp',
# or bare 'b{NUM}_llama.cpp' / 'llama.cpp'. To match a profile's fork hint
# (e.g. "2b_llama") against the versioned dir actually on disk
# (e.g. "2b_b8840_llama.cpp"), we normalize both to a family form by
# stripping the '.cpp' suffix and any '_b<NUM>' / leading 'b<NUM>' version
# segment:
#     '2b_b8840_llama.cpp' -> '2b_llama'
#     '2b_llama.cpp'       -> '2b_llama'
#     'b9840_llama.cpp'    -> 'llama'
#     'tq_b9625_llama.cpp' -> 'tq_llama'
#     '1b_llama.cpp'       -> '1b_llama'
# This keeps the 1-bit ('1b_') and 2-bit/Ternary ('2b_') families distinct
# while tolerating both versioned and unversioned directory names.
_FORK_VERSION_RE = re.compile(r"(?:^|_)b\d+(?=_|$)")


def _fork_family(name: str) -> str:
    """Normalize a fork dir/name to its family form for fuzzy matching."""
    n = name.lower()
    if n.endswith(".cpp"):
        n = n[:-4]
    n = _FORK_VERSION_RE.sub("", n).lstrip("_")
    return n


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
            for root in _candidate_search_roots():
                root_base = root.name.lower()
                if root_base.endswith(".cpp"):
                    root_base = root_base[:-4]
                if (
                    root_base.startswith(fork_name)
                    or fork_name.startswith(root_base)
                    # Fuzzy fallback: match after normalizing version
                    # segments out of both names, so a profile hint '2b_llama'
                    # resolves to an on-disk dir '2b_b8840_llama.cpp'.
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
        "--no-prompt-cache",
        action="store_true",
        dest="no_prompt_cache",
        help="Disable host-memory prompt caching (--cache-ram 0). By default "
        "prompt caching is auto-enabled; Vision caching requires llama.cpp "
        "b10045+ and safely falls back to off on older/unprobeable builds.",
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
        help="Force --mlock / --no-mmap even for full-GPU-offload models "
        "(prevents VRAM paging when enough free VRAM is available)",
    )
    p.add_argument(
        "--performance-target",
        choices=list_target_names(),
        default=None,
        metavar="{safe,balanced,throughput,low_vram}",
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
                # Compare on the family form so a versioned fork dir
                # (e.g. '2b_b8840_llama.cpp') is recognized as matching a
                # profile requirement of '2b_llama.cpp'.
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
            force_gpu=getattr(args, "gpu", None) or app_settings.get_forced_gpu(),
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
              1. server_binary from YAML profile
              2. Whatever the user selected / args.server — for Gemma 4 WITH
                 external draft this is probed for --spec-type first: mainline
                 runs the drafter natively since PR #23398 (b9190+), so only a
                 build too old to advertise --spec-type still falls back to
                 the legacy ik_llama.cpp fork.
            """
            # Explicit server_binary in YAML always wins
            if profile.server_binary:
                return profile.server_binary

            # Gemma 4 + external drafter: use the selected build when it
            # advertises --spec-type (mainline b9190+ handles the
            # gemma4-assistant head via --spec-type draft-mtp); redirect to
            # ik_llama.cpp only for genuinely old builds.
            if gemma_draft_needs_ik_fork(
                model_name, use_draft_flag, _resolve_server_binary(args.server)
            ):
                print(
                    "[AutoTuner] Selected build advertises no --spec-type "
                    "(pre-b9190) — Gemma-4 drafter falls back to ik_llama.cpp."
                )
                return "ik_llama.cpp/llama-server"

            # Default: let _resolve_server_binary find it in LLAMA_CPP_DIR
            return args.server

        # ── Runner selection: diffusion vs server ────────────────────────
        # Diffusion text models (Dream/LLaDA/RND1 mainline; DiffusionGemma
        # fork) are NOT served by llama-server — they run single-shot via
        # llama-diffusion-cli with --diffusion-* flags and no /health/API.
        # The scanner detects this from general.architecture; a profile may
        # also force it with `runner: llama-diffusion-cli`.
        is_diffusion_run = (
            model.is_diffusion or profile.runner == "llama-diffusion-cli"
        ) and profile.runner != "llama-diffusion-gemma-server"

        extra = args.passthrough or []

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
            effective_server = resolve_specialized_binary(
                profile, use_draft, model.name
            )
            server = _resolve_server_binary(effective_server)

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
                # --nodraft disables all draft-based speculative decoding,
                # including embedded MTP (which has no external file and so
                # isn't covered by effective_draft=None alone). n-gram is
                # independent (--ngram).
                enable_speculative=not args.nodraft,
                enable_ngram=use_ngram,
                enable_prompt_cache=not getattr(args, "no_prompt_cache", False),
                enable_metrics=not getattr(args, "no_metrics", False),
                enable_slots_api=bool(getattr(args, "slots_api", False)),
            )

        cmd, removed_args = prepare_command_for_binary(cmd)
        if removed_args:
            print(
                "[AutoTuner] Compatibility: selected llama.cpp binary does not "
                "advertise these argument(s); removed them: "
                + "; ".join(removed_args)
            )

        if args.dry_run:
            if is_diffusion_run:
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
                "Run diffusion generation now?"
                if is_diffusion_run
                else "Launch llama-server now?"
            )
            launch_now = args.yes or _confirm(prompt_label)
        except KeyboardInterrupt:
            print("\n[AutoTuner] Aborted by user.")
            return 0

        if not launch_now:
            print("[AutoTuner] Launch skipped — back to the model menu.")
            first_iteration = False
            continue

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
