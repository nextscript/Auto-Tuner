from __future__ import annotations

import re
import platform
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from hardware import SystemInfo, GPUInfo
from scanner import ModelEntry
from settings_loader import ModelProfile
from performance_target import (
    PerformanceTarget,
    PERFORMANCE_TARGETS,
    resolve_performance_target,
    DEFAULT_TARGET_NAME,
)

ctypes: Any = None
try:
    import ctypes as _ctypes

    ctypes = _ctypes
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Tunables. Kept at module scope so tests / callers can override them.
#
# These are now thin compat shims — the real values come from the active
# PerformanceTarget. Keeping the module constants means external callers
# (tests, scripts) that monkey-patched them in the past keep working, and
# reading the constants still gives the "balanced" defaults.

DEFAULT_VRAM_SAFETY_GB = PERFORMANCE_TARGETS[DEFAULT_TARGET_NAME].dense_vram_safety_gb
DEFAULT_RAM_SAFETY_GB = PERFORMANCE_TARGETS[DEFAULT_TARGET_NAME].ram_safety_gb

# MoE-specific knobs. Read from the "balanced" preset for back-compat.
MOE_VRAM_SAFETY_GB = PERFORMANCE_TARGETS[DEFAULT_TARGET_NAME].moe_vram_safety_gb
MOE_PLACEMENT_CTX_TARGET = PERFORMANCE_TARGETS[
    DEFAULT_TARGET_NAME
].moe_placement_ctx_target
MOE_KV_RESERVE_FRAC = 0.06


# ---------------------------------------------------------------------------
# Model-size helpers


def extract_params_billion(name: str) -> float:
    """Extract parameter count in billions from a model filename."""
    matches = re.findall(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*B(?![a-zA-Z0-9_])", name)
    if matches:
        return max(float(m) for m in matches)
    m = re.search(r"E(\d+(?:\.\d+)?)B", name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return 0.0


def kv_per_token_mb_f16(params_billion: float) -> float:
    """Approximate KV-cache memory per token at f16 quant, in MB.

    Fallback heuristic when GGUF metadata is unavailable. NOTE: this is
    calibrated for *dense* models. For MoE models (e.g. Qwen3.6-35B-A3B
    and Gemma-4-26B-A4B where only ~3B and ~4B params are active per token),
    this heuristic overestimates KV by roughly an order of magnitude — prefer
    `kv_per_token_mb_from_metadata()` whenever metadata is present.
    """
    if params_billion <= 0:
        return 0.20
    if params_billion < 1.5:
        return 0.04
    if params_billion < 4:
        return 0.10
    if params_billion < 9:
        return 0.18
    if params_billion < 16:
        return 0.30
    if params_billion < 32:
        return 0.50
    if params_billion < 70:
        return 0.85
    return 1.40


def _kv_per_token_for_interleaved_attention(
    md: Dict[str, Any],
    arch: str,
    n_kv_heads_per_layer: List[Any],
) -> float:
    """KV per-token (MB) for models with per-layer KV-head arrays.

    The Gemma-4 family (26B-A4B, 31B, and likely future siblings)
    stores ``<arch>.attention.head_count_kv`` as a **per-layer array**
    rather than a single scalar, because each layer alternates between
    full attention and sliding-window attention (SWA) with different
    head/dim configurations. Example from gemma-4-26B-A4B:

        head_count_kv          = [8,8,8,8,8,2, 8,8,8,8,8,2, …]   # 30 layers
        sliding_window_pattern = [T,T,T,T,T,F, T,T,T,T,T,F, …]   # 30 layers
        sliding_window         = 1024

    The pattern array tells us which layers do SWA (True) and which
    do full attention (False). SWA layers cap their KV at
    ``sliding_window`` tokens — they contribute a **constant** overhead
    that does NOT scale with ctx. Full-attention layers (the False
    entries) scale linearly with ctx.

    For the AutoTuner's "per-token KV size" estimate at large ctx, we
    return only the asymptotic part: the sum over full-attention
    layers. At typical ctx >> sliding_window (e.g. 32k >> 1024) the
    constant SWA overhead is well under 100 MB and can be ignored.

    Fallback: if ``sliding_window_pattern`` is missing or shorter than
    the head array, sum every entry (treat all as full-attention).
    That overshoots, but on the safe side — better the AutoTuner
    reserves too much KV than too little.
    """

    def _int_md(key: str) -> int:
        v = md.get(f"{arch}.{key}", 0)
        try:
            return int(v) if v is not None else 0
        except (TypeError, ValueError):
            return 0

    sliding_pattern = md.get(f"{arch}.attention.sliding_window_pattern")
    kl = _int_md("attention.key_length")
    vl = _int_md("attention.value_length")
    n_heads = _int_md("attention.head_count")
    embd = _int_md("embedding_length")

    # Fall back to embd/n_heads if explicit head dims absent.
    if kl <= 0 or vl <= 0:
        if n_heads > 0 and embd > 0:
            head_size = max(1, embd // n_heads)
            if kl <= 0:
                kl = head_size
            if vl <= 0:
                vl = head_size
        else:
            return 0.0

    full_kv = 0
    if isinstance(sliding_pattern, list) and sliding_pattern:
        # Sum KV-heads only for full-attention layers (pattern entry False).
        for i, h in enumerate(n_kv_heads_per_layer):
            if i >= len(sliding_pattern):
                break
            # Pattern entry True = SWA → skip (constant overhead).
            if not bool(sliding_pattern[i]):
                try:
                    full_kv += int(h)
                except (TypeError, ValueError):
                    continue
    else:
        # No pattern info — treat every layer as full attention.
        for h in n_kv_heads_per_layer:
            try:
                full_kv += int(h)
            except (TypeError, ValueError):
                continue

    if full_kv <= 0:
        return 0.0

    bytes_per_token = full_kv * (kl + vl) * 2
    return bytes_per_token / (1024.0 * 1024.0)


def kv_per_token_mb_from_metadata(md: Dict[str, Any]) -> float:
    """Compute exact f16 K+V cache size per token (MB) from GGUF metadata.

    Standard transformer formula:
        bytes/token = n_attention_layers * n_kv_heads * (key_length + value_length) * 2

    Three special cases are handled before the formula:

    1. **Interleaved-attention models** (Gemma-4 family) — when
       ``head_count_kv`` is stored as a *per-layer array* instead of a
       scalar, we route to :func:`_kv_per_token_for_interleaved_attention`
       which uses the sliding-window pattern to count only the
       full-attention layers (those that actually scale with ctx).

    2. **Hybrid Mamba/Transformer models** (Nemotron-H, Jamba, …) —
       only a fraction of layers carry KV cache. We multiply by that
       fraction via :func:`metadata_attention_layer_count`. Otherwise
       we'd over-reserve VRAM by 4–5× on these architectures.

    3. **GQA** — when ``head_count_kv`` is present and a positive scalar,
       it's already smaller than ``head_count`` and the formula uses
       it directly. When the value is missing (``head_count_kv = 0``)
       we fall back to ``head_count``, which over-estimates KV for
       any modern GQA model. Such over-estimates are visible in the
       config preview; if you see them, the GGUF likely stored the
       value under a non-canonical key or as an array (case 1).

    Returns 0.0 when metadata is incomplete; the caller should then
    fall back to the params-billion heuristic.
    """
    if not md:
        return 0.0
    arch = md.get("general.architecture") or ""
    if not arch:
        return 0.0

    # ── Case 1: interleaved attention with per-layer KV-head array ─────
    n_kv_raw = md.get(f"{arch}.attention.head_count_kv")
    if isinstance(n_kv_raw, list) and n_kv_raw:
        return _kv_per_token_for_interleaved_attention(md, arch, n_kv_raw)

    # ── Standard scalar path (existing behaviour) ─────────────────────
    def _int(key: str) -> int:
        v = md.get(f"{arch}.{key}", 0)
        try:
            return int(v) if v is not None else 0
        except (TypeError, ValueError):
            return 0

    # Use the attention-bearing layer count for hybrids; for pure
    # Transformer this equals block_count and behaves as before.
    from scanner import metadata_attention_layer_count

    n_layers = metadata_attention_layer_count(md)
    if n_layers <= 0:
        # Fallback for older models / incomplete metadata: use total blocks.
        n_layers = _int("block_count")

    n_heads = _int("attention.head_count")
    n_kv_heads = _int("attention.head_count_kv")
    embd = _int("embedding_length")
    key_length = _int("attention.key_length")
    value_length = _int("attention.value_length")

    if n_layers <= 0:
        return 0.0

    # Default head dim = embedding_length / head_count when not explicit.
    if key_length <= 0 or value_length <= 0:
        if n_heads > 0 and embd > 0:
            head_size = max(1, embd // n_heads)
            if key_length <= 0:
                key_length = head_size
            if value_length <= 0:
                value_length = head_size
        else:
            return 0.0

    # No GQA → KV heads == query heads.
    if n_kv_heads <= 0:
        n_kv_heads = n_heads if n_heads > 0 else 1

    bytes_per_token = n_layers * n_kv_heads * (key_length + value_length) * 2
    return bytes_per_token / (1024.0 * 1024.0)


def _resolve_kv_per_token_mb(model: ModelEntry, params_billion: float) -> float:
    """Pick the best KV-per-token estimate available.

    Preference: exact GGUF metadata first (precise; works for MoE),
    falling back to params-based heuristic (for tests / metadata-less
    models).
    """
    md_estimate = kv_per_token_mb_from_metadata(model.metadata)
    if md_estimate > 0:
        return md_estimate
    return kv_per_token_mb_f16(params_billion)


def kv_quant_factor(quant: str) -> float:
    """Memory factor of a given KV-cache quant, relative to f16.

    Covers the upstream cache types (f16/q8/q5/q4 + iq4_nl) plus the
    TurboQuant-fork labels (turbo2/turbo3/turbo4). The turbo factors
    come from Google's TurboQuant paper + the TheTom/AtomicBot fork
    measurements (ICLR 2026 / b9082+ branches): turbo3 ≈ 4.3× vs f16,
    turbo4 ≈ 3.8×, turbo2 ≈ 6.4×.
    """
    q = quant.lower()
    if q in ("f16", "fp16", "bf16"):
        return 1.0
    if q in ("q8_0", "q8_1", "q8"):
        return 0.55
    if q in ("q5_0", "q5_1", "q5"):
        return 0.40
    if q in ("q4_0", "q4_1", "q4", "iq4_nl"):
        return 0.32
    # TurboQuant labels (TheTom/turboquant_plus, AtomicBot, spiritbuun).
    # The Google paper quotes compression ratios vs F16; we convert
    # 1 / ratio = factor. Slightly conservative (rounded up) so the
    # auto-tuner does not over-promise context length.
    if q == "turbo4":
        return 0.27  # ~3.8× → 1/3.8 = 0.263, rounded up
    if q in ("turbo3", "tq3_0"):
        return 0.24  # ~4.3× → 1/4.3 = 0.233
    if q in ("turbo3_tcq",):
        # 3-bit Viterbi-coded, ~5x at same quality as turbo3 scalar.
        return 0.20
    if q == "turbo2":
        return 0.16  # ~6.4× → 1/6.4 = 0.156
    if q in ("turbo2_tcq",):
        return 0.13  # ~7-8× in spiritbuun benchmarks
    return 0.55


# ---------------------------------------------------------------------------
# MoE detection

# Common alternate metadata keys some quantizers emit instead of the
# canonical "<arch>.expert_count". Order matters: more specific first.
_MOE_ALT_KEY_SUFFIXES = (
    ".expert_count",  # canonical (qwen3moe.expert_count, etc.)
    ".num_local_experts",  # HF-style fallback
    ".num_experts",  # plain
    ".moe.expert_count",  # some hybrid/MTP forks
)

# Filename-level MoE marker: the "A{N}B" suffix that vendors use to
# advertise the active-parameter count of an MoE model
# (e.g. Qwen3.5-30B-A3B, Gemma-4-26B-A4B, Qwen3.5-122B-A10B). This is
# a *fallback only* — when GGUF metadata declares no expert count but
# the filename clearly says "active 3B of 30B total", we trust the
# filename and route the model through the MoE placement path.
_MOE_FILENAME_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(\d+(?:\.\d+)?)B"  # total params, e.g. "30B"
    r"[-_.]?A(\d+(?:\.\d+)?)B"  # active params, e.g. "A3B"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _moe_expert_count(model: ModelEntry) -> int:
    """Return expert_count from GGUF metadata, or 0 if dense / unknown.

    Detection order:
      1. ``<arch>.expert_count`` — canonical GGUF key.
      2. Any ``*.expert_count`` key in metadata (older quantizers).
      3. Common alternate suffixes (``num_local_experts`` etc.) — some
         MTP and hybrid forks emit these instead of the canonical name.
      4. Filename heuristic: ``{Total}B-A{Active}B`` pattern returns 1
         (sentinel "MoE confirmed by filename, exact count unknown" —
         enough to enter the MoE placement branch). This catches GGUFs
         where the metadata writer dropped the expert count entirely.
    """
    md = model.metadata
    if md:
        arch = md.get("general.architecture")
        # Step 1+3: try every alt suffix on the model's own architecture.
        if arch:
            for suffix in _MOE_ALT_KEY_SUFFIXES:
                key = f"{arch}{suffix}"
                if key in md:
                    try:
                        n = int(md[key])
                        if n > 0:
                            return n
                    except (TypeError, ValueError):
                        pass
        # Step 2+3: scan all keys for any of the alt suffixes.
        for k, v in md.items():
            if any(k.endswith(s) for s in _MOE_ALT_KEY_SUFFIXES):
                try:
                    n = int(v)
                    if n > 0:
                        return n
                except (TypeError, ValueError):
                    continue

    # Step 4: filename fallback. Returns 1 (sentinel) — caller treats
    # any value > 1 as "definitely MoE". We bump the sentinel to 2 so
    # `is_moe = expert_count > 1` triggers correctly without lying about
    # the real count, which we don't know.
    if _MOE_FILENAME_RE.search(model.name):
        return 2
    return 0


# ---------------------------------------------------------------------------
# Configuration result


@dataclass
class TunedConfig:
    ctx: int
    ngl: int
    threads: int
    batch_threads: int
    batch: int
    ubatch: int
    cache_k: str
    cache_v: str
    flash_attn: bool
    sampling: Dict[str, Any] = field(default_factory=dict)

    mlock: bool = False
    no_mmap: bool = False
    numa: Optional[str] = None
    tensor_split: Optional[str] = None
    main_gpu: Optional[int] = None

    n_cpu_moe: Optional[int] = None
    is_moe: bool = False
    expert_count: int = 0

    estimated_model_vram_gb: float = 0.0
    estimated_model_ram_gb: float = 0.0
    estimated_kv_gb: float = 0.0
    full_offload: bool = False

    # ---- New (display fidelity) ---------------------------------------
    # VRAM that vision (mmproj) and draft model consume on the GPU.
    # The main model placement subtracts these from `free_vram_gb` to
    # decide layer placement, but until now the display only showed the
    # main-model VRAM number — so toggling vision/draft produced
    # counter-intuitive context changes the user could not explain.
    # Surfacing both here lets the GUI render the FULL GPU picture.
    vision_vram_gb: float = 0.0
    draft_vram_gb: float = 0.0
    # KV split between VRAM and RAM (set by compute_config). For
    # full-offload / MoE-on-GPU the entire KV cache lives in VRAM and
    # `kv_ram_gb == 0`. For dense-hybrid placement the small RAM share
    # is shown so the user can see why context is throttled.
    kv_vram_gb: float = 0.0
    kv_ram_gb: float = 0.0
    # KV-quant labels actually applied (may differ from cache_k/cache_v
    # when an explicit Expert override was used — kept for diagnostics).
    kv_quant_strategy: str = (
        "symmetric"  # "symmetric" | "asymmetric" | "manual" | "turbo"
    )

    no_context_shift: bool = False

    # RoPE-Scaling: aktiviert wenn ctx > native_ctx und YaRN/rope-scaling
    # verwendet werden soll (optional, nur für Modelle die es unterstützen).
    rope_scaling: bool = False
    rope_scale_factor: float = 1.0  # z.B. 4.0 für yarn mit 4x scaling

    # Optional CLI extras the GUI's Expert mode injects. Examples:
    # "--jinja", "--verbose". Built-in defaults stay empty so the
    # auto-mode behaviour is unchanged.
    extra_cli_flags: List[str] = field(default_factory=list)

    # Environment variables to set when spawning llama-server.
    # Primarily used to set HIP_VISIBLE_DEVICES on Windows AMD multi-GPU
    # setups where the Windows registry GPU order differs from HIP order.
    env_overrides: Dict[str, str] = field(default_factory=dict)

    # Active performance target name ("safe" / "balanced" / "throughput").
    # Set by compute_config so display code can show what was applied.
    performance_target: str = DEFAULT_TARGET_NAME

    # --parallel N for llama-server.  Controls how many inference slots
    # the server allocates simultaneously.  Always passed explicitly so
    # llama-server's "auto" heuristic cannot over-provision KV cache.
    # Sized by the resolved PerformanceTarget (1 / 2 / 4 for throughput /
    # balanced / safe).  The ctx calculation in compute_config divides
    # kv_budget_gb by n_parallel so each slot gets a correctly-sized KV
    # window instead of the server silently multiplying KV by N slots.
    n_parallel: int = 1

    warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers


def _decide_offload(
    model_size_gb: float,
    free_vram_gb: float,
    n_layers: int,
    has_gpu: bool,
    vram_headroom_gb: float = DEFAULT_VRAM_SAFETY_GB,
) -> Tuple[int, float, float, bool]:
    if not has_gpu or free_vram_gb < 1.0:
        return 0, 0.0, model_size_gb, False

    usable = max(0.0, free_vram_gb - vram_headroom_gb)
    if usable >= model_size_gb:
        return 999, model_size_gb, 0.0, True

    if usable < 0.5:
        return 0, 0.0, model_size_gb, False

    if n_layers > 0:
        per_layer_gb = model_size_gb / n_layers
        ngl = int(usable / per_layer_gb)
        ngl = max(0, min(n_layers, ngl))
        model_vram = ngl * per_layer_gb
        residual_overhead = model_size_gb * 0.02  # Reduced overhead
        model_ram = (n_layers - ngl) * per_layer_gb + residual_overhead
        return ngl, model_vram, model_ram, False

    estimated_layers = 50
    fraction = usable / model_size_gb
    ngl = max(0, int(fraction * estimated_layers))
    return ngl, usable, max(0.0, model_size_gb - usable), False


def _decide_moe_offload(
    model_size_gb: float,
    free_vram_gb: float,
    free_ram_gb: float,
    n_layers: int,
    expert_count: int,
    params_billion: float,
    target_ctx: int,
    base_kv_per_token_mb: float = 0.0,
    ram_safety_gb: float = DEFAULT_RAM_SAFETY_GB,
    moe_vram_safety_gb: float = MOE_VRAM_SAFETY_GB,
    moe_placement_ctx_target: int = MOE_PLACEMENT_CTX_TARGET,
) -> Tuple[int, Optional[int], float, float, bool]:
    """Decide how to split an MoE model between GPU and CPU.

    Strategy:
      1. Reserve VRAM for the KV cache up front (Vulkan requires KV to
         live entirely in VRAM for MoE — RAM-resident KV crashes with
         GGML_ASSERT(addr) on the AMD/Vulkan backend).
      2. Reserve VRAM for shared (non-expert) tensors.
      3. Pack as many expert layers as possible into the leftover VRAM;
         everything else goes to CPU via `--n-cpu-moe`.

    A practical KV target of ``moe_placement_ctx_target`` is used
    instead of the profile maximum, so we don't reserve VRAM for context
    the user is unlikely to need on this run. The actual ctx in
    compute_config can still be larger if the remaining VRAM allows it.
    The target is supplied by the active PerformanceTarget — "safe"
    keeps the legacy 128k value, "throughput" shrinks it to 32k.
    """
    if base_kv_per_token_mb <= 0:
        base_kv_per_token_mb = kv_per_token_mb_f16(params_billion)

    shared_overhead_gb = model_size_gb * 0.08
    per_layer_expert_gb = max(0.001, (model_size_gb - shared_overhead_gb) / n_layers)

    # ---- KV reservation in VRAM (q5_0 assumption) -----------------------
    # Cap at moe_placement_ctx_target so we don't pessimise layer placement
    # for huge profile_max values (Qwen3.6 → 262k, but most users run 32k).
    kv_reservation_ctx = max(2048, min(target_ctx, moe_placement_ctx_target))
    kv_reserve_gb = (
        kv_reservation_ctx * base_kv_per_token_mb * kv_quant_factor("q5_0")
    ) / 1024.0

    # Layer placement uses VRAM left over AFTER KV + shared overhead.
    usable_for_experts = (
        free_vram_gb - moe_vram_safety_gb - shared_overhead_gb - kv_reserve_gb
    )

    if usable_for_experts < 0:
        # Not even the shared overhead + KV fits → everything via mmap/RAM.
        if free_ram_gb - ram_safety_gb < model_size_gb - shared_overhead_gb:
            return 999, n_layers, shared_overhead_gb, model_size_gb, False
        return (
            999,
            n_layers,
            shared_overhead_gb,
            model_size_gb - shared_overhead_gb,
            False,
        )

    layers_on_gpu = int(usable_for_experts / per_layer_expert_gb)
    layers_on_gpu = max(0, min(n_layers, layers_on_gpu))
    n_cpu_moe = n_layers - layers_on_gpu

    model_vram = shared_overhead_gb + layers_on_gpu * per_layer_expert_gb

    if n_cpu_moe == 0:
        # All experts on GPU.
        return 999, 0, model_size_gb, 0.0, True

    # Some experts on CPU — they live in RAM via mmap.
    model_ram = n_cpu_moe * per_layer_expert_gb
    return 999, n_cpu_moe, model_vram, model_ram, False


# ---- Turbo-Quant labels --------------------------------------------------
# The TurboQuant family of forks (TheTom/turboquant_plus,
# AtomicBot/atomic-llama-cpp-turboquant, spiritbuun/buun-llama-cpp)
# adds three new -ctk / -ctv labels that pack the KV-cache much
# tighter than the stock f16 → q4_0 ladder:
#
#     turbo4   ~3.8× vs F16   (4-bit, highest accuracy, default fallback)
#     turbo3   ~4.3× vs F16   (3-bit, the "sweet spot" — recommended default)
#     turbo2   ~6.4× vs F16   (2-bit, max compression, quality drops)
#
# We map upstream quant choices to their turbo equivalent **at the
# bit-width tier the algorithm already picked** — q8_0 → turbo4 (both
# are the "high-accuracy" tier), q5_0 → turbo3 (mid), q4_0 → turbo3
# (low, but turbo3 is still measurably better than q4_0 at long ctx).
# turbo2 is intentionally not auto-selected; users who really want it
# pick it manually in the Expert panel.
_TURBO_QUANT_MAP: Dict[str, str] = {
    "f16": "turbo4",  # if someone runs f16 on a turbo fork, give them headroom
    "q8_0": "turbo4",  # 4-bit, ~3.8x, highest-accuracy turbo tier
    "q5_0": "turbo3",  # 3-bit, ~4.3x, the canonical default
    "q5_1": "turbo3",
    "q4_0": "turbo3",  # 3-bit beats q4_0 noticeably at long context
    "q4_1": "turbo3",
    "iq4_nl": "turbo3",
}


def _turbo_quant_for(label: str) -> str:
    """Map a normal KV quant label to its TurboQuant equivalent.

    Falls back to the input label when no mapping is known — that keeps
    Turbo a *safe* toggle: the worst case is "same quant as before".
    """
    return _TURBO_QUANT_MAP.get(label.lower(), label)


def _pick_kv_quant(
    profile_recommended: str,
    target_ctx: int,
    base_kv_per_token_mb: float,
    kv_budget_gb: float,
    model_max_ctx: int = 0,  # native_ctx aus GGUF-Metadata (0 = keine Begrenzung)
    *,
    turbo: bool = False,
    asymmetric: bool = True,  # Vulkan b9106+ supports asymmetric FA
) -> Tuple[str, str]:
    """Pick best (K, V) quants that fit target_ctx into kv_budget_gb.

    With ``asymmetric=True`` (default — Vulkan ≥ b9106, ROCm, CUDA all
    support it) K and V may use *different* quants. The strategy is to
    keep K at the highest quality that still fits, and step V down one
    bucket — because K-quantisation hurts attention recall more than
    V-quantisation hurts output quality. When K and V cannot live at
    different levels (e.g. an older fork without asymmetric FA), pass
    ``asymmetric=False`` for the legacy K=V behaviour.

    With ``turbo=True`` the chosen labels are mapped via
    :data:`_TURBO_QUANT_MAP` to the TurboQuant equivalents. Practical
    effect: ~10 % smaller KV at the same nominal precision.

    Order tried (top → bottom = best → worst quality):

        K=q8_0  V=q8_0     # symmetric high
        K=q8_0  V=q5_0     # asymmetric mid     (only if asymmetric=True)
        K=q5_0  V=q5_0     # symmetric mid
        K=q5_0  V=q4_0     # asymmetric low     (only if asymmetric=True)
        K=q4_0  V=q4_0     # symmetric low
    """
    # Beschränke target_ctx auf Modell-Maximum wenn nötig.
    if model_max_ctx > 0 and target_ctx > model_max_ctx:
        target_ctx = model_max_ctx

    # Quality-ranked pairs. The list is *static*; the profile-recommended
    # quant only nudges the starting index forward when its symmetric
    # variant is in the list (so e.g. recommended_kv_quant=q8_0 still
    # tries q8_0/q8_0 first even though it's the default top-of-list).
    pairs: List[Tuple[str, str]] = []
    if asymmetric:
        pairs = [
            ("q8_0", "q8_0"),
            ("q8_0", "q5_0"),
            ("q5_0", "q5_0"),
            ("q5_0", "q4_0"),
            ("q4_0", "q4_0"),
        ]
    else:
        pairs = [
            ("q8_0", "q8_0"),
            ("q5_0", "q5_0"),
            ("q4_0", "q4_0"),
        ]

    # Honour profile_recommended as the starting *floor* — never go above
    # what the model author tested. If the recommended quant is q5_0 we
    # skip the q8_0 rows.
    rec = profile_recommended.lower()
    if rec in ("q8_0", "q5_0", "q4_0"):
        # Drop pairs whose K-quant is strictly better than the recommended.
        order_rank = {"q4_0": 0, "q5_0": 1, "q8_0": 2}
        rec_rank = order_rank[rec]
        pairs = [p for p in pairs if order_rank[p[0]] <= rec_rank]
        if not pairs:
            pairs = [(rec, rec)]  # defensive — should never happen

    budget_mb = kv_budget_gb * 1024 * 0.98

    # When the user enabled Turbo-KV, the quants we are about to test
    # are NOT the labels that will actually end up in the cmd line —
    # we map (q8_0, q5_0, q4_0) → (turbo4, turbo3, turbo3) just below.
    # The Turbo labels are denser than their q-counterparts, so the
    # budget check has to use the turbo factor or the AutoTuner will
    # leave a lot of context on the table (Basti's complaint: "the
    # token count doesn't change when switching to Turbo").
    def _factor_for_pair(k_label: str, v_label: str) -> float:
        if turbo:
            k_label = _turbo_quant_for(k_label)
            v_label = _turbo_quant_for(v_label)
        return (kv_quant_factor(k_label) + kv_quant_factor(v_label)) / 2

    for k, v in pairs:
        per_tok = base_kv_per_token_mb * _factor_for_pair(k, v)
        if per_tok <= 0:
            continue
        max_fit = int(budget_mb / per_tok)
        if max_fit >= target_ctx:
            chosen_k, chosen_v = k, v
            break
    else:
        # Nothing in the table fit — fall back to the most aggressive entry.
        chosen_k, chosen_v = pairs[-1]

    if turbo:
        chosen_k = _turbo_quant_for(chosen_k)
        chosen_v = _turbo_quant_for(chosen_v)

    return chosen_k, chosen_v


# ---------------------------------------------------------------------------
# Main entry


def compute_config(
    model: ModelEntry,
    system: SystemInfo,
    profile: ModelProfile,
    draft_model: Optional[ModelEntry] = None,
    user_ctx: Optional[int] = None,
    ram_safety_gb: Optional[float] = None,
    vram_safety_gb: Optional[float] = None,
    force_mlock: bool = False,
    perf_target: Optional[PerformanceTarget] = None,
    mode: str = "chat",
    *,
    # ---- Expert-mode (auto-cascade) overrides --------------------------
    # When any of these is set, the AutoTuner respects the user-supplied
    # value and lets the rest of the configuration cascade around it.
    # Manual mode bypasses compute_config entirely and builds a
    # TunedConfig directly from widget values, so these only apply to
    # the cascading Auto branch of the Expert panel.
    turbo_kv: bool = False,  # Map quants → TurboQuant equivalents
    force_cache_k: Optional[str] = None,  # Pin K-quant; ctx adjusts
    force_cache_v: Optional[str] = None,  # Pin V-quant; ctx adjusts
    force_ngl: Optional[int] = None,  # Pin layer offload count
    force_n_cpu_moe: Optional[int] = None,  # Pin MoE CPU-layer count
    force_rope_scale: Optional[bool] = None,  # Force YaRN on/off
    # ---- GPU priority overrides ----------------------------------------
    # Optional mapping of GPU name → user-assigned priority (≥1).
    # When provided, the GPU with the highest priority×VRAM score is
    # selected as the primary compute device (main_gpu). Priorities are
    # read from autotuner_settings.json → gpu_overrides.priority and
    # exposed through app_settings.get_gpu_priorities(). When absent or
    # None, pure VRAM size determines the primary GPU (legacy behaviour).
    gpu_priorities: Optional[Dict[str, int]] = None,
) -> TunedConfig:
    """Compute a TunedConfig that fits this model on this system.

    Priority order for VRAM allocation:
      1. Vision model (mmproj) — always placed on GPU first
      2. Draft model (speculative decoding) — always placed on GPU first
      3. Main model (weights + KV cache)

    The ``perf_target`` argument controls the safety/headroom regime
    (see ``performance_target.py``). If ``None``, it is resolved from
    ``profile.performance_target`` — falling back to "balanced" if the
    profile doesn't specify one. Callers (CLI, GUI) typically resolve
    the target themselves so a user override beats the YAML default.

    Explicit ``ram_safety_gb`` / ``vram_safety_gb`` arguments still win
    over the perf_target's values; pass ``None`` (the default) to use
    whatever the resolved target prescribes.

    Expert overrides (keyword-only)
    --------------------------------
    These are exposed primarily for the GUI's Expert panel. The plain
    CLI path keeps using the auto-tuned defaults — only set these when
    you have a specific reason to pin a value. The cascading rule:
    *whatever you pin stays; everything not pinned recomputes around it*.
    """
    # ---- Resolve performance target. Caller-supplied wins; otherwise we
    # fall back to whatever the profile recommends (or "balanced").
    if perf_target is None:
        perf_target = resolve_performance_target(
            cli_choice=None,
            profile_choice=getattr(profile, "performance_target", "") or None,
        )

    # ---- Apply the target's safety values where the caller didn't override.
    if ram_safety_gb is None:
        ram_safety_gb = perf_target.ram_safety_gb
    if vram_safety_gb is None:
        vram_safety_gb = perf_target.dense_vram_safety_gb

    # Number of parallel inference slots — always passed as --parallel N
    # to llama-server to prevent auto-detection from over-provisioning KV.
    n_parallel: int = max(1, perf_target.n_parallel)

    has_gpu = bool(system.gpus) and system.total_vram_gb > 1
    free_vram = max(0.0, system.free_vram_gb)
    n_layers = model.n_layers

    # ---- (0) MoE detection
    expert_count = _moe_expert_count(model)
    is_moe = expert_count > 1
    params_b = extract_params_billion(model.name)

    # ---- (0.2) Primary inference GPU selection (multi-GPU only)
    # The user's preferred main GPU is the one with the highest
    # priority×VRAM score (e.g. R9700 32 GB @ priority 2 beats RX 9070 XT
    # 16 GB @ priority 1).  Two things are computed against THIS card:
    #   • MoE expert placement (n_cpu_moe) — experts never spread onto the
    #     secondary GPU, they spill to CPU/RAM, so only the primary's free
    #     VRAM is relevant.  Using the *summed* free VRAM of all GPUs (the
    #     old behaviour) overcommits and crashes with ErrorOutOfDeviceMemory
    #     once the KV cache grows past what the primary alone can hold.
    #   • Single-GPU pinning via device-visibility env vars (section 4d).
    # Falls back gracefully to the summed value on single-GPU / CPU systems.
    _prio_map = gpu_priorities or {}

    def _gpu_score(g: GPUInfo) -> float:
        return max(1, _prio_map.get(g.name, 1)) * g.total_vram_mb

    primary_gpu: Optional[GPUInfo] = None
    primary_free_vram_gb = free_vram  # default = summed (single-GPU / CPU)
    if has_gpu and system.gpus:
        primary_gpu = max(system.gpus, key=_gpu_score)
        if len(system.gpus) > 1:
            primary_free_vram_gb = max(0.0, primary_gpu.free_vram_mb / 1024.0)

    # ---- (0.1) KV per-token: MUST be defined before any branch uses it.
    # This is the bug that caused crashes on selection of any non-Qwen
    # model in v3.x — base_kv_mb was previously only set inside the
    # rope-scaling branch, but referenced unconditionally further below.
    base_kv_mb = _resolve_kv_per_token_mb(model, params_b)

    native_ctx = model.native_context  # GGUF metadata: model's native ctx

    # RoPE-Scaling Konfiguration aus Profil lesen
    profile_rope_scale = profile.rope_scale_enabled
    profile_rope_max = profile.rope_scale_max_ctx  # Standard: 1M
    profile_rope_factor = profile.rope_scale_factor  # Standard: 4.0

    rope_scaled_ctx = (
        0  # Wird später berechnet (braucht free_vram_after/free_ram_after)
    )
    rope_scaling_active = False  # Flag für build_command

    profile_max = profile.max_context
    if native_ctx > 0:
        profile_max = min(profile_max, native_ctx)
    target_ctx_for_placement = user_ctx if user_ctx is not None else profile_max

    # ---- (0.5) Calculate VRAM reserved for Vision + Draft models
    # These MUST be on GPU for optimal performance.
    vision_vram_gb = 0.0
    draft_vram_gb = 0.0

    if model.mmproj is not None:
        # Vision model (mmproj) — estimate from file size
        try:
            mmproj_size_bytes = model.mmproj.stat().st_size
            vision_vram_gb = mmproj_size_bytes / (1024**3)
        except (OSError, AttributeError):
            # Fallback: ~6 GB for typical F16 mmproj files
            vision_vram_gb = 6.0

    if draft_model is not None:
        # Draft model — must fit in VRAM for speculative decoding to work well
        draft_vram_gb = draft_model.size_gb

    # Effective VRAM available for main model placement
    effective_free_vram = free_vram - vision_vram_gb - draft_vram_gb
    if effective_free_vram < 0:
        effective_free_vram = 0.0

    # Same, but scoped to the PRIMARY GPU only — MoE expert placement must
    # use this (experts spill to CPU, never to the secondary GPU). On
    # single-GPU systems this equals effective_free_vram.
    effective_primary_free_vram = primary_free_vram_gb - vision_vram_gb - draft_vram_gb
    if effective_primary_free_vram < 0:
        effective_primary_free_vram = 0.0

    # For MoE models with multiple GPUs, use the combined VRAM for expert
    # placement. This allows large MoE models like Qwen3.5-122B-A10B to
    # utilise both GPUs (R9700 + RX 9070 XT = 48 GB total) instead of
    # being restricted to the primary GPU only.
    has_multiple_gpus = has_gpu and len(system.gpus) > 1
    if has_multiple_gpus:
        # Combined free VRAM across all GPUs for MoE expert placement.
        combined_free_vram_gb = sum(g.free_vram_mb / 1024.0 for g in system.gpus)
        effective_moe_vram = combined_free_vram_gb - vision_vram_gb - draft_vram_gb
    else:
        effective_moe_vram = effective_primary_free_vram
    if effective_moe_vram < 0:
        effective_moe_vram = 0.0

    # ---- (1) Model placement
    n_cpu_moe: Optional[int] = None
    if is_moe and has_gpu and n_layers > 0:
        ngl, n_cpu_moe, model_vram, model_ram, full_off = _decide_moe_offload(
            model_size_gb=model.size_gb,
            free_vram_gb=effective_moe_vram,
            free_ram_gb=system.free_ram_gb,
            n_layers=n_layers,
            expert_count=expert_count,
            params_billion=params_b,
            target_ctx=target_ctx_for_placement,
            base_kv_per_token_mb=base_kv_mb,
            ram_safety_gb=ram_safety_gb,
            moe_vram_safety_gb=perf_target.moe_vram_safety_gb,
            moe_placement_ctx_target=perf_target.moe_placement_ctx_target,
        )

        # ---- Two-pass placement fallback ---------------------------------
        # If the first pass dumped *every* expert layer to CPU but >4 GB
        # of VRAM is still free, the KV reservation was clearly too
        # pessimistic for this model. Retry once with the placement
        # target halved (down to a 16k floor). This is a defensive net
        # for hybrid architectures we don't recognise yet, or for
        # quantisations where our heuristic mis-estimates KV footprint.
        if (
            n_cpu_moe is not None
            and n_layers > 0
            and n_cpu_moe >= n_layers
            and effective_moe_vram > 4.0
            and perf_target.moe_placement_ctx_target > 16384
        ):
            shrunk_target = max(16384, perf_target.moe_placement_ctx_target // 2)
            ngl_2, cpu_moe_2, vram_2, ram_2, full_2 = _decide_moe_offload(
                model_size_gb=model.size_gb,
                free_vram_gb=effective_moe_vram,
                free_ram_gb=system.free_ram_gb,
                n_layers=n_layers,
                expert_count=expert_count,
                params_billion=params_b,
                target_ctx=target_ctx_for_placement,
                base_kv_per_token_mb=base_kv_mb,
                ram_safety_gb=ram_safety_gb,
                moe_vram_safety_gb=perf_target.moe_vram_safety_gb,
                moe_placement_ctx_target=shrunk_target,
            )
            # Only adopt the second pass if it actually placed layers on GPU.
            if cpu_moe_2 is not None and cpu_moe_2 < n_cpu_moe:
                ngl, n_cpu_moe, model_vram, model_ram, full_off = (
                    ngl_2,
                    cpu_moe_2,
                    vram_2,
                    ram_2,
                    full_2,
                )

        if n_cpu_moe == 0:
            n_cpu_moe = None
    else:
        ngl, model_vram, model_ram, full_off = _decide_offload(
            model_size_gb=model.size_gb,
            free_vram_gb=effective_free_vram,
            n_layers=n_layers,
            has_gpu=has_gpu,
            vram_headroom_gb=vram_safety_gb,
        )

    # ---- (1.5) Expert overrides: force_ngl / force_n_cpu_moe -----------
    # Applied AFTER the automatic placement so model_vram / model_ram
    # estimates reflect the user's pinned values. The user owns the
    # consequences (over/undercommit); we only redistribute the model
    # size estimate to match the new layer split.
    if force_n_cpu_moe is not None and is_moe and has_gpu and n_layers > 0:
        new_cpu_moe = max(0, min(n_layers, int(force_n_cpu_moe)))
        # Re-derive model_vram/ram from the new split, holding shared
        # overhead constant (it scales with model size, not layer
        # placement).
        shared_overhead_gb = model.size_gb * 0.08
        per_layer_expert_gb = max(
            0.001, (model.size_gb - shared_overhead_gb) / n_layers
        )
        layers_on_gpu = n_layers - new_cpu_moe
        model_vram = shared_overhead_gb + layers_on_gpu * per_layer_expert_gb
        model_ram = new_cpu_moe * per_layer_expert_gb
        n_cpu_moe = new_cpu_moe if new_cpu_moe > 0 else None
        full_off = new_cpu_moe == 0
        ngl = 999

    if force_ngl is not None and n_layers > 0 and not (is_moe and has_gpu):
        new_ngl = max(0, min(n_layers, int(force_ngl)))
        per_layer_gb = model.size_gb / n_layers
        ngl = new_ngl if new_ngl < n_layers else 999
        if new_ngl >= n_layers:
            model_vram = model.size_gb
            model_ram = 0.0
            full_off = True
        else:
            model_vram = new_ngl * per_layer_gb
            residual_overhead = model.size_gb * 0.02
            model_ram = (n_layers - new_ngl) * per_layer_gb + residual_overhead
            full_off = False

    # ---- (2) Remaining KV budget — include vision/draft VRAM in total
    effective_vram_safety = (
        perf_target.moe_vram_safety_gb if n_cpu_moe is not None else vram_safety_gb
    )
    free_vram_after = max(
        0.0,
        free_vram - effective_vram_safety - model_vram - vision_vram_gb - draft_vram_gb,
    )
    free_ram_after = max(0.0, system.free_ram_gb - ram_safety_gb - model_ram)

    # KV-cache placement rules:
    #   - MoE on GPU: KV must live in VRAM only. The Vulkan backend
    #     crashes with GGML_ASSERT(addr) when MoE KV spills to RAM.
    #   - Dense full-offload: KV in VRAM only (it's already on GPU).
    #   - Dense partial: KV split MIRRORS layer split. The VRAM portion
    #     limits the total budget; RAM portion is intentionally capped
    #     so we never bleed multi-GB KV cache into slow main memory.
    #     This was the root cause of the gemma-31B-Q3-with-draft bug:
    #     the old code added free_ram_after wholesale and produced an
    #     11 GB KV cache living in RAM, dragging inference to a crawl.
    #   - CPU-only: KV lives entirely in RAM.
    #
    # Cap RAM-resident KV at this many GB for hybrid placements. The
    # value is chosen so even a tight 30B hybrid (free_vram_after ≈ 0.25 GB)
    # can reach the 32k context floor when enough system RAM is available.
    # (2 GB was too small: 0.25 + 2.0 = 2.25 GB → ~27k tokens for a
    # 64-layer model with 8 GQA heads at q4_0, just below the 32k target.)
    # 4 GB raises that to ~53k so the floor clamps correctly to 32k.
    # The cap still prevents the multi-GB-KV-in-RAM trap from the old
    # uncapped formula.
    HYBRID_KV_RAM_CAP_GB = 4.0

    if is_moe and has_gpu:
        kv_budget_gb = free_vram_after
    elif full_off:
        # Dense model fully on GPU, but the model may nearly fill VRAM
        # (e.g. a 27B Q3 occupying 14 of 16 GB), leaving almost nothing
        # for KV. Layer computation stays on GPU; allow the KV cache to
        # use system RAM so the server reaches a useful context length.
        # Cap prevents consuming all available RAM for KV alone.
        # The VRAM share is always included — if VRAM has headroom, KV
        # goes there first and is fast; the RAM portion only matters when
        # VRAM is nearly full.
        DENSE_FULL_KV_RAM_CAP_GB = 8.0
        ram_supplement = min(free_ram_after, DENSE_FULL_KV_RAM_CAP_GB)
        kv_budget_gb = free_vram_after + ram_supplement
    elif ngl > 0 and n_layers > 0:
        # Dense hybrid: derive max total-KV budget so neither the GPU
        # nor the (capped) RAM share blows past its limit. The actual
        # ctx in step (3) picks whichever quant fits this total budget.
        gpu_layer_fraction = ngl / n_layers
        if gpu_layer_fraction >= 0.99:
            # Effectively full offload — treat KV as VRAM-only.
            kv_budget_gb = free_vram_after
        elif gpu_layer_fraction <= 0.0:
            # No GPU layers — should not happen (ngl > 0) but stay defensive.
            kv_budget_gb = min(free_ram_after, HYBRID_KV_RAM_CAP_GB)
        else:
            # Dense hybrid: KV budget = whatever VRAM headroom remains
            # PLUS a capped RAM supplement.
            #
            # Old formula: min(free_vram/gpu_frac, ram_cap/cpu_frac)
            # Problem: when VRAM is nearly full (e.g. model + vision leaves
            # only 0.05 GB), the VRAM term collapses to 0.06 GB and the
            # min drives kv_budget to ~0 → context 2048. The proportional
            # formula assumed VRAM is never the exhausted resource.
            #
            # Additive approach: use every byte of remaining VRAM for KV,
            # then top up with up to HYBRID_KV_RAM_CAP_GB from system RAM.
            # The cap still prevents the "10 GB KV in RAM" trap that
            # motivated the previous formula; it just handles the VRAM-
            # exhausted case gracefully.
            ram_supplement = min(free_ram_after, HYBRID_KV_RAM_CAP_GB)
            kv_budget_gb = free_vram_after + ram_supplement
    else:
        # CPU-only — KV lives entirely in RAM.
        kv_budget_gb = free_ram_after

    # ---- (2.5) RoPE-Scaling (YaRN) auto-detection
    # Aktiviere RoPE-Scaling automatisch wenn:
    # 1. Modell RoPE-Scaling unterstützt (qwen2 etc.) ODER das YAML-Profil
    #    rope_scale.enabled=true setzt (erlaubt Profil-Autoren, RoPE-Scaling
    #    für Architekturen zu aktivieren die nicht in _ROPE_SCALE_SUPPORTED_ARCHS
    #    stehen — z.B. phi3/Phi-4 mit nativem 16k-Kontext aber 128k-Kapazität)
    # 2. Genügend Speicher für Context > native_ctx vorhanden ist
    # 3. Entweder profil-configured (rope_scale.enabled=true) ODER
    #    berechneter max_fit_ctx überschreitet native_ctx
    rope_scaled_ctx = 0
    rope_scaling_active = False

    if (
        (model.supports_rope_scale or profile_rope_scale)
        and native_ctx > 0
        and native_ctx < profile_rope_max
    ):
        # KV-Speicherbedarf pro Token (q5_0 als Entscheidungsgrundlage)
        kv_per_tok_q5 = base_kv_mb * kv_quant_factor("q5_0")

        # Gewünschter Context: user-specified oder Profil-Maximum.
        # WICHTIG: Hier das UNCAPPED profile.max_context verwenden (nicht das
        # native_ctx-beschränkte profile_max), denn der Sinn von RoPE-Scaling
        # ist ja, über native_ctx hinaus zu gehen. Wenn profile.max_context
        # bereits <= native_ctx ist, wird desired_ctx <= native_ctx und die
        # Aktivierungsbedingung unten bleibt False (korrekt).
        desired_ctx = user_ctx if user_ctx is not None else profile.max_context

        # Wenn gewünschter Context das native Limit überschreitet
        if desired_ctx > native_ctx:
            # Prüfe ob Speicher vorhanden ist
            rope_kv_gb = (desired_ctx * kv_per_tok_q5) / 1024
            total_available = free_vram_after + free_ram_after

            # Aktiviere RoPE-Scaling wenn >= 90% des Bedarfs verfügbar
            if profile_rope_scale or total_available >= rope_kv_gb * 1.1:
                rope_scaled_ctx = min(desired_ctx, profile_rope_max)
                rope_scaling_active = True

    # Expert override: force_rope_scale = True turns it on unconditionally;
    # force_rope_scale = False turns it off. Either choice respects native_ctx
    # as a hard upper bound.
    if force_rope_scale is True:
        rope_scaled_ctx = min(
            (user_ctx if user_ctx is not None else profile_rope_max),
            profile_rope_max,
        )
        rope_scaling_active = True
    elif force_rope_scale is False:
        rope_scaled_ctx = 0
        rope_scaling_active = False

    # ---- (3) Context + KV quant
    target_ctx = user_ctx if user_ctx is not None else profile_max

    # Bestimme das effektive Modell-Maximum für die KV-Quantisierung:
    # - rope_scaled_ctx: erweiterbares Maximum via YaRN (wenn aktiviert)
    # - native_ctx: natives Maximum des Modells (aus GGUF)
    model_ctx_limit = rope_scaled_ctx if rope_scaled_ctx > 0 else native_ctx
    if model_ctx_limit <= 0:
        model_ctx_limit = profile_max

    # Expert overrides for KV-quant: when both K and V are pinned we
    # respect the user's pair as-is; when only one is pinned we still
    # let _pick_kv_quant decide the other within budget.
    kv_quant_strategy = "symmetric"
    if force_cache_k is not None and force_cache_v is not None:
        cache_k, cache_v = force_cache_k, force_cache_v
        if turbo_kv:
            cache_k = _turbo_quant_for(cache_k)
            cache_v = _turbo_quant_for(cache_v)
            kv_quant_strategy = "manual+turbo"
        else:
            kv_quant_strategy = "manual"
    else:
        cache_k, cache_v = _pick_kv_quant(
            profile.recommended_kv_quant,
            target_ctx,
            base_kv_mb,
            kv_budget_gb,
            model_ctx_limit,
            turbo=turbo_kv,
            asymmetric=True,
        )
        if force_cache_k is not None:
            cache_k = _turbo_quant_for(force_cache_k) if turbo_kv else force_cache_k
        if force_cache_v is not None:
            cache_v = _turbo_quant_for(force_cache_v) if turbo_kv else force_cache_v
        if cache_k != cache_v:
            kv_quant_strategy = "asymmetric"
        if turbo_kv:
            kv_quant_strategy = (
                f"{kv_quant_strategy}+turbo"
                if kv_quant_strategy != "symmetric"
                else "turbo"
            )

    actual_per_tok_mb = (
        base_kv_mb * (kv_quant_factor(cache_k) + kv_quant_factor(cache_v)) / 2
    )

    max_fit_ctx: int = 0  # computed only in auto mode; needed for floor guard
    if user_ctx is not None:
        # User-specified context — respect it but clamp to model limits
        ctx = user_ctx
        if model_ctx_limit > 0 and ctx > model_ctx_limit:
            ctx = model_ctx_limit
    else:
        # Berechne den maximal möglichen Kontext basierend auf dem verfügbaren
        # KV-Cache-Budget. Dividiere durch n_parallel, da llama-server N Slots
        # anlegt (jeder Slot braucht einen vollen KV-Buffer der angeforderten
        # Größe). Ohne diese Division würde llama-server "auto" n_parallel auf
        # z.B. 4 setzen und 4× das Budget belegen.
        #
        # Beispiel: 21 GB KV-Budget, n_parallel=1 →
        #   max_fit_ctx bei Q8 (0.060 MB/tok) = 356k → cap auf 262k ✓
        #   RAM-Nutzung ~3 GB statt ~60 GB (bei n_parallel=4 ohne diesen Fix).
        kv_budget_per_slot_gb = kv_budget_gb / n_parallel
        if actual_per_tok_mb > 0:
            max_fit_ctx = int((kv_budget_per_slot_gb * 1024 * 0.995) / actual_per_tok_mb)
        else:
            max_fit_ctx = profile_max

        # Beschränke auf das Modell-Maximum (native oder rope-scaled)
        if model_ctx_limit > 0:
            ctx = min(max_fit_ctx, model_ctx_limit)
        else:
            ctx = min(max_fit_ctx, profile_max * 3)

    # Minimum context floor — AUTO MODE ONLY.
    # When the user explicitly sets a context (user_ctx is not None) we
    # respect that value as-is (already clamped to model limits above).
    # The 32k floor is a quality-of-life default for the auto calculation
    # so that system-prompts + tool scaffolding (e.g. zoo-code starts at
    # ~10-12k) leave meaningful room for the actual conversation.
    # Two guards prevent over-promising in auto mode:
    #   (a) model cap  — if the model's native context is below 32k, use that
    #   (b) VRAM cap   — never exceed what the KV budget can actually fit
    if user_ctx is None:
        _PREF_MIN_CTX = 32768
        effective_min = _PREF_MIN_CTX
        if model_ctx_limit > 0 and model_ctx_limit < effective_min:
            effective_min = (model_ctx_limit // 1024) * 1024  # model too small for 32k
        if max_fit_ctx > 0 and max_fit_ctx < effective_min:
            effective_min = max(2048, (max_fit_ctx // 1024) * 1024)  # budget too tight
        ctx = max(effective_min, (ctx // 1024) * 1024)
    else:
        ctx = max(2048, (ctx // 1024) * 1024)  # absolute safety floor only
    estimated_kv_gb = (ctx * actual_per_tok_mb) / 1024

    # ---- (3b) VRAM Overcommit Warning
    warning: Optional[str] = None
    if n_cpu_moe is not None or full_off:
        gpu_total = model_vram + estimated_kv_gb + effective_vram_safety
        if gpu_total > free_vram * 0.98:
            warning = (
                f"VRAM budget tight: model {model_vram:.1f} GB + KV "
                f"{estimated_kv_gb:.1f} GB + safety "
                f"{effective_vram_safety:.1f} GB ≈ {gpu_total:.1f} GB of "
                f"{free_vram:.1f} GB free."
            )

    # ---- (4) Threads — weniger Threads für bessere Performance
    # start_llama.py verwendet: cpu_count // 2 (max 8 bei <16 cores)
    physical = system.cpu_cores_physical
    logical = system.cpu_cores_logical
    optimal_threads = (logical // 2) if logical > 8 else logical

    if full_off:
        threads = min(optimal_threads, 16)
        batch_threads = min(physical, 16)
    elif n_cpu_moe is not None and n_cpu_moe > 0:
        threads = min(optimal_threads, 24)
        batch_threads = min(logical, 32)
    elif ngl > 0:
        threads = min(optimal_threads, 20)
        batch_threads = min(logical, 32)
    else:
        threads = min(optimal_threads, 32)
        batch_threads = min(logical, 64)

    # ---- (4b) Batch / ubatch sizing
    # Three regimes, picked in order:
    #
    #   1. MoE with CPU-resident experts (`--n-cpu-moe` > 0): use the
    #      perf_target's moe_hybrid_batch/ubatch. Larger batches let
    #      llama.cpp's op-offload prompt processing copy CPU-resident
    #      expert tensors to the GPU as a single batched operation,
    #      which is much faster than per-token round-trips. Reference:
    #      HuggingFace MoE-offload guide (Doctor-Shotgun, Feb 2026) and
    #      the gfx1151 ROCm/Vulkan benchmark in llama.cpp issue #21284,
    #      both showing near-linear PP scaling up to -ub 2048/4096.
    #
    #   2. Full GPU offload of a large dense model (>30 GB) OR long
    #      context (>32k): 1024/1024 — keeps the compute buffer modest
    #      so the model itself doesn't get squeezed.
    #
    #   3. Everything else (small-to-mid dense, short ctx): 2048/512 —
    #      the historical default that's optimal for pure GPU inference.
    if n_cpu_moe is not None and n_cpu_moe > 0:
        batch = perf_target.moe_hybrid_batch
        ubatch = perf_target.moe_hybrid_ubatch
        # When integrated MTP is active on a MoE model, the speculative hook
        # fires at every ubatch boundary during generation. With moe_hybrid_ubatch
        # at 2048 or 4096 the D2H transfer overhead per token grows and write speed
        # regresses below baseline. Cap ubatch at 512 for MTP MoE models so the
        # generation phase has the same granularity the community uses (b 2048 ub 512).
        # Prompt processing (PP) is unaffected because PP fills full batches anyway.
        if model.has_embedded_mtp and ubatch > 512:
            ubatch = 512
    elif model.size_gb > 30 or ctx > 32768 or model.size_gb > 10:
        batch, ubatch = 1024, 1024
    else:
        batch, ubatch = 2048, 512

    # ---- (4c) mlock + no_mmap (Windows Admin Check)
    ram_resident_gb = model_ram

    is_windows = platform.system() == "Windows"
    is_admin = False
    if is_windows:
        if ctypes:
            try:
                is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            except Exception:
                is_admin = False
    else:
        # Auf Linux/Mac prüfen wir auf Root
        try:
            # Benutze getattr, damit Pylance nicht direkt nach dem Attribut sucht
            getuid = getattr(os, "getuid", None)
            is_admin = getuid() == 0 if getuid else True
        except Exception:
            is_admin = True

    # Option A: VRAM-basierte Bedingung für full-off Modelle
    # Wenn das Modell vollständig auf der GPU ist (full_off=True), kann mlock/no-mmap
    # trotzdem sinnvoll sein, um VRAM-Paging zu verhindern.
    vram_resident_gb = model_vram
    has_enough_vram = system.total_vram_gb > 8

    if force_mlock:
        # Option B: User-Override — aktiviert mlock/no-mmap wenn System-Ressourcen reichen
        mlock = (has_enough_vram or vram_resident_gb > 0) and (
            is_windows and is_admin or not is_windows
        )
    else:
        # Automatische Logik: zwei Fälle
        if full_off:
            # Full GPU offload: prüfe VRAM statt RAM
            mlock = (
                has_enough_vram
                and vram_resident_gb > 0
                and vram_resident_gb < (system.free_vram_gb - 2)
                and (not is_windows or is_admin)
            )
        else:
            # Partial/CPU offload: prüfe RAM
            mlock = (
                system.total_ram_gb > 32
                and ram_resident_gb > 0
                and ram_resident_gb < (system.free_ram_gb - 8)
                and (not is_windows or is_admin)
            )
    no_mmap = mlock

    # ---- (4d) Multi-GPU placement & device visibility.
    #
    # Runs for BOTH dense and MoE configs.  The previous version skipped
    # MoE entirely (`n_cpu_moe is None` gate), which left llama.cpp to
    # default to Vulkan0 — the 16 GB gaming GPU — and crash with
    # ErrorOutOfDeviceMemory while building the (MTP) draft context, even
    # though the 32 GB R9700 sat idle at Vulkan1.
    #
    # Fill strategy (matches the requested target: ~30/32 GB on the R9700,
    # ~13/16 GB on the RX 9070 XT, *then* system RAM):
    #
    #   1. Compute a per-card usable cap = total_vram − headroom, where the
    #      headroom keeps a card breathing for the OS/compositor/OBS.  The
    #      primary keeps ~2 GB, secondary cards keep ~3 GB (OBS encode).
    #   2. If the whole GPU footprint (weights + KV + vision + draft) fits in
    #      the PRIMARY's cap → pin everything to the primary and hide the
    #      secondary GPU completely, so it stays free for gaming/OBS.
    #   3. Otherwise → SEQUENTIALLY fill the primary up to its cap, then spill
    #      the remainder onto the secondary (and so on).  Only once every GPU
    #      cap is exhausted does llama.cpp fall back to RAM (dense: reduced
    #      ngl handled upstream; MoE: --n-cpu-moe).
    #
    # Device visibility / indices come from gpu.hip_index, resolved in
    # hardware.py by PCI-device-id (vulkaninfo --summary) → --list-devices →
    # vulkaninfo name match.  We NEVER use the Windows registry/detection
    # position as a device index (it is the opposite order on this system).
    # Both env vars are emitted so the config is backend-agnostic:
    #   - HIP_VISIBLE_DEVICES        → ROCm/HIP builds
    #   - GGML_VK_VISIBLE_DEVICES    → Vulkan builds (PR #5321+)
    tensor_split: Optional[str] = None
    main_gpu: Optional[int] = None
    env_overrides: Dict[str, str] = {}

    if has_gpu and len(system.gpus) > 1 and primary_gpu is not None:
        primary_pos = system.gpus.index(primary_gpu)
        hip_known = all(g.hip_index is not None for g in system.gpus)
        is_moe_cfg = n_cpu_moe is not None

        # Per-card usable VRAM cap (GB). Keep a little headroom so the card
        # never runs bone-dry: 2 GB on the primary, 3 GB on secondaries
        # (the RX 9070 XT also drives the desktop + OBS encode). Clamp by the
        # card's *free* VRAM so we never plan to use memory that other apps
        # already hold — "inclusive was schon genutzt wird".
        def _usable_cap_gb(gpu: GPUInfo, is_primary: bool) -> float:
            headroom = 2.0 if is_primary else 3.0
            cap_by_total = gpu.total_vram_mb / 1024.0 - headroom
            free_gb = gpu.free_vram_mb / 1024.0
            return max(0.0, min(cap_by_total, free_gb))

        primary_cap = _usable_cap_gb(primary_gpu, True)

        # Full GPU footprint we need to place: weights + KV + vision + draft.
        # For pinning decisions, we only consider model_vram (weights) because
        # the KV cache is dynamically allocated and won't consume the entire
        # budget. This allows smaller models to be pinned to the primary GPU
        # even when their theoretical max-KV footprint exceeds the cap.
        model_footprint_gb = model_vram + vision_vram_gb + draft_vram_gb

        # MoE: experts already spilled to CPU via --n-cpu-moe, so model_vram is
        # the GPU-resident portion. Pin to primary ONLY if it fits; otherwise
        # allow distribution across all GPUs to utilise the full VRAM pool.
        # Dense: pin when the model weights fit the primary cap.
        if is_moe_cfg:
            pin_to_primary = model_footprint_gb <= primary_cap
        else:
            pin_to_primary = model_footprint_gb <= primary_cap

        if pin_to_primary:
            if hip_known:
                # Expose ONLY the primary GPU. After this remap the primary is
                # the sole visible device (index 0), so EVERY allocation —
                # including the draft/MTP context that was crashing on Vulkan0 —
                # lands on the intended card (the R9700).
                vis = str(primary_gpu.hip_index)
                env_overrides["HIP_VISIBLE_DEVICES"] = vis
                env_overrides["GGML_VK_VISIBLE_DEVICES"] = vis
                main_gpu = 0  # only one device visible after remapping
            else:
                # Index unknown — pin weights via a position-based split. This
                # steers the main model to the primary but cannot HIDE the
                # secondary GPU. Ensure the llama binary / Vulkan SDK is
                # reachable so hardware.py can resolve hip_index next time.
                parts = ["0.000"] * len(system.gpus)
                parts[primary_pos] = "1.000"
                tensor_split = ",".join(parts)
                main_gpu = primary_pos
        else:
            # Spread: priority-weighted allocation across GPUs.
            # The resulting per-GPU GB amounts are turned into tensor_split
            # fractions (proportions of the whole model). llama.cpp distributes
            # BOTH weights and KV by these fractions.
            ordered = sorted(
                system.gpus,
                key=lambda g: max(1, _prio_map.get(g.name, 1)) * g.total_vram_mb,
                reverse=True,  # primary (highest score) first
            )
            caps = [_usable_cap_gb(g, g is primary_gpu) for g in ordered]
            total_cap = sum(caps)

            # Priority-weighted allocation: each GPU gets a share proportional
            # to its priority×VRAM score, capped by its usable cap.
            scores = [max(1, _prio_map.get(g.name, 1)) * g.total_vram_mb for g in ordered]
            total_score = sum(scores)

            # First pass: allocate proportionally by score, respecting caps.
            alloc: List[float] = []
            for i, cap in enumerate(caps):
                proportion = scores[i] / total_score if total_score > 0 else 0
                alloc.append(min(cap, model_footprint_gb * proportion))

            # Second pass: if there's remaining footprint, distribute it
            # proportionally among GPUs that haven't hit their cap.
            remaining = model_footprint_gb - sum(alloc)
            if remaining > 0.01:  # small epsilon to avoid floating-point noise
                for _ in range(3):  # iterate a few times to converge
                    if remaining <= 0.01:
                        break
                    for i, cap in enumerate(caps):
                        if alloc[i] < cap and remaining > 0.01:
                            space = cap - alloc[i]
                            take = min(space, remaining * 0.5)  # give half to each available GPU
                            alloc[i] += take
                            remaining -= take

            denom = sum(alloc) if sum(alloc) > 0 else (total_cap or 1.0)

            if hip_known:
                # Order by ascending device index so the visibility env vars and
                # the tensor_split fractions line up with what llama.cpp sees.
                idx_order = sorted(
                    range(len(ordered)),
                    key=lambda i: int(ordered[i].hip_index),  # type: ignore[arg-type]
                )
                vis_str = ",".join(str(ordered[i].hip_index) for i in idx_order)
                env_overrides["HIP_VISIBLE_DEVICES"] = vis_str
                env_overrides["GGML_VK_VISIBLE_DEVICES"] = vis_str
                tensor_split = ",".join(f"{alloc[i] / denom:.3f}" for i in idx_order)
                # main_gpu is the index (within the visible/sorted list) of the
                # primary card — where llama.cpp keeps the small shared tensors.
                main_gpu = idx_order.index(ordered.index(primary_gpu))
            else:
                # Indices unknown — position-based split in the system.gpus
                # order (may be wrong on Windows AMD; keep the llama binary /
                # vulkaninfo reachable so hip_index resolves).
                pos_alloc = {id(g): a for g, a in zip(ordered, alloc)}
                tensor_split = ",".join(
                    f"{pos_alloc.get(id(g), 0.0) / denom:.3f}" for g in system.gpus
                )
                main_gpu = primary_pos

    # ---- (4d) NUMA — immer aktivieren bei genügend Kernen für bessere Performance
    numa = None
    if system.cpu_cores_physical >= 16:
        numa = "distribute"

    # ---- (4f) Sampling
    # Two YAML schema variants are supported:
    #   New (chat/coding split):   sampling: { chat: {...}, coding: {...} }
    #   Old (flat / shared):       sampling: { temperature: ..., top_k: ... }
    #
    # The flat form is detected by the ABSENCE of both "chat" and
    # "coding" sub-dicts — in that case we use the flat dict for every
    # mode. New-format profiles that define only one of the two modes
    # still fall back to the flat dict for the missing mode, so a
    # half-migrated file behaves predictably.
    raw_sampling = profile.sampling or {}
    has_chat_block = isinstance(raw_sampling.get("chat"), dict)
    has_coding_block = isinstance(raw_sampling.get("coding"), dict)
    has_split = has_chat_block or has_coding_block

    # Resolve to a concrete dict that the rest of the function can
    # call .get() on. Done in two passes (mode → fallback) so the
    # type checker can narrow sd to `dict` after the assignment.
    sd: Dict[str, Any] = {}
    if has_split:
        mode_block = raw_sampling.get(mode)
        if isinstance(mode_block, dict):
            sd = mode_block
        else:
            # Mode not defined in this profile — fall back to the other
            # mode if present.
            other = "coding" if mode == "chat" else "chat"
            other_block = raw_sampling.get(other)
            if isinstance(other_block, dict):
                sd = other_block
    else:
        # Old flat format: every mode shares the same sampling block.
        sd = {k: v for k, v in raw_sampling.items() if not isinstance(v, dict)}

    sampling = {
        "temperature": float(sd.get("temperature", 0.7)),
        "top_k": int(sd.get("top_k", 40)),
        "top_p": float(sd.get("top_p", 0.9)),
        "min_p": float(sd.get("min_p", 0.05)),
        "repeat_penalty": float(sd.get("repeat_penalty", 1.05)),
        "presence_penalty": float(sd.get("presence_penalty", 0.0)),
    }

    # no_context_shift für bessere Performance bei grossen Kontexten aktivieren
    no_context_shift = (ctx >= 32768) or full_off

    # ---- KV split between VRAM and RAM for display fidelity -----------
    # Mirrors the budget logic in step (2): MoE/full_off keep KV on GPU
    # entirely; dense-hybrid splits proportionally to the layer split;
    # CPU-only keeps it all in RAM.
    if is_moe and has_gpu:
        kv_vram_gb = estimated_kv_gb
        kv_ram_gb = 0.0
    elif full_off:
        kv_vram_gb = estimated_kv_gb
        kv_ram_gb = 0.0
    elif ngl > 0 and n_layers > 0:
        gpu_layer_fraction = min(1.0, ngl / n_layers)
        kv_vram_gb = estimated_kv_gb * gpu_layer_fraction
        kv_ram_gb = estimated_kv_gb * (1.0 - gpu_layer_fraction)
    else:
        kv_vram_gb = 0.0
        kv_ram_gb = estimated_kv_gb

    # ---- Seed extra_cli_flags with whatever the profile declares ------
    # Until now, profile.extra_args (e.g. "--jinja" for the reasoning
    # families) were appended directly in build_cmd, never landing in
    # cfg.extra_cli_flags. Result: the Expert panel's "--jinja" checkbox
    # stayed unchecked even for models whose profile demands it. We
    # surface them here so the GUI reflects the truth, and build_cmd
    # de-dupes when it emits the final argv.
    seed_extras: List[str] = []
    if getattr(profile, "extra_args", None):
        seed_extras = [str(a) for a in profile.extra_args if a]

    return TunedConfig(
        ctx=ctx,
        ngl=ngl,
        threads=threads,
        batch_threads=batch_threads,
        batch=batch,
        ubatch=ubatch,
        cache_k=cache_k,
        cache_v=cache_v,
        flash_attn=True,
        sampling=sampling,
        mlock=mlock,
        no_mmap=no_mmap,
        numa=numa,
        tensor_split=tensor_split,
        main_gpu=main_gpu,
        n_cpu_moe=n_cpu_moe,
        is_moe=is_moe,
        expert_count=expert_count,
        estimated_model_vram_gb=model_vram,
        estimated_model_ram_gb=model_ram,
        estimated_kv_gb=estimated_kv_gb,
        full_offload=full_off,
        vision_vram_gb=vision_vram_gb,
        draft_vram_gb=draft_vram_gb,
        kv_vram_gb=kv_vram_gb,
        kv_ram_gb=kv_ram_gb,
        kv_quant_strategy=kv_quant_strategy,
        no_context_shift=no_context_shift,
        rope_scaling=rope_scaling_active,
        rope_scale_factor=float(profile_rope_factor) if rope_scaling_active else 1.0,
        performance_target=perf_target.name,
        n_parallel=n_parallel,
        warning=warning,
        extra_cli_flags=seed_extras,
        env_overrides=env_overrides,
    )


def _has_integrated_mtp(model: ModelEntry) -> bool:
    """Detect models that ship an integrated MTP drafter inside the GGUF.

    Delegates to ``ModelEntry.has_embedded_mtp`` in scanner.py, which is
    the canonical source of truth for this detection.  Detection is
    metadata-first (``<arch>.nextn_predict_layers > 0`` or tensor-info
    scan) with a filename pattern (``MTP`` token) as fallback.  See that
    property for the full rationale and examples.
    """
    return model.has_embedded_mtp


def build_command(
    model: ModelEntry,
    config: TunedConfig,
    profile: ModelProfile,
    draft_model: Optional[ModelEntry] = None,
    server_binary: str = "llama-server",
    host: str = "127.0.0.1",
    port: int = 1234,
    extra_args: Optional[List[str]] = None,
    use_thinking: bool = False,
    enable_speculative: bool = True,
    enable_ngram: bool = False,
) -> List[str]:
    """Build the llama-server command line for ``model`` and ``config``.

    Speculative decoding paths
    --------------------------
    * ``draft_model`` is set → sibling-drafter path. Adds ``-md`` plus
      ``--spec-draft-n-max`` (no ``--spec-type``; mainline auto-detects from ``-md``).
      Skipped when vision is loaded (three model graphs in VRAM simultaneously
      is too risky on 16-GB-class cards).
    * ``draft_model`` is None and the model has embedded MTP (detected
      via ``<arch>.nextn_predict_layers`` metadata or tensor-info scan,
      with filename token ``MTP`` as fallback) →
      ``--spec-type draft-mtp`` + ``--spec-draft-n-max`` only (the drafter
      rides inside the GGUF). Compatible with ``--mmproj`` / vision since
      mainline b9180 (PR #22673, merged 2026-05-16).
    * ``enable_speculative=False`` overrides both paths and emits no
      speculative flags at all — for the case where the user explicitly
      unchecked Draft on an MTP-named model.
    * ``enable_ngram=True`` adds ``ngram-mod`` self-speculative decoding
      (Path C). It is model-agnostic — no draft model required — so it can
      run standalone on any GGUF, or be combined with Path A/B. ``--spec-type``
      is a comma-separated list, and llama.cpp explicitly allows mixing a
      draft-model path with a draftless one, so e.g. ``draft-mtp,ngram-mod``
      is valid; the draftless path takes precedence when both fire.
    """
    cmd: List[str] = [
        server_binary,
        "-m",
        str(model.path),
        "-c",
        str(config.ctx),
        "-ngl",
        str(config.ngl),
        "-t",
        str(config.threads),
        "-tb",
        str(config.batch_threads),
        "-b",
        str(config.batch),
        "-ub",
        str(config.ubatch),
        "-ctk",
        config.cache_k,
        "-ctv",
        config.cache_v,
        "--host",
        host,
        "--port",
        str(port),
    ]

    # ---- AutoTuner authority over memory placement --------------------
    # Mainline llama.cpp gained an auto-fit pass (`--fit`, default 'on')
    # that silently adjusts UNSET arguments to fit device memory. The
    # AutoTuner deliberately computes ngl / ctx / n-cpu-moe / tensor-split,
    # so we turn auto-fit OFF: the values we computed and logged are the
    # ones that run. If they overcommit we want a visible, debuggable OOM
    # — not a silent ctx/ngl downscale that desyncs the running config
    # from what the launcher reported.
    # NOTE: `--fit` is a recent flag. It is present in current mainline
    # (b9297). If a server binary predates it, this will abort with
    # "unknown argument"; in that case drop the two tokens below.
    cmd += ["--fit", "off"]

    # ---- Prometheus metrics endpoint ----------------------------------
    # Exposes GET /metrics on the SAME host:port as the inference API
    # (no separate port). Scraped by the System Tricorder for live
    # tokens/s (llamacpp:predicted_tokens_seconds) and KV-cache fill
    # (llamacpp:kv_cache_usage_ratio). Negligible overhead.
    cmd.append("--metrics")

    # Speculative decoding — composable paths combined into one --spec-type:
    #   - sibling drafter passed in        → Path A (-md, auto-detected type)
    #   - integrated MTP                    → Path B (--spec-type draft-mtp)
    #   - n-gram (enable_ngram)             → Path C (--spec-type ngram-mod)
    #   - enable_speculative=False          → suppresses Path A and B; Path C
    #                                         (ngram) is independent and still
    #                                         honours its own enable_ngram flag.
    #
    # --spec-type accepts a comma-separated list and llama.cpp allows mixing a
    # draft-model path (draft-mtp) with a draftless one (ngram-mod), so we
    # assemble the active types and emit a single token (e.g. "draft-mtp,ngram-mod").
    #
    # Vision / draft compatibility:
    #   - External draft (Path A, -md) conflicts with --mmproj in llama.cpp:
    #     both try to load a second model and the server aborts. When vision
    #     is loaded, we skip Path A entirely.
    #   - Integrated MTP (Path B) embeds the drafter inside the main GGUF —
    #     no second model-load conflict. Vision and embedded MTP can coexist;
    #     Qwen3.6-MTP models in fact require the mmproj to work correctly.
    #   - n-gram (Path C) loads no model at all → always compatible.
    draft_val = getattr(profile, "draft_max", 0) or 2
    draft_p_min = getattr(profile, "draft_p_min", 0.75) or 0.75
    vision_loaded = model.mmproj is not None
    # Path A: sibling drafter — skip when vision is active.
    # Loading *two separate model files* (-m + -md) while also loading a
    # vision encoder (--mmproj) puts three large graphs in VRAM at once and
    # can exhaust memory. Integrated MTP (Path B) does not have this problem
    # because the draft head is already inside the main GGUF — there is no
    # second model file to load.
    use_external = enable_speculative and draft_model is not None and not vision_loaded
    # Path B: integrated MTP (draft-mtp) — compatible with vision (--mmproj)
    # since mainline b9180 (PR #22673, merged 2026-05-16). The MTP draft head
    # lives inside the same GGUF as the main model; llama.cpp loads it as part
    # of the same graph so there is no second-model-load conflict.
    use_integrated = (
        enable_speculative
        and _has_integrated_mtp(model)
        and draft_model is None
    )

    # Assemble the --spec-type list (embedded-draft + draftless types) and emit
    # it BEFORE the per-path parameter flags. -md (Path A) is auto-detected by
    # mainline, so it contributes no type token — only its parameter flags.
    spec_types: List[str] = []
    if use_integrated:
        spec_types.append("draft-mtp")
    if enable_ngram:
        spec_types.append("ngram-mod")
    if spec_types:
        cmd += ["--spec-type", ",".join(spec_types)]

    if use_external:
        # Path A — sibling drafter file.
        # -md MUST come before --spec-draft-n-max (llama-server parses
        # left-to-right). No --spec-type token: mainline auto-detects the
        # draft path from -md. If ngram is also enabled it was added to
        # --spec-type above and runs as the draftless path alongside -md.
        assert draft_model is not None  # guaranteed by use_external condition
        cmd += ["-md", str(draft_model.path)]
        cmd += ["--spec-draft-ngl", "99"]
        cmd += ["--spec-draft-n-max", str(draft_val)]
        cmd += ["--spec-draft-p-min", str(draft_p_min)]
    elif use_integrated:
        # Path B — integrated MTP drafter inside the main GGUF.
        # `--spec-draft-ngl 99` keeps the MTP head on GPU; without it the
        # drafter layers fall back to CPU and the speedup vanishes.
        # `--spec-draft-p-min` is emitted explicitly: with p_min=0.0 the MTP
        # hook fires on every decode step regardless of confidence, adding
        # constant D2H-transfer overhead that can make write-speed slower than
        # baseline on Vulkan/ROCm. The mainline default is 0.75.
        cmd += ["--spec-draft-n-max", str(draft_val)]
        cmd += ["--spec-draft-ngl", "99"]
        cmd += ["--spec-draft-p-min", str(draft_p_min)]

    if enable_ngram:
        # Path C — ngram-mod self-speculative decoding (no draft model).
        # Builds a rolling-hash lookup table from the live context (~16 MB,
        # constant memory). Parameters per llama.cpp docs/speculative.md:
        #   n-match = lookup length, n-min/n-max = draft length bounds.
        ngram_match = getattr(profile, "ngram_n_match", 24) or 24
        ngram_min = getattr(profile, "ngram_n_min", 48) or 48
        ngram_max = getattr(profile, "ngram_n_max", 64) or 64
        cmd += ["--spec-ngram-mod-n-match", str(ngram_match)]
        cmd += ["--spec-ngram-mod-n-min", str(ngram_min)]
        cmd += ["--spec-ngram-mod-n-max", str(ngram_max)]

    if config.flash_attn:
        cmd += ["-fa", "on"]
    if config.numa:
        cmd += ["--numa", config.numa]
    if config.mlock:
        cmd.append("--mlock")
    if config.no_mmap:
        cmd.append("--no-mmap")
    if config.no_context_shift:
        cmd.append("--no-context-shift")

    # RoPE-Scaling (YaRN) optional aktivieren für erweiterte Context-Längen
    # Bei Qwen3.5/3.6 möglich: native 262144 → bis 1048576 mit yarn scaling
    if config.rope_scaling and config.rope_scale_factor > 1.0:
        cmd += ["--rope-scaling", "yarn"]
        cmd += ["--rope-scale", str(int(config.rope_scale_factor))]

    if config.n_cpu_moe is not None and config.n_cpu_moe > 0:
        cmd += ["--n-cpu-moe", str(config.n_cpu_moe)]
    if config.tensor_split:
        cmd += ["--tensor-split", config.tensor_split]
    if config.main_gpu is not None:
        cmd += ["--main-gpu", str(config.main_gpu)]

    # Always pass --parallel explicitly.  llama-server's "auto" mode infers
    # n_parallel from the total KV budget ÷ per-slot KV cost.  On large
    # dual-GPU systems this can produce n_parallel=4 or more, multiplying
    # the actual KV allocation by that factor and filling all available RAM
    # (confirmed on R9700 32 GB + RX 9070 XT 16 GB with Qwen3.6-27B-Q8).
    # Passing the value explicitly prevents the server from picking a
    # different N than what compute_config budgeted for.
    cmd += ["--parallel", str(max(1, config.n_parallel))]

    s = config.sampling
    cmd += [
        "--temp",
        str(s["temperature"]),
        "--top-k",
        str(s["top_k"]),
        "--top-p",
        str(s["top_p"]),
        "--min-p",
        str(s["min_p"]),
        "--repeat-penalty",
        str(s["repeat_penalty"]),
    ]
    pp = s.get("presence_penalty", 0.0)
    if pp:
        cmd += ["--presence-penalty", str(pp)]

    if model.mmproj is not None:
        cmd += ["--mmproj", str(model.mmproj)]

    # Thinking/Reasoning-Modus (Gemma 4, DeepSeek, etc.)
    # Thinking wird über Prompt-Tags gesteuert (<|think|>), nicht über CLI-Argumente.
    # use_thinking ist ein internes Flag - extra_args werden immer angehängt:

    # ---- Extra-flag merge (de-duplicated, order-preserving) -----------
    # Two sources feed `cmd` here:
    #   1. profile.extra_args  — declared in the YAML (e.g. "--jinja")
    #   2. cfg.extra_cli_flags — what the GUI's Expert panel emitted
    # Until v3.1 we appended both blindly, which produced duplicate
    # flags whenever compute_config (correctly) seeded extra_cli_flags
    # from profile.extra_args so the Expert checkbox would reflect it.
    # Walk both lists with a "seen" set so a flag appears at most once,
    # and the relative ordering of first occurrences is preserved.
    #
    # The seen set is pre-populated with the entire cmd built so far —
    # this also catches the case where a profile lists "--no-context-shift"
    # in extra_args *and* the tuner separately decided to emit it (line
    # 1408): without prepopulating, the same flag would land twice.
    seen: set = set(cmd)

    def _append_unique(src: Optional[List[str]]) -> None:
        if not src:
            return
        for arg in src:
            if arg in seen:
                continue
            seen.add(arg)
            cmd.append(arg)

    _append_unique(getattr(profile, "extra_args", None))
    _append_unique(config.extra_cli_flags)
    _append_unique(extra_args)

    return cmd
