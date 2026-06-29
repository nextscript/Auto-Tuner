"""Performance target presets for the AutoTuner.

Each target is a coherent bundle of safety/headroom values that controls
how aggressively the AutoTuner uses GPU VRAM and system RAM. The goal
is to give users one knob ("safe / balanced / throughput") instead of
forcing them to tune four interacting numbers individually.

Four tiers
-----------
- **safe**: conservative reservations. Best for long-context sessions
  (>64k tokens) and users who prefer "it just works" over peak speed.
  Equivalent to the AutoTuner's pre-perf-target behaviour.
- **balanced** (default): moderate reservations. Mild VRAM
  optimisation that benefits everyone on most workloads.
- **throughput**: aggressive reservations. Optimised for short-context
  inference (~32k) where you want every available expert layer to sit
  in VRAM. Trades context headroom for tokens-per-second.
- **low_vram**: the LOW-VRAM / high-RAM escape hatch (e.g. 8 GB VRAM,
  64 GB RAM). Forces the KV cache into system RAM via ``--no-kv-offload``
  so context is drawn from abundant system memory instead of scarce
  VRAM. The only tier that lets a 20 GB MoE reach 90k+ context on a
  GPU that can barely hold the experts. Trades generation speed for
  context — attention compute follows the KV onto the CPU.

Resolution priority (highest wins)
----------------------------------
1. Explicit user choice (CLI flag, GUI dropdown).
2. YAML profile field ``performance_target:``. Lets a model author
   recommend a target appropriate for the architecture (e.g. a tiny
   3B dense model rarely benefits from "safe").
3. Module default (``balanced``).

Unknown values fall back silently to the default — we never want a
typo in YAML to crash the tuner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PerformanceTarget:
    """Coherent bundle of placement parameters."""

    name: str
    moe_vram_safety_gb: float
    moe_placement_ctx_target: int
    dense_vram_safety_gb: float
    ram_safety_gb: float
    description: str

    # Batch sizes for MoE + CPU hybrid placements. Pure GPU and CPU-only
    # placements keep the legacy 1024/1024 path (see tuner.py step 4b).
    # Rationale: the HuggingFace MoE-offload guide (Doctor-Shotgun, Feb
    # 2026) and the gfx1151 ROCm/Vulkan benchmark in llama.cpp issue
    # #21284 both demonstrate near-linear PP scaling up to -ub 2048 on
    # MoE-hybrid setups, because op-offload batches the CPU-resident
    # expert tensors as a single GPU operation. With -b 1024 the
    # round-trip overhead dominates. Larger batches cost ~0.5–1.5 GB
    # extra compute buffer, so we only step up when the perf target
    # already accepts a tighter VRAM safety band.
    moe_hybrid_batch: int = 2048
    moe_hybrid_ubatch: int = 2048

    # --parallel N passed to llama-server.
    #
    # llama-server's default is "auto": it infers N from the total KV
    # budget divided by the per-slot cost at the requested context.  On a
    # dual-GPU system (e.g. R9700 32 GB + RX 9070 XT 16 GB) with a large
    # dense model (e.g. Qwen3.6-27B-Q8, ~33 GB) the free VRAM after
    # weights is ~13 GB plus the 8 GB RAM supplement = ~21 GB.  With a
    # 262k context at Q8 KV (~0.060 MB/token) that budget fits ~354k
    # tokens, so "auto" happily sets n_parallel = 4 (354k / 262k ≈ 1.35
    # rounded up to 4 by llama-server's heuristic).  The result: 4 slots
    # × 15.4 GB KV/slot = 61.6 GB reserved, filling ALL 47 GB of RAM and
    # part of VRAM even though the model barely fits.
    #
    # Fix: always pass --parallel N explicitly so the server cannot
    # over-provision KV cache.  The value must also feed into the ctx
    # calculation (kv_budget_per_slot = kv_budget / n_parallel) so the
    # auto-tuned context is sized correctly for a SINGLE slot.
    #
    # Tier defaults:
    #   throughput  1  — single-user, every token as fast as possible
    #   balanced    2  — dual-user or light agentic workflows
    #   safe        4  — traditional multi-slot, long-context sessions
    #   low_vram    1  — single slot, every spare RAM byte goes to KV
    n_parallel: int = 1

    # -- LOW-VRAM escape hatch (``low_vram`` tier only).
    # When True, compute_config moves the *entire* KV cache into system
    # RAM by emitting ``--no-kv-offload`` (attention compute follows the
    # KV onto the CPU). This is the one lever that gives a low-VRAM /
    # high-RAM box a usable long context on a model whose KV would never
    # fit in the leftover VRAM. It trades generation speed for context:
    # attention reads/writes KV over the host-memory bus instead of VRAM.
    #
    # Only meaningful when there IS a GPU (CPU-only configs already keep
    # the KV in RAM). Existing tiers keep the default False, so high-end
    # systems on safe/balanced/throughput are completely unaffected.
    kv_to_ram: bool = False

    # Upper bound (GB) on RAM-resident KV when ``kv_to_ram`` is active.
    # Caps the context the tuner will promise from system RAM; the
    # model's native_ctx is a separate (usually lower) ceiling. Chosen
    # generously so abundant system RAM is effectively the only KV
    # limit, while still leaving headroom for the model weights and OS.
    kv_ram_cap_gb: float = 32.0

    def __str__(self) -> str:  # pragma: no cover — trivial
        return self.name


# ---------------------------------------------------------------------------
# Registry. Add new tiers here; nothing else needs to change.

PERFORMANCE_TARGETS: Dict[str, PerformanceTarget] = {
    "safe": PerformanceTarget(
        name="safe",
        moe_vram_safety_gb=0.30,
        moe_placement_ctx_target=131072,  # 128k — full long-context budget
        dense_vram_safety_gb=0.30,
        ram_safety_gb=1.50,
        # Safe keeps the legacy 1024/1024 to maximise headroom for the
        # 128k KV reservation. Users who care about throughput pick a
        # different tier anyway.
        moe_hybrid_batch=1024,
        moe_hybrid_ubatch=1024,
        n_parallel=4,
        description=(
            "Conservative. KV reserved for 128k context, generous "
            "safety bands. 4 parallel slots. Pick this for long-context "
            "sessions or multi-user setups."
        ),
    ),
    "balanced": PerformanceTarget(
        name="balanced",
        moe_vram_safety_gb=0.25,
        moe_placement_ctx_target=65536,  # 64k — typical working ceiling
        dense_vram_safety_gb=0.25,
        ram_safety_gb=1.25,
        # Balanced bumps to 2048/2048 — empirically the sweet spot for
        # 16 GB-class GPUs on Qwen3.6-A3B / Gemma-4-26B-A4B / GLM-4.7-MoE.
        moe_hybrid_batch=2048,
        moe_hybrid_ubatch=2048,
        n_parallel=2,
        description=(
            "Default. KV reserved for 64k context — enough headroom "
            "for most chats while letting more expert layers fit on GPU. "
            "2 parallel slots."
        ),
    ),
    "throughput": PerformanceTarget(
        name="throughput",
        moe_vram_safety_gb=0.15,
        moe_placement_ctx_target=32768,  # 32k — short reasoning / coding
        dense_vram_safety_gb=0.15,
        ram_safety_gb=1.00,
        # Throughput goes all-in: 4096/4096 maximises op-offload prompt
        # processing on MoE-hybrid setups. Costs ~1.5 GB extra VRAM for
        # the compute buffer but throughput already reserves the
        # smallest KV budget (32k) so there's room.
        moe_hybrid_batch=4096,
        moe_hybrid_ubatch=4096,
        n_parallel=1,
        description=(
            "Aggressive. KV reserved for only 32k — every spare GB "
            "of VRAM goes to expert layers. Single parallel slot "
            "(--parallel 1) for max tokens/s. Not recommended above ~32k "
            "context or multi-user setups."
        ),
    ),
    "low_vram": PerformanceTarget(
        name="low_vram",
        moe_vram_safety_gb=0.15,
        # KV lives in system RAM (kv_to_ram below), so we do NOT reserve
        # VRAM for it during MoE placement — only a token 4k reservation
        # so the placement heuristic's formula stays positive. Every
        # spare MB of VRAM goes to expert weights instead, maximising
        # how many experts run on the GPU before spilling to CPU.
        moe_placement_ctx_target=4096,
        dense_vram_safety_gb=0.15,
        ram_safety_gb=1.00,
        # 1024/1024 keeps the compute buffer small — on a low-VRAM card
        # every MB matters, and prompt-processing throughput is already
        # limited by CPU-resident experts + CPU attention.
        moe_hybrid_batch=1024,
        moe_hybrid_ubatch=1024,
        # Single slot: the whole point is maximising per-slot context,
        # so the KV budget is not divided across slots.
        n_parallel=1,
        # The defining lever: force the KV cache into system RAM.
        kv_to_ram=True,
        kv_ram_cap_gb=32.0,
        description=(
            "Max context for low-VRAM / high-RAM systems (e.g. 8 GB VRAM, "
            "64 GB RAM). KV cache moves to system RAM (--no-kv-offload), "
            "trading generation speed for far larger context — the only "
            "way to reach 90k+ on a 20 GB MoE that barely fits the GPU. "
            "Single parallel slot."
        ),
    ),
}

DEFAULT_TARGET_NAME = "balanced"


# ---------------------------------------------------------------------------
# Public API


def list_target_names() -> List[str]:
    """Return target names in display order.

    safe → balanced → throughput → low_vram. The first three are the
    performance tiers; ``low_vram`` is a special-purpose escape hatch
    kept last so it never changes the default selection (``balanced``).
    """
    return ["safe", "balanced", "throughput", "low_vram"]


def get_target(name: str) -> Optional[PerformanceTarget]:
    """Return target by name (case-insensitive); ``None`` if unknown."""
    if not name:
        return None
    return PERFORMANCE_TARGETS.get(name.lower().strip())


def resolve_performance_target(
    cli_choice: Optional[str] = None,
    profile_choice: Optional[str] = None,
    default: str = DEFAULT_TARGET_NAME,
) -> PerformanceTarget:
    """Resolve which target to use.

    Tries ``cli_choice`` first, then ``profile_choice``, then ``default``.
    Unknown / empty values are silently skipped so a single bad source
    never breaks the chain. Always returns a valid ``PerformanceTarget``.
    """
    for choice in (cli_choice, profile_choice, default):
        target = get_target(choice) if choice else None
        if target is not None:
            return target
    # Defensive fallback. ``DEFAULT_TARGET_NAME`` must exist in the registry.
    return PERFORMANCE_TARGETS[DEFAULT_TARGET_NAME]


def describe_targets() -> str:
    """Multiline summary for ``--help`` text and GUI tooltips."""
    lines = []
    for name in list_target_names():
        t = PERFORMANCE_TARGETS[name]
        lines.append(f"  {name:<11} {t.description}")
    return "\n".join(lines)
