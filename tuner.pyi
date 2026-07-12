"""Type stub for tuner.py.

Placing a .pyi file next to tuner.py makes both mypy and pyright use
these declarations unconditionally.  This eliminates the 'Module "tuner"
has no attribute X' errors that appear when a PyPI package named "tuner"
is installed in the venv and shadows the local module.

Keep this file in sync with tuner.py whenever new public symbols are added.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hardware import GPUInfo, SystemInfo
from performance_target import PerformanceTarget
from scanner import ModelEntry
from settings_loader import ModelProfile

# ---------------------------------------------------------------------------
# Module-level constants

DEFAULT_VRAM_SAFETY_GB: float
DEFAULT_RAM_SAFETY_GB: float
MOE_VRAM_SAFETY_GB: float
MOE_PLACEMENT_CTX_TARGET: int
MOE_KV_RESERVE_FRAC: float

_MOE_FILENAME_RE: re.Pattern[str]

# ---------------------------------------------------------------------------
# Public helpers


def extract_params_billion(name: str) -> float: ...
def kv_per_token_mb_f16(params_billion: float) -> float: ...
def kv_per_token_mb_from_metadata(md: Dict[str, Any]) -> float: ...
def kv_quant_factor(quant: str) -> float: ...
def _moe_expert_count(model: ModelEntry) -> int: ...
def prepare_command_for_binary(cmd: List[str]) -> Tuple[List[str], List[str]]: ...
def gemma_draft_needs_ik_fork(
    model_name: str, use_draft: bool, resolved_binary: str
) -> bool: ...
def match_gpu_by_token(
    token: Optional[str], gpus: List[GPUInfo]
) -> Optional[GPUInfo]: ...
def veto_unsafe_mlock(
    config: TunedConfig, system: SystemInfo, force_mlock: bool = ...
) -> bool: ...

# ---------------------------------------------------------------------------
# TunedConfig


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

    mlock: bool = ...
    no_mmap: bool = ...
    numa: Optional[str] = ...
    tensor_split: Optional[str] = ...
    main_gpu: Optional[int] = ...

    n_cpu_moe: Optional[int] = ...
    is_moe: bool = ...
    expert_count: int = ...

    estimated_model_vram_gb: float = ...
    estimated_model_ram_gb: float = ...
    estimated_kv_gb: float = ...
    full_offload: bool = ...

    vision_vram_gb: float = ...
    draft_vram_gb: float = ...
    kv_vram_gb: float = ...
    kv_ram_gb: float = ...
    kv_quant_strategy: str = ...

    no_context_shift: bool = ...
    # True when --no-kv-offload is emitted (low_vram perf-target lever):
    # the KV cache lives in system RAM, attention compute follows to CPU.
    no_kv_offload: bool = ...
    rope_scaling: bool = ...
    rope_scale_factor: float = ...
    # Expert override for --spec-draft-n-max (0 = profile draft_max).
    draft_n_max: int = ...

    extra_cli_flags: List[str] = field(default_factory=list)
    env_overrides: Dict[str, str] = field(default_factory=dict)
    performance_target: str = ...
    # Number of parallel inference slots (--parallel N).
    # Sourced from PerformanceTarget.n_parallel; always emitted explicitly
    # so llama-server cannot over-provision KV cache via its "auto" mode.
    n_parallel: int = ...
    # True when n_parallel was pinned by the Expert panel override.
    n_parallel_forced: bool = ...
    metrics_enabled: bool = ...
    slots_api_enabled: bool = ...
    warning: Optional[str] = ...

# ---------------------------------------------------------------------------
# Main API


def compute_config(
    model: ModelEntry,
    system: SystemInfo,
    profile: ModelProfile,
    draft_model: Optional[ModelEntry] = ...,
    user_ctx: Optional[int] = ...,
    ram_safety_gb: Optional[float] = ...,
    vram_safety_gb: Optional[float] = ...,
    force_mlock: bool = ...,
    perf_target: Optional[PerformanceTarget] = ...,
    mode: str = ...,
    *,
    turbo_kv: bool = ...,
    force_cache_k: Optional[str] = ...,
    force_cache_v: Optional[str] = ...,
    force_ngl: Optional[int] = ...,
    force_n_cpu_moe: Optional[int] = ...,
    force_n_parallel: Optional[int] = ...,
    force_rope_scale: Optional[bool] = ...,
    gpu_priorities: Optional[Dict[str, int]] = ...,
    force_gpu: Optional[str] = ...,
) -> TunedConfig: ...


def build_command(
    model: ModelEntry,
    config: TunedConfig,
    profile: ModelProfile,
    draft_model: Optional[ModelEntry] = ...,
    server_binary: str = ...,
    host: str = ...,
    port: int = ...,
    extra_args: Optional[List[str]] = ...,
    use_thinking: bool = ...,
    enable_speculative: bool = ...,
    enable_ngram: bool = ...,
    enable_prompt_cache: bool = ...,
    prompt_cache_ram_mib: int = ...,
    enable_metrics: Optional[bool] = ...,
    enable_slots_api: Optional[bool] = ...,
) -> List[str]: ...


def build_diffusion_server_command(
    model: ModelEntry,
    config: TunedConfig,
    profile: ModelProfile,
    server_binary: str = ...,
    host: str = ...,
    port: int = ...,
    alias: Optional[str] = ...,
    extra_args: Optional[List[str]] = ...,
    enable_metrics: Optional[bool] = ...,
    enable_slots_api: Optional[bool] = ...,
) -> List[str]: ...
def build_diffusion_command(
    model: ModelEntry,
    config: TunedConfig,
    profile: ModelProfile,
    diffusion_binary: str = ...,
    prompt: Optional[str] = ...,
    extra_args: Optional[List[str]] = ...,
) -> List[str]: ...
