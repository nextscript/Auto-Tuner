"""Load YAML profiles from the settings/ folder and match them
against model filenames."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError as e:
    raise SystemExit(
        "PyYAML is required. Install with:  pip install -r requirements.txt"
    ) from e


@dataclass
class ModelProfile:
    display_name: str
    patterns: List[str] = field(default_factory=list)
    # Architectures this profile also covers as a *fallback* when no
    # filename pattern matched. These are matched against the GGUF
    # ``general.architecture`` string (exact, case-insensitive). Filename
    # patterns always take precedence; arch_fallback only resolves the
    # "no pattern matched" case so that e.g. a community gpt-oss merge with
    # an unrecognised filename still gets the gpt-oss profile (and its
    # mandatory --jinja) instead of falling through to _default. See
    # match_profile() for the precedence rules.
    arch_fallback: List[str] = field(default_factory=list)
    max_context: int = 8192
    sampling: Dict[str, Any] = field(default_factory=dict)
    recommended_kv_quant: str = "q8_0"
    # Optional profile override for Flash Attention. None keeps AutoTuner's
    # normal default; OCR/reference-sensitive models can require it off.
    flash_attn: Optional[bool] = None
    # Minimum numeric llama.cpp build needed by this model/runtime contract.
    # Launchers surface this before model load; 0 means no build gate.
    min_llama_build: int = 0
    # ASCII capability markers that must exist in the selected llama runtime
    # executable or its sibling shared library. This safely gates community
    # architectures that have no meaningful numeric mainline build minimum.
    required_runtime_markers: List[str] = field(default_factory=list)
    # Recognize a model without claiming an unimplemented inference runtime.
    # Cleared only after its loader, memory plan and template are validated.
    runtime_block_reason: str = ""
    # When GGUF metadata names one of this profile's ``arch_fallback``
    # architectures, this profile wins over a filename match elsewhere
    # (DeepSeek-V4.1 files named "DeepSeek-V4-..." carry deepseek41).
    arch_authoritative: bool = False
    # False: a paired external drafter stays selectable but is not chosen
    # by default (measured slower than no draft for this family).
    auto_draft: bool = True
    extra_args: List[str] = field(default_factory=list)
    notes: str = ""
    source_file: Optional[str] = None  # which YAML this came from
    # Optional: override the llama-server binary for this model family.
    # Used e.g. by Ternary-Bonsai (BitNet) to invoke a 1bllama build.
    server_binary: Optional[str] = None
    draft_max: int = 16
    draft_p_min: float = 0.75
    # Draftless self-speculative decoding method. As of llama.cpp b9334 the
    # --spec-type vocabulary grew into a family:
    #   ngram-mod      — rolling-hash pool (the original; default here)
    #   ngram-map-k    — key-only n-gram map
    #   ngram-map-k4v  — key+value n-gram map; the method ggerganov's MTP
    #                    clean-up (PR #23269) wired into --spec-default and
    #                    explicitly designed to COEXIST with draft-mtp
    #   ngram-simple / ngram-cache — older draftless variants
    # Default stays "ngram-mod" so existing profiles behave exactly as before.
    # IMPORTANT: only "ngram-mod" conflicts with integrated MTP (draft-mtp,
    # ngram-mod -> mid-gen crashes, llama.cpp #23154, stale-closed without
    # a verified fix as of the b11105 audit). The
    # ngram-map-* methods are allowed alongside draft-mtp by the tuner. So to
    # actually combine MTP + ngram on an MTP model, set this to ngram-map-k4v.
    ngram_method: str = "ngram-mod"
    # ngram-mod tuning — model-agnostic, needs no draft model. Defaults mirror
    # llama.cpp docs/speculative.md (n-match 24, n-min 48, n-max 64). Tunable
    # per profile for repetitive code/text or reasoning workloads; MoE models
    # benefit from longer drafts.
    ngram_n_match: int = 24
    ngram_n_min: int = 48
    ngram_n_max: int = 64
    # ngram-map-k4v tuning (only used when ngram_method == "ngram-map-k4v").
    # Names/defaults follow PR #23269's example: --spec-ngram-map-k4v-size-n 16,
    # -size-m 24, -min-hits 1.
    ngram_k4v_size_n: int = 16
    ngram_k4v_size_m: int = 24
    ngram_k4v_min_hits: int = 1
    # RoPE-Scaling (YaRN): aktiviert wenn ctx > native_ctx und genügend Speicher
    rope_scale_enabled: bool = False  # YAML-Konfiguration: rope_scale: true
    rope_scale_max_ctx: int = 0  # maximales Context mit RoPE-Scaling (0=auto 1M)
    rope_scale_factor: float = 4.0  # Standard Scaling-Faktor für Qwen3.5/3.6

    # Performance target preset suggested by the profile author.
    # Empty string = no profile-level recommendation, use the global
    # default ("balanced"). Recognised values: "safe" / "balanced" /
    # "throughput". Unknown values are ignored.
    performance_target: str = ""

    # ---- Diffusion-LLM support ----------------------------------------
    # Runner selection. "" / "llama-server" → the normal OpenAI-compatible
    # server path (default, unchanged for every existing profile).
    # "llama-diffusion-cli" → single-shot diffusion generation: the tuner
    # builds a diffusion command instead of a server command and the
    # launcher does NOT expect a /health endpoint or chat API. Normally you
    # don't need to set this by hand — the scanner detects diffusion
    # architectures (dream / llada / rnd1 / diffusion-gemma) and the
    # launcher switches automatically; the field is here so a profile can
    # force it (e.g. for an arch mainline doesn't recognise yet).
    runner: str = ""
    # Diffusion generation parameters (only consulted on the diffusion
    # path). Mainline b9700 flags map as:
    #   steps         -> --diffusion-steps        (default 256)
    #   algorithm     -> --diffusion-algorithm    (0..4; 4=confidence)
    #   eps           -> --diffusion-eps          (timestep schedule)
    #   block_length  -> --diffusion-block-length (block schedule; pick ONE
    #                    of eps / block_length, not both)
    #   visual        -> --diffusion-visual       (live token visualisation)
    #   n_predict     -> -n / --predict           (max tokens to generate)
    # Fork-only extras (NOT in mainline b9700 — DiffusionGemma build) are
    # passed through verbatim from the ``fork_args`` list, e.g.
    #   fork_args: ["--diffusion-eb", "--diffusion-kv-cache"]
    # so the same profile schema works for both mainline and your fork.
    diffusion: Dict[str, Any] = field(default_factory=dict)


# The draftless --spec-type methods llama.cpp accepts as of b9334. Used to
# validate the profile-level ngram_method so a typo in YAML fails loudly at
# load time instead of producing an "unknown speculative type" abort when
# llama-server starts.
_VALID_NGRAM_METHODS = (
    "ngram-mod",
    "ngram-map-k",
    "ngram-map-k4v",
    "ngram-simple",
    "ngram-cache",
)


def _validate_ngram_method(value: str, yml_name: str) -> str:
    v = value.lower().strip()
    if v not in _VALID_NGRAM_METHODS:
        print(
            f"[AutoTuner] {yml_name}: unknown ngram_method '{value}', "
            f"falling back to 'ngram-mod'. Valid: {', '.join(_VALID_NGRAM_METHODS)}"
        )
        return "ngram-mod"
    return v


def load_profiles(settings_dir: Path) -> List[ModelProfile]:
    """Load every *.yaml / *.yml file in settings_dir."""
    profiles: List[ModelProfile] = []
    if not settings_dir.exists():
        return profiles

    files = sorted(list(settings_dir.glob("*.yaml")) + list(settings_dir.glob("*.yml")))
    for yml in files:
        try:
            with yml.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            print(f"[AutoTuner] Warning: failed to load {yml.name}: {e}")
            continue
        if not isinstance(data, dict):
            print(f"[AutoTuner] Warning: {yml.name} is not a mapping; skipped.")
            continue
        for list_key in ("patterns", "arch_fallback"):
            # ``patterns: qwen`` would otherwise iterate character-wise and
            # match nearly every model.
            if isinstance(data.get(list_key), str):
                data[list_key] = [data[list_key]]

        sampling = data.get("sampling") or {}
        if not isinstance(sampling, dict):
            sampling = {}
        extra = data.get("extra_args") or []
        if not isinstance(extra, list):
            extra = []
        runtime_markers = data.get("required_runtime_markers") or []
        if not isinstance(runtime_markers, list):
            runtime_markers = []

        flash_raw = data.get("flash_attn", None)
        flash_attn: Optional[bool]
        if isinstance(flash_raw, bool):
            flash_attn = flash_raw
        elif isinstance(flash_raw, int) and flash_raw in {0, 1}:
            flash_attn = bool(flash_raw)
        elif isinstance(flash_raw, str) and flash_raw.strip().lower() in {
            "on",
            "true",
            "yes",
            "1",
        }:
            flash_attn = True
        elif isinstance(flash_raw, str) and flash_raw.strip().lower() in {
            "off",
            "false",
            "no",
            "0",
        }:
            flash_attn = False
        else:
            flash_attn = None
        try:
            min_llama_build = max(0, int(data.get("min_llama_build", 0) or 0))
        except (TypeError, ValueError):
            print(
                f"[AutoTuner] {yml.name}: invalid min_llama_build; ignoring build gate."
            )
            min_llama_build = 0

        # RoPE-Scaling Konfiguration (optional)
        rope_scale_cfg = data.get("rope_scale") or {}
        if not isinstance(rope_scale_cfg, dict):
            rope_scale_cfg = {}
        rope_scale_enabled = bool(rope_scale_cfg.get("enabled", False))
        rope_scale_max_ctx = int(rope_scale_cfg.get("max_context", 0))
        rope_scale_factor = float(rope_scale_cfg.get("scale_factor", 4.0))

        # Performance target preset (optional). Validate softly: any string
        # outside the four GUI targets is treated as "use global default".
        perf_target_raw = str(data.get("performance_target", "") or "").lower().strip()
        if perf_target_raw not in ("safe", "balanced", "throughput", "low_vram", ""):
            print(
                f"[AutoTuner] {yml.name}: unknown performance_target "
                f"'{perf_target_raw}', ignoring (using global default)."
            )
            perf_target_raw = ""

        # Runner override (optional). Soft-validate: only the known
        # values are honoured; anything else falls back to the server path.
        #   ""/"llama-server"               → normal OpenAI HTTP server
        #   "llama-diffusion-cli"           → single-shot CLI (dream/llada/rnd1)
        #   "llama-diffusion-gemma-server"  → PR #24427 DiffusionGemma HTTP server
        runner_raw = str(data.get("runner", "") or "").lower().strip()
        if runner_raw not in (
            "",
            "llama-server",
            "llama-diffusion-cli",
            "llama-diffusion-gemma-server",
        ):
            print(
                f"[AutoTuner] {yml.name}: unknown runner '{runner_raw}', "
                f"ignoring (using llama-server)."
            )
            runner_raw = ""

        diffusion_cfg = data.get("diffusion") or {}
        if not isinstance(diffusion_cfg, dict):
            diffusion_cfg = {}

        try:
            profile = ModelProfile(
                display_name=str(data.get("display_name", yml.stem)),
                patterns=[str(p).lower() for p in (data.get("patterns") or [])],
                arch_fallback=[
                    str(a).lower() for a in (data.get("arch_fallback") or [])
                ],
                max_context=int(data.get("max_context", 8192)),
                sampling=sampling,
                recommended_kv_quant=str(data.get("recommended_kv_quant", "q8_0")),
                flash_attn=flash_attn,
                min_llama_build=min_llama_build,
                required_runtime_markers=[
                    str(marker).strip().lower()
                    for marker in runtime_markers
                    if str(marker).strip()
                ],
                runtime_block_reason=str(
                    data.get("runtime_block_reason", "") or ""
                ).strip(),
                arch_authoritative=bool(data.get("arch_authoritative", False)),
                auto_draft=bool(data.get("auto_draft", True)),
                extra_args=[str(x) for x in extra],
                notes=str(data.get("notes", "") or ""),
                source_file=yml.name,
                server_binary=(
                    str(data["server_binary"]) if data.get("server_binary") else None
                ),
                draft_max=int(data.get("draft_max", 16)),
                draft_p_min=float(data.get("draft_p_min", 0.75)),
                ngram_method=_validate_ngram_method(
                    str(data.get("ngram_method", "ngram-mod") or "ngram-mod"),
                    yml.name,
                ),
                ngram_n_match=int(data.get("ngram_n_match", 24)),
                ngram_n_min=int(data.get("ngram_n_min", 48)),
                ngram_n_max=int(data.get("ngram_n_max", 64)),
                ngram_k4v_size_n=int(data.get("ngram_k4v_size_n", 16)),
                ngram_k4v_size_m=int(data.get("ngram_k4v_size_m", 24)),
                ngram_k4v_min_hits=int(data.get("ngram_k4v_min_hits", 1)),
                rope_scale_enabled=rope_scale_enabled,
                rope_scale_max_ctx=rope_scale_max_ctx
                if rope_scale_max_ctx > 0
                else 1048576,
                rope_scale_factor=rope_scale_factor,
                performance_target=perf_target_raw,
                runner=runner_raw,
                diffusion=diffusion_cfg,
            )
        except (TypeError, ValueError) as e:
            # One hand-edited field (e.g. an empty ``max_context:``) must
            # not take every other profile down with it.
            print(f"[AutoTuner] Warning: failed to load {yml.name}: {e}")
            continue
        profiles.append(profile)
    return profiles


def match_profile(
    model_name: str,
    profiles: List[ModelProfile],
    arch: Optional[str] = None,
) -> ModelProfile:
    """Pick the best-matching profile for the given model filename.

    Precedence:
      1. Filename pattern — case-insensitive substring match on each
         ``patterns:`` entry; the longest matching pattern wins. This is
         the primary mechanism and is unchanged.
      2. Architecture fallback — only when NO filename pattern matched and
         an ``arch`` is supplied: a profile whose ``arch_fallback:`` list
         contains that exact architecture string is used. This catches
         community merges / re-quants with unrecognised filenames whose
         ``general.architecture`` is nonetheless a known family (e.g. a
         gpt-oss MXFP4 merge that must still get --jinja). If several
         profiles claim the same arch the first one loaded wins
         (load order is alphabetical by filename).
      3. The generic ``_default.yaml`` (empty ``patterns:``), else a
         built-in default.

    ``arch`` is optional so existing callers that pass only
    ``(model_name, profiles)`` keep their exact previous behaviour.
    """
    name_lower = model_name.lower()
    arch_lower = (arch or "").lower().strip()
    best: Optional[ModelProfile] = None
    best_len = -1
    fallback: Optional[ModelProfile] = None

    for p in profiles:
        if not p.patterns:
            # A profile with no filename patterns is only the generic
            # fallback when it also declares no arch_fallback. A pattern-less
            # profile that exists purely to claim an arch (unusual, but
            # allowed) must not hijack the _default slot.
            if fallback is None and not p.arch_fallback:
                fallback = p
            continue
        for pat in p.patterns:
            if pat and pat in name_lower and len(pat) > best_len:
                best = p
                best_len = len(pat)
                # Don't break — a later pattern in the same file might be longer

    # Known-unsupported architecture metadata must not be hidden by a generic
    # filename match. Conversely, a blocked filename must not fall back to an
    # older architecture (early V4.1 converters incorrectly wrote deepseek4).
    if arch_lower:
        for p in profiles:
            if p.runtime_block_reason and arch_lower in p.arch_fallback:
                return p

    if best is not None:
        if best.runtime_block_reason:
            return best
        # A fork-specific filename can also name a newly supported upstream
        # conversion (HY4: hyv4 versus hy_v4). When metadata explicitly names
        # another claimed architecture, do not require the wrong fork marker.
        if (
            best.required_runtime_markers
            and arch_lower
            and arch_lower not in best.arch_fallback
        ):
            for p in profiles:
                if arch_lower in p.arch_fallback:
                    return p
        # A profile that declares its architecture authoritative wins over a
        # filename match: "DeepSeek-V4-..." carrying deepseek41 (V4.1 Flash,
        # b11870+) must not inherit V4's floor, sampling and draft settings.
        if arch_lower and arch_lower not in best.arch_fallback:
            for p in profiles:
                if p.arch_authoritative and arch_lower in p.arch_fallback:
                    return p
        return best

    # No filename pattern matched — try the architecture fallback.
    if arch_lower:
        for p in profiles:
            if arch_lower in p.arch_fallback:
                return p

    return fallback or ModelProfile(
        display_name="builtin-default",
        max_context=8192,
        sampling={
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "min_p": 0.05,
            "repeat_penalty": 1.05,
        },
    )
