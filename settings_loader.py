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
    recommended_kv_quant: str = "q5_0"
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
    # ngram-mod -> mid-gen crashes, llama.cpp #23154, still open at b9334). The
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
    # path). Mainline b9672 flags map as:
    #   steps         -> --diffusion-steps        (default 256)
    #   algorithm     -> --diffusion-algorithm    (0..4; 4=confidence)
    #   eps           -> --diffusion-eps          (timestep schedule)
    #   block_length  -> --diffusion-block-length (block schedule; pick ONE
    #                    of eps / block_length, not both)
    #   visual        -> --diffusion-visual       (live token visualisation)
    #   n_predict     -> -n / --predict           (max tokens to generate)
    # Fork-only extras (NOT in mainline b9672 — DiffusionGemma build) are
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

        sampling = data.get("sampling") or {}
        if not isinstance(sampling, dict):
            sampling = {}
        extra = data.get("extra_args") or []
        if not isinstance(extra, list):
            extra = []

        # RoPE-Scaling Konfiguration (optional)
        rope_scale_cfg = data.get("rope_scale") or {}
        if not isinstance(rope_scale_cfg, dict):
            rope_scale_cfg = {}
        rope_scale_enabled = bool(rope_scale_cfg.get("enabled", False))
        rope_scale_max_ctx = int(rope_scale_cfg.get("max_context", 0))
        rope_scale_factor = float(rope_scale_cfg.get("scale_factor", 4.0))

        # Performance target preset (optional). Validate softly: any string
        # other than the three known names is treated as "use global default".
        perf_target_raw = str(data.get("performance_target", "") or "").lower().strip()
        if perf_target_raw not in ("safe", "balanced", "throughput", ""):
            print(
                f"[AutoTuner] {yml.name}: unknown performance_target "
                f"'{perf_target_raw}', ignoring (using global default)."
            )
            perf_target_raw = ""

        # Runner override (optional). Soft-validate: only the two known
        # values are honoured; anything else falls back to the server path.
        runner_raw = str(data.get("runner", "") or "").lower().strip()
        if runner_raw not in ("", "llama-server", "llama-diffusion-cli"):
            print(
                f"[AutoTuner] {yml.name}: unknown runner '{runner_raw}', "
                f"ignoring (using llama-server)."
            )
            runner_raw = ""

        diffusion_cfg = data.get("diffusion") or {}
        if not isinstance(diffusion_cfg, dict):
            diffusion_cfg = {}

        profiles.append(
            ModelProfile(
                display_name=str(data.get("display_name", yml.stem)),
                patterns=[str(p).lower() for p in (data.get("patterns") or [])],
                arch_fallback=[
                    str(a).lower() for a in (data.get("arch_fallback") or [])
                ],
                max_context=int(data.get("max_context", 8192)),
                sampling=sampling,
                recommended_kv_quant=str(data.get("recommended_kv_quant", "q5_0")),
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
        )
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

    if best is not None:
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
