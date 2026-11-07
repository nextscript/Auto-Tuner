"""Model metadata diagnostics for the AutoTuner.

Wraps and replaces the standalone ``diag_kv.py`` / ``diag_kv_v2.py``
scripts with a callable API. Two entry points:

* :func:`audit_model_metadata` — fast, returns a list of *actionable*
  warnings only (kv-heads missing, MoE detected by filename only,
  block_count zero). Empty list means "the tuner has a complete picture
  of this model" — silence is good news. The audit is intentionally
  conservative: heuristic fallbacks that work fine (e.g. deriving
  ``key_length`` from ``embd / n_heads``) are NOT warnings.

* :func:`format_diagnostic_report` — verbose, multi-line block with
  every KV-relevant input plus capacity estimates at 32k/64k/128k.
  Equivalent to ``diag_kv.py``'s output for a single model. Use from
  the CLI ``--diagnose`` path or the GUI Diagnose button.

The standalone ``diag_kv.py`` / ``diag_kv_v2.py`` scripts are kept for
power-user forensics (full key-name dump for the v2 case); they share
no code with this module but produce overlapping information.

Design notes:
* No side effects: ``audit_model_metadata`` is pure. The caller decides
  whether to display the warnings, log them, or ignore them.
* Stable warning IDs: each warning is a (id, message) tuple so callers
  can deduplicate, filter by severity, or look up explanations later.
  Ids are prefixed ``KV-`` (KV-related), ``MOE-`` (MoE detection), or
  ``GGUF-`` (general GGUF integrity).
"""

from __future__ import annotations

import os
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from autotuner_version import VERSION
from scanner import (
    ModelEntry,
    metadata_cache_stats,
    metadata_layer_count,
    metadata_attention_layer_count,
    metadata_is_hybrid_architecture,
    _RECURRENT_ARCHS,
)
from tuner import (
    _moe_expert_count,
    _KNOWN_MOE_ARCHS,
    _MOE_ALT_KEY_SUFFIXES,
    _MOE_FILENAME_RE,
    _KV_HEAD_COUNT_SUFFIXES,
    _metadata_arch_int,
    _resolve_kv_per_token_parts_mb,
    extract_params_billion,
    kv_per_token_mb_from_metadata,
    kv_per_token_mb_f16,
    kv_quant_factor,
    recurrent_state_gb_from_metadata,
)


# ---------------------------------------------------------------------------
# Warning dataclass


@dataclass(frozen=True)
class DiagnosticWarning:
    """A single actionable hint about a model's metadata quality.

    ``id`` is stable across releases; UI code can map it to richer help
    text later. ``message`` is the short human-readable form for logs.
    """

    id: str
    message: str

    def __str__(self) -> str:  # pragma: no cover — trivial
        return f"[{self.id}] {self.message}"


# ---------------------------------------------------------------------------
# Internal: small typed metadata accessors


def _md_int(md: dict, key: str) -> int:
    v = md.get(key, 0)
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Public API


def audit_model_metadata(model: ModelEntry) -> List[DiagnosticWarning]:
    """Return *actionable* warnings about this model's metadata.

    Empty list = silence = the tuner has a complete picture. Only flags
    conditions where the tuner is forced to guess and the guess could
    measurably hurt placement or KV sizing.

    Warning catalogue (stable IDs):

    * ``GGUF-NO-METADATA`` — metadata dict is empty or unreadable. The
      tuner falls back to params-based heuristics for everything.
    * ``KV-HEAD-COUNT-MISSING`` — ``head_count_kv`` is zero or absent.
      KV per-token estimate falls back to ``head_count`` (no GQA
      reduction), which can over-estimate KV by 4-8x on modern models.
    * ``KV-LAYERS-MISSING`` — ``block_count`` is zero. Per-layer split
      and KV sizing both rely on this number.
    * ``MOE-FILENAME-FALLBACK`` — model name says "A{N}B" (MoE marker)
      but the GGUF metadata declares no expert count. The tuner routes
      it through the MoE placement path but cannot reason about expert
      counts precisely.
    """
    warnings: List[DiagnosticWarning] = []
    md = model.metadata or {}

    if not md:
        warnings.append(
            DiagnosticWarning(
                "GGUF-NO-METADATA",
                f"{model.name}: GGUF metadata empty or unreadable. "
                "Tuner falls back to filename + params heuristics; "
                "KV sizing and MoE placement will be approximate.",
            )
        )
        return warnings  # No metadata → no further checks meaningful.

    arch = md.get("general.architecture", "")
    block_count = metadata_layer_count(md)
    arch_l = str(arch or "").lower()

    # qwen4exp is a normal mainline architecture from b10660. Runtime
    # compatibility is enforced by the profile's numeric build gate, where the
    # selected binary is available; a metadata-only diagnostic must not keep
    # emitting the obsolete pre-merge issue #27741 warning.

    # --- KV-HEAD-COUNT-MISSING --------------------------------------------
    # head_count_kv = 0 is the single most-impactful metadata bug. Without
    # it the tuner can't account for GQA (8x-32x KV reduction on modern
    # models) and will pessimistically reserve too much KV VRAM, leading
    # directly to fewer expert layers on GPU and lower t/s.
    n_kv = _metadata_arch_int(md, str(arch), *_KV_HEAD_COUNT_SUFFIXES) if arch else 0
    if n_kv <= 0 and str(arch).lower() not in _RECURRENT_ARCHS:
        # Some quantizers write under non-canonical key names. Scan the
        # rest of the metadata for any *.head_count_kv variant before
        # raising the warning.
        alternates = [
            k
            for k in md.keys()
            if k.lower().endswith(".head_count_kv")
            or k.lower().endswith(".kv_head_count")
            or k.lower().endswith(".num_key_value_heads")
        ]
        if not alternates:
            warnings.append(
                DiagnosticWarning(
                    "KV-HEAD-COUNT-MISSING",
                    f"{model.name}: head_count_kv not found in metadata. "
                    "KV size estimate will use head_count instead "
                    "(GQA reduction lost). Run diag_kv_v2.py to find the "
                    "real key name if one exists.",
                )
            )

    # --- KV-LAYERS-MISSING ------------------------------------------------
    if block_count <= 0:
        warnings.append(
            DiagnosticWarning(
                "KV-LAYERS-MISSING",
                f"{model.name}: block_count is zero. Layer placement "
                "falls back to a 50-layer estimate which may over- or "
                "under-commit VRAM significantly.",
            )
        )

    # --- MOE-ARCH/FILENAME-FALLBACK ---------------------------------------
    expert_count = _moe_expert_count(model)
    has_expert_key = any(str(key).lower().endswith(_MOE_ALT_KEY_SUFFIXES) for key in md)
    if arch_l in _KNOWN_MOE_ARCHS and expert_count == 2 and not has_expert_key:
        warnings.append(
            DiagnosticWarning(
                "MOE-ARCH-FALLBACK",
                f"{model.name}: architecture '{arch_l}' is MoE but expert_count "
                "is missing. Expert-aware placement is enabled with an unknown "
                "count; re-quantize from authoritative metadata if possible.",
            )
        )

    is_moe_by_filename = bool(_MOE_FILENAME_RE.search(model.name))
    # _moe_expert_count returns the sentinel 2 when only the filename
    # tells us it's MoE. If we get 2 *and* the filename matches AND no
    # canonical expert_count key is present, flag it.
    if is_moe_by_filename and expert_count == 2 and arch_l not in _KNOWN_MOE_ARCHS:
        canonical_keys = [k for k in md.keys() if k.endswith(".expert_count")]
        if not canonical_keys:
            warnings.append(
                DiagnosticWarning(
                    "MOE-FILENAME-FALLBACK",
                    f"{model.name}: detected as MoE from filename only "
                    "(no expert_count in metadata). Placement routed "
                    "through MoE path but expert distribution is "
                    "approximate. Consider re-quantizing from upstream "
                    "if performance is unsatisfactory.",
                )
            )

    return warnings


def format_diagnostic_report(model: ModelEntry) -> str:
    """Return a multi-line, human-readable diagnostic block for ``model``.

    Equivalent to running ``diag_kv.py`` on a single GGUF. Includes the
    KV-relevant metadata, the calculated bytes/token, capacity estimates
    at common context lengths, and at the bottom any active warnings
    from :func:`audit_model_metadata`.

    Safe to call on any ModelEntry — degrades gracefully when metadata
    is missing.
    """
    md = model.metadata or {}
    arch = md.get("general.architecture", "?")
    block_count = metadata_layer_count(md)
    att_layers = metadata_attention_layer_count(md)
    is_hybrid = metadata_is_hybrid_architecture(md)
    n_heads = _metadata_arch_int(md, str(arch), "attention.head_count")
    n_kv = _metadata_arch_int(md, str(arch), *_KV_HEAD_COUNT_SUFFIXES)
    embd = _metadata_arch_int(md, str(arch), "embedding_length")
    kl = _metadata_arch_int(md, str(arch), "attention.key_length")
    vl = _metadata_arch_int(md, str(arch), "attention.value_length")
    ctx_native = _md_int(md, f"{arch}.context_length")
    expert_count_md = _metadata_arch_int(
        md,
        str(arch),
        *(suffix.lstrip(".") for suffix in _MOE_ALT_KEY_SUFFIXES),
    )
    expert_count_eff = _moe_expert_count(model)

    # If head dims missing, scanner derives from embd / n_heads. We
    # surface this so the user knows whether the number is exact.
    if (kl <= 0 or vl <= 0) and n_heads > 0 and embd > 0:
        kl_eff = vl_eff = max(1, embd // n_heads)
        head_dim_note = " (derived from embd/n_heads)"
    else:
        kl_eff, vl_eff = kl, vl
        head_dim_note = ""

    # KV per token: metadata-derived if possible, else params heuristic.
    params_b = extract_params_billion(model.name)
    f16_per_tok_md = kv_per_token_mb_from_metadata(md) if md else 0.0
    f16_per_tok_heur = kv_per_token_mb_f16(params_b)
    if str(arch).lower() in _RECURRENT_ARCHS:
        f16_per_tok = sum(_resolve_kv_per_token_parts_mb(model, params_b))
        kv_source = "pure recurrent (no context-growing attention KV)"
    else:
        f16_per_tok = f16_per_tok_md if f16_per_tok_md > 0 else f16_per_tok_heur
        kv_source = "metadata" if f16_per_tok_md > 0 else "heuristic (no metadata)"
    q4_per_tok = f16_per_tok * kv_quant_factor("q4_0")
    q5_per_tok = f16_per_tok * kv_quant_factor("q5_0")
    q8_per_tok = f16_per_tok * kv_quant_factor("q8_0")

    lines: List[str] = []
    lines.append(f"━━━ {model.name} ━━━")
    lines.append(f"  file size             : {model.size_gb:.2f} GB")
    lines.append(f"  architecture          : {arch}")
    lines.append(f"  block_count           : {block_count}")
    line = f"  attention_layer_count : {att_layers}"
    if is_hybrid:
        line += "  (HYBRID architecture detected)"
    lines.append(line)
    line = f"  head_count            : {n_heads}"
    lines.append(line)
    line = f"  head_count_kv         : {n_kv}"
    if n_kv > 0 and n_heads > 0:
        line += f"   (GQA ratio {n_heads / n_kv:.1f}:1)"
    lines.append(line)
    lines.append(f"  embedding_length      : {embd}")
    lines.append(f"  key_length            : {kl_eff}{head_dim_note}")
    lines.append(f"  value_length          : {vl_eff}{head_dim_note}")
    lines.append(f"  context_length        : {ctx_native:,}")
    line = f"  expert_count (md)     : {expert_count_md}"
    if expert_count_eff > 1 and expert_count_md == 0:
        if str(arch).lower() in _KNOWN_MOE_ARCHS:
            line += "   ← from architecture fallback"
        else:
            line += "   ← from filename fallback"
    lines.append(line)

    # KV capacity table
    lines.append("  ── KV size per token (source: " + kv_source + ") ──")
    lines.append(f"  f16 base              : {f16_per_tok:.4f} MB/token")
    lines.append(f"  q8_0                  : {q8_per_tok:.4f} MB/token")
    lines.append(f"  q5_0                  : {q5_per_tok:.4f} MB/token")
    lines.append(f"  q4_0                  : {q4_per_tok:.4f} MB/token")
    lines.append("  ── Capacity at q5_0 (the AutoTuner's preferred quant) ──")
    for ctx in (32768, 65536, 131072, 262144):
        gb = ctx * q5_per_tok / 1024.0
        lines.append(f"  → {ctx:>7,} ctx          : {gb:6.2f} GB KV cache")
    recurrent_gb = recurrent_state_gb_from_metadata(md)
    if recurrent_gb > 0:
        lines.append(f"  recurrent state (1 slot): {recurrent_gb:6.2f} GB fixed F32")

    # Companions
    if model.mmproj is not None:
        try:
            mmproj_gb = model.mmproj.stat().st_size / (1024**3)
            lines.append(
                f"  vision projector      : {model.mmproj.name} ({mmproj_gb:.2f} GB)"
            )
        except OSError:
            lines.append(f"  vision projector      : {model.mmproj.name}")
    if model.draft is not None:
        lines.append(f"  draft model           : {model.draft.name}")

    # Active warnings — embedded at the bottom of the report so callers
    # don't need a second pass.
    warnings = audit_model_metadata(model)
    if warnings:
        lines.append("  ── Active warnings ──")
        for w in warnings:
            # Wrap long messages at 70 chars for terminal readability.
            msg = w.message
            prefix = f"  ⚠ [{w.id}] "
            lines.append(prefix + msg)
    else:
        lines.append("  ── No warnings (tuner has a complete picture) ──")

    return "\n".join(lines)


def find_model_by_substring(models: List[ModelEntry], needle: str) -> List[ModelEntry]:
    """Helper for the ``--diagnose <substring>`` CLI path.

    Case-insensitive contains-match on ``model.name``. Returns all
    matches (the caller decides whether to diagnose one or all).
    """
    if not needle:
        return list(models)
    n = needle.lower()
    return [m for m in models if n in m.name.lower()]


# ---------------------------------------------------------------------------
# Privacy-conscious support report


def _tail_text(path: Optional[Path], *, max_bytes: int = 512 * 1024) -> str:
    if path is None:
        return ""
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            raw = handle.read(max_bytes)
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace")


def _redact_support_text(text: str, roots: Iterable[Path]) -> str:
    replacements: List[Tuple[str, str]] = []
    try:
        replacements.append((str(Path.home().resolve(strict=False)), "~"))
    except (OSError, RuntimeError):
        pass
    for index, raw_root in enumerate(roots, start=1):
        try:
            root = str(Path(raw_root).expanduser().resolve(strict=False))
        except (OSError, RuntimeError, ValueError):
            continue
        replacements.append((root, f"<MODEL_ROOT_{index}>"))
    try:
        app_root = str(Path(__file__).resolve().parent)
        replacements.append((app_root, "<AUTOTUNER_APP>"))
    except (OSError, RuntimeError):
        pass
    out = text
    for source, replacement in sorted(
        replacements, key=lambda item: len(item[0]), reverse=True
    ):
        if not source:
            continue
        variants = {
            source,
            source.replace("\\", "/"),
            source.replace("/", "\\"),
        }
        for variant in variants:
            out = re.sub(
                re.escape(variant),
                lambda _match, value=replacement: value,
                out,
                flags=re.IGNORECASE if os.name == "nt" else 0,
            )
    # Defensive second boundary: application logs should never contain secrets,
    # but redact common accidental key/value and Bearer-token spellings before a
    # user shares the report.
    out = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", out)
    # Command-line spellings: "--api-key VALUE", "--connect-code VALUE" (the
    # b12187 peer-to-peer tunnel code), "--hf-token VALUE".
    out = re.sub(
        r"(?i)(--(?:api-key|connect-code|hf-token))(?:\s+|=)(?!<redacted>)[^\s\"']+",
        r"\1 <redacted>",
        out,
    )
    out = re.sub(
        r"(?i)\b(api[_-]?key|access[_-]?token|authorization|password|secret)"
        r"\s*[:=]\s*[^\r\n,;]+",
        r"\1=<redacted>",
        out,
    )
    return out


def _format_system_support(system: Any) -> List[str]:
    if system is None:
        return ["Hardware snapshot: unavailable"]
    lines = [
        f"OS snapshot: {getattr(system, 'os_name', '?')}",
        f"CPU: {getattr(system, 'cpu_name', '?')}",
        (
            "CPU cores: "
            f"{getattr(system, 'cpu_cores_physical', '?')} physical / "
            f"{getattr(system, 'cpu_cores_logical', '?')} logical"
        ),
        (
            f"RAM: {float(getattr(system, 'free_ram_gb', 0.0)):.1f} GiB free / "
            f"{float(getattr(system, 'total_ram_gb', 0.0)):.1f} GiB total"
        ),
    ]
    gpus = list(getattr(system, "gpus", []) or [])
    if not gpus:
        lines.append("GPUs: none selected/detected")
    for index, gpu in enumerate(gpus):
        lines.append(
            f"GPU {index}: {getattr(gpu, 'name', '?')} · "
            f"{float(getattr(gpu, 'free_vram_gb', 0.0)):.1f}/"
            f"{float(getattr(gpu, 'total_vram_gb', 0.0)):.1f} GiB free/total · "
            f"backend={getattr(gpu, 'runtime_backend', None) or '?'} · "
            f"device={getattr(gpu, 'runtime_device', None) or '?'}"
        )
    return lines


def format_support_report(
    models: Sequence[ModelEntry],
    *,
    system: Any = None,
    forks: Sequence[Tuple[str, Path]] = (),
    active_fork: Optional[Path] = None,
    model_roots: Sequence[Path] = (),
    app_log_path: Optional[Path] = None,
    debug_enabled: bool = False,
) -> str:
    """Return a shareable report without settings contents or server prompts."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cache = metadata_cache_stats()
    lines = [
        "AutoTuner support report",
        "=" * 80,
        f"Generated (UTC): {now}",
        f"AutoTuner: v{VERSION}",
        f"Python: {platform.python_version()} ({sys.executable and Path(sys.executable).name})",
        f"Runtime: {'frozen binary' if getattr(sys, 'frozen', False) else 'source'}",
        f"Platform: {platform.platform()}",
        f"Machine: {platform.machine() or '?'}",
        f"Debug mode: {'on' if debug_enabled else 'off'}",
        "",
        "Privacy boundary",
        "-" * 80,
        "No prompts, API payloads, credentials, environment-variable values, raw settings,",
        "or llama-server output are included. Known home/model paths are redacted in the",
        "bounded AutoTuner application-log tail below.",
        "",
        "Hardware",
        "-" * 80,
        *_format_system_support(system),
        "",
        "llama.cpp builds",
        "-" * 80,
        f"Discovered builds: {len(forks)}",
    ]
    active_name = active_fork.name if active_fork is not None else "none"
    lines.append(f"Active build folder: {active_name}")
    lines.extend(f"- {name}" for name, _path in forks[:128])
    lines.extend(
        [
            "",
            "Scanner",
            "-" * 80,
            f"Models: {len(models)}",
            f"Configured active roots: {len(model_roots)}",
            (
                f"Metadata cache: {cache['entries']} entries, {cache['hits']} hit(s), "
                f"{cache['misses']} miss(es), up to {cache['workers']} worker(s)"
            ),
            "",
            "Model diagnostics",
            "-" * 80,
        ]
    )
    if not models:
        lines.append("No scanned models were available.")
    for model in models:
        lines.extend([format_diagnostic_report(model), ""])

    app_log = _tail_text(app_log_path)
    lines.extend(["Application log tail (redacted)", "-" * 80])
    if app_log:
        redaction_roots = [*model_roots, *(path for _name, path in forks)]
        redacted = _redact_support_text(app_log, redaction_roots)
        lines.extend(redacted.splitlines()[-400:])
    else:
        lines.append("No application log was available.")
    lines.extend(["", "End of report", ""])
    return "\n".join(lines)


def write_support_report(
    models: Sequence[ModelEntry],
    *,
    output_dir: Optional[Path] = None,
    **kwargs: Any,
) -> Path:
    """Atomically write :func:`format_support_report` under reports/."""
    if output_dir is None:
        import app_settings

        output_dir = app_settings.app_data_dir() / "reports"
    directory = Path(output_dir).expanduser().resolve(strict=False)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = directory / f"AutoTuner-support-{stamp}.txt"
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(format_support_report(models, **kwargs), encoding="utf-8")
    os.replace(tmp, output)
    return output
