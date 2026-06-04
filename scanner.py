"""Scan a models folder for GGUF files, pair them with mmproj projectors,
and pull a few useful fields from GGUF metadata when available.

GGUF format reference (v3): https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
We read only the header (KV pairs), never tensor data, so this is fast even
for 100+ GB files.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# GGUF metadata reader (minimal, no external deps)

_GGUF_MAGIC = b"GGUF"

# GGUF value type IDs
_GT_UINT8, _GT_INT8 = 0, 1
_GT_UINT16, _GT_INT16 = 2, 3
_GT_UINT32, _GT_INT32 = 4, 5
_GT_FLOAT32 = 6
_GT_BOOL = 7
_GT_STRING = 8
_GT_ARRAY = 9
_GT_UINT64, _GT_INT64 = 10, 11
_GT_FLOAT64 = 12

_SCALAR_FMT = {
    _GT_UINT8: ("<B", 1),
    _GT_INT8: ("<b", 1),
    _GT_UINT16: ("<H", 2),
    _GT_INT16: ("<h", 2),
    _GT_UINT32: ("<I", 4),
    _GT_INT32: ("<i", 4),
    _GT_FLOAT32: ("<f", 4),
    _GT_BOOL: ("<?", 1),
    _GT_UINT64: ("<Q", 8),
    _GT_INT64: ("<q", 8),
    _GT_FLOAT64: ("<d", 8),
}


def _read_value(f, vtype: int, want_array_elements: bool = True) -> Any:
    """Read one GGUF value of given type. Skips array contents to save
    memory if `want_array_elements` is False."""
    if vtype in _SCALAR_FMT:
        fmt, size = _SCALAR_FMT[vtype]
        data = f.read(size)
        if len(data) < size:
            raise EOFError("Unexpected EOF in GGUF value")
        return struct.unpack(fmt, data)[0]
    if vtype == _GT_STRING:
        ln = struct.unpack("<Q", f.read(8))[0]
        return f.read(ln).decode("utf-8", errors="replace")
    if vtype == _GT_ARRAY:
        atype = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        # Token vocab arrays can be huge — skip them silently.
        if not want_array_elements or n > 256:
            for _ in range(n):
                _read_value(f, atype, want_array_elements=False)
            return None
        return [_read_value(f, atype, True) for _ in range(n)]
    raise ValueError(f"Unknown GGUF value type {vtype}")


# Pre-compiled: match "blk.{N}." tensor names — used by MTP tensor scan.
_BLK_IDX_RE = re.compile(r"^blk\.(\d+)\.")


def read_gguf_metadata(path: Path) -> Dict[str, Any]:
    """Read GGUF header KV pairs and scan tensor info for MTP detection.

    In addition to the standard KV pairs this function reads the tensor
    info section (names only — no data) and stores a synthetic *tri-state*
    flag describing what the tensor scan concluded:

      ``__mtp_scan__: "found"``        — an MTP/draft-head tensor was seen,
        identified either by a block index ``>= <arch>.block_count`` or by
        a tensor name containing ``nextn`` / ``mtp`` (the canonical llama.cpp
        nextn naming, e.g. ``blk.N.nextn.eh_proj.weight``).

      ``__mtp_scan__: "absent"``       — the scan ran to completion over the
        whole model, ``block_count`` was known, the file was *not* a shard,
        and no MTP tensors were found.  Only in this high-confidence state
        may a positive ``<arch>.nextn_predict_layers`` key be treated as a
        false positive (the UD/unsloth case where the metadata value is kept
        but the MTP weights are stripped during quantisation).

      ``__mtp_scan__: "inconclusive"`` — the scan could not reliably cover
        the whole model: the file is one shard of a split GGUF (the nextn
        block lives in the *last* shard, not shard 1), ``block_count`` was
        unreadable, or the tensor-info parse hit EOF / a struct error.  In
        this state the scan must NOT veto the metadata key — doing so was the
        root cause of "sometimes detected, sometimes not" on sharded MoE MTP
        models (GLM-4.6, DeepSeek-V3) and on conversions whose nextn block is
        numbered differently from ``block_count``.

    Synthetic keys start with ``__`` and can never collide with real GGUF
    keys (the GGUF spec forbids leading underscores in key names).
    """
    try:
        with path.open("rb") as f:
            magic = f.read(4)
            if magic != _GGUF_MAGIC:
                return {}
            version = struct.unpack("<I", f.read(4))[0]
            if version < 2:
                return {}  # v1 layout differed; not worth supporting
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            md: Dict[str, Any] = {}
            for _ in range(n_kv):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8", errors="replace")
                vtype = struct.unpack("<I", f.read(4))[0]
                md[key] = _read_value(f, vtype)

            # ------------------------------------------------------------------
            # Tensor info scan — GGUF layout after KV section:
            #   For each tensor: name (u64-len string), n_dims (u32),
            #                    dims (u64 * n_dims), type (u32), offset (u64)
            # No padding before this section; data padding is after.
            #
            # Goal: detect block indices beyond block_count which indicate
            # extra MTP/draft heads (e.g. blk.28.* when block_count == 28).
            # The official converter writes <arch>.nextn_predict_layers for
            # this, but inject-style community GGUFs often skip that key.
            # ------------------------------------------------------------------
            arch = str(md.get("general.architecture", "") or "")
            block_count: int = 0
            if arch:
                bc = md.get(f"{arch}.block_count")
                if bc is not None:
                    try:
                        block_count = int(bc)
                    except (TypeError, ValueError):
                        pass

            # Is this one shard of a split GGUF?  The nextn/MTP block is the
            # LAST transformer block (blk.{block_count}.*) and therefore almost
            # always lives in the final shard — never in shard 1, which is what
            # we read here.  So a negative scan on a shard tells us nothing.
            split_count = 0
            for sk in ("split.count", "general.split_count"):
                sv = md.get(sk)
                if sv is not None:
                    try:
                        split_count = int(sv)
                        break
                    except (TypeError, ValueError):
                        pass
            is_sharded = split_count > 1

            has_mtp_tensors = False
            scan_complete = False
            try:
                for _ in range(n_tensors):
                    tname_len = struct.unpack("<Q", f.read(8))[0]
                    tname = f.read(tname_len).decode("utf-8", errors="replace")
                    n_dims = struct.unpack("<I", f.read(4))[0]
                    # skip: dims (u64 * n_dims) + type (u32) + offset (u64)
                    f.read(8 * n_dims + 4 + 8)
                    if not has_mtp_tensors:
                        # (a) Name-based: the canonical llama.cpp nextn tensors
                        #     are named "blk.{N}.nextn.*" (eh_proj, embed_tokens,
                        #     enorm, hnorm, shared_head_*). This catch is
                        #     independent of block_count, so it works even when
                        #     block_count is unreadable or the block is numbered
                        #     unexpectedly. Some forks emit "mtp" in the name.
                        tl = tname.lower()
                        if "nextn" in tl or "mtp" in tl:
                            has_mtp_tensors = True
                        # (b) Index-based: a block index at/after block_count is
                        #     an extra draft head grafted past the main stack.
                        elif block_count > 0:
                            m = _BLK_IDX_RE.match(tname)
                            if m:
                                try:
                                    if int(m.group(1)) >= block_count:
                                        has_mtp_tensors = True
                                except (TypeError, ValueError):
                                    pass
                else:
                    # Loop ran to completion without break/exception → the whole
                    # tensor-info section of THIS file was parsed successfully.
                    scan_complete = True
            except (OSError, struct.error, EOFError):
                pass  # non-fatal; KV data already collected

            # Record a tri-state confidence value. Only a *complete* scan over
            # a *non-sharded* file with a known block_count can authoritatively
            # assert absence; anything else is inconclusive and must not veto
            # the metadata key downstream.
            if has_mtp_tensors:
                md["__mtp_scan__"] = "found"
            elif scan_complete and not is_sharded and block_count > 0:
                md["__mtp_scan__"] = "absent"
            else:
                md["__mtp_scan__"] = "inconclusive"

            return md
    except (OSError, struct.error, EOFError, ValueError, UnicodeDecodeError):
        return {}


def metadata_has_embedded_mtp(md: Dict[str, Any]) -> bool:
    """Return True iff the GGUF contains an integrated MTP/draft-head.

    Detection order (most to least authoritative):

    1. ``<arch>.nextn_predict_layers > 0`` — the official GGUF key written
       by ``convert_hf_to_gguf.py --mtp`` (llama.cpp gguf-py constants
       ``Keys.LLM.NEXTN_PREDICT_LAYERS``) and present in all standard
       MTP GGUFs from the mainstream converter.  A value > 0 is normally
       definitive proof of embedded draft heads.

    2. ``__mtp_scan__ == "found"`` — the tensor-info scan in
       :func:`read_gguf_metadata` saw an MTP tensor (nextn-named or block
       index beyond ``block_count``).  Covers community / inject-style
       GGUFs that graft MTP weights without writing the metadata key.

    3. Generic KV scan for any ``*.nextn_predict_layers > 0`` — forward-
       compat for new architecture prefixes.

    Cross-check (false-positive guard)
    ----------------------------------
    The only known false positive is UD/unsloth quantisation that keeps a
    base-architecture ``nextn_predict_layers`` value in metadata while
    stripping the MTP weights.  We suppress the key in checks 1 and 3
    **only** when ``__mtp_scan__ == "absent"`` — i.e. a complete scan over a
    non-sharded file with a known ``block_count`` confirmed the tensors are
    gone.  An ``"inconclusive"`` scan (split GGUF read from shard 1, missing
    ``block_count``, or a parse error) NEVER vetoes the key — that overly
    aggressive veto was the cause of intermittent detection on sharded MoE
    MTP models whose nextn block sits in the last shard.
    """
    if not md:
        return False

    scan = md.get("__mtp_scan__")  # "found" / "absent" / "inconclusive" / None
    scan_absent = scan == "absent"

    # 1. Official key: <arch>.nextn_predict_layers
    arch = str(md.get("general.architecture", "") or "")
    arch_nextn_key = f"{arch}.nextn_predict_layers" if arch else None
    if arch_nextn_key is not None:
        v = md.get(arch_nextn_key)
        if v is not None:
            try:
                if int(v) > 0 and not scan_absent:
                    # Trust the key unless a high-confidence scan proved the
                    # weights are absent (UD/unsloth stripped quant).
                    return True
            except (TypeError, ValueError):
                pass

    # 2. Tensor scan positively identified an MTP tensor.
    if scan == "found":
        return True

    # 3. Generic KV scan — forward-compat for new arch prefixes. Skip the
    #    arch-specific key only when check 1 deliberately suppressed it
    #    (scan == "absent"), so we don't re-introduce that false positive.
    for key, val in md.items():
        if key.startswith("__"):
            continue  # skip synthetic keys
        if "nextn_predict" in key.lower():
            if scan_absent and key == arch_nextn_key:
                continue
            try:
                if int(val) > 0:
                    return True
            except (TypeError, ValueError):
                pass

    return False


def metadata_layer_count(md: Dict[str, Any]) -> int:
    """Find architecture's `block_count` (number of transformer layers)."""
    if not md:
        return 0
    arch = md.get("general.architecture")
    if arch:
        key = f"{arch}.block_count"
        if key in md:
            try:
                return int(md[key])
            except (TypeError, ValueError):
                pass
    # Fallback: scan all keys for *.block_count
    for k, v in md.items():
        if k.endswith(".block_count"):
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return 0


def metadata_native_context(md: Dict[str, Any]) -> int:
    """Find architecture's training context length."""
    if not md:
        return 0
    arch = md.get("general.architecture")
    if arch:
        key = f"{arch}.context_length"
        if key in md:
            try:
                return int(md[key])
            except (TypeError, ValueError):
                pass
    for k, v in md.items():
        if k.endswith(".context_length"):
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return 0


# Keys the converters write to record the model author's recommended
# sampler defaults. Qwen3.5/3.6, GLM-4.x and several others embed these
# (e.g. general.sampling.temp = 1.0, general.sampling.top_k = 20). They
# are the single most reliable per-model sampling source — better than a
# generic family profile and far better than the global defaults.
_SAMPLING_MD_KEYS = {
    "temperature": ("general.sampling.temp", "general.sampling.temperature"),
    "top_k": ("general.sampling.top_k",),
    "top_p": ("general.sampling.top_p",),
    "min_p": ("general.sampling.min_p",),
    "repeat_penalty": (
        "general.sampling.repeat_penalty",
        "general.sampling.repetition_penalty",
    ),
    "presence_penalty": ("general.sampling.presence_penalty",),
}


def metadata_sampling(md: Dict[str, Any]) -> Dict[str, float]:
    """Extract the author-recommended sampler settings from GGUF metadata.

    Returns a dict with whatever subset of
    ``temperature / top_k / top_p / min_p / repeat_penalty /
    presence_penalty`` the file actually declares (missing keys are simply
    absent — the caller decides how to merge). ``top_k`` is returned as an
    int; everything else as a float.

    These ``general.sampling.*`` keys are emitted by the mainstream
    converter for models whose authors ship recommended defaults
    (Qwen3.5/3.6: temp 1.0 / top_k 20 / top_p 0.95; GLM-4.x; etc.). They
    were previously ignored, so a model with no matching YAML profile fell
    back to the generic ``temp 0.7 / top_k 40`` defaults — a frequent cause
    of repetition loops and broken tool-calls on models tuned for a low
    top_k with a non-zero min_p.
    """
    out: Dict[str, float] = {}
    if not md:
        return out
    for field_name, keys in _SAMPLING_MD_KEYS.items():
        for k in keys:
            v = md.get(k)
            if v is None:
                continue
            # GGUF scalars may arrive as 0-d values or 1-element lists
            # depending on the reader; coerce defensively.
            if isinstance(v, (list, tuple)):
                if not v:
                    continue
                v = v[0]
            try:
                if field_name == "top_k":
                    out[field_name] = float(int(v))
                else:
                    out[field_name] = float(v)
            except (TypeError, ValueError):
                continue
            break  # first present key wins
    return out


# Architektur-Namen die RoPE-Scaling (YaRN) unterstützen bis zu 1M tokens.
# Gematcht wird via arch.startswith() in metadata_supports_rope_scale(), daher
# deckt das Prefix "qwen" ALLE Qwen-Arch-Strings ab: qwen2/qwen2moe/qwen2vl
# UND qwen3/qwen3moe/qwen3next/qwen3vl/qwen3vlmoe/qwen35/qwen35moe. Vorher
# stand hier "qwen2", was die neueren qwen3*/qwen35*-Strings NICHT traf —
# die ganze Qwen3/3.5/3.6-Familie wäre so vom automatischen YaRN
# ausgeschlossen gewesen (nur noch via rope_scale.enabled=true im Profil).
_ROPE_SCALE_SUPPORTED_ARCHS = frozenset(
    {
        "qwen",  # Qwen / Qwen2 / Qwen2.5 / Qwen3 / Qwen3.5 / Qwen3.6 Familie
    }
)


def metadata_supports_rope_scale(md: Dict[str, Any]) -> bool:
    """Prüft ob das Modell RoPE-Scaling (YaRN) unterstützt.

    Returns True wenn die Architektur RoPE-Scaling bis zu 1M tokens unterstützt.
    """
    if not md:
        return False
    arch = md.get("general.architecture")
    if not arch:
        return False
    # Prüfe ob die Architektur in der Support-Liste ist
    for supported in _ROPE_SCALE_SUPPORTED_ARCHS:
        if arch.startswith(supported):
            return True
    return False


# ---------------------------------------------------------------------------
# Hybrid Mamba+Transformer detection
#
# Pure Transformer models keep KV cache for every layer. Hybrid models
# (Mamba/SSM blocks interleaved with Transformer blocks) only allocate
# KV for the attention layers, which is typically 1/4 to 1/8 of all
# blocks. Our params-based KV estimate dramatically overshoots on these
# unless we know the real attention-layer count.

# Architectures known to be hybrid (Mamba/SSM + Transformer).
_HYBRID_ARCHS = frozenset(
    {
        "nemotron_h",
        "nemotron-h",
        "granitemoehybrid",
        "granite-h",
        "granite_h",
        "jamba",
        "bamba",
        "falcon_h1",
        "plamo2",  # Plamo-2 hybrid
        "zamba2",  # Zamba2 hybrid
        "rwkv6",  # RWKV — pure SSM, but treated similarly for KV
        "rwkv7",
    }
)


def metadata_is_hybrid_architecture(md: Dict[str, Any]) -> bool:
    """Detect Mamba/SSM-Transformer hybrid models from GGUF metadata.

    A model is "hybrid" when only a fraction of its layers carry KV
    cache. We detect this two ways:
      1. Architecture name matches a known hybrid (cheap, reliable).
      2. Any ``<arch>.ssm.*`` keys exist in the metadata (catches new
         hybrid architectures we don't have on the allow-list yet).
    """
    if not md:
        return False
    arch = str(md.get("general.architecture", "") or "").lower()
    if arch in _HYBRID_ARCHS:
        return True
    # Generic SSM-state detection — any *.ssm.* key signals hybrid.
    for k in md.keys():
        if ".ssm." in k:
            return True
    return False


def metadata_attention_layer_count(md: Dict[str, Any]) -> int:
    """Return the number of layers that actually carry KV cache.

    For pure Transformer models this equals ``block_count``. For hybrid
    Mamba/Transformer models we look for an explicit attention-layer
    count first, then fall back to a conservative ratio of total blocks.

    Returns 0 when the answer can't be determined (caller should treat
    as "use total block count" — i.e. assume non-hybrid).
    """
    if not md:
        return 0
    arch = str(md.get("general.architecture", "") or "")
    total = metadata_layer_count(md)
    if total <= 0:
        return 0

    if not metadata_is_hybrid_architecture(md):
        return total  # pure Transformer — every layer has KV

    # Hybrid model — try explicit metadata keys.
    explicit_keys = (
        f"{arch}.attention.block_count",
        f"{arch}.attention.layer_count",
        f"{arch}.transformer.block_count",
        f"{arch}.n_attention_layers",
    )
    for key in explicit_keys:
        v = md.get(key)
        if v is not None:
            try:
                n = int(v)
                if 0 < n <= total:
                    return n
            except (TypeError, ValueError):
                pass

    # No explicit count — apply a per-architecture heuristic. These
    # ratios come from each model's published architecture diagrams.
    # When in doubt we err high (more attention layers ↔ larger KV
    # estimate ↔ safer placement).
    arch_l = arch.lower()
    if "nemotron" in arch_l:
        # Nemotron-H: roughly 1 attention block per 4 Mamba blocks.
        ratio = 0.25
    elif "jamba" in arch_l:
        # Jamba: 1 attention per 7 Mamba (~14%).
        ratio = 0.15
    elif "granite" in arch_l and "hybrid" in arch_l:
        # Granite-Hybrid: ~25% attention.
        ratio = 0.25
    elif "bamba" in arch_l:
        ratio = 0.20
    elif "rwkv" in arch_l:
        # Pure SSM — no real KV cache. Use 1 to keep estimates small
        # but non-zero so the rest of the code doesn't divide by zero.
        return 1
    else:
        # Unknown hybrid — assume 25% attention layers (conservative).
        ratio = 0.25

    return max(1, int(total * ratio))


# ---------------------------------------------------------------------------
# Thinking / reasoning capability detection
#
# Detecting reasoning support purely by filename ("qwen3" → has thinking)
# false-positives on non-thinking siblings like Qwen3-Coder, Qwen3-VL-
# Captioner, or Qwen3-Embedding. The chat template is the authoritative
# source: models built for thinking embed <think> tokens or
# enable_thinking flags in their template.

# Filename markers that exclude a model from thinking even if its base
# family supports it (Qwen3-Coder is a Qwen3 model with no <think>).
_NON_THINKING_NAME_HINTS = (
    "coder",
    "embedding",
    "reranker",
    "captioner",
    "instruct-2507",  # Qwen3-2507-Instruct is the non-thinking branch
    "non-thinking",
)

# Markers that indicate thinking support inside a chat template.
_THINKING_TEMPLATE_MARKERS = (
    "<think>",
    "</think>",
    "<|think|>",
    "enable_thinking",
    "reasoning_content",
    "thinking_budget",
    "preserve_thinking",
)

# Filename keywords used as a fallback when no chat template is present.
_THINKING_NAME_HINTS = (
    "gemma",
    "deepseek-r",
    "qwq",
    "reasoning",
    "thinking",
)


def metadata_supports_thinking(md: Dict[str, Any], filename: str = "") -> bool:
    """Return True iff this model supports reasoning / thinking output.

    Decision order:
      1. Filename excludes thinking explicitly (e.g. "Qwen3-Coder") →
         False, even if the chat template has <think>. The non-thinking
         siblings sometimes inherit a generic template that mentions
         thinking but they don't actually emit it.
      2. Chat template contains a thinking marker → True.
      3. No template available → fall back to filename keywords.
      4. Otherwise → False.
    """
    name_l = (filename or "").lower()
    if any(hint in name_l for hint in _NON_THINKING_NAME_HINTS):
        return False

    if md:
        for key in ("tokenizer.chat_template", "tokenizer.chat_template.default"):
            template = md.get(key)
            if isinstance(template, str) and template:
                if any(m in template for m in _THINKING_TEMPLATE_MARKERS):
                    return True
                # Template was present but had no thinking marker —
                # this is informative enough to stop here.
                return False

    # No template at all — fall back to filename heuristic.
    return any(hint in name_l for hint in _THINKING_NAME_HINTS)


# ---------------------------------------------------------------------------
# Tool-use / function-calling capability detection
#
# Modern instruct/chat models advertise tool-calling support inside their
# chat template — older models (Llama-2 base, original Mistral, Phi-2,
# many GGUFs from 2023) genuinely cannot do tool calls and llama-server
# will refuse `--jinja` workflows that need them. Detection mirrors the
# thinking-detection: scan the chat template for known markers, with a
# small filename allow-list as a fallback.

# Markers that indicate tool/function-calling support in a chat template.
_TOOLUSE_TEMPLATE_MARKERS = (
    "<tool_call>",
    "</tool_call>",
    "<|tool_call_begin|>",  # DeepSeek
    "<|tool_calls_begin|>",  # DeepSeek-V3
    "<|tool|>",
    "<|tool_results|>",
    "tool_calls",  # OpenAI-style; common in modern templates
    "function_call",
    "<|im_start|>tool",  # Hermes / Qwen tool role
    "[TOOL_CALLS]",  # Mistral
    "[AVAILABLE_TOOLS]",  # Mistral
    "{{ tools }}",  # Jinja variable — only present when supported
    "{%- if tools",
    "{% if tools",
    "if tools is defined",
)

# Filenames that strongly imply tool-use even without a template
# (rare; mostly for older quants stripped of their chat template).
_TOOLUSE_NAME_HINTS = (
    "hermes",
    "functionary",
    "tool",
)

# Architectures known to NOT support tool calls regardless of template
# heuristics (embedding-only, captioner-only, base completion models).
_NON_TOOLUSE_NAME_HINTS = (
    "embedding",
    "reranker",
    "captioner",
    "base",  # raw-base completion models
)


def metadata_supports_tool_use(md: Dict[str, Any], filename: str = "") -> bool:
    """Return True iff this model can invoke tools / call functions.

    Decision order mirrors :func:`metadata_supports_thinking`:
      1. Filename excludes tool-use explicitly → False.
      2. Chat template contains a tool-call marker → True.
      3. Template was present but had no marker → False (informative).
      4. No template available → fall back to filename hints.
      5. Otherwise → False.
    """
    name_l = (filename or "").lower()
    if any(hint in name_l for hint in _NON_TOOLUSE_NAME_HINTS):
        return False

    if md:
        for key in ("tokenizer.chat_template", "tokenizer.chat_template.default"):
            template = md.get(key)
            if isinstance(template, str) and template:
                if any(m in template for m in _TOOLUSE_TEMPLATE_MARKERS):
                    return True
                # Template present but no tool marker — authoritative no.
                return False

    # No template — fall back to filename heuristic.
    return any(hint in name_l for hint in _TOOLUSE_NAME_HINTS)


# ---------------------------------------------------------------------------
# Model entries + scanner

# Strip quant + extension when normalizing for mmproj pairing.
# Recognises the standard llama.cpp quant tails plus a few non-standard
# community ones (mxfp4 / mxfp4_moe — the MXFP4 micro-scaled FP4 format
# used by e.g. the qwen3.6-…-mxfp4_moe GGUFs, which carry no Q*/IQ* token).
_QUANT_PATTERN = re.compile(
    r"[-._]"
    r"(?:UD-)?"
    r"(?:i\d+-)?"  # i1- prefix (imatrix variants)
    r"(?:Q\d+(?:_[A-Z0-9]+)*"
    r"|IQ\d+(?:_[A-Z0-9]+)*"
    r"|MXFP4(?:_MOE)?"  # MXFP4 / MXFP4_MOE (no Q-prefix)
    r"|BF16|F16|F32"
    r")"
    r"(?:[-._][0-9.]+bpw)?"
    r"(?:[-._](?:bf16|f16|f32))?"
    r"\.gguf$",
    re.IGNORECASE,
)


# Matches llama.cpp split-GGUF naming: "model-00002-of-00003.gguf"
# llama-gguf-split always zero-pads to 5 digits on both sides.
_SPLIT_PART_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def _split_gguf_key(filename: str) -> Optional[Tuple[str, int, int]]:
    """Return ``(base_stem, part_idx, total_parts)`` for a split-GGUF shard.

    Recognises the ``-NNNNN-of-NNNNN.gguf`` suffix produced by
    ``llama-gguf-split`` (e.g. ``Qwen3.5-122B-A10B-UD-Q3_K_XL-00002-of-00003.gguf``).
    Returns ``None`` for ordinary single-file GGUFs.
    """
    m = _SPLIT_PART_RE.search(filename)
    if m:
        return filename[: m.start()], int(m.group(1)), int(m.group(2))
    return None


@dataclass
class ModelEntry:
    path: Path
    name: str  # display name (filename stem)
    group: str  # parent folder relative to scan root (e.g. "Alibaba/Qwen3.6")
    size_bytes: int
    mmproj: Optional[Path] = None
    # All mmproj projectors found in the model's own directory that match
    # this model's base name, sorted best-first (the same ranking used to
    # pick `mmproj`). Lets the GUI offer a manual dropdown when a model
    # ships several precisions (bf16 / f16 / f32). `mmproj` is just
    # `mmproj_candidates[0]` when any matched.
    mmproj_candidates: List[Path] = field(default_factory=list)
    draft: Optional[Path] = None  # paired assistant/draft model (if any)
    metadata: Dict[str, Any] = field(default_factory=dict)
    part_paths: List[Path] = field(default_factory=list)
    """All shard paths in order; length > 1 for split GGUFs, otherwise [path]."""

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024**3)

    @property
    def is_split(self) -> bool:
        """True when this entry represents a multi-part (sharded) GGUF."""
        return len(self.part_paths) > 1

    @property
    def part_count(self) -> int:
        """Number of GGUF shards on disk (1 for single-file models)."""
        return len(self.part_paths) if self.part_paths else 1

    @property
    def has_vision(self) -> bool:
        return self.mmproj is not None

    @property
    def has_draft(self) -> bool:
        """True iff a paired assistant/draft model was found in the same folder."""
        return self.draft is not None

    @property
    def n_layers(self) -> int:
        return metadata_layer_count(self.metadata)

    @property
    def native_context(self) -> int:
        return metadata_native_context(self.metadata)

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "") or "")

    @property
    def supports_rope_scale(self) -> bool:
        """Prüft ob das Modell RoPE-Scaling (YaRN) unterstützt."""
        return metadata_supports_rope_scale(self.metadata)

    @property
    def is_hybrid(self) -> bool:
        """True for Mamba/Transformer hybrids (Nemotron-H, Jamba, …)."""
        return metadata_is_hybrid_architecture(self.metadata)

    @property
    def n_attention_layers(self) -> int:
        """Number of layers carrying KV cache. Equals ``n_layers`` for
        pure Transformer; smaller for hybrids."""
        return metadata_attention_layer_count(self.metadata)

    @property
    def supports_thinking(self) -> bool:
        """True if the chat template signals thinking/reasoning support."""
        return metadata_supports_thinking(self.metadata, self.name)

    @property
    def supports_tool_use(self) -> bool:
        """True if the chat template signals tool-call / function support."""
        return metadata_supports_tool_use(self.metadata, self.name)

    @property
    def recommended_sampling(self) -> Dict[str, float]:
        """Author-recommended sampler settings from GGUF metadata, if any.

        Subset of temperature/top_k/top_p/min_p/repeat_penalty/
        presence_penalty actually declared in ``general.sampling.*``.
        Empty when the file carries none. The tuner uses this to fill any
        sampling value a matched YAML profile leaves unspecified, so models
        without a tailored profile still run on their intended samplers
        instead of the generic defaults.
        """
        return metadata_sampling(self.metadata)

    @property
    def has_embedded_mtp(self) -> bool:
        """True if this GGUF contains an integrated MTP/draft-head.

        Detection is metadata-first with a filename-pattern fallback:

        **Primary** — :func:`metadata_has_embedded_mtp` checks:
          1. ``<arch>.nextn_predict_layers > 0`` (official GGUF key set by
             ``convert_hf_to_gguf.py --mtp`` and all standard converters),
             trusted unless a complete non-sharded tensor scan proved the
             weights were stripped (``__mtp_scan__ == "absent"``).
          2. ``__mtp_scan__ == "found"`` from the tensor-info scan in
             :func:`read_gguf_metadata` — catches nextn-named tensors and
             inject-style community GGUFs that add MTP blocks without
             updating the metadata key, independent of ``block_count``.
          3. Generic scan for any ``*.nextn_predict_layers > 0``.

        **Fallback** — filename regex ``(?:^|[-_.])\\ mtp(?:[-_.]|$)``
        (case-insensitive) for rare GGUFs that predate the standardised
        metadata key and carry no standard keys.

        Examples that are detected by metadata alone (no "MTP" in name):
            ``Qwen3.6-27B-Q4_K_M.gguf``  (if ``qwen2.nextn_predict_layers=1``)
        Examples detected by filename fallback only:
            ``Qwen3.6-27B-MTP-UD-Q3_K_XL.gguf``  (legacy community inject)
        Examples that never match (correct negative):
            ``prometheus-13b.gguf``  (contains 'mtp' but not bounded)
        """
        # Primary: authoritative GGUF metadata
        if self.metadata and metadata_has_embedded_mtp(self.metadata):
            return True
        # Fallback: filename-based for GGUFs missing the standard key
        return bool(re.search(r"(?:^|[-_.])mtp(?:[-_.]|$)", self.name, re.IGNORECASE))

    @property
    def has_speculative_draft(self) -> bool:
        """True when any form of speculative decoding is available.

        Covers both an external sibling-assistant GGUF (``self.draft``) and
        an embedded MTP drafter detected from the filename.
        """
        return self.draft is not None or self.has_embedded_mtp


def _strip_quant(filename: str) -> str:
    """Strip the quant tail and the GGUF/mmproj extension from a filename.

    Two stages:
      1. Try the standard quant-tail pattern (``…-Q6_K_XL.gguf`` etc.) and
         strip the whole tail if it matches.
      2. If the name carries no recognised quant token (e.g. the
         ``…-mxfp4_moe.gguf`` community files, or a bare
         ``Model.gguf`` / ``…-f32.mmproj``), still remove the trailing
         ``.gguf`` / ``.mmproj`` extension so the stem is usable for
         prefix matching. The previous version left ``.gguf`` attached
         whenever the quant pattern missed, which broke mmproj pairing
         for any non-standard quant label.
    """
    low = filename.lower()
    if low.endswith(".gguf"):
        stripped = _QUANT_PATTERN.sub("", filename)
        if stripped != filename:
            return stripped.rstrip(".-_")
        # No quant tail matched — just drop the extension.
        return filename[: -len(".gguf")].rstrip(".-_")
    if low.endswith(".mmproj"):
        # ".mmproj"-extension projectors (e.g. LFM2.5-Audio-…-f32.mmproj).
        # Strip a trailing precision token (f16/f32/bf16) too.
        stem = filename[: -len(".mmproj")]
        stem = re.sub(r"[-._](?:bf16|f16|f32)$", "", stem, flags=re.IGNORECASE)
        return stem.rstrip(".-_")
    return filename


def _canonical_sep(s: str) -> str:
    """Collapse all separators (``- _ .``) to a single ``-`` and lowercase.

    Used for separator-tolerant prefix matching between a model and its
    projector: the mxfp4 pair, for instance, mixes ``-moe`` (in the
    projector name) with ``_moe`` (in the model name), so a literal
    ``startswith`` fails. Canonicalising both sides to ``-moe`` fixes it
    without loosening the match to an unrelated model.
    """
    return re.sub(r"[-_.]+", "-", s.lower()).strip("-")


def _normalize_model(filename: str) -> str:
    return _strip_quant(filename).lower()


# Matches the "mmproj" marker anywhere in a filename stem, bounded by a
# separator or the string ends — so it catches both the canonical
# "mmproj-Model-F16.gguf" prefix form AND the mid-name form
# "Model-mxfp4-moe-mmproj-f16.gguf". A bare substring check would also
# fire on an unrelated word containing "mmproj", which never occurs in
# practice, but the boundary keeps it strict regardless.
_MMPROJ_TOKEN_RE = re.compile(r"(?:^|[-_.])mmproj(?:[-_.]|$)", re.IGNORECASE)


def _is_mmproj_filename(name: str) -> bool:
    """True if *name* is a vision/audio projector file.

    Two independent signals:
      1. A ``.mmproj`` extension (e.g. ``LFM2.5-Audio-1.5B-f32.mmproj``) —
         these never reach the ``*.gguf`` glob, so the scanner picks them
         up via a separate ``*.mmproj`` pass.
      2. The ``mmproj`` token anywhere in a ``.gguf`` filename — covers the
         standard ``mmproj-…`` prefix and the embedded ``…-mmproj-…`` form
         the MXFP4 GGUFs use.
    """
    low = name.lower()
    if low.endswith(".mmproj"):
        return True
    return bool(_MMPROJ_TOKEN_RE.search(low))


def _normalize_mmproj(filename: str) -> str:
    """Normalize an mmproj filename to its bare model base for matching.

    The ``mmproj`` marker is removed wherever it appears — not only as a
    leading ``mmproj-`` / ``mmproj_`` prefix but also embedded mid-name,
    e.g. ``qwen3.6-35b-a3b-mxfp4-moe-mmproj-f16.gguf`` (the projector for
    ``qwen3.6-35b-a3b-mxfp4_moe.gguf``), where the vendor put the quant
    label before the ``mmproj`` token. The previous prefix-only strip left
    those files unmatched, so the model showed up without its projector.
    """
    base = _strip_quant(filename)
    # Remove a leading mmproj prefix first (mmproj-… / mmproj_…).
    base = re.sub(r"^mmproj[-_.]", "", base, flags=re.IGNORECASE)
    # Then remove any embedded/trailing mmproj token (…-mmproj-… / …-mmproj).
    base = re.sub(r"[-_.]mmproj(?=[-_.]|$)", "", base, flags=re.IGNORECASE)
    return base.lower().rstrip(".-_")


def _find_mmproj(model: Path, candidates: List[Path]) -> Optional[Path]:
    """Pick the most-specific mmproj that matches the given model.

    A candidate matches if its normalized base is a prefix of the model's
    normalized name (same directory only). The longest matching prefix wins.
    """
    ranked = _find_mmproj_candidates(model, candidates)
    return ranked[0] if ranked else None


# Precision tokens we use ONLY as a tie-breaker when two projectors match
# the same model with an equally-long name prefix (e.g. a model ships
# mmproj-…-bf16, mmproj-…-f16 and mmproj-…-f32 side by side). Without an
# explicit user choice we prefer the higher-precision file, because the
# projector is small and quality matters more than the few hundred MB
# saved — this also flips Basti's complaint where bf16 was always picked
# first purely by sort order. The GUI dropdown lets the user override.
_MMPROJ_PRECISION_RANK = {
    "f32": 3,
    "fp32": 3,
    "f16": 2,
    "fp16": 2,
    "bf16": 1,
}


def _mmproj_precision_score(name: str) -> int:
    low = name.lower()
    for tok, score in _MMPROJ_PRECISION_RANK.items():
        if re.search(rf"(?<![a-z0-9]){tok}(?![a-z0-9])", low):
            return score
    return 0


def _find_mmproj_candidates(model: Path, candidates: List[Path]) -> List[Path]:
    """Return ALL mmproj projectors that match ``model``, best-first.

    Same matching rule as :func:`_find_mmproj` (normalized base of the
    projector is a prefix of the model's normalized name, same directory
    only). The list is sorted by: (1) longest matching prefix first — the
    most-specific projector — then (2) higher precision first (f32 > f16 >
    bf16) as a tie-breaker, then (3) filename for a stable order. The GUI
    exposes this list as a manual dropdown so the user can switch between
    precisions instead of being stuck with whatever sorted first.
    """
    model_norm = _normalize_model(model.name)
    model_canon = _canonical_sep(model_norm)
    scored: List[Tuple[int, int, str, Path]] = []
    for c in candidates:
        if c.parent != model.parent:
            continue
        c_norm = _normalize_mmproj(c.name)
        if not c_norm:
            continue
        c_canon = _canonical_sep(c_norm)
        # Separator-tolerant, BIDIRECTIONAL prefix match. Normally the
        # projector base is a prefix of the model name. But the two sides
        # don't always strip the same tokens: for the mxfp4 pair the model
        # ("qwen3.6-35b-a3b-mxfp4_moe") loses "mxfp4_moe" as a quant tail
        # while the projector ("…-mxfp4-moe-mmproj-f16") keeps "mxfp4-moe"
        # (its quant token is the trailing "f16"), leaving the PROJECTOR
        # base longer than the model base. Accept the match when either
        # canonical base is a prefix of the other, so the pair survives
        # regardless of which side carried the extra quant token.
        if model_canon.startswith(c_canon) or c_canon.startswith(model_canon):
            # Rank by the length of the OVERLAP (the shorter of the two
            # bases), so a more-specific projector still outranks a generic
            # one and an unrelated short prefix can't hijack the pairing.
            overlap = min(len(model_canon), len(c_canon))
            scored.append(
                (overlap, _mmproj_precision_score(c.name), c.name.lower(), c)
            )
    # Sort: longest prefix first, then highest precision, then name.
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [t[3] for t in scored]


# ---------------------------------------------------------------------------
# Draft / assistant pairing
#
# Speculative decoding (and llama.cpp's `--model-draft` flag) needs a
# small "assistant" sibling that shares the main model's tokenizer.
# Distributors like Unsloth, Bartowski, and ggml-org publish these as
# files named e.g. "Qwen3.6-32B-Assistant-Q4_K_M.gguf" alongside the
# main "Qwen3.6-32B-Q4_K_M.gguf". They aren't useful on their own —
# loading just the draft yields gibberish — so the GUI/Terminal must
# never offer them as standalone choices, mirroring the mmproj rule.

# Filename markers that identify a draft/assistant model.
_DRAFT_FILENAME_TOKENS = ("assistant", "draft")


def _is_draft_filename(name: str) -> bool:
    """Cheap pre-filter: does this filename look like a draft/assistant file?"""
    n = name.lower()
    # Match token surrounded by separators OR at the end of stem
    # (e.g. "qwen3.5-30b-a3b-assistant-q4_k_m.gguf" is a draft;
    #  "rooks_assistant_v2.gguf" — a fictional case — would also match,
    #  which is acceptable, false-positives just cost a draft pairing).
    for tok in _DRAFT_FILENAME_TOKENS:
        if (
            re.search(rf"[-_.]{tok}[-_.]", n)
            or n.startswith(tok + "-")
            or n.startswith(tok + "_")
        ):
            return True
    return False


def _strip_draft_token(stem: str) -> str:
    """Remove ``-assistant-…`` / ``_draft_…`` segments from a filename stem
    so the remaining base can be matched against the main model's stem.
    """
    s = stem.lower()
    for tok in _DRAFT_FILENAME_TOKENS:
        # remove ``-assistant`` segment (and any quant tail attached to it)
        s = re.sub(rf"[-_.]{tok}(?=[-_.]|$)", "", s)
    return s.strip("-_.")


def _find_draft(model: Path, candidates: List[Path]) -> Optional[Path]:
    """Pick the smallest draft whose base prefix matches the main model.

    Same directory only — drafts only count if they sit beside the main
    model. Among multiple matches, pick the smallest on disk (drafts are
    speculative; smaller is faster to evaluate).
    """
    main_norm = _normalize_model(model.name)
    best: Optional[Path] = None
    best_size = -1
    for c in candidates:
        if c.parent != model.parent:
            continue
        # Normalize the draft: strip its quant tail AND the assistant/draft token,
        # then check whether the result is a prefix of the main model's base.
        c_base = _strip_quant(c.name).lower()
        c_norm = _strip_draft_token(c_base.removesuffix(".gguf"))
        if not c_norm:
            continue
        if not main_norm.startswith(c_norm + "-") and main_norm != c_norm:
            continue
        try:
            sz = c.stat().st_size
        except OSError:
            continue
        if best is None or sz < best_size:
            best = c
            best_size = sz
    return best


def scan_models(
    root: Path,
    read_metadata: bool = True,
) -> List[ModelEntry]:
    """Walk `root` recursively and return all loadable GGUF models.

    Multi-part (sharded) GGUFs produced by ``llama-gguf-split`` — e.g.
    ``model-00001-of-00003.gguf`` — are merged into a single
    :class:`ModelEntry` whose :attr:`~ModelEntry.path` points to shard 1,
    :attr:`~ModelEntry.size_bytes` is the sum of all shards, and
    :attr:`~ModelEntry.part_paths` lists every shard in index order.

    Two kinds of files get filtered out of the main list and attached
    to their "big-brother" model instead:
      * mmproj projectors (vision encoders) → :attr:`ModelEntry.mmproj`
      * assistant / draft models             → :attr:`ModelEntry.draft`

    Both kinds are useless on their own — loading a bare mmproj file
    fails outright, and a draft model alone produces garbage — so the
    UI should never present them as choosable models.
    """
    if not root.exists() or not root.is_dir():
        return []

    # Projectors can ship either as ".gguf" (the common case) or with a
    # ".mmproj" extension (some audio projectors, e.g. LFM2.5-Audio). The
    # ".mmproj" files are NOT matched by the "*.gguf" glob, so collect them
    # explicitly and feed them into the same projector pool.
    all_gguf = list(root.rglob("*.gguf"))
    all_mmproj_ext = list(root.rglob("*.mmproj"))
    mmprojs: List[Path] = list(all_mmproj_ext)
    drafts: List[Path] = []
    models: List[Path] = []
    for f in all_gguf:
        if _is_mmproj_filename(f.name):
            mmprojs.append(f)
        elif _is_draft_filename(f.name):
            drafts.append(f)
        else:
            models.append(f)

    # ------------------------------------------------------------------
    # Separate single-file models from multi-part (sharded) GGUFs.
    # Split key: (parent_dir_str, base_stem) → {part_index: Path}
    # ------------------------------------------------------------------
    split_parts: Dict[Tuple[str, str], Dict[int, Path]] = {}
    single_models: List[Path] = []

    for m in models:
        info = _split_gguf_key(m.name)
        if info is None:
            single_models.append(m)
        else:
            base, part_idx, _total = info
            split_key = (str(m.parent), base)
            split_parts.setdefault(split_key, {})[part_idx] = m

    def _group_for(path: Path) -> str:
        """Return the group label (relative sub-directory) for *path*."""
        try:
            rel = path.relative_to(root)
            rel_parts = rel.parts
            return "/".join(rel_parts[:-1]) if len(rel_parts) > 1 else "."
        except ValueError:
            return str(path.parent)

    entries: List[ModelEntry] = []

    # --- Single-file models -------------------------------------------
    for m in sorted(single_models):
        try:
            size = m.stat().st_size
        except OSError:
            continue
        md = read_gguf_metadata(m) if read_metadata else {}
        entries.append(
            ModelEntry(
                path=m,
                name=m.stem,
                group=_group_for(m),
                size_bytes=size,
                mmproj=_find_mmproj(m, mmprojs),
                mmproj_candidates=_find_mmproj_candidates(m, mmprojs),
                draft=_find_draft(m, drafts),
                metadata=md,
                part_paths=[m],
            )
        )

    # --- Multi-part (sharded) models ----------------------------------
    for (parent_str, base), parts_dict in sorted(
        split_parts.items(), key=lambda kv: kv[0][1].lower()
    ):
        # Use shard 1 as the primary path (llama.cpp auto-discovers the rest).
        # Fall back to the lowest-indexed shard if shard 1 is missing.
        part1 = parts_dict.get(1) or parts_dict[min(parts_dict)]
        total_size = 0
        for p in parts_dict.values():
            try:
                total_size += p.stat().st_size
            except OSError:
                pass
        if total_size == 0:
            continue
        ordered_parts = [parts_dict[i] for i in sorted(parts_dict)]
        # Build a synthetic Path whose .name == "<base>.gguf" so the
        # mmproj / draft pairing functions get the correct base stem.
        pairing_path = part1.parent / (base + ".gguf")
        md = read_gguf_metadata(part1) if read_metadata else {}
        entries.append(
            ModelEntry(
                path=part1,
                name=base,
                group=_group_for(part1),
                size_bytes=total_size,
                mmproj=_find_mmproj(pairing_path, mmprojs),
                mmproj_candidates=_find_mmproj_candidates(pairing_path, mmprojs),
                draft=_find_draft(pairing_path, drafts),
                metadata=md,
                part_paths=ordered_parts,
            )
        )

    return entries


def group_entries(entries: List[ModelEntry]) -> Dict[str, List[ModelEntry]]:
    """Group entries by their `group` field, preserving discovery order."""
    out: Dict[str, List[ModelEntry]] = {}
    for e in entries:
        out.setdefault(e.group, []).append(e)
    return out
