"""Scan a models folder for GGUF files, pair them with mmproj projectors,
and pull a few useful fields from GGUF metadata when available.

GGUF format reference (v3): https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
We read only the header (KV pairs), never tensor data, so this is fast even
for 100+ GB files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


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
        # Token vocab arrays can be huge — skip them silently. Fixed-width
        # scalar arrays can be skipped in one seek instead of one Python call
        # per element (important for metadata-heavy model collections).
        if not want_array_elements or n > 256:
            scalar = _SCALAR_FMT.get(atype)
            if scalar is not None:
                f.seek(scalar[1] * n, os.SEEK_CUR)
            elif atype == _GT_STRING:
                # Tokenizer vocab/merges: ~150k strings each. Seek past the
                # bytes instead of reading and decoding every element.
                for _ in range(n):
                    ln = struct.unpack("<Q", f.read(8))[0]
                    f.seek(ln, os.SEEK_CUR)
            else:
                for _ in range(n):
                    _read_value(f, atype, want_array_elements=False)
            return None
        return [_read_value(f, atype, True) for _ in range(n)]
    raise ValueError(f"Unknown GGUF value type {vtype}")


# Pre-compiled: match "blk.{N}." tensor names — used by MTP tensor scan.
_BLK_IDX_RE = re.compile(r"^blk\.(\d+)\.")

# llama.cpp b10653+ can leave selected giant row-gather tensors mmap-backed and
# read their rows on demand. qwen4exp and Gemma 4 currently mark this exact
# tensor; ``auto`` applies only above 4 GiB. Its on-disk bytes must not be
# mistaken for ordinary layer weights that can be split across GPUs/CPU.
_READ_LAZY_TENSOR_NAMES = frozenset({"per_layer_token_embd.weight"})
_READ_LAZY_AUTO_MIN_BYTES = 4 * 1024**3


def _read_lazy_tensor_span_bytes(
    tensor_offsets: List[Tuple[str, int]], data_start: int, file_size: int
) -> int:
    """Return storage bytes of llama.cpp auto-lazy tensors in one GGUF.

    GGUF tensor offsets are relative to the aligned data section. Using the
    next tensor offset (or EOF for the final tensor) gives an exact bounded
    storage span without duplicating ggml's growing quant-type table here.
    Alignment padding is at most a few bytes and is intentionally included.
    """
    ordered = sorted(
        ((name.lower(), max(0, int(offset))) for name, offset in tensor_offsets),
        key=lambda item: item[1],
    )
    total = 0
    data_bytes = max(0, int(file_size) - max(0, int(data_start)))
    for index, (name, offset) in enumerate(ordered):
        if name not in _READ_LAZY_TENSOR_NAMES:
            continue
        next_offset = ordered[index + 1][1] if index + 1 < len(ordered) else data_bytes
        span = max(0, min(data_bytes, next_offset) - min(data_bytes, offset))
        if span > _READ_LAZY_AUTO_MIN_BYTES:
            total += span
    return total


def _moe_expert_spans(
    tensor_offsets: List[Tuple[str, int]],
    data_start: int,
    file_size: int,
    n_layers: int,
) -> List[int]:
    """Storage moved by --n-cpu-moe, indexed by block; no tensor data read.

    Include alignment padding conservatively. Shared experts, attention and
    embeddings are NOT movable with --n-cpu-moe. Invalid spans disable the
    precise estimate rather than silently falling back to zero-sized weights.
    """
    if not 0 < n_layers <= 4096:
        return []
    ordered = sorted(tensor_offsets, key=lambda item: item[1])
    data_bytes = file_size - data_start
    result = [0] * n_layers
    for index, (name, offset) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else data_bytes
        if not 0 <= offset < end <= data_bytes:
            return []
        # Same tensor family as llama.cpp's LLM_FFN_EXPS_REGEX (incl. chexps).
        match = re.match(
            r"^blk\.(\d+)\.ffn_(?:gate|up|down|gate_up)_(?:ch)?exps\.", name
        )
        if match and int(match.group(1)) < n_layers:
            result[int(match.group(1))] += end - offset
    return result if any(result) else []


def _read_gguf_metadata_uncached(path: Path) -> Dict[str, Any]:
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

      ``__ggml_types__`` — sorted, bounded list of the distinct ggml tensor
        type ids seen in the tensor-info section. Fork-private weight
        formats live outside mainline's ``[0, GGML_TYPE_COUNT)`` range
        (ROCmFPX 100..111, PrismML 141..143), so the launch preflight can
        refuse them on a stock runtime without reopening the file.

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
            has_dspark_tensors = False
            max_block_index = -1
            # Lowest ``blk.N`` index; an MTP-only head (no trunk) starts at
            # the NextN block instead of blk.0.
            min_block_index = -1
            scan_complete = False
            tensor_offsets: List[Tuple[str, int]] = []
            # Root (non-``blk.N``) tensor names. llama.cpp loads a standalone
            # MTP/NextN sidecar with the full architecture loader, which
            # requires the same root tensors as the target model, so this list
            # lets the launch preflight reject an incomplete sidecar before
            # llama-server aborts with ``check_tensor_dims``.
            root_tensors: List[str] = []
            ggml_types: Set[int] = set()
            try:
                for _ in range(n_tensors):
                    tname_len = struct.unpack("<Q", f.read(8))[0]
                    tname = f.read(tname_len).decode("utf-8", errors="replace")
                    n_dims = struct.unpack("<I", f.read(4))[0]
                    # Dimensions/type are not needed for the metadata scan, but
                    # retain each relative data offset so giant auto-lazy row
                    # tables can be sized from the next tensor boundary.
                    f.read(8 * n_dims)
                    ggml_type = struct.unpack("<I", f.read(4))[0]
                    if len(ggml_types) < 64:
                        ggml_types.add(ggml_type)
                    tensor_offset = struct.unpack("<Q", f.read(8))[0]
                    tensor_offsets.append((tname, tensor_offset))
                    tl = tname.lower()
                    if not tname.startswith("blk.") and len(root_tensors) < 512:
                        root_tensors.append(tname)
                    # DSpark uses the DFlash architecture plus an additional
                    # Markov/confidence head, so ``general.architecture`` alone
                    # cannot distinguish it. b10329 names those root tensors
                    # markov_w1 / markov_w2 / conf_proj. Retain a synthetic
                    # marker so command generation can select draft-dspark.
                    if not has_dspark_tensors and (
                        "markov_w1" in tl
                        or "markov_w2" in tl
                        or "conf_proj" in tl
                        or "markov_head" in tl
                    ):
                        has_dspark_tensors = True
                    block_match = _BLK_IDX_RE.match(tname)
                    if block_match:
                        try:
                            block_index = int(block_match.group(1))
                            max_block_index = max(max_block_index, block_index)
                            if min_block_index < 0 or block_index < min_block_index:
                                min_block_index = block_index
                        except (TypeError, ValueError):
                            pass
                    if not has_mtp_tensors:
                        # (a) Name-based: the canonical llama.cpp nextn tensors
                        #     are named "blk.{N}.nextn.*" (eh_proj, embed_tokens,
                        #     enorm, hnorm, shared_head_*). This catch is
                        #     independent of block_count, so it works even when
                        #     block_count is unreadable or the block is numbered
                        #     unexpectedly. Some forks emit "mtp" in the name.
                        if "nextn" in tl or "mtp" in tl:
                            has_mtp_tensors = True
                        # (b) Index-based: a block index at/after block_count is
                        #     an extra draft head grafted past the main stack.
                        elif block_count > 0 and max_block_index >= block_count:
                            has_mtp_tensors = True
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
            if has_dspark_tensors:
                md["__dspark_scan__"] = "found"
            # Preserve enough bounded scan evidence for ``scan_models`` to
            # combine all headers of a split GGUF. This distinguishes a real
            # last-shard MTP block from stale ``nextn_predict_layers`` metadata.
            md["__tensor_scan_complete__"] = scan_complete
            md["__max_block_index__"] = max_block_index
            md["__min_block_index__"] = min_block_index
            md["__root_tensors__"] = root_tensors
            md["__ggml_types__"] = sorted(ggml_types)

            if scan_complete and tensor_offsets:
                try:
                    alignment = max(1, int(md.get("general.alignment", 32) or 32))
                except (TypeError, ValueError):
                    alignment = 32
                data_start = ((f.tell() + alignment - 1) // alignment) * alignment
                lazy_bytes = _read_lazy_tensor_span_bytes(
                    tensor_offsets,
                    data_start,
                    os.fstat(f.fileno()).st_size,
                )
                if lazy_bytes > 0:
                    md["__read_lazy_tensor_bytes__"] = lazy_bytes
                # A partial shard is not a complete placement map. Keep the
                # heuristic for split files until all shard maps are merged.
                if not is_sharded:
                    expert_bytes = _moe_expert_spans(
                        tensor_offsets,
                        data_start,
                        os.fstat(f.fileno()).st_size,
                        block_count,
                    )
                    if expert_bytes:
                        md["__moe_expert_bytes_by_layer__"] = expert_bytes
                        ordered = sorted(tensor_offsets, key=lambda item: item[1])
                        for i, (name, start) in enumerate(ordered):
                            if name == "token_embd.weight":
                                end = (
                                    ordered[i + 1][1]
                                    if i + 1 < len(ordered)
                                    else os.fstat(f.fileno()).st_size - data_start
                                )
                                md["__input_embedding_bytes__"] = max(0, end - start)
                                md["__tied_output_embedding__"] = (
                                    "output.weight" not in root_tensors
                                )
                                break

            return md
    except (OSError, struct.error, EOFError, ValueError, UnicodeDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Metadata cache + bounded parallel reader

# Schema 6 adds MoE expert spans and input/tied-output storage accounting.
# Schema 7 adds the lowest block index (MTP-only head detection).
_METADATA_CACHE_SCHEMA = 7
_METADATA_CACHE_MAX_ENTRIES = 2048
_METADATA_CACHE_MAX_BYTES = 64 * 1024 * 1024
_METADATA_CACHE_LOCK = threading.RLock()
_METADATA_CACHE_LOADED = False
_METADATA_CACHE_DIRTY = False
_METADATA_CACHE_ENTRIES: Dict[str, Dict[str, Any]] = {}
_METADATA_CACHE_HITS = 0
_METADATA_CACHE_MISSES = 0


def _metadata_cache_path() -> Optional[Path]:
    """Return the private persistent cache path, or ``None`` when disabled."""
    explicit = os.environ.get("AUTOTUNER_METADATA_CACHE", "").strip()
    if explicit.lower() in {"0", "off", "false", "none"}:
        return None
    if explicit:
        return Path(explicit).expanduser()
    # Pytest scans many temporary fixtures. Keep those in memory unless a test
    # explicitly supplies its own cache path, never in the developer's home.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    override = os.environ.get("AUTOTUNER_DATA_DIR", "").strip()
    base = Path(override).expanduser() if override else Path.home() / ".autotuner"
    return base / "cache" / "model-metadata-v3.json"


def _metadata_cache_key(path: Path) -> Optional[str]:
    try:
        stat = path.stat()
        resolved = path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None
    identity = (
        f"{os.path.normcase(str(resolved))}\0{stat.st_size}\0{stat.st_mtime_ns}"
        f"\0parser-{_METADATA_CACHE_SCHEMA}"
    )
    return hashlib.sha256(identity.encode("utf-8", errors="surrogatepass")).hexdigest()


def _load_metadata_cache() -> None:
    global _METADATA_CACHE_LOADED, _METADATA_CACHE_ENTRIES
    with _METADATA_CACHE_LOCK:
        if _METADATA_CACHE_LOADED:
            return
        _METADATA_CACHE_LOADED = True
        path = _metadata_cache_path()
        if path is None:
            return
        try:
            if path.stat().st_size > _METADATA_CACHE_MAX_BYTES:
                return
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != _METADATA_CACHE_SCHEMA
        ):
            return
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, dict):
            return
        valid: Dict[str, Dict[str, Any]] = {}
        for key, value in raw_entries.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            metadata = value.get("metadata")
            if not isinstance(metadata, dict):
                continue
            valid[key] = {
                "metadata": metadata,
                "cached_at": int(value.get("cached_at", 0) or 0),
            }
        if len(valid) > _METADATA_CACHE_MAX_ENTRIES:
            ordered = sorted(
                valid.items(),
                key=lambda item: int(item[1].get("cached_at", 0)),
                reverse=True,
            )
            valid = dict(ordered[:_METADATA_CACHE_MAX_ENTRIES])
        _METADATA_CACHE_ENTRIES = valid


def read_gguf_metadata(path: Path) -> Dict[str, Any]:
    """Read metadata with stat-based in-memory and persistent caching.

    The signature includes normalized path, byte size, nanosecond mtime, and
    parser schema. Replacing or editing a GGUF therefore invalidates the entry;
    unchanged models avoid repeating the expensive tensor-name scan on every
    application start and performance-suite preparation.
    """
    global _METADATA_CACHE_DIRTY, _METADATA_CACHE_HITS, _METADATA_CACHE_MISSES
    model_path = Path(path)
    key = _metadata_cache_key(model_path)
    if key is None:
        return {}
    _load_metadata_cache()
    with _METADATA_CACHE_LOCK:
        cached = _METADATA_CACHE_ENTRIES.get(key)
        if cached is not None:
            metadata = cached.get("metadata")
            if isinstance(metadata, dict):
                _METADATA_CACHE_HITS += 1
                return dict(metadata)
        _METADATA_CACHE_MISSES += 1

    metadata = _read_gguf_metadata_uncached(model_path)
    with _METADATA_CACHE_LOCK:
        # Another scan thread may have completed the same path meanwhile; the
        # deterministic parse result is equivalent, so replacing is harmless.
        _METADATA_CACHE_ENTRIES[key] = {
            "metadata": dict(metadata),
            "cached_at": int(time.time()),
        }
        _METADATA_CACHE_DIRTY = True
    return dict(metadata)


def _metadata_worker_count(item_count: int) -> int:
    if item_count <= 1:
        return 1
    raw = os.environ.get("AUTOTUNER_SCAN_WORKERS", "").strip()
    try:
        requested = int(raw) if raw else 0
    except ValueError:
        requested = 0
    if requested <= 0:
        requested = min(8, max(2, os.cpu_count() or 2))
    return max(1, min(item_count, requested, 32))


def _read_metadata_many(paths: Iterable[Path]) -> Dict[Path, Dict[str, Any]]:
    ordered = list(dict.fromkeys(Path(path) for path in paths))
    if not ordered:
        return {}
    workers = _metadata_worker_count(len(ordered))
    if workers <= 1:
        return {path: read_gguf_metadata(path) for path in ordered}
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="autotuner-metadata"
    ) as executor:
        values = list(executor.map(read_gguf_metadata, ordered))
    return dict(zip(ordered, values))


def read_gguf_metadata_many(paths: Iterable[Path]) -> Dict[Path, Dict[str, Any]]:
    """Read independent GGUF headers concurrently with deterministic results."""
    return _read_metadata_many(paths)


def flush_gguf_metadata_cache() -> None:
    """Atomically persist bounded cache entries after a completed scan."""
    global _METADATA_CACHE_DIRTY, _METADATA_CACHE_ENTRIES
    _load_metadata_cache()
    path = _metadata_cache_path()
    with _METADATA_CACHE_LOCK:
        if not _METADATA_CACHE_DIRTY:
            return
        if path is None:
            _METADATA_CACHE_DIRTY = False
            return
        ordered = sorted(
            _METADATA_CACHE_ENTRIES.items(),
            key=lambda item: int(item[1].get("cached_at", 0)),
            reverse=True,
        )[:_METADATA_CACHE_MAX_ENTRIES]
        entries = dict(ordered)

    payload = {"schema": _METADATA_CACHE_SCHEMA, "entries": entries}
    try:
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), default=str
        )
        # A few unusually large chat templates must not let a cache grow without
        # bound. Retain the newest half repeatedly until it fits the hard cap.
        while (
            len(encoded.encode("utf-8")) > _METADATA_CACHE_MAX_BYTES
            and len(entries) > 1
        ):
            entries = dict(list(entries.items())[: max(1, len(entries) // 2)])
            payload["entries"] = entries
            encoded = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), default=str
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(encoded, encoding="utf-8")
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError):
        return
    with _METADATA_CACHE_LOCK:
        _METADATA_CACHE_ENTRIES = entries
        _METADATA_CACHE_DIRTY = False


def metadata_cache_stats() -> Dict[str, int]:
    """Return process-local cache counters for diagnostics and tests."""
    with _METADATA_CACHE_LOCK:
        return {
            "entries": len(_METADATA_CACHE_ENTRIES),
            "hits": _METADATA_CACHE_HITS,
            "misses": _METADATA_CACHE_MISSES,
            "workers": _metadata_worker_count(max(1, len(_METADATA_CACHE_ENTRIES))),
        }


def _reset_metadata_cache_for_tests() -> None:
    global _METADATA_CACHE_LOADED, _METADATA_CACHE_DIRTY
    global _METADATA_CACHE_ENTRIES, _METADATA_CACHE_HITS, _METADATA_CACHE_MISSES
    with _METADATA_CACHE_LOCK:
        _METADATA_CACHE_LOADED = False
        _METADATA_CACHE_DIRTY = False
        _METADATA_CACHE_ENTRIES = {}
        _METADATA_CACHE_HITS = 0
        _METADATA_CACHE_MISSES = 0


def metadata_is_standalone_drafter(md: Dict[str, Any]) -> bool:
    """Return True iff the GGUF is a *standalone* speculative-draft head.

    These are the small "assistant" / MTP-head files that ship as a
    SEPARATE GGUF next to a normal target model (Gemma 4 MTP, e.g.
    ``google/gemma-4-12B-it-assistant`` and the Frankenmerger
    ``…-MTP-BF16`` / ``mtp-…`` builds). They are NOT runnable on their own
    — stock llama.cpp aborts with ``unknown model architecture:
    'gemma4-assistant'`` / ``Gemma4Assistant requires ctx_other to be
    set`` — so they must be attached to their big-brother target and
    passed via ``-md`` + ``--spec-type draft-mtp``, never offered as a
    choosable model. This mirrors the mmproj rule.

    The authoritative signal is the architecture name: the Gemma 4 MTP
    drafter declares ``general.architecture = gemma4-assistant`` (llama.cpp
    PR #23398 — ``LLM_ARCH_GEMMA4_ASSISTANT``, renamed from
    ``GEMMA4_MTP``). We generalise to any architecture whose name ends in
    ``-assistant`` / ``_assistant`` so future vendors' draft-head archs are
    caught without a code change. A standalone drafter is additionally
    tiny (here: 4 blocks, ``general.size_label`` like ``423M``) and always
    carries ``<arch>.nextn_predict_layers > 0`` — but the arch suffix
    alone is the reliable discriminator and is what we key on.

    NOTE — this is deliberately distinct from
    :func:`metadata_has_embedded_mtp`, which detects a *target* model that
    has an MTP head fused INSIDE its own GGUF (Qwen3.6-27B-MTP, run via
    ``--spec-type draft-mtp`` with no second file). A standalone drafter is
    the opposite case: the head lives in its own file. ``has_embedded_mtp``
    must NOT fire on a standalone drafter (it would mislabel the bare
    drafter as a self-speculating target), so callers check
    :func:`metadata_is_standalone_drafter` first and treat the file as a
    draft, not a model.
    """
    if not md:
        return False
    arch = str(md.get("general.architecture", "") or "").lower().strip()
    return arch.endswith("-assistant") or arch.endswith("_assistant")


# Architecture strings that mark a GGUF as a *standalone drafter* file — i.e.
# not runnable as a target model on its own, but loaded via ``-md`` next to a
# target. Used by the scan reclassification so such files are paired to a
# target instead of appearing in the chooser. Covers the Gemma 4 MTP
# assistant heads AND the EAGLE-3 / DFlash draft models (PR #18039 / #22105).
_DRAFTER_ARCHS = frozenset({"eagle3", "dflash", "dspark"})


def metadata_is_mtp_only_head(md: Dict[str, Any]) -> bool:
    """True for a same-architecture MTP/NextN head GGUF without a trunk.

    Converters write separate MTP heads with the target's architecture
    (``qwen35``, ``qwen4exp``, ``mimo2``) and ``<arch>.block_count`` of the
    full model plus ``<arch>.nextn_predict_layers``, but store only the
    NextN block(s) and a few root tensors. Such a file cannot serve requests
    by itself; it is a ``-md`` draft for its target. The tensor layout is
    the evidence, independent of the file name: the original EschaLabs head
    ``Escha-Qwen3.8-27B-W2-MTP-F16-headQ4.gguf`` (2.9 GB, ``blk.64`` only)
    carries no ``mtp-`` prefix and exceeds the infix-name size guard.

    Only a complete scan of a single-file GGUF qualifies; a shard or an
    unreadable tensor section never does.
    """
    if not md or not md.get("__tensor_scan_complete__"):
        return False
    try:
        if int(md.get("split.count", 0) or 0) > 1:
            return False
    except (TypeError, ValueError):
        return False
    arch = str(md.get("general.architecture", "") or "").strip()
    if not arch:
        return False
    try:
        block_count = int(md.get(f"{arch}.block_count", 0) or 0)
        nextn = int(md.get(f"{arch}.nextn_predict_layers", 0) or 0)
        min_block = int(md.get("__min_block_index__", -1))
    except (TypeError, ValueError):
        return False
    if nextn <= 0 or block_count <= nextn or min_block < 0:
        return False
    return min_block >= block_count - nextn


def metadata_is_drafter_file(md: Dict[str, Any]) -> bool:
    """True if *md* marks a standalone speculative-draft GGUF of ANY kind.

    This is the union of :func:`metadata_is_standalone_drafter` (Gemma 4
    MTP ``*-assistant`` heads), same-architecture MTP-only heads
    (:func:`metadata_is_mtp_only_head`) and the EAGLE-3 / DFlash draft
    architectures (``general.architecture == 'eagle3'`` / ``'dflash'``). All
    of these are attached to a target via ``-md`` and never listed as a
    choosable model.
    """
    if not md:
        return False
    if metadata_is_standalone_drafter(md) or metadata_is_mtp_only_head(md):
        return True
    arch = str(md.get("general.architecture", "") or "").lower().strip()
    return arch in _DRAFTER_ARCHS


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
# die ganze Qwen3/3.5/3.6/3.8-Familie wäre so vom automatischen YaRN
# ausgeschlossen gewesen (nur noch via rope_scale.enabled=true im Profil).
_ROPE_SCALE_SUPPORTED_ARCHS = frozenset(
    {
        "qwen",  # Qwen / Qwen2 / Qwen2.5 / Qwen3 / Qwen3.5–3.8 family
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
#
# The arch strings here are matched against ``general.architecture`` from
# the GGUF (the runtime reader decodes that to a clean lowercase string).
# As of llama.cpp b9672 ``llm_arch_is_hybrid()`` returns true for a larger
# family than the classic Mamba hybrids — it now also covers the
# linear-/gated-delta-net attention models. We mirror that list so KV is
# only budgeted for the full-attention layers:
#   llama-arch.cpp:  JAMBA, FALCON_H1, PLAMO2, GRANITE_HYBRID, LFM2,
#                    LFM2MOE, NEMOTRON_H, NEMOTRON_H_MOE, QWEN3NEXT,
#                    KIMI_LINEAR, KIMI_K3, BAILINGMOE3, QWEN35, QWEN35MOE
# Qwen3.5/3.6/3.8 use SSM-style metadata (ssm.conv_kernel / ssm.state_size /
# ssm.group_count) so the generic ``.ssm.`` fallback in
# ``metadata_is_hybrid_architecture`` already catches them — but listing
# them explicitly is cheaper and survives GGUFs that name the recurrent
# keys differently. The exact recurrent-layer count is read from the
# ``<arch>.attention.recurrent_layers`` key (b9672) in
# ``metadata_attention_layer_count`` below; the names here are only the
# hybrid *gate*, not the count.
_RECURRENT_ARCHS = frozenset(
    {"mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7"}
)


_HYBRID_ARCHS = frozenset(
    {
        "nemotron_h",
        "nemotron-h",
        "nemotron_h_moe",  # Nemotron-H MoE (b9672 hybrid)
        "nemotron-h-moe",
        "granitemoehybrid",
        "granite-h",
        "granite_h",
        "jamba",
        "bamba",
        "bailing_hybrid",  # Ling-2.6 fork: KDA/short-conv + attention
        "bailingmoe2.5",
        "bailingmoe2_5",
        "bailingmoe3",  # Ling 3.0: KDA + gated MLA (b10460)
        "falcon_h1",
        "falcon-h1",  # exact llama.cpp arch string (LLM_ARCH_FALCON_H1)
        "granitehybrid",  # exact llama.cpp arch string (LLM_ARCH_GRANITE_HYBRID)
        "deepseek4",  # DeepSeek-V4: llm_arch_is_hybrid() since the hashed
        # sliding-window/recurrent-state design; KV accounting stays
        # full-layer because no recurrent_layers key is written (see
        # metadata_attention_layer_count).
        "plamo2",  # Plamo-2 hybrid
        "zamba2",  # Zamba2 hybrid
        "lfm2",  # LFM2 dense hybrid (short-conv + attention)
        "lfm2moe",  # LFM2.5-MoE hybrid (b9672 hybrid)
        "lfm3",  # LFM3 (b11938): short-conv + SWA/global attention
        "metis16",  # Metis-1.6 (b12089): Mamba-2 + GQA, depth recurrence
        "clover_moe",  # Clover (b12060): Qwen3.6-MoE GDN hybrid
        "deepseek41",  # DeepSeek-V4.1 (b11870): like deepseek4, full-layer KV
        "qwen3next",  # Qwen3-Next: gated-delta-net + full attention
        "kimi-linear",  # Kimi-Linear hybrid
        "kimi_linear",
        "kimi-k3",  # Kimi-K3: KDA + gated MLA (b10448, text path)
        "kimi_k3",
        "kimi-k31",  # Kimi-K3.1 (b12180): 69 KDA + 24 gated-DSA layers
        "glm5next",  # GLM-5.3-Flash KDA/DSA (full-layer safety accounting below)
        "glm5-next",
        "hyv4",  # HY4 Gated DSA/IndexCache (fork-only allocation contract)
        "hy_v4",
        "qwen35",  # Qwen3.5–3.8 dense: linear + full attention
        "qwen35moe",  # Qwen3.5–3.8 MoE: linear + full attention
        "qwen4exp",  # Qwen3.8 Flash Next preview: GDN + 1-in-4 attention
        "minimax-01",  # MiniMax-Text-01: lightning + full attention (b10441)
        "minimax_01",
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
    if arch in _RECURRENT_ARCHS or arch in _HYBRID_ARCHS:
        return True
    # Generic SSM-state detection — any *.ssm.* key signals hybrid.
    for k in md.keys():
        if ".ssm." in k:
            return True
    return False


# ---------------------------------------------------------------------------
# Diffusion-LLM detection
#
# Diffusion text models (Dream, LLaDA, LLaDA-MoE, RND1 in llama.cpp
# mainline; DiffusionGemma in a dedicated fork) are NOT served by
# ``llama-server``. As of b9700 the server has no diffusion code path at
# all — these run only through the single-shot ``llama-diffusion-cli``
# example binary (prompt in → text out → exit), with their own
# ``--diffusion-*`` flags and no OpenAI API / /health endpoint.
#
# llama.cpp's own classifier ``llm_arch_is_diffusion()`` returns true for
# dream / llada / llada-moe / rnd1. ``diffusion-gemma`` is fork-only (not
# in mainline's arch table) so we add it explicitly — the runtime arch
# string is what the fork's converter wrote into general.architecture.
_DIFFUSION_ARCHS = frozenset(
    {
        "dream",
        "llada",
        "llada-moe",
        "llada_moe",
        "rnd1",
        "diffusion-gemma",  # fork-only (Unsloth DiffusionGemma build)
        "diffusion_gemma",
        "diffusiongemma",
    }
)


def metadata_is_diffusion_architecture(md: Dict[str, Any]) -> bool:
    """Detect a diffusion-LLM from GGUF metadata.

    A diffusion model must be launched with ``llama-diffusion-cli`` and a
    set of ``--diffusion-*`` flags rather than ``llama-server``; the caller
    uses this to switch the runner and command builder. Detection is by
    architecture name (the only reliable signal — diffusion GGUFs carry no
    distinctive KV key the way SSM hybrids carry ``*.ssm.*``).
    """
    if not md:
        return False
    arch = str(md.get("general.architecture", "") or "").lower().strip()
    return arch in _DIFFUSION_ARCHS


def metadata_attention_layer_count(md: Dict[str, Any]) -> int:
    """Return the layers whose KV cache grows with the full context.

    For ordinary Transformers this equals ``block_count``. Interleaved
    sliding-window architectures count only their global-attention layers;
    the bounded SWA cache is a small constant handled like Gemma's SWA cache
    in :func:`tuner._kv_per_token_for_interleaved_attention`. Hybrid
    Mamba/Transformer models use explicit recurrent-layer metadata when
    available, then a conservative architecture ratio.

    Returns 0 when the answer can't be determined (caller should treat
    as "use total block count" — i.e. assume non-hybrid).
    """
    if not md:
        return 0
    arch = str(md.get("general.architecture", "") or "")
    total = metadata_layer_count(md)
    if total <= 0:
        return 0

    # HY4 and GLM-5.3-Flash use fork-only DSA/KDA index pools whose exact
    # allocation is not represented by ordinary KV metadata yet. Budget every
    # layer conservatively until a selected runtime exposes a validated pool
    # formula; applying the generic 25% hybrid heuristic could otherwise OOM.
    if arch.lower() in {"hyv4", "hy_v4", "glm5next", "glm5-next"}:
        return total

    # Interleaved sliding-window attention: only global-attention layers
    # grow with the requested context. A per-layer pattern stores True for
    # SWA and False for global attention. Laguna GGUFs currently omit the
    # optional pattern key; llama.cpp's Laguna loader then defaults to a
    # period of 4 with the first layer global (FULL/SWA/SWA/SWA).
    # The 512-token SWA caches remain a small bounded constant, matching the
    # existing Gemma interleaved-attention estimator's treatment.
    sliding_window = md.get(f"{arch}.attention.sliding_window")
    try:
        has_sliding_window = int(sliding_window or 0) > 0
    except (TypeError, ValueError):
        has_sliding_window = False
    if has_sliding_window:
        pattern = md.get(f"{arch}.attention.sliding_window_pattern")
        if isinstance(pattern, (list, tuple)) and len(pattern) >= total:
            n_global = sum(1 for is_swa in pattern[:total] if not bool(is_swa))
            if n_global > 0:
                return n_global
        try:
            period = int(pattern) if pattern is not None else 0
        except (TypeError, ValueError):
            period = 0
        if period <= 0 and arch.lower() == "laguna":
            period = 4
        if period > 1:
            return max(1, (total + period - 1) // period)

    if not metadata_is_hybrid_architecture(md):
        return total  # pure full-attention Transformer

    # Highest priority: the authoritative recurrent-layer count.
    #
    # llama.cpp b9672 added ``LLM_KV_ATTENTION_RECURRENT_LAYERS`` =
    # ``<arch>.attention.recurrent_layers``. For a hybrid model this is the
    # number of linear-/SSM-attention layers that do NOT carry a standard
    # KV cache, so the full-attention (KV-bearing) count is exactly
    # ``block_count - recurrent_layers``. This is precise for the whole
    # Qwen3.5/3.6, LFM2-MoE, Nemotron-H(-MoE) and Qwen3-Next family and
    # replaces the coarse per-arch ratio guess below. We also scan for any
    # ``*.attention.recurrent_layers`` key so a GGUF whose arch prefix does
    # not match (community re-converts) still benefits.
    recurrent = None
    rec_key = f"{arch}.attention.recurrent_layers"
    if rec_key in md:
        recurrent = md.get(rec_key)
    else:
        for k, v in md.items():
            if k.endswith(".attention.recurrent_layers"):
                recurrent = v
                break
    if recurrent is not None:
        try:
            if isinstance(recurrent, (list, tuple)):
                # Current llama.cpp may serialize this as one bool/int per
                # layer (not only as a scalar count), notably MiniMax-Text-01.
                n_rec = sum(1 for value in recurrent[:total] if bool(value))
            else:
                n_rec = int(recurrent)
            # Pure recurrent/SSM models legitimately have n_rec == total and
            # no context-growing attention KV. Return the existing one-layer
            # sentinel so downstream sizing stays non-zero without inventing
            # a 25%-of-layers KV cache.
            if 0 <= n_rec < total:
                return max(1, total - n_rec)
            if n_rec == total and arch.lower() in _RECURRENT_ARCHS:
                return 1
        except (TypeError, ValueError):
            pass

    # Some preview converters describe the interleave directly instead of
    # serializing a recurrent-layer list. Qwen3.8 Flash Next's qwen4exp GGUF,
    # for example, declares full_attention_interval=4: exactly one of every
    # four blocks carries context-growing KV (48 blocks -> 12 attention layers).
    # This remains a fallback because current llama.cpp gives the explicit
    # recurrent-layer list precedence when both keys are present.
    interval_value = md.get(f"{arch}.full_attention_interval")
    if interval_value is not None:
        try:
            interval = int(interval_value)
        except (TypeError, ValueError):
            interval = 0
        if interval > 1:
            return max(1, math.ceil(total / interval))

    # Next: explicit attention-layer-count keys some converters emit.
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
    # estimate ↔ safer placement). This path only runs when the
    # authoritative ``attention.recurrent_layers`` key (b9672) is absent —
    # i.e. older or community GGUFs predating that key.
    arch_l = arch.lower()
    if arch_l in _RECURRENT_ARCHS:
        return 1
    if "qwen3next" in arch_l:
        # Qwen3-Next: gated-delta-net on most layers, full attention on a
        # 1-in-4 minority (~25%).
        ratio = 0.25
    elif arch_l.startswith("qwen35"):
        # Qwen3.5–3.8 dense/MoE: linear attention interleaved with a
        # minority of full-attention layers. Published layouts put full
        # attention at roughly 1 in 4 (~25%); err high for KV safety.
        ratio = 0.25
    elif "lfm2" in arch_l:
        # LFM2 / LFM2.5-MoE: short-conv recurrent blocks with a small
        # number of GQA attention layers (~1 in 6, ~17%).
        ratio = 0.20
    elif ("kimi" in arch_l and "linear" in arch_l) or arch_l in {
        "kimi-k3",
        "kimi_k3",
        "kimi-k31",
    }:
        # Kimi-Linear/K3/K3.1: three KDA layers per attention layer (~25%;
        # K3.1: 24 gated-DSA of 93 layers).
        ratio = 0.25
    elif "bailingmoe3" in arch_l:
        # Ling 3.0 Flash: 35 KDA + 7 gated-MLA layers (one in six).
        ratio = 1.0 / 6.0
    elif "nemotron" in arch_l:
        # Nemotron-H / Nemotron-H-MoE: roughly 1 attention block per 4
        # Mamba blocks.
        ratio = 0.25
    elif "jamba" in arch_l:
        # Jamba: 1 attention per 7 Mamba (~14%).
        ratio = 0.15
    elif "minimax-01" in arch_l or "minimax_01" in arch_l:
        # Upstream fallback marks every eighth layer as full attention.
        ratio = 0.125
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

    # Round upward: these are safety estimates of context-growing KV layers;
    # flooring 93 × 1/4 miscounted Kimi-K3's known 24 MLA layers as 23.
    return max(1, math.ceil(total * ratio))


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
    "granite-4.2",
    "granite4.2",
    "granite-4-2",
    "granite4_2",
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
    "granite-4.2",
    "granite4.2",
    "granite-4-2",
    "granite4_2",
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
    r"|NVFP4|F8_E4M3|FP8"  # NVFP4, OCP FP8 E4M3 (ggml type 43, b12000)
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


def _coerce_positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


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
    # Every mmproj / draft file found in this model's own directory,
    # REGARDLESS of whether it matches this model. The GUI shows these as
    # always-available manual dropdowns so the user can override the auto
    # pick or experiment across models; entries the auto-logic considers
    # incompatible are flagged (not hidden). `mmproj_candidates` / `draft`
    # above remain the auto-resolved subset used for headless launches.
    folder_mmprojs: List[Path] = field(default_factory=list)
    folder_drafts: List[Path] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    part_paths: List[Path] = field(default_factory=list)
    """All shard paths in order; length > 1 for split GGUFs, otherwise [path]."""

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024**3)

    @property
    def read_lazy_size_bytes(self) -> int:
        """Bytes kept as an on-demand mmap row table by current llama.cpp."""
        try:
            value = int((self.metadata or {}).get("__read_lazy_tensor_bytes__", 0))
        except (TypeError, ValueError):
            value = 0
        return max(0, min(self.size_bytes, value))

    @property
    def read_lazy_size_gb(self) -> float:
        return self.read_lazy_size_bytes / (1024**3)

    @property
    def placement_size_gb(self) -> float:
        """Ordinary resident weights eligible for CPU/GPU placement."""
        return max(0, self.size_bytes - self.read_lazy_size_bytes) / (1024**3)

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
    def is_diffusion(self) -> bool:
        """True for diffusion text models (Dream, LLaDA, RND1,
        DiffusionGemma). These must run via ``llama-diffusion-cli``, not
        ``llama-server`` — the launcher routes on this flag."""
        return metadata_is_diffusion_architecture(self.metadata)

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
        # A standalone drafter (its own gemma4-assistant GGUF) is NOT a
        # self-speculating target — its MTP head is the whole file, not an
        # extra block fused into a full model. Never report embedded MTP for
        # it, or the launcher would try to run the bare drafter as a model
        # with --spec-type draft-mtp (it can't run alone). It is handled as a
        # draft attached to its target instead.
        if self.metadata and (
            metadata_is_standalone_drafter(self.metadata)
            or metadata_is_mtp_only_head(self.metadata)
        ):
            return False
        # EAGLE-3 / DFlash / DSpark draft models are drafts by architecture.
        # NVIDIA's LiLiCorr drafters (arch dflash) store their lattice
        # layers as blk.5..7 beyond dflash.block_count = 5, which the tensor
        # scan reports as an MTP block; treating the draft as an MTP head
        # sent "--spec-type draft-mtp" and llama-server stopped with
        # "failed to create MTP context".
        if self.metadata and (
            str(self.metadata.get("general.architecture", "") or "").lower().strip()
            in _DRAFTER_ARCHS
        ):
            return False
        # Primary: authoritative GGUF metadata
        if self.metadata and metadata_has_embedded_mtp(self.metadata):
            return True
        # A complete scan proving absence outranks the filename as well as
        # stale nextn metadata. Otherwise an MTP-named trunk re-enables the
        # head that metadata_has_embedded_mtp deliberately rejected.
        if self.metadata and self.metadata.get("__mtp_scan__") == "absent":
            return False
        # Fallback: filename-based for GGUFs missing the standard key. Guard
        # the "mtp" token so it does NOT fire on a standalone drafter named
        # with an "mtp-" prefix / "-MTP-" infix (e.g. mtp-gemma-4-12B-it-qat-UD,
        # gemma-4-12B-…-MTP-BF16) — those are draft files, matched separately.
        # The size argument keeps a large MTP-named TARGET (integrated draft)
        # from being mistaken for a draft file, so this fallback correctly
        # reports it as embedded-MTP instead.
        if _is_draft_filename(self.name, self.size_bytes):
            return False
        return bool(re.search(r"(?:^|[-_.])mtp(?:[-_.]|$)", self.name, re.IGNORECASE))

    @property
    def is_standalone_drafter(self) -> bool:
        """True if this GGUF is a separate speculative-draft head (Gemma 4
        ``gemma4-assistant`` MTP drafter and similar). Such files attach to a
        target via :attr:`draft` and are launched with ``-md`` +
        ``--spec-type draft-mtp``; they are never choosable on their own.
        Same-architecture MTP-only heads (:func:`metadata_is_mtp_only_head`)
        count as well, whatever their file name."""
        return metadata_is_standalone_drafter(
            self.metadata
        ) or metadata_is_mtp_only_head(self.metadata)

    @property
    def is_dflash2_drafter(self) -> bool:
        """True for a DFlash2 sidecar with convolution + selector weights.

        DFlash2 deliberately keeps ``general.architecture=dflash`` and the
        ``draft-dflash`` CLI type used by first-generation DFlash.  The
        checkpoint is distinguished by its extra GGUF metadata: grouped local
        convolution and the candidate selector. Pre-b10658 stock llama.cpp
        creates the old 58-tensor graph for these 81-tensor sidecars; launch
        preflight uses this property to require current mainline before a
        multi-gigabyte target starts loading.
        """
        if self.architecture.lower().strip() != "dflash":
            return False
        md = self.metadata or {}
        return bool(
            md.get("dflash.conv_kernel_size")
            and md.get("dflash.conv_group_size")
            and md.get("dflash.selector_rank")
            and md.get("dflash.selector_top_k")
        )

    @property
    def is_lilicorr_drafter(self) -> bool:
        """True for an NVIDIA LiLiCorr drafter (``dflash`` + ``lilicorr.*``).

        llama.cpp recognises it by ``lilicorr.candidate_topk`` and runs it as
        ``draft-dflash`` from b11641 (PR #30506); b11668 fixes its lattice
        walk, so launch preflight requires that build.
        """
        if self.architecture.lower().strip() != "dflash":
            return False
        try:
            return int((self.metadata or {}).get("lilicorr.candidate_topk") or 0) > 0
        except (TypeError, ValueError):
            return False

    @property
    def drafter_spec_type(self) -> Optional[str]:
        """The ``--spec-type`` token this drafter needs, or ``None``.

        ``None``    — not a drafter, OR a plain sibling drafter that mainline
                     auto-detects from ``-md`` (no explicit ``--spec-type``
                     token emitted).
        ``"mtp"``  — standalone MTP head (Gemma 4 ``gemma4-assistant``),
                     launched as ``-md`` + ``--spec-type draft-mtp``.
        ``"eagle3"``— EAGLE-3 draft model (arch ``eagle3``), reads the
                     target's hidden states; ``-md`` + ``--spec-type
                     draft-eagle3`` (PR #18039, merged; Qwen3.5/3.6 support
                     in PR #24593 since b9723).
        ``"dflash"``— DFlash block-diffusion draft model (arch ``dflash``),
                     ``-md`` + ``--spec-type draft-dflash`` (PR #22105).
        ``"dspark"``— DSpark's DFlash-derived draft with Markov head,
                     ``-md`` + ``--spec-type draft-dspark`` (b10164+).
        """
        if not self.metadata:
            return None
        arch = str(self.metadata.get("general.architecture", "") or "").lower().strip()
        if arch.endswith("-assistant") or arch.endswith("_assistant"):
            return "mtp"
        if arch == "eagle3":
            return "eagle3"
        if arch == "dspark" or (
            arch == "dflash"
            and (
                self.metadata.get("__dspark_scan__") == "found"
                or bool(
                    re.search(
                        r"(?:^|[-_.])dspark(?:[-_.]|$)",
                        self.name,
                        re.IGNORECASE,
                    )
                )
            )
        ):
            return "dspark"
        if arch == "dflash":
            return "dflash"
        return None

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
            scored.append((overlap, _mmproj_precision_score(c.name), c.name.lower(), c))
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
#   - "assistant" / "draft": the conventional distributor naming
#     (Qwen3.6-32B-Assistant-…, …-draft-…).
#   - "mtp": Gemma 4 MTP drafters ship as a SEPARATE small head named with
#     an "mtp-" prefix (mtp-gemma-4-12B-it-qat-UD) or a "-MTP-" infix
#     (gemma-4-…-MTP-BF16). The bounded match below keeps this from firing
#     on an unrelated word, and the embedded-MTP filename fallback skips any
#     name caught here so a real self-speculating target (Qwen3.6-27B-MTP,
#     detected from metadata) is never misread as a draft file.
# This is only the cheap filename pre-filter; the authoritative signal for
# Gemma-4-style drafters is the architecture (see
# :func:`metadata_is_standalone_drafter`), applied after the metadata read.
_DRAFT_FILENAME_TOKENS = (
    "assistant",
    "draft",
    "mtp",
    # EAGLE-3 / DFlash / DSpark draft models ship as separate GGUFs named
    # ``…-eagle3`` / ``…-dflash`` / ``dspark-…``; treat them as drafts so
    # they pair to a target via -md instead of showing up as choosable models.
    "eagle3",
    "eagle",
    "dflash",
    "dspark",
    # NVIDIA's LiLiCorr drafters (b11641, PR #30506) are ``dflash`` GGUFs
    # named ``lilicorr-<target>-…`` / ``<target>-LiLiCorr.gguf``.
    "lilicorr",
)

# Maximum size for an ambiguously named GGUF to be treated as a standalone
# draft head based on its filename alone. Infix names such as
# ``Qwen3.6-27B-MTP-UD-Q3_K_XL.gguf`` commonly denote a large TARGET with an
# integrated head, so those retain the size guard. A leading ``mtp-`` /
# ``mtp_`` / ``mtp.`` is different: distributors use that form for an external
# head and newer Qwen-based heads (for example Tess-4) can be ~2.9 GiB. Those
# explicit prefix names bypass the guard and must never appear as runnable
# models. Metadata-based reclassification remains the authoritative fallback.
_DRAFT_MAX_SIZE_BYTES = int(1.5 * 1024**3)  # 1.5 GiB


# Drafter-base matching: strip the speculative-draft marker AND the quant
# tail, then compare the canonical (separator-normalised) stems. Unlike the
# generic _strip_quant, this PRESERVES a standalone "UD" variant token: in
# the QAT family both gemma-4-12b-it-qat-q4_0 and gemma-4-12B-it-qat-UD-Q4_K_XL
# collapse to "…-it-qat" once UD is treated as part of the quant tail, which
# makes the drafter mtp-gemma-4-12B-it-qat-UD ambiguous between them. Keeping
# UD as an identity token lets the drafter bind to the UD target it names.
_QUANT_CORE_PATTERN = re.compile(
    r"[-._]"
    r"(?:i\d+-)?"
    r"(?:Q\d+(?:_[A-Z0-9]+)*"
    r"|IQ\d+(?:_[A-Z0-9]+)*"
    r"|MXFP4(?:_MOE)?"
    r"|NVFP4|F8_E4M3|FP8"
    r"|BF16|F16|F32"
    r")"
    r"(?:[-._][0-9.]+bpw)?"
    r"(?:[-._](?:bf16|f16|f32))?"
    r"\.gguf$",
    re.IGNORECASE,
)
_DRAFT_MARKER_RE = re.compile(
    r"(?:^|[-_.])(?:assistant|draft|mtp|eagle3|eagle|dflash|dspark|lilicorr)(?=[-_.]|$)",
    re.IGNORECASE,
)


def _strip_quant_keep_variant(filename: str) -> str:
    """Strip the quant token but keep a standalone UD identity marker."""
    s = _QUANT_CORE_PATTERN.sub("", filename)
    if s == filename and filename.lower().endswith(".gguf"):
        s = filename[: -len(".gguf")]
    return s.rstrip(".-_")


def _draft_match_base(filename: str) -> str:
    """Canonical base of a drafter: quant stripped (UD kept), marker removed."""
    return _canonical_sep(_DRAFT_MARKER_RE.sub("", _strip_quant_keep_variant(filename)))


# Parameter-size labels such as "27b", "74b-a9b", "0.8b" or "30b_a3b".
_SIZE_LABEL_RE = re.compile(
    r"(?<![a-z0-9.])(\d+(?:\.\d+)?)b(?:[-_.]?a(\d+(?:\.\d+)?)b)?(?![a-z0-9])",
    re.IGNORECASE,
)


def _size_labels(text: str) -> Set[str]:
    """Return the normalized parameter-size labels in a filename stem."""
    return {
        f"{total}b" + (f"-a{active}b" if active else "")
        for total, active in _SIZE_LABEL_RE.findall(text.lower())
    }


def _size_labels_conflict(model_name: str, draft_name: str) -> bool:
    """True when a draft head names a different model size than the target.

    DFlash/DSpark/MTP/EAGLE heads are trained for one target. Two family
    names can share a long prefix ("NVIDIA-Nemotron-3-Nano-Omni-30B-A3B" and
    a "dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B" head); different size labels
    on both sides mean the head belongs to another model. Files without a
    size label are not judged.
    """
    model_sizes = _size_labels(_strip_quant_keep_variant(model_name))
    draft_sizes = _size_labels(_strip_quant_keep_variant(draft_name))
    return bool(model_sizes and draft_sizes and not (model_sizes & draft_sizes))


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _is_draft_filename(name: str, size_bytes: Optional[int] = None) -> bool:
    """Cheap pre-filter: does this filename look like a draft/assistant file?

    A file counts as a draft when its stem carries a bounded draft token.
    Ambiguous infix names are accepted only up to
    :data:`_DRAFT_MAX_SIZE_BYTES`, because a large ``…-MTP-…`` GGUF normally
    contains an integrated head. A leading ``mtp-`` / ``mtp_`` / ``mtp.`` is
    an explicit external-head convention and bypasses that size guard; modern
    Qwen-based draft heads can legitimately be several GiB.
    """
    n = name.lower()
    # Match token surrounded by separators OR at the end of stem
    # (e.g. "qwen3.5-30b-a3b-assistant-q4_k_m.gguf" is a draft;
    #  "rooks_assistant_v2.gguf" — a fictional case — would also match,
    #  which is acceptable, false-positives just cost a draft pairing).
    matched = False
    for tok in _DRAFT_FILENAME_TOKENS:
        if (
            re.search(rf"[-_.]{tok}[-_.]", n)
            or n.startswith(tok + "-")
            or n.startswith(tok + "_")
            or n.startswith(tok + ".")
        ):
            matched = True
            break
    if not matched:
        return False
    # Size guard: a large file with an ambiguous infix draft token is usually
    # a target with an integrated head. A leading MTP marker explicitly names
    # a separate head (including newer ~2.9 GiB Qwen/Tess heads), so keep it
    # in the draft pool regardless of size.
    explicit_mtp_head = n.startswith(("mtp-", "mtp_", "mtp."))
    if (
        size_bytes is not None
        and size_bytes > _DRAFT_MAX_SIZE_BYTES
        and not explicit_mtp_head
    ):
        return False
    return True


def _find_draft(model: Path, candidates: List[Path]) -> Optional[Path]:
    """Pick the draft whose stem best matches the main model.

    Same directory only — drafts only count if they sit beside the main
    model. Matching is by longest common canonical-stem prefix after both
    sides have their quant tail stripped (UD preserved as an identity token)
    and the drafter's speculative marker removed. This is robust to the
    fragile cases that a strict ``startswith`` missed:

      * Frankenmerger ``…-MTP-BF16`` drafters whose base equals the target's
        full stem (``gemma-4-12B-agentic-…-tau2``), where two unrelated
        targets (agentic vs coder) share only a short prefix.
      * QAT ``mtp-…-it-qat-UD`` drafters where ``_strip_quant`` erases the UD
        token from both candidate targets, making them tie; keeping UD as an
        identity token binds the drafter to the UD target it names.

    A candidate must share at least the model-family prefix to qualify; the
    longest common prefix wins, and among equal-length matches the smallest
    file on disk is preferred (drafts are speculative — smaller evaluates
    faster). The threshold (half the shorter base) guards against an
    unrelated drafter in the same folder binding to the wrong target.
    """
    main_base = _canonical_sep(_strip_quant_keep_variant(model.name))
    best: Optional[Path] = None
    best_score = -1
    best_size = -1
    for c in candidates:
        if c.parent != model.parent:
            continue
        c_base = _draft_match_base(c.name)
        if not c_base:
            continue
        if _size_labels_conflict(model.name, c.name):
            continue
        cp = _common_prefix_len(main_base, c_base)
        # Require a meaningful shared prefix: at least half of the shorter
        # base (so "gemma-4-12b" alone can't bind a coder drafter to an
        # agentic target, but a full-stem match always wins).
        if cp < max(8, min(len(main_base), len(c_base)) // 2):
            continue
        try:
            sz = c.stat().st_size
        except OSError:
            continue
        if cp > best_score or (cp == best_score and sz < best_size):
            best = c
            best_score = cp
            best_size = sz
    return best


def is_mmproj_compatible(model: Path, mmproj: Path) -> bool:
    """True if ``mmproj`` is a plausible projector for ``model``.

    Used by the GUI to flag (not hide) incompatible picks in the always-on
    mmproj dropdown. Reuses the same separator-tolerant bidirectional prefix
    match as :func:`_find_mmproj_candidates`: the projector's normalized base
    and the model's normalized name must share a prefix. Cross-model files in
    the same folder (e.g. another model's projector) correctly fail this.
    """
    model_canon = _canonical_sep(_normalize_model(model.name))
    c_canon = _canonical_sep(_normalize_mmproj(mmproj.name))
    if not c_canon or not model_canon:
        return False
    return model_canon.startswith(c_canon) or c_canon.startswith(model_canon)


def is_draft_compatible(
    model: Path, draft: Path, draft_md: Optional[Dict[str, Any]] = None
) -> bool:
    """True if ``draft`` can plausibly drive speculative decoding for ``model``.

    Two independent acceptance paths (either suffices), mirroring how
    llama.cpp actually pairs drafts:

      1. **Name affinity** — the draft's canonical base (quant stripped, UD
         kept, speculative marker removed) shares a meaningful prefix with the
         model's base. This is the standard sibling-drafter case
         (``…-Assistant-Q4`` next to the full model) and the Frankenmerger /
         QAT MTP heads.
      2. **Architecture family** — a standalone MTP head declares
         ``<family>-assistant`` (Gemma 4 ``gemma4-assistant``). When the
         drafter's metadata is available and its architecture family matches
         the model's (``gemma4-assistant`` ↔ ``gemma4``), accept it even if
         the filenames diverge, since the head binds by tokenizer/backbone,
         not by name.

    The GUI uses this only to decide whether to prefix an entry with "!" — an
    incompatible pick still launches (with a logged warning), so this stays
    deliberately permissive: it answers "does this look like a real pair?",
    not "is this guaranteed to load".
    """
    main_base = _canonical_sep(_strip_quant_keep_variant(model.name))
    d_base = _draft_match_base(draft.name)
    if d_base and main_base and not _size_labels_conflict(model.name, draft.name):
        cp = _common_prefix_len(main_base, d_base)
        if cp >= max(8, min(len(main_base), len(d_base)) // 2):
            return True
    # Architecture-family fallback for standalone MTP heads.
    if draft_md and metadata_is_standalone_drafter(draft_md):
        d_arch = str(draft_md.get("general.architecture", "") or "").lower()
        # "gemma4-assistant" → backbone family "gemma4".
        family = d_arch.rsplit("-assistant", 1)[0].rsplit("_assistant", 1)[0]
        # We don't always have the target's metadata here, so match on the
        # family token appearing in the model's filename base (gemma4 ↔
        # "gemma-4"/"gemma4"). Canonicalise both to compare separator-free.
        if family:
            fam_canon = re.sub(r"[-_.]", "", family)
            model_canon = re.sub(r"[-_.]", "", _normalize_model(model.name))
            if fam_canon and fam_canon in model_canon:
                return True
    return False


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
    file_sizes: Dict[Path, int] = {}
    for f in all_gguf:
        if _is_mmproj_filename(f.name):
            mmprojs.append(f)
        else:
            # Stat once so the size guard in _is_draft_filename can tell a
            # real (small) draft head apart from a large MTP-named target
            # model that carries an integrated draft. A failed stat falls
            # back to the size-agnostic check so the file is not lost.
            try:
                f_size = f.stat().st_size
                file_sizes[f] = f_size
            except OSError:
                f_size = None
            if _is_draft_filename(f.name, f_size):
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

    # Metadata/tensor-name parsing is independent per file. Read single models,
    # filename-classified draft heads, and each split model's first shard in a
    # bounded pool. Executor.map preserves deterministic path/result ordering;
    # the persistent stat-signature cache makes later starts mostly cache hits.
    split_primary_paths = [
        parts.get(1) or parts[min(parts)] for parts in split_parts.values()
    ]
    metadata_by_path = (
        _read_metadata_many([*single_models, *drafts, *split_primary_paths])
        if read_metadata
        else {}
    )

    def _group_for(path: Path) -> str:
        """Return the group label (relative sub-directory) for *path*."""
        try:
            rel = path.relative_to(root)
            rel_parts = rel.parts
            return "/".join(rel_parts[:-1]) if len(rel_parts) > 1 else "."
        except ValueError:
            return str(path.parent)

    # Per-directory index of EVERY projector / drafter, used to populate the
    # always-on manual dropdowns (folder_mmprojs / folder_drafts). These hold
    # all same-folder files regardless of whether they match the model; the
    # GUI flags non-matching ones rather than hiding them. Note: `drafts` is
    # finalised below in phase 1 (after standalone-drafter reclassification),
    # so the draft index is built afterwards.
    def _by_parent(paths: List[Path]) -> Dict[str, List[Path]]:
        idx: Dict[str, List[Path]] = {}
        for p in sorted(paths, key=lambda x: x.name.lower()):
            idx.setdefault(str(p.parent), []).append(p)
        return idx

    mmproj_by_parent = _by_parent(mmprojs)

    entries: List[ModelEntry] = []

    # --- Single-file models -------------------------------------------
    # Phase 1: read metadata and split off any file whose architecture marks
    # it as a standalone speculative drafter (Gemma 4 gemma4-assistant and
    # friends). The filename pre-filter already catches the common "mtp-" /
    # "-MTP-" / "-assistant-" names, but the architecture check is the
    # authoritative backstop — it reclassifies a drafter that slipped through
    # with an unconventional name so it is paired via -md, never listed as a
    # choosable model. Reclassified drafters join the `drafts` pool so the
    # real targets in the same folder can bind them in phase 2.
    single_meta: List[Tuple[Path, int, Dict[str, Any]]] = []
    for m in sorted(single_models):
        size = file_sizes.get(m)
        if size is None:
            try:
                size = m.stat().st_size
            except OSError:
                continue
        md = dict(metadata_by_path.get(m, {})) if read_metadata else {}
        if md and metadata_is_drafter_file(md):
            # Standalone drafter (Gemma 4 MTP assistant head, EAGLE-3,
            # DFlash, or DSpark) — reclassify into the draft pool so phase 2 pairs it
            # to its target via -md instead of listing it as a model.
            if m not in drafts:
                drafts.append(m)
            continue
        single_meta.append((m, size, md))

    # Phase 2: build entries for the genuine models, pairing against the
    # (now complete) draft pool. The draft index is built here because phase
    # 1 may have moved standalone drafters into `drafts`.
    draft_by_parent = _by_parent(drafts)
    for m, size, md in single_meta:
        parent = str(m.parent)
        local_mmprojs = list(mmproj_by_parent.get(parent, []))
        local_drafts = list(draft_by_parent.get(parent, []))
        entries.append(
            ModelEntry(
                path=m,
                name=m.stem,
                group=_group_for(m),
                size_bytes=size,
                mmproj=_find_mmproj(m, local_mmprojs),
                mmproj_candidates=_find_mmproj_candidates(m, local_mmprojs),
                draft=_find_draft(m, local_drafts),
                folder_mmprojs=local_mmprojs,
                folder_drafts=local_drafts,
                metadata=md,
                part_paths=[m],
            )
        )

    # --- Multi-part (sharded) models ----------------------------------
    # Identify every split model whose primary metadata declares MTP, then read
    # all required remaining shards in one bounded pool instead of serially per
    # model. Models without nextn metadata retain the cheap shard-1-only path.
    extra_shards: List[Path] = []
    if read_metadata:
        for parts_dict in split_parts.values():
            ordered = [parts_dict[i] for i in sorted(parts_dict)]
            if len(ordered) <= 1:
                continue
            part1 = parts_dict.get(1) or ordered[0]
            primary_md = metadata_by_path.get(part1, {})
            declares_nextn = False
            for key, value in primary_md.items():
                if "nextn_predict" not in key.lower():
                    continue
                try:
                    declares_nextn = int(value) > 0
                except (TypeError, ValueError):
                    declares_nextn = False
                if declares_nextn:
                    break
            arch = str(primary_md.get("general.architecture", "") or "").lower()
            declares_read_lazy_table = arch in {"qwen4exp", "gemma4"} or any(
                str(key).endswith(".embedding_length_per_layer_input")
                for key in primary_md
            )
            if declares_nextn or declares_read_lazy_table:
                extra_shards.extend(part for part in ordered if part != part1)
        metadata_by_path.update(_read_metadata_many(extra_shards))

    for (_parent_str, base), parts_dict in sorted(
        split_parts.items(), key=lambda kv: kv[0][1].lower()
    ):
        # Use shard 1 as the primary path (llama.cpp auto-discovers the rest).
        # Fall back to the lowest-indexed shard if shard 1 is missing.
        part1 = parts_dict.get(1) or parts_dict[min(parts_dict)]
        total_size = 0
        for part in parts_dict.values():
            size = file_sizes.get(part)
            if size is None:
                try:
                    size = part.stat().st_size
                except OSError:
                    size = 0
            total_size += size
        if total_size == 0:
            continue
        ordered_parts = [parts_dict[i] for i in sorted(parts_dict)]
        # Build a synthetic Path whose .name == "<base>.gguf" so the
        # mmproj / draft pairing functions get the correct base stem.
        pairing_path = part1.parent / (base + ".gguf")
        parent = str(part1.parent)
        md = dict(metadata_by_path.get(part1, {})) if read_metadata else {}
        if read_metadata and len(ordered_parts) > 1:
            shard_metadata = [
                dict(metadata_by_path.get(part, {})) for part in ordered_parts
            ]
            lazy_bytes = sum(
                max(0, int(item.get("__read_lazy_tensor_bytes__", 0) or 0))
                for item in shard_metadata
            )
            if lazy_bytes > 0:
                md["__read_lazy_tensor_bytes__"] = lazy_bytes

            # Some quantizers preserve ``nextn_predict_layers`` after dropping
            # the actual MTP tensors. A primary-shard scan is inconclusive by
            # design, so verify every shard header before advertising internal
            # MTP in the GUI. Only models declaring nextn reached the pooled
            # extra-shard read above.
            declares_nextn = any(
                "nextn_predict" in key.lower() and _coerce_positive_int(value)
                for key, value in md.items()
            )
            if declares_nextn:
                scans_complete = all(
                    item.get("__tensor_scan_complete__") is True
                    for item in shard_metadata
                )
                if scans_complete:
                    block_count = 0
                    arch = str(md.get("general.architecture", "") or "")
                    try:
                        block_count = int(md.get(f"{arch}.block_count", 0) or 0)
                    except (TypeError, ValueError):
                        pass
                    found = any(
                        item.get("__mtp_scan__") == "found" for item in shard_metadata
                    ) or (
                        block_count > 0
                        and any(
                            int(item.get("__max_block_index__", -1)) >= block_count
                            for item in shard_metadata
                        )
                    )
                    md["__mtp_scan__"] = "found" if found else "absent"
        local_mmprojs = list(mmproj_by_parent.get(parent, []))
        local_drafts = list(draft_by_parent.get(parent, []))
        entries.append(
            ModelEntry(
                path=part1,
                name=base,
                group=_group_for(part1),
                size_bytes=total_size,
                mmproj=_find_mmproj(pairing_path, local_mmprojs),
                mmproj_candidates=_find_mmproj_candidates(pairing_path, local_mmprojs),
                draft=_find_draft(pairing_path, local_drafts),
                folder_mmprojs=local_mmprojs,
                folder_drafts=local_drafts,
                metadata=md,
                part_paths=ordered_parts,
            )
        )

    # A target whose own tensors already contain the NextN/MTP block must not
    # default to a separate ``mtp-*`` head lying next to it (for example after
    # merge_mtp_head.py): the embedded head is the path llama.cpp tests, and
    # an external draft would take precedence over it at launch. The head
    # stays selectable through ``folder_drafts``.
    for entry in entries:
        if (
            entry.draft is not None
            and (entry.metadata or {}).get("__mtp_scan__") == "found"
            and entry.has_embedded_mtp
            and entry.draft.name.lower().startswith(("mtp-", "mtp_", "mtp."))
        ):
            entry.draft = None

    if read_metadata:
        flush_gguf_metadata_cache()
    return entries


def group_entries(entries: List[ModelEntry]) -> Dict[str, List[ModelEntry]]:
    """Group entries by their `group` field, preserving discovery order."""
    out: Dict[str, List[ModelEntry]] = {}
    for e in entries:
        out.setdefault(e.group, []).append(e)
    return out
