"""b11538 audit: --moe-cache-mib, EschaLabs W2 GGUFs on the Escha fork (issue
#7), MTP-only head detection, decision-type/tokenizer build floors, the DFlash
embedding-head fallback (b11132-b11511) and K2 Horizon on mainline."""

import json
import struct
from pathlib import Path

import pytest

import scanner
import tuner
from settings_loader import load_profiles, match_profile
from test_smoke import _fake_model_md

ROOT = Path(__file__).resolve().parent


def _profiles():
    return load_profiles(ROOT / "settings")


def _manifest(tag: str) -> dict:
    return json.loads(
        (ROOT / f"docs/llama-{tag}-server-flags.json").read_text(encoding="utf-8")
    )


# ------------------------------------------------------------------ manifest


def test_b11538_manifest_only_adds_moe_cache_mib():
    current = _manifest("b11538")
    previous = _manifest("b11371")
    assert current["tag"] == "b11538"
    assert current["commit"] == "79e2e74eb"
    assert set(current["flags"]) - set(previous["flags"]) == {"--moe-cache-mib"}
    assert set(previous["flags"]) <= set(current["flags"])
    assert len(current["flags"]) == 418
    assert set(current["binaries"]) == {"vulkan", "hip"}
    for profile in _profiles():
        for arg in profile.extra_args:
            if arg.startswith("--"):
                assert arg.split("=", 1)[0] in current["flags"]


def test_moe_cache_mib_is_a_value_flag_pruned_with_its_value():
    assert "--moe-cache-mib" in tuner._ARG_FLAGS_WITH_VALUES
    cmd = ["llama-server", "-m", "m.gguf", "--moe-cache-mib", "4096", "-c", "4096"]
    kept, removed = tuner._filter_command_for_supported_flags(
        cmd, set(_manifest("b11371")["flags"])
    )
    assert kept == ["llama-server", "-m", "m.gguf", "-c", "4096"]
    assert removed == ["--moe-cache-mib 4096"]
    assert tuner._filter_command_for_supported_flags(
        cmd, set(_manifest("b11538")["flags"])
    ) == (cmd, [])


# ------------------------------------------------------- Escha (issue #7)

_ESCHA_ROOTS = [
    "output.weight",
    "token_embd.weight",
    "output_norm.weight",
    "escha_lut",
    "escha_dep_k2",
    "escha_dep_k3",
]


def _escha_target(tmp_path, *, arch="qwen35", moe=False, version=True):
    metadata = {
        "general.architecture": arch,
        f"{arch}.block_count": 64,
        "__tensor_scan_complete__": True,
        "__root_tensors__": list(_ESCHA_ROOTS),
        "__min_block_index__": 0,
        "__max_block_index__": 63,
        "__mtp_scan__": "absent",
    }
    if version:
        metadata[f"{arch}.escha.version"] = 1
    if moe:
        metadata[f"{arch}.expert_count"] = 256
    return _fake_model_md(tmp_path, "Escha-Qwen3.8-27B-W2", 11.8, metadata)


def _mtp_head(tmp_path, name, roots, *, arch="qwen35", min_block=64, extra=None):
    metadata = {
        "general.architecture": arch,
        f"{arch}.block_count": 65,
        f"{arch}.nextn_predict_layers": 1,
        "__tensor_scan_complete__": True,
        "__root_tensors__": roots,
        "__min_block_index__": min_block,
        "__max_block_index__": 64,
        "__mtp_scan__": "found",
    }
    metadata.update(extra or {})
    return _fake_model_md(tmp_path, name, 2.7, metadata)


def _markers(present):
    seen = []

    def fake(binary, markers):
        seen.append((binary, tuple(markers)))
        return present

    return fake, seen


def test_escha_target_is_refused_without_the_escha_loader(tmp_path, monkeypatch):
    target = _escha_target(tmp_path)
    fake, seen = _markers(False)
    monkeypatch.setattr(tuner, "_runtime_has_required_markers", fake)
    assert tuner.is_escha_model(target)
    ok, reason, _ = tuner.check_model_build(target, "b11538/llama-server")
    assert not ok
    assert "Escha fork" in reason and "escha-w2-dense" in reason
    assert "Ajay9o9/llama.cpp-escha" in reason
    assert ("b11538/llama-server", ("escha_lut",)) in seen
    moe = _escha_target(tmp_path, moe=True)
    assert "branch escha-w2)" in tuner.check_model_build(moe, "x")[1]


def test_escha_codec_table_alone_identifies_the_target(tmp_path, monkeypatch):
    monkeypatch.setattr(tuner, "_runtime_has_required_markers", lambda *_: False)
    target = _escha_target(tmp_path, version=False)
    assert tuner.is_escha_model(target)
    assert not tuner.check_model_build(target, "x")[0]


def test_escha_target_and_its_head_load_on_the_escha_fork(tmp_path, monkeypatch):
    """Issue #7: the official head lacks the codec tables and output_norm."""
    monkeypatch.setattr(tuner, "_runtime_has_required_markers", lambda *_: True)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 10372)
    target = _escha_target(tmp_path)
    head = _mtp_head(
        tmp_path,
        "Escha-Qwen3.8-27B-W2-MTP-F16-headQ4",
        ["output.weight", "token_embd.weight"],
    )
    assert tuner.mtp_sidecar_missing_root_tensors(target, head) == [
        "escha_dep_k2",
        "escha_dep_k3",
        "escha_lut",
        "output_norm.weight",
    ]
    assert tuner.check_model_build(target, "escha/llama-server") == (True, "", None)
    assert tuner.check_draft_model_build(head, "escha/llama-server", target) == (
        True,
        "",
        None,
    )
    # The fork's qwen35 MTP-only mode also makes token_embd optional.
    bare = _mtp_head(tmp_path, "mtp-bare", ["output.weight"])
    assert tuner.check_draft_model_build(bare, "escha/llama-server", target)[0]


def test_escha_fork_still_needs_what_its_loader_requires(tmp_path, monkeypatch):
    monkeypatch.setattr(tuner, "_runtime_has_required_markers", lambda *_: True)
    target = _escha_target(tmp_path)
    # A trunk-carrying draft gets no MTP-only exemption.
    trunk = _mtp_head(tmp_path, "trunk", ["token_embd.weight"], min_block=0)
    ok, reason, _ = tuner.check_draft_model_build(trunk, "escha", target)
    assert not ok and "output_norm.weight" in reason and "Escha fork" in reason
    assert "escha_lut" not in reason
    # A head with its own Escha code needs the codec tables.
    coded = _mtp_head(
        tmp_path,
        "coded",
        ["token_embd.weight", "output.weight"],
        extra={"qwen35.escha.version": 1},
    )
    ok, reason, _ = tuner.check_draft_model_build(coded, "escha", target)
    assert not ok and "escha_lut" in reason
    # The fork's qwen35moe loader requires output_norm even for MTP-only heads.
    moe_target = _escha_target(tmp_path, arch="qwen35moe", moe=True)
    moe_head = _mtp_head(
        tmp_path, "moe-head", ["token_embd.weight", "output.weight"], arch="qwen35moe"
    )
    ok, reason, _ = tuner.check_draft_model_build(moe_head, "escha", moe_target)
    assert not ok and reason.count("output_norm.weight") == 1


def test_escha_target_on_mainline_defers_to_the_model_check(tmp_path, monkeypatch):
    monkeypatch.setattr(tuner, "_runtime_has_required_markers", lambda *_: False)
    target = _escha_target(tmp_path)
    head = _mtp_head(tmp_path, "head", ["output.weight", "token_embd.weight"])
    # No misleading "sidecar lacks escha_lut" message: the model check names
    # the fork the target needs.
    assert tuner.check_draft_model_build(head, "mainline", target) == (True, "", None)
    assert "Escha fork" in tuner.check_model_build(target, "mainline")[1]


def test_mainline_message_unchanged_for_ordinary_heads(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        tuner,
        "_runtime_has_required_markers",
        lambda b, m: seen.append(m) or False,
    )
    target = _fake_model_md(
        tmp_path,
        "Tess-4-27B-Q6_K",
        20,
        {
            "general.architecture": "qwen35",
            "__tensor_scan_complete__": True,
            "__root_tensors__": ["token_embd.weight", "output_norm.weight"],
        },
    )
    head = _mtp_head(tmp_path, "mtp-Tess-4-27B-Q4_K_M", ["token_embd.weight"])
    ok, reason, _ = tuner.check_draft_model_build(head, "mainline", target)
    assert not ok and "cannot load on mainline llama.cpp" in reason
    assert "output_norm.weight" in reason
    assert seen == [["escha_lut"]]


def test_ordinary_models_never_scan_for_the_escha_marker(tmp_path, monkeypatch):
    def explode(*_):
        raise AssertionError("marker scan for an ordinary GGUF")

    monkeypatch.setattr(tuner, "_runtime_has_required_markers", explode)
    model = _fake_model_md(
        tmp_path,
        "Qwen3.8-27B-UD-Q4_K_XL",
        16,
        {"general.architecture": "qwen35", "__root_tensors__": ["token_embd.weight"]},
    )
    assert not tuner.is_escha_model(model)
    assert tuner.check_model_build(model, "x") == (True, "", None)


# ------------------------------------------------- MTP-only head (scanner)


def _kv(key, value):
    def s(text):
        data = text.encode("utf-8")
        return struct.pack("<Q", len(data)) + data

    if isinstance(value, bool):
        return s(key) + struct.pack("<I", 7) + struct.pack("<?", value)
    if isinstance(value, int):
        return s(key) + struct.pack("<I", 4) + struct.pack("<I", value)
    return s(key) + struct.pack("<I", 8) + s(str(value))


def _write_gguf(path: Path, kv: dict, tensors, pad_bytes: int = 0) -> None:
    def s(text):
        data = text.encode("utf-8")
        return struct.pack("<Q", len(data)) + data

    with path.open("wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3))
        f.write(struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv)))
        for key, value in kv.items():
            f.write(_kv(key, value))
        offset = 0
        for name in tensors:
            f.write(s(name) + struct.pack("<I", 1) + struct.pack("<Q", 32))
            f.write(struct.pack("<I", 1) + struct.pack("<Q", offset))
            offset += 64
        f.write(b"\0" * (offset + pad_bytes))


def test_scanner_records_the_lowest_block_index(tmp_path):
    head = tmp_path / "head.gguf"
    _write_gguf(
        head,
        {
            "general.architecture": "qwen35",
            "qwen35.block_count": 65,
            "qwen35.nextn_predict_layers": 1,
        },
        ["token_embd.weight", "blk.64.nextn.eh_proj.weight", "blk.64.attn_q.weight"],
    )
    md = scanner._read_gguf_metadata_uncached(head)
    assert md["__min_block_index__"] == 64 and md["__max_block_index__"] == 64
    assert scanner.metadata_is_mtp_only_head(md)
    assert scanner.metadata_is_drafter_file(md)


@pytest.mark.parametrize(
    "change",
    [
        {"__min_block_index__": 0},  # carries the trunk: a target with MTP
        {"__min_block_index__": -1},  # no block tensors at all
        {"__tensor_scan_complete__": False},
        {"split.count": 3},
        {"qwen35.nextn_predict_layers": 0},
    ],
)
def test_mtp_only_head_needs_complete_single_file_evidence(change):
    md = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 65,
        "qwen35.nextn_predict_layers": 1,
        "__tensor_scan_complete__": True,
        "__min_block_index__": 64,
    }
    assert scanner.metadata_is_mtp_only_head(md)
    md.update(change)
    assert not scanner.metadata_is_mtp_only_head(md)


def test_unprefixed_large_escha_head_pairs_as_draft(tmp_path, monkeypatch):
    """The original EschaLabs head name has no mtp- prefix and is 2.9 GB, so
    the filename rules kept it in the model list as an embedded-MTP model."""
    monkeypatch.setattr(scanner, "_DRAFT_MAX_SIZE_BYTES", 1024)
    monkeypatch.setenv("AUTOTUNER_METADATA_CACHE", str(tmp_path / "cache.json"))
    target = tmp_path / "Escha-Qwen3.8-27B-W2.gguf"
    _write_gguf(
        target,
        {
            "general.architecture": "qwen35",
            "qwen35.block_count": 2,
            "qwen35.escha.version": 1,
        },
        [
            "token_embd.weight",
            "escha_lut",
            "blk.0.attn_norm.weight",
            "blk.1.attn_norm.weight",
        ],
    )
    head = tmp_path / "Escha-Qwen3.8-27B-W2-MTP-F16-headQ4.gguf"
    _write_gguf(
        head,
        {
            "general.architecture": "qwen35",
            "qwen35.block_count": 3,
            "qwen35.nextn_predict_layers": 1,
        },
        ["token_embd.weight", "output.weight", "blk.2.nextn.eh_proj.weight"],
        pad_bytes=4096,
    )
    entries = scanner.scan_models(tmp_path)
    assert [e.name for e in entries] == ["Escha-Qwen3.8-27B-W2"]
    assert entries[0].draft == head
    assert not entries[0].has_embedded_mtp
    draft = scanner.ModelEntry(
        path=head,
        name=head.stem,
        group=".",
        size_bytes=head.stat().st_size,
        metadata=scanner.read_gguf_metadata(head),
    )
    assert draft.is_standalone_drafter and not draft.has_embedded_mtp


def test_external_mtp_only_head_still_launches_with_draft_mtp(tmp_path):
    from test_smoke import _fake_system

    profile = match_profile("Qwen3.8-27B", _profiles(), "qwen35")
    target = _fake_model_md(
        tmp_path,
        "Tess-4-27B-Q6_K",
        20,
        {"general.architecture": "qwen35", "qwen35.block_count": 64},
    )
    head = _mtp_head(tmp_path, "mtp-Tess-4-27B-Q4_K_M", ["token_embd.weight"])
    assert head.is_standalone_drafter and not head.has_embedded_mtp
    cfg = tuner.compute_config(
        target,
        _fake_system(vram_total=48, vram_free=46),
        profile,
        draft_model=head,
        prompt_cache_ram_mib=0,
    )
    cmd = tuner.build_command(
        target,
        cfg,
        profile,
        draft_model=head,
        enable_speculative=True,
        enable_ngram=False,
        enable_prompt_cache=False,
    )
    assert "-md" in cmd
    assert "draft-mtp" in cmd[cmd.index("--spec-type") + 1]


# ------------------------------------------- decision types and tokenizers


@pytest.mark.parametrize(
    ("arch", "kind", "floor"),
    [
        ("qwen35", "pplx-decider", 11447),
        ("lfm2", "lfm2-d1", 11483),
        ("lfm2", "lfm2-d1-omni", 11488),
        ("clef", "clef", 11371),
        ("qwen35", "nimble", 11364),
    ],
)
def test_decision_types_need_their_upstream_build(
    tmp_path, monkeypatch, arch, kind, floor
):
    model = _fake_model_md(
        tmp_path,
        f"{kind}-Q8_0",
        2,
        {"general.architecture": arch, f"{arch}.decision.type": kind},
    )
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: floor - 1)
    ok, reason, detected = tuner.check_model_build(model, "x")
    assert not ok and f"b{floor}+" in reason and kind in reason
    assert detected == floor - 1
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: floor)
    assert tuner.check_model_build(model, "x")[0]


def test_unknown_decision_types_and_unprobeable_builds_are_not_gated(
    tmp_path, monkeypatch
):
    model = _fake_model_md(
        tmp_path,
        "future-decider",
        2,
        {"general.architecture": "qwen35", "qwen35.decision.type": "future"},
    )
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11000)
    assert tuner.check_model_build(model, "x") == (True, "", None)
    known = _fake_model_md(
        tmp_path,
        "d1-3B-Q8_0",
        2,
        {"general.architecture": "lfm2", "lfm2.decision.type": "lfm2-d1"},
    )
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: None)
    assert tuner.check_model_build(known, "x")[0]


def test_plamo3_tokenizer_needs_b11455(tmp_path, monkeypatch):
    model = _fake_model_md(
        tmp_path,
        "plamo-3-8b",
        8,
        {"general.architecture": "plamo3", "tokenizer.ggml.model": "plamo3"},
    )
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11454)
    ok, reason, _ = tuner.check_model_build(model, "x")
    assert not ok and "b11455+" in reason and "plamo3" in reason
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11455)
    assert tuner.check_model_build(model, "x")[0]


@pytest.mark.parametrize(
    "name",
    [
        "pplx-decider-v1.1-27b-Q4_K_M",
        "d1-3B-Q8_0",
        "d1-omni-600M-Q8_0",
    ],
)
def test_new_decision_models_match_the_decision_profile(name):
    arch = "qwen35" if name.startswith("pplx") else "lfm2"
    assert (
        match_profile(name, _profiles(), arch).source_file == "decision-systemone.yaml"
    )


# --------------------------------------------------- DFlash head (#30111)


def _dflash(tmp_path, roots, **extra):
    metadata = {
        "general.architecture": "dflash",
        "__tensor_scan_complete__": True,
        "__root_tensors__": roots,
    }
    metadata.update(extra)
    return _fake_model_md(tmp_path, "dflash-Nemotron-3.5-Lightning", 1.2, metadata)


@pytest.mark.parametrize(
    ("build", "warned"), [(11131, False), (11132, True), (11511, True), (11512, False)]
)
def test_dflash_draft_with_embeddings_but_no_head_warns_on_affected_builds(
    tmp_path, monkeypatch, build, warned
):
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: build)
    draft = _dflash(tmp_path, ["token_embd.weight", "fc.weight", "output_norm.weight"])
    ok, message, _ = tuner.check_draft_model_build(draft, "x")
    assert ok
    assert ("PR #30111" in message) is warned
    assert ("b11512+" in message) is warned


def test_dflash_drafts_without_embeddings_or_gemma_backbones_are_not_warned(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11371)
    plain = _dflash(tmp_path, ["fc.weight", "output_norm.weight"])
    assert tuner.check_draft_model_build(plain, "x") == (True, "", None)
    headed = _dflash(tmp_path, ["token_embd.weight", "output.weight", "fc.weight"])
    assert tuner.check_draft_model_build(headed, "x") == (True, "", None)
    gemma = _dflash(
        tmp_path,
        ["token_embd.weight", "fc.weight"],
        **{"dflash.hidden_activation": "gelu_pytorch_tanh"},
    )
    assert tuner.check_draft_model_build(gemma, "x") == (True, "", None)


# ------------------------------------------------------- K2 Horizon


def test_k2_horizon_profile_accepts_mainline_marker(monkeypatch):
    profile = match_profile("K2-Horizon-MoVA-36B-A4B-Q8_0", _profiles(), "k2-horizon")
    assert profile.source_file == "k2-horizon.yaml"
    assert profile.required_runtime_markers == ["k2-horizon"]
    assert "b11454" in profile.notes
    seen = []
    monkeypatch.setattr(
        tuner,
        "_runtime_has_required_markers",
        lambda binary, markers: seen.append(tuple(markers)) or True,
    )
    assert tuner.check_profile_build(profile, "b11538/llama-server")[0]
    assert seen == [("k2-horizon",)]
    dense = match_profile("K2-Horizon-1B-Q8_0", _profiles(), "k2-horizon")
    assert dense.source_file == "k2-horizon.yaml"


# ------------------------------------------------- qwen4exp MTP recheck


def test_qwen4exp_draft_mtp_gate_was_rechecked_on_b11538(tmp_path, monkeypatch):
    """b11538 HIP/Vulkan: the separate Flash Next head still aborts with
    GGML_ASSERT(buffer); the embedded head stays unused. Re-checked on
    b11669 (v5.6.3) and b12187 (v5.6.4) with the same result."""
    assert tuner._QWEN4EXP_MTP_HEAD_VERIFIED_BUILD == 12187
    head = _mtp_head(
        tmp_path,
        "mtp-Qwen3.8-Flash-Next-Q4_K_M",
        ["token_embd.weight", "output.weight"],
        arch="qwen4exp",
    )
    assert head.is_standalone_drafter and tuner.is_qwen4exp_mtp_sidecar(head)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11538)
    ok, reason, detected = tuner.check_draft_model_build(head, "llama-server")
    assert not ok and detected == 11538
    assert "GGML_ASSERT(buffer)" in reason and "b12187 HIP and Vulkan" in reason


# ------------------------------------------------------- EmbeddingGemma 2


def _encoder(tmp_path, name, arch, native_ctx):
    return _fake_model_md(
        tmp_path,
        name,
        0.3,
        {
            "general.architecture": arch,
            f"{arch}.block_count": 24,
            f"{arch}.context_length": native_ctx,
            f"{arch}.embedding_length": 512,
            f"{arch}.attention.head_count": 4,
            f"{arch}.attention.head_count_kv": 1,
            f"{arch}.attention.causal": False,
            f"{arch}.pooling_type": 1,
        },
    )


def test_embeddinggemma2_gets_an_embedding_plan_within_one_batch(tmp_path):
    from test_smoke import _fake_system

    model = _encoder(tmp_path, "embeddinggemma-2-Q8_0", "gemma-embedding2", 262144)
    profile = match_profile(model.name, _profiles(), model.architecture)
    assert profile.source_file == "embeddinggemma-2.yaml"
    assert profile.min_llama_build == 11452
    assert (
        match_profile("my-embedder", _profiles(), "gemma-embedding2").source_file
        == "embeddinggemma-2.yaml"
    )
    cfg = tuner.compute_config(
        model,
        _fake_system(vram_total=16, vram_free=15),
        profile,
        prompt_cache_ram_mib=0,
    )
    # The native 262k window cannot be used: the encoder reads one batch.
    assert cfg.ctx == cfg.batch == cfg.ubatch == 8192
    cmd = tuner.build_command(
        model, cfg, profile, enable_speculative=False, enable_ngram=False
    )
    assert "--embeddings" in cmd
    # An explicit context is still honoured.
    explicit = tuner.compute_config(
        model,
        _fake_system(vram_total=16, vram_free=15),
        profile,
        user_ctx=16384,
        prompt_cache_ram_mib=0,
    )
    assert explicit.ctx == 16384


def test_single_batch_cap_keeps_profiles_that_already_fit(tmp_path):
    from test_smoke import _fake_system

    granite = _encoder(
        tmp_path, "granite-embedding-311m-multilingual-r2", "modern-bert", 32768
    )
    profile = match_profile(granite.name, _profiles(), granite.architecture)
    assert profile.source_file == "granite-embedding-r2.yaml"
    cfg = tuner.compute_config(
        granite,
        _fake_system(vram_total=16, vram_free=15),
        profile,
        prompt_cache_ram_mib=0,
    )
    assert cfg.ctx == 32768 and cfg.ubatch == 8192


# ------------------------------------------------------- Clef images


def _clef(tmp_path, template, with_mmproj=True):
    model = _fake_model_md(
        tmp_path,
        "Clef-Flash-Q4_K_M",
        6,
        {
            "general.architecture": "clef",
            "clef.decision.type": "clef",
            "tokenizer.chat_template.systemone": template,
        },
    )
    if with_mmproj:
        mmproj = tmp_path / "mmproj-Clef-Flash-Q8_0.gguf"
        mmproj.write_bytes(b"GGUF")
        model.mmproj = mmproj
    return model


_OLD_CLEF = 'STATE:\n" }}{{ sep }}{{ state }}'
_NEW_CLEF = 'STATE:\n" }}{% if images %}{{ sep }}{% for image in images %}{{ image }}{% endfor %}{% endif %}{{ sep }}{{ state }}'


def test_clef_gguf_without_image_slot_warns_when_its_mmproj_is_attached(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11538)
    ok, note, _ = tuner.check_model_build(_clef(tmp_path, _OLD_CLEF), "x")
    assert ok and "HTTP 500" in note and "2026-10-05" in note
    assert tuner.check_model_build(_clef(tmp_path, _NEW_CLEF), "x") == (True, "", None)
    text_only = _clef(tmp_path, _OLD_CLEF, with_mmproj=False)
    assert tuner.check_model_build(text_only, "x") == (True, "", None)


def test_clef_images_need_b11418(tmp_path, monkeypatch):
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11417)
    ok, note, detected = tuner.check_model_build(_clef(tmp_path, _NEW_CLEF), "x")
    assert ok and "b11418+" in note and detected == 11417
