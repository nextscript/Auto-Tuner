"""b11030 audit: unchanged CLI surface, the new DFM Mimir (hrm_text) profile,
the Nemotron latent-MoE MTP loader gate (PR #29018) and the still-open
vision+DFlash2 gate."""

import json
from pathlib import Path

import pytest

import tuner
from settings_loader import load_profiles, match_profile
from test_kv_policy import _model
from test_smoke import _fake_model_md, _fake_system

ROOT = Path(__file__).resolve().parent


def _manifest(tag: str) -> dict:
    return json.loads(
        (ROOT / f"docs/llama-{tag}-server-flags.json").read_text(encoding="utf-8")
    )


def _profiles():
    return load_profiles(ROOT / "settings")


def _config():
    return tuner.TunedConfig(
        ctx=4096,
        ngl=0,
        threads=4,
        batch_threads=4,
        batch=256,
        ubatch=128,
        cache_k="q8_0",
        cache_v="q8_0",
        flash_attn=True,
    )


def test_pinned_b11030_manifest_keeps_the_b10977_option_set():
    manifest = _manifest("b11030")
    assert manifest["tag"] == "b11030"
    assert manifest["commit"] == "bdcbaaf6e7520b68c8c60ff724c67409970d70e1"
    flags = set(manifest["flags"])
    # b10977..b11030 (53 commits) touches neither common/arg.cpp nor the
    # server option table, so the option set and the help wording are the
    # same as on b10948 and b10977 (--help SHA-256 unchanged since b10948).
    assert flags == set(_manifest("b10977")["flags"])
    assert len(flags) == 415 and sum(f.startswith("--") for f in flags) == 328
    assert "--rpc" not in flags  # local GGML_RPC=OFF, not an upstream removal
    binaries = manifest["binaries"]
    assert set(binaries) == {"vulkan", "hip"}
    hashes = {b["help_sha256"] for b in binaries.values()}
    assert len(hashes) == 1  # one shared help text for both backends
    previous = {b["help_sha256"] for b in _manifest("b10977")["binaries"].values()}
    assert hashes == previous  # byte-identical --help on both backends
    assert not flags.intersection(tuner._LEGACY_LOAD_MODES)
    for profile in _profiles():
        for token in profile.extra_args:
            if token.startswith("--"):
                assert token.split("=", 1)[0] in flags, (profile.source_file, token)


# ------------------------------------------------------------ DFM Mimir ----


def _mimir(tmp_path, name="DFM-Mimir-Q8_0"):
    model = _model(tmp_path)
    model.name = name
    # danish-foundation-models/DFM-Mimir config.json: 16 layers per stack,
    # H_cycles 2, L_cycles 3 -> block_count = 16 * 2 * (3 + 1) = 128 cache
    # slots over 32 physical blocks; MHA (12 heads, 12 KV heads, head 128).
    model.metadata = {
        "general.architecture": "hrm_text",
        "hrm_text.block_count": 128,
        "hrm_text.context_length": 4096,
        "hrm_text.embedding_length": 1536,
        "hrm_text.feed_forward_length": 4096,
        "hrm_text.attention.head_count": 12,
        "hrm_text.attention.head_count_kv": 12,
        "hrm_text.attention.key_length": 128,
        "hrm_text.attention.value_length": 128,
        "hrm_text.attention.layer_norm_rms_epsilon": 1e-6,
        "hrm_text.rope.dimension_count": 128,
        "hrm_text.hrm.layers_per_stack": 16,
        "hrm_text.hrm.h_cycles": 2,
        "hrm_text.hrm.l_cycles": 3,
        "hrm_text.hrm.prefix_lm": True,
    }
    return model


def test_mimir_profile_matches_by_name_and_by_architecture():
    profiles = _profiles()
    by_name = match_profile("DFM-Mimir-Q8_0", profiles)
    assert by_name.source_file == "dfm-mimir.yaml"
    assert by_name.arch_fallback == ["hrm_text"]
    # PR #27625 merged on 2026-09-16; b11003 is the first tag that loads it.
    assert by_name.min_llama_build == 11003
    assert by_name.max_context == 4096
    assert by_name.extra_args == ["--jinja"]
    assert by_name.runtime_block_reason == ""
    assert by_name.sampling["chat"]["temperature"] == 0.7
    assert by_name.sampling["coding"]["temperature"] == 0.2
    # Opaque filenames still resolve through the architecture ...
    assert match_profile("opaque-hrm-checkpoint.gguf", profiles, "hrm_text") is by_name
    # ... but the fallback is exact: other "mimir" names on other archs keep
    # their generic handling.
    assert (
        match_profile("mimir-bert-base.gguf", profiles, "bert").source_file
        == "_default.yaml"
    )
    # Every language pack explains the new profile.
    for pack in sorted((ROOT / "assets/languages").glob("*.json")):
        notes = json.loads(pack.read_text(encoding="utf-8"))["profile_notes"]
        note = notes["dfm-mimir.yaml"]
        assert "b11003" in note and "hrm_text" in note, pack.name
        assert "--jinja" in note and "enable_thinking" in note, pack.name


@pytest.mark.parametrize(
    "build,allowed",
    [(10977, False), (11002, False), (11003, True), (11030, True), (None, True)],
)
def test_mimir_requires_b11003(monkeypatch, build, allowed):
    profile = match_profile("DFM-Mimir-Q8_0", _profiles())
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: build)
    ok, message, detected = tuner.check_profile_build(profile, "llama-server")
    assert ok is allowed
    assert detected == build
    if not allowed:
        assert "b11003+" in message and f"b{build}" in message
    elif build is None:
        assert "b11003+" in message  # unprobeable wrapper: warn, allow


def test_mimir_command_is_capped_at_the_native_4k_and_uses_jinja(tmp_path, monkeypatch):
    model = _mimir(tmp_path)
    profile = match_profile(model.name, _profiles(), "hrm_text")
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11030)
    monkeypatch.setattr(
        tuner, "_probe_supported_flags", lambda _: {"--jinja", "--spec-type"}
    )
    config = tuner.compute_config(
        model, _fake_system(vram_total=32, vram_free=31), profile, user_ctx=16384
    )
    assert config.ctx <= 4096
    cmd = tuner.build_command(model, config, profile, server_binary="llama-server")
    assert "--jinja" in cmd
    assert cmd[cmd.index("--temp") + 1] == "0.7"
    assert "-md" not in cmd  # no MTP head, no draft model
    assert "--spec-type" not in cmd or "draft-mtp" not in cmd


def test_mimir_kv_estimate_counts_all_128_cache_slots(tmp_path):
    model = _mimir(tmp_path)
    per_token_mib = tuner._kv_per_token_total_mb_from_metadata(model.metadata)
    # llama.cpp keeps one KV entry per pass: 128 slots x 12 KV heads x
    # (128 + 128) x 2 bytes = 786,432 bytes per token ...
    assert per_token_mib == pytest.approx(786432 / 1024**2)
    # ... i.e. about 3,072 MiB at the native 4,096 context in F16, exactly the
    # figure PR #27625 documents for Mimir 1B.
    assert per_token_mib * 4096 == pytest.approx(3072)


# ------------------------------------------------ Nemotron latent MoE MTP ----


def _nemotron(tmp_path, latent: bool, mtp: bool = True):
    metadata = {
        "general.architecture": "nemotron_h_moe",
        "nemotron_h_moe.block_count": 89 if mtp else 88,
        "nemotron_h_moe.context_length": 1048576,
        "nemotron_h_moe.embedding_length": 4096,
        "nemotron_h_moe.attention.head_count": 32,
        "nemotron_h_moe.attention.head_count_kv": 8,
        "nemotron_h_moe.attention.key_length": 128,
        "nemotron_h_moe.attention.value_length": 128,
        "nemotron_h_moe.expert_count": 512,
        "nemotron_h_moe.expert_used_count": 6,
    }
    if mtp:
        metadata["nemotron_h_moe.nextn_predict_layers"] = 1
    if latent:
        # Nemotron 3 Super: the MoE FFN works in a 1024-wide latent space
        # behind ffn_latent_down / ffn_latent_up (PR #29018).
        metadata["nemotron_h_moe.moe_latent_size"] = 1024
    return _fake_model_md(
        tmp_path, "NVIDIA-Nemotron-3-Super-120B-A12B-Q4_K_M", 60.0, metadata
    )


@pytest.mark.parametrize(
    "build,allowed",
    [(10977, False), (11024, False), (11025, True), (11030, True), (None, True)],
)
def test_nemotron_super_mtp_block_needs_b11025(tmp_path, monkeypatch, build, allowed):
    model = _nemotron(tmp_path, latent=True)
    assert model.has_embedded_mtp
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: build)
    ok, message, detected = tuner.check_model_build(model, "llama-server")
    assert ok is allowed
    assert detected == build
    if allowed:
        assert message == ""
    else:
        # Upstream's own reproduction before PR #29018:
        # "wrong number of tensors; expected 781, got 779".
        assert f"b{build}" in message and "b11025" in message
        assert "PR #29018" in message and "moe_latent_size 1024" in message
        assert "wrong number of tensors" in message


def test_nemotron_without_latent_moe_or_without_mtp_block_is_not_gated(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 10977)
    # Lightning / Nano: PR #26725 MTP without the latent step loads on b10977.
    lightning = _nemotron(tmp_path, latent=False)
    assert lightning.has_embedded_mtp
    assert tuner.check_model_build(lightning, "llama-server") == (True, "", 10977)
    # A Super GGUF whose MTP block was stripped never reaches the gate.
    stripped = _nemotron(tmp_path, latent=True, mtp=False)
    assert not stripped.has_embedded_mtp
    assert tuner.check_model_build(stripped, "llama-server") == (True, "", None)
    # The gate is architecture-specific: another latent-MoE arch is untouched.
    other = _nemotron(tmp_path, latent=True)
    other.metadata = {
        (
            k.replace("nemotron_h_moe", "kimi-k3") if k != "general.architecture" else k
        ): ("kimi-k3" if k == "general.architecture" else v)
        for k, v in other.metadata.items()
    }
    assert tuner.check_model_build(other, "llama-server")[0]


# ------------------------------------------------------------- deepseek ----


# ------------------------------------------------------- vision + dflash2 ----


def test_vision_dflash2_gate_is_unchanged_and_names_b11030():
    # Nothing in b10977..b11030 touches tools/server, the speculative helpers
    # or the recurrent memory; the actual image + DFlash2 request was re-run
    # on b11030 and still fails with HTTP 500 on HIP and Vulkan.
    assert tuner.QWEN35_VISION_DFLASH2_BROKEN_SINCE == 10896


def _qwen_vision_dflash2(tmp_path):
    model = _model(tmp_path)
    model.name = "Qwen3.8-27B"
    model.metadata["general.architecture"] = "qwen35"
    model.mmproj = tmp_path / "mmproj.gguf"
    draft = _model(tmp_path)
    draft.metadata = {
        "general.architecture": "dflash",
        "dflash.conv_kernel_size": 2,
        "dflash.conv_group_size": 16,
        "dflash.selector_rank": 256,
        "dflash.selector_top_k": 16,
    }
    profile = match_profile(model.name, _profiles(), "qwen35")
    config = _config()
    config.sampling = profile.sampling["chat"]
    return model, draft, profile, config


@pytest.mark.parametrize(
    "build,blocked",
    [(10895, False), (10896, True), (10977, True), (11003, True), (11030, True)],
)
def test_vision_with_dflash2_stays_gated_through_b11030(
    tmp_path, monkeypatch, build, blocked
):
    model, draft, profile, config = _qwen_vision_dflash2(tmp_path)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: build)
    monkeypatch.setattr(
        tuner, "_probe_supported_flags", lambda _: {"--spec-type", "--mmproj"}
    )
    if blocked:
        with pytest.raises(ValueError, match=rf"b{build}.*since b10896.*b11249"):
            tuner.build_command(model, config, profile, draft_model=draft)
    else:
        cmd = tuner.build_command(model, config, profile, draft_model=draft)
        assert "--mmproj" in cmd and "-md" in cmd


def test_documented_alternatives_still_build_on_b11030(tmp_path, monkeypatch):
    model, draft, profile, config = _qwen_vision_dflash2(tmp_path)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11030)
    monkeypatch.setattr(
        tuner, "_probe_supported_flags", lambda _: {"--spec-type", "--mmproj"}
    )
    text_only = tuner.build_command(
        model, config, profile, draft_model=draft, enable_speculative=False
    )
    assert "--mmproj" in text_only and "-md" not in text_only
    model.mmproj = None
    assert "-md" in tuner.build_command(model, config, profile, draft_model=draft)
