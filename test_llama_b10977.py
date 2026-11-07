"""b10977 audit: unchanged CLI surface, the new maple profile and the still-open
vision+DFlash2 gate."""

import json
from pathlib import Path

import pytest

import tuner
from settings_loader import load_profiles, match_profile
from test_kv_policy import _model
from test_smoke import _fake_system

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


def test_pinned_b10977_manifest_keeps_the_b10948_option_set():
    manifest = _manifest("b10977")
    assert manifest["tag"] == "b10977"
    assert manifest["commit"] == "0ecb159c9e93056a4742afe4195d05a2912b1746"
    flags = set(manifest["flags"])
    # b10948..b10977 (29 commits) touches neither common/arg.cpp nor the
    # server's option table, so the option set and even the help wording are
    # unchanged; v0.4.1 stable is b10964 inside this range.
    assert flags == set(_manifest("b10948")["flags"])
    assert len(flags) == 415 and sum(f.startswith("--") for f in flags) == 328
    assert "--rpc" not in flags  # local GGML_RPC=OFF, not an upstream removal
    binaries = manifest["binaries"]
    assert set(binaries) == {"vulkan", "hip"}
    hashes = {b["help_sha256"] for b in binaries.values()}
    assert len(hashes) == 1  # one shared help text for both backends
    previous = {b["help_sha256"] for b in _manifest("b10948")["binaries"].values()}
    assert hashes == previous  # byte-identical --help on both backends
    assert not flags.intersection(tuner._LEGACY_LOAD_MODES)
    for profile in _profiles():
        for token in profile.extra_args:
            if token.startswith("--"):
                assert token.split("=", 1)[0] in flags, (profile.source_file, token)


# ---------------------------------------------------------------- maple ----


def _maple(tmp_path, name="maple-preview-TQ2_0-head-Q4_K"):
    model = _model(tmp_path)
    model.name = name
    model.metadata = {
        "general.architecture": "maple",
        "maple.block_count": 24,
        "maple.context_length": 131072,
        "maple.embedding_length": 2048,
        "maple.attention.head_count": 16,
        "maple.attention.head_count_kv": 4,
        "maple.attention.key_length": 128,
        "maple.attention.value_length": 128,
        "maple.attention.sliding_window": 512,
        "maple.attention.sliding_window_pattern": [
            True,
            True,
            True,
            False,
        ]
        * 6,
        "maple.expert_count": 256,
        "maple.expert_used_count": 8,
        "maple.expert_feed_forward_length": 512,
    }
    return model


def test_maple_profile_matches_by_name_and_by_architecture():
    profiles = _profiles()
    by_name = match_profile("maple-preview-TQ1_0-head-F16", profiles)
    assert by_name.source_file == "maple.yaml"
    assert by_name.arch_fallback == ["maple"]
    # PR #27000 merged on 2026-09-14; b10964 (= v0.4.1) is the first tag.
    assert by_name.min_llama_build == 10964
    assert by_name.max_context == 131072
    assert by_name.extra_args == ["--jinja", "--reasoning-preserve"]
    assert by_name.runtime_block_reason == ""
    for mode in ("chat", "coding"):
        assert by_name.sampling[mode]["temperature"] == 0.6
        assert by_name.sampling[mode]["top_p"] == 0.95
        assert by_name.sampling[mode]["top_k"] == 20
        assert by_name.sampling[mode]["min_p"] == 0.0
        assert by_name.sampling[mode]["repeat_penalty"] == 1.0
    # A re-quant with an opaque filename still resolves through the arch.
    assert match_profile("opaque-ternary-moe.gguf", profiles, "maple") is by_name
    # The arch fallback stays exact: unrelated "maple" filenames on other
    # architectures keep their generic handling.
    assert (
        match_profile("maplestory-bert-large.gguf", profiles, "bert").source_file
        == "_default.yaml"
    )
    # Every language pack explains the new profile.
    for pack in sorted((ROOT / "assets/languages").glob("*.json")):
        notes = json.loads(pack.read_text(encoding="utf-8"))["profile_notes"]
        assert "b10964" in notes["maple.yaml"], pack.name


@pytest.mark.parametrize(
    "build,allowed",
    [(10948, False), (10963, False), (10964, True), (10977, True), (None, True)],
)
def test_maple_requires_b10964(monkeypatch, build, allowed):
    profile = match_profile("maple-preview-TQ2_0-head-Q4_K", _profiles())
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: build)
    ok, message, detected = tuner.check_profile_build(profile, "llama-server")
    assert ok is allowed
    assert detected == build
    if not allowed:
        assert "b10964+" in message and f"b{build}" in message
    elif build is None:
        assert "b10964+" in message  # unprobeable wrapper: warn, allow


def test_maple_command_uses_the_profile_sampling_and_template_flags(
    tmp_path, monkeypatch
):
    model = _maple(tmp_path)
    profile = match_profile(model.name, _profiles(), "maple")
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 10977)
    monkeypatch.setattr(
        tuner,
        "_probe_supported_flags",
        lambda _: {"--jinja", "--reasoning-preserve", "--spec-type"},
    )
    config = tuner.compute_config(
        model, _fake_system(vram_total=32, vram_free=31), profile, user_ctx=32768
    )
    assert config.ctx == 32768
    cmd = tuner.build_command(model, config, profile, server_binary="llama-server")
    assert "--jinja" in cmd and "--reasoning-preserve" in cmd
    assert cmd[cmd.index("--temp") + 1] == "0.6"
    assert cmd[cmd.index("--top-p") + 1] == "0.95"
    assert cmd[cmd.index("--top-k") + 1] == "20"
    assert "-md" not in cmd  # no MTP head, no draft model


def test_maple_context_is_capped_at_the_native_131k(tmp_path):
    model = _maple(tmp_path)
    profile = match_profile(model.name, _profiles(), "maple")
    config = tuner.compute_config(
        model, _fake_system(vram_total=32, vram_free=31), profile, user_ctx=262144
    )
    assert config.ctx <= 131072


# ------------------------------------------------------------- deepseek ----


# ------------------------------------------------------- vision + dflash2 ----


def test_vision_dflash2_gate_is_unchanged_and_names_b10977():
    # Nothing in b10948..b10977 touches tools/server/server-context.cpp, the
    # speculative helpers or the recurrent memory; the actual image + DFlash2
    # request was re-run on b10977 and still fails with HTTP 500 on HIP and
    # Vulkan.
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
    [(10895, False), (10896, True), (10948, True), (10964, True), (10977, True)],
)
def test_vision_with_dflash2_stays_gated_through_b10977(
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


def test_documented_alternatives_still_build_on_b10977(tmp_path, monkeypatch):
    model, draft, profile, config = _qwen_vision_dflash2(tmp_path)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 10977)
    monkeypatch.setattr(
        tuner, "_probe_supported_flags", lambda _: {"--spec-type", "--mmproj"}
    )
    text_only = tuner.build_command(
        model, config, profile, draft_model=draft, enable_speculative=False
    )
    assert "--mmproj" in text_only and "-md" not in text_only
    model.mmproj = None
    assert "-md" in tuner.build_command(model, config, profile, draft_model=draft)
