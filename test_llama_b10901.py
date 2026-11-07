"""b10901 model/profile compatibility and conservative indexer budgeting."""

import json
from pathlib import Path

import pytest

import tuner
from settings_loader import ModelProfile, load_profiles, match_profile
from test_kv_policy import _model
from test_smoke import _fake_system

ROOT = Path(__file__).resolve().parent


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


@pytest.mark.parametrize(
    "name",
    [
        "DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf",
        "DeepSeek_V4.1_Flash-Q4.gguf",
        "DeepSeekV4.1-Flash.gguf",
        "DeepSeek-V4_1-Flash.gguf",
        "DeepSeek_V41_Flash.gguf",
        "deepseek41-flash.gguf",
    ],
)
@pytest.mark.parametrize("arch", [None, "deepseek4", "deepseek41"])
def test_v41_never_inherits_v4_runtime(name, arch, monkeypatch):
    profile = match_profile(name, _profiles(), arch)
    assert profile.source_file == "deepseek-v4_1.yaml"
    assert profile.max_context == 1048576
    assert profile.sampling["chat"]["top_p"] == 0.95
    # b11870 (PR #31086) added the deepseek41 loader; the block became a floor.
    assert not profile.runtime_block_reason
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 11869)
    allowed, message, _ = tuner.check_profile_build(profile, "server")
    assert not allowed and "b11870" in message
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 11870)
    assert tuner.check_profile_build(profile, "server")[0]


def test_v41_metadata_overrides_generic_filename():
    profile = match_profile("DeepSeek-V4-Custom", _profiles(), "deepseek41")
    assert profile.source_file == "deepseek-v4_1.yaml"
    assert profile.min_llama_build == 11870


@pytest.mark.parametrize(
    "name,metadata",
    [
        ("renamed", {"general.architecture": "deepseek_v41"}),
        ("renamed", {"general.architecture": "deepseek_v41_text"}),
        (
            "renamed",
            {"general.architecture": "deepseek4", "deepseek4.engram.head_count": 8},
        ),
        ("DeepSeek-V4.1-Flash", {"general.architecture": "deepseek4"}),
    ],
)
def test_v41_model_gate_catches_legacy_conversion_and_generic_profile(
    tmp_path, name, metadata
):
    model = _model(tmp_path)
    model.name = name
    model.metadata = metadata
    assert not tuner.check_model_build(model, "server")[0]
    with pytest.raises(ValueError, match="PR #28696"):
        tuner.build_command(model, _config(), ModelProfile("custom"))
    with pytest.raises(ValueError, match="PR #28696"):
        tuner.build_command(
            _model(tmp_path),
            _config(),
            ModelProfile("custom"),
            draft_model=model,
        )


def test_current_v41_conversion_passes_the_model_gate(tmp_path, monkeypatch):
    model = _model(tmp_path)
    model.name = "DeepSeek-V4.1-Flash-Q2_K"
    model.metadata = {"general.architecture": "deepseek41"}
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 11870)
    assert tuner.check_model_build(model, "server")[0]
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 11869)
    ok, message, _ = tuner.check_model_build(model, "server")
    assert not ok and "b11870" in message


@pytest.mark.parametrize("name", ["DeepSeek-V4-Flash", "DeepSeek-V4-Pro", "custom"])
def test_original_v4_remains_unchanged(tmp_path, name):
    profile = match_profile(name, _profiles(), "deepseek4")
    assert profile.source_file == "deepseek-v4.yaml"
    assert not profile.runtime_block_reason
    assert profile.min_llama_build == 10254
    assert profile.sampling["chat"]["top_p"] == 1.0
    model = _model(tmp_path)
    model.name = name
    model.metadata = {"general.architecture": "deepseek4"}
    assert tuner.check_model_build(model, "server")[0]


@pytest.mark.parametrize("mode", ["chat", "coding"])
def test_nex_sampling_and_template_contract(tmp_path, mode):
    profile = match_profile("Nex-N2.5-mini-Q4_K_M", _profiles(), "qwen35moe")
    assert profile.source_file == "nex-n2_5-mini.yaml"
    assert profile.max_context == 262144 and not profile.rope_scale_enabled
    assert profile.sampling[mode] == {
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
    }
    model = _model(tmp_path)
    cfg = tuner.compute_config(model, _fake_system(), profile, mode=mode, user_ctx=4096)
    cmd = tuner.build_command(model, cfg, profile, enable_speculative=False)
    assert "--jinja" in cmd
    assert "--chat-template" not in cmd and "--chat-template-file" not in cmd
    assert "--tool-call-parser" not in cmd and "--reasoning-parser" not in cmd
    assert profile.ngram_method == "ngram-map-k4v"
    assert (
        match_profile("renamed", _profiles(), "qwen35moe").source_file
        == "qwen3_5-3_6.yaml"
    )
    assert (
        match_profile("Qwen3.6-35B", _profiles()).sampling["chat"]["temperature"] == 1.0
    )


@pytest.mark.parametrize(
    "name",
    ["GLM-5.3-CYBERSECURITY-FP8-Q4", "glm5.3-cybersecurity", "glm_5_3_cybersecurity"],
)
def test_cybersecurity_keeps_distinct_generation_defaults(name, tmp_path):
    profiles = _profiles()
    cyber = match_profile(name, profiles, "glm-dsa")
    base = match_profile("GLM-5.3", profiles, "glm-dsa")
    assert cyber.source_file == "glm-5_3-cybersecurity.yaml"
    assert cyber.max_context == base.max_context == 1048576
    assert cyber.min_llama_build == base.min_llama_build == 10174
    for mode in ("chat", "coding"):
        assert cyber.sampling[mode]["repeat_penalty"] == 1.1
        assert base.sampling[mode]["repeat_penalty"] == 1.0
    model = _model(tmp_path)
    config = tuner.compute_config(model, _fake_system(), cyber, user_ctx=4096)
    cmd = tuner.build_command(model, config, cyber)
    assert cmd[cmd.index("--repeat-penalty") + 1] == "1.1"
    assert "--no-reasoning-preserve" in cmd
    assert "--reasoning-preserve" not in cmd
    assert "--reasoning-preserve" in base.extra_args
    assert match_profile("renamed", profiles, "glm-dsa").source_file == base.source_file
    assert match_profile("GLM-5.3-Flash", profiles).source_file == "glm-5_3_flash.yaml"


@pytest.mark.parametrize(
    "build,vision,speculative,dflash2,blocked",
    [
        (10901, True, True, True, True),
        (10901, False, True, True, False),
        (10901, True, False, True, False),
        (10901, True, True, False, False),
        (10878, True, True, True, False),
        (None, True, True, True, False),
    ],
)
def test_b10901_vision_dflash2_gate_is_narrow(
    tmp_path, monkeypatch, build, vision, speculative, dflash2, blocked
):
    model = _model(tmp_path)
    model.name = "Qwen3.8-27B"
    model.metadata["general.architecture"] = "qwen35"
    model.mmproj = tmp_path / "mmproj.gguf" if vision else None
    draft = _model(tmp_path)
    draft.metadata = {"general.architecture": "dflash"}
    if dflash2:
        draft.metadata.update(
            {
                "dflash.conv_kernel_size": 2,
                "dflash.conv_group_size": 16,
                "dflash.selector_rank": 256,
                "dflash.selector_top_k": 16,
            }
        )
    profile = match_profile(model.name, _profiles(), "qwen35")
    config = _config()
    config.sampling = profile.sampling["chat"]
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: build)
    monkeypatch.setattr(
        tuner, "_probe_supported_flags", lambda _: {"--spec-type", "--mmproj"}
    )
    if blocked:
        with pytest.raises(ValueError, match="Disable Draft.*disable Vision"):
            tuner.build_command(model, config, profile, draft_model=draft)
    else:
        cmd = tuner.build_command(
            model, config, profile, draft_model=draft, enable_speculative=speculative
        )
        assert ("--mmproj" in cmd) == vision
        assert ("-md" in cmd) == speculative


def test_b10901_indexer_estimate_stays_safe_for_older_runtimes():
    # PR #28330 / b10889 removes indexer V. Without a runtime-qualified
    # memory contract the old reserve must stay; a global subtraction would
    # underbudget exported configs used with b10878/unknown forks.
    md = {
        "general.architecture": "qwen4exp",
        "qwen4exp.block_count": 48,
        "qwen4exp.full_attention_interval": 4,
        "qwen4exp.attention.head_count": 16,
        "qwen4exp.attention.head_count_kv": 2,
        "qwen4exp.attention.key_length": 256,
        "qwen4exp.attention.value_length": 256,
        "qwen4exp.attention.indexer.key_length": 128,
    }
    k_mb, v_mb = tuner.kv_per_token_parts_mb_from_metadata(md)
    assert k_mb * 1024**2 == pytest.approx(15360)
    assert v_mb * 1024**2 == pytest.approx(18432)
    b10901_v_bytes = 12288
    assert v_mb * 1024**2 - b10901_v_bytes == pytest.approx(6144)


def test_pinned_b10901_local_flags_and_profiles():
    manifest = json.loads((ROOT / "docs/llama-b10901-server-flags.json").read_text())
    flags = set(manifest["flags"])
    previous = set(
        json.loads((ROOT / "docs/llama-b10878-server-flags.json").read_text())["flags"]
    )
    # --rpc is a local CMake feature difference, not a removal in upstream.
    assert previous - flags == {"--rpc"}
    assert not flags - previous
    assert not flags.intersection(tuner._LEGACY_LOAD_MODES)
    for profile in _profiles():
        for token in profile.extra_args:
            if token.startswith("--"):
                assert token.split("=", 1)[0] in flags, (profile.source_file, token)
