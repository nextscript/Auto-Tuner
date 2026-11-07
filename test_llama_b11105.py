"""b11105 CLI, sampling environment precedence and model-inventory regressions."""

import json
from pathlib import Path

import pytest

import tuner
from settings_loader import ModelProfile, load_profiles, match_profile
from test_kv_policy import _model
from test_llama_b11063 import _config, _qwen_vision_dflash2
from test_smoke import _fake_model_md, _fake_system

ROOT = Path(__file__).resolve().parent


def _profiles():
    return load_profiles(ROOT / "settings")


def test_b11105_manifest_keeps_flags_but_updates_help():
    current = json.loads((ROOT / "docs/llama-b11105-server-flags.json").read_text())
    previous = json.loads((ROOT / "docs/llama-b11063-server-flags.json").read_text())
    assert current["tag"] == "b11105"
    assert current["commit"] == "348f853b7"
    assert current["flags"] == previous["flags"]
    assert len(current["flags"]) == 415
    assert set(current["binaries"]) == {"vulkan", "hip"}
    for backend in current["binaries"]:
        assert (
            current["binaries"][backend]["help_sha256"]
            != previous["binaries"][backend]["help_sha256"]
        )  # Six sampling env vars and --host documentation, not new flags.
    assert len({x["help_sha256"] for x in current["binaries"].values()}) == 1
    for profile in _profiles():
        for arg in profile.extra_args:
            if arg.startswith("--"):
                assert arg.split("=", 1)[0] in current["flags"]


@pytest.mark.parametrize("presence", [0.0, 0.4])
def test_presence_penalty_including_zero_is_explicit(tmp_path, presence):
    model = _model(tmp_path)
    profile = match_profile("Qwen3-4B", _profiles())
    config = tuner.compute_config(model, _fake_system(), profile, user_ctx=4096)
    config.sampling["presence_penalty"] = presence
    command = tuner.build_command(
        model, config, profile, enable_speculative=False, enable_ngram=False
    )
    assert command.count("--presence-penalty") == 1
    assert command[command.index("--presence-penalty") + 1] == str(presence)
    # No frequency UI exists; explicit Extra CLI frequency remains usable.
    command = tuner.build_command(
        model, config, profile, extra_args=["--frequency-penalty", "0.2"]
    )
    assert command[command.index("--frequency-penalty") + 1] == "0.2"


@pytest.mark.parametrize(
    "name", ["MiMo-V2.6-Flash-RL-MXFP4", "MiMo-V2.6-Pro-RL-Q4_K_M"]
)
def test_mimo_v26_profile_gate_and_command(tmp_path, monkeypatch, name):
    profile = match_profile(name, _profiles(), "mimo2")
    assert profile.source_file == "mimo-v2_6.yaml"
    assert profile.min_llama_build == 11102
    assert profile.max_context == 1048576
    assert not profile.arch_fallback
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11101)
    assert not tuner.check_profile_build(profile, "llama-server")[0]
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11102)
    assert tuner.check_profile_build(profile, "llama-server")[0]
    model = _fake_model_md(
        tmp_path,
        name,
        160,
        {
            "general.architecture": "mimo2",
            "mimo2.block_count": 51,
            "mimo2.context_length": 1048576,
            "mimo2.nextn_predict_layers": 3,
            "__mtp_scan__": "absent",  # Trunk-only GGUF, not an MTP promise.
        },
    )
    assert not model.has_embedded_mtp
    cfg = _config()
    cfg.sampling = profile.sampling["chat"]
    command = tuner.build_command(model, cfg, profile, enable_ngram=False)
    assert "--jinja" in command
    assert "--spec-type" not in command
    assert command[command.index("--temp") + 1] == "1.0"
    assert command[command.index("--top-p") + 1] == "0.95"
    assert command[command.index("--top-k") + 1] == "0"
    assert command[command.index("--min-p") + 1] == "0.0"
    assert profile.sampling["coding"] == profile.sampling["chat"]
    assert (
        match_profile("MiMo-V2.5", _profiles(), "mimo2").source_file
        != profile.source_file
    )


def test_mimo_v26_interleaved_kv_uses_full_attention_and_asymmetric_heads():
    # Flash converter: 48 trunk + 3 NextN; 9 GA layers (4 KV heads),
    # 39 SWA + 3 NextN layers (8 KV heads). K=192, V=128, not embd/heads=64.
    pattern = [False] * 9 + [True] * 42
    md = {
        "general.architecture": "mimo2",
        "mimo2.block_count": 51,
        "mimo2.attention.head_count_kv": [4] * 9 + [8] * 42,
        "mimo2.attention.sliding_window_pattern": pattern,
        "mimo2.attention.sliding_window": 128,
        "mimo2.attention.key_length": 192,
        "mimo2.attention.value_length": 128,
    }
    assert tuner._kv_per_token_total_mb_from_metadata(md) == pytest.approx(
        9 * 4 * (192 + 128) * 2 / 1024**2
    )


@pytest.mark.parametrize("variant", ["RL", "SFT"])
def test_fastcontext_4b_follows_gguf_defaults(tmp_path, variant):
    name = f"FastContext-1.0-4B-{variant}-Q8_0"
    profile = match_profile(name, _profiles(), "qwen3")
    assert profile.source_file == "fastcontext-1_0-4b.yaml"
    assert profile.max_context == 262144
    assert profile.extra_args == ["--jinja"]
    assert not profile.arch_fallback
    model = _fake_model_md(
        tmp_path,
        name,
        4,
        {"general.architecture": "qwen3", "qwen3.context_length": 262144},
    )
    cfg = tuner.compute_config(model, _fake_system(), profile, user_ctx=16384)
    cmd = tuner.build_command(model, cfg, profile, enable_speculative=False)
    for flag, value in [("--temp", "0.7"), ("--top-p", "0.8"), ("--top-k", "20")]:
        assert cmd[cmd.index(flag) + 1] == value
    assert (
        match_profile("FastContext-1.0-30B", _profiles(), "qwen3moe").source_file
        != profile.source_file
    )


@pytest.mark.parametrize("name", ["xing4_0-29b-IQ4_NL", "renamed-model"])
def test_xing_architecture_needs_the_b12156_loader(tmp_path, name, monkeypatch):
    # b11105 recognised xing4_0 without a loader; mainline added it in
    # b12156 (PR #31589), so the block became a build floor.
    model = _fake_model_md(tmp_path, name, 19, {"general.architecture": "xing4_0"})
    profile = match_profile(name, _profiles(), "xing4_0")
    assert profile.source_file == "xing-4_0.yaml"
    assert not profile.runtime_block_reason
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 12155)
    assert not tuner.check_profile_build(profile, "llama-server")[0]
    ok, message, _ = tuner.check_model_build(model, "llama-server")
    assert not ok and "b12156" in message and "xing4_0" in message
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 12156)
    assert tuner.check_profile_build(profile, "llama-server")[0]
    assert tuner.check_model_build(model, "llama-server")[0]


def test_voxcpm2_is_not_a_standalone_chat_model(tmp_path):
    model = _fake_model_md(
        tmp_path, "VoxCPM2-BaseLM-F16", 3, {"general.architecture": "minicpm4"}
    )
    profile = match_profile(model.name, _profiles(), model.architecture)
    assert profile.source_file == "voxcpm2.yaml"
    assert not profile.arch_fallback
    with pytest.raises(ValueError, match="Voice Lab TTS component"):
        tuner.build_command(model, _config(), ModelProfile("custom"))
    model.name = "custom-name"
    model.metadata["general.name"] = "VoxCPM2-BaseLM"
    assert tuner._model_runtime_block_reason(model)
    model.metadata.pop("general.name")
    assert not tuner._model_runtime_block_reason(model)  # No minicpm4-wide veto.
    assert (
        match_profile(model.name, _profiles(), "minicpm4").source_file
        != profile.source_file
    )


def test_muse_parser_note_and_existing_loader_contract():
    profile = match_profile("Muse-Glimmer-30B", _profiles(), "muse-glimmer")
    assert profile.min_llama_build == 10353
    assert "--jinja" in profile.extra_args
    assert "b11100" in profile.notes and "#29242" in profile.notes
    for pack in sorted((ROOT / "assets/languages").glob("*.json")):
        notes = json.loads(pack.read_text(encoding="utf-8"))["profile_notes"]
        assert "b11100" in notes["muse-glimmer.yaml"]
        assert "#29242" in notes["muse-glimmer.yaml"]
        for key in [
            "mimo-v2_6.yaml",
            "fastcontext-1_0-4b.yaml",
            "xing-4_0.yaml",
            "voxcpm2.yaml",
        ]:
            assert notes[key]


def test_vision_dflash2_gate_reverified_b11105(tmp_path, monkeypatch):
    model, draft, profile, config = _qwen_vision_dflash2(tmp_path)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11105)
    with pytest.raises(ValueError, match="verified through b11249"):
        tuner.build_command(model, config, profile, draft_model=draft)


def test_voice_lab_recipe_is_isolated():
    recipe = (ROOT / "building llama.cpp/voicelab_voxcpm2_vulkan_build.ps1").read_text()
    assert "L:/LAB/ai-local/voicelab_llama.cpp-omni" in recipe
    assert "build-voicelab" in recipe
    assert "--target voxcpm2-cli llama-tts-server" in recipe
    assert "-DGGML_VULKAN=ON" in recipe
    assert "Remove-Item" not in recipe
