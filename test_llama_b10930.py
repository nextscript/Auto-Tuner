"""b10930 audit: unchanged CLI surface and the open-ended vision+DFlash2 gate."""

import json
from pathlib import Path

import pytest

import tuner
from settings_loader import load_profiles, match_profile
from test_kv_policy import _model

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


def test_pinned_b10930_manifest_matches_b10901_surface():
    manifest = _manifest("b10930")
    assert manifest["tag"] == "b10930"
    assert manifest["commit"] == "56381e407c0ccfb3a6f71e668a27a901001d22ce"
    flags = set(manifest["flags"])
    # No commit between b10901 and b10930 touches common/arg.cpp: the local
    # HIP and Vulkan help texts are byte-identical to b10901.
    assert flags == set(_manifest("b10901")["flags"])
    assert len(flags) == 415 and sum(f.startswith("--") for f in flags) == 328
    assert "--rpc" not in flags  # local GGML_RPC=OFF, not an upstream removal
    hashes = {b["help_sha256"] for b in manifest["binaries"].values()}
    assert set(manifest["binaries"]) == {"vulkan", "hip"} and len(hashes) == 1
    assert not flags.intersection(tuner._LEGACY_LOAD_MODES)
    for profile in _profiles():
        for token in profile.extra_args:
            if token.startswith("--"):
                assert token.split("=", 1)[0] in flags, (profile.source_file, token)


def test_vision_dflash2_gate_starts_at_the_regressing_build():
    # PR #28587 landed in b10896. PR #28715 (b10906) did not resolve the
    # recurrent draft-memory position gap: b10930 still fails the actual
    # image+DFlash2 request on both backends, so the gate stays open-ended.
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
    [
        (10878, False),
        (10895, False),
        (10896, True),
        (10901, True),
        (10903, True),
        (10906, True),
        (10930, True),
        # The b11319 qualification now bounds the historical open-ended gate.
        (11500, False),
        (None, False),
    ],
)
def test_vision_with_dflash2_stays_gated_through_b10930(
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
        assert cmd[cmd.index("--spec-type") + 1] == "draft-dflash"


def test_gate_stays_narrow_on_a_gated_build(tmp_path, monkeypatch):
    model, draft, profile, config = _qwen_vision_dflash2(tmp_path)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 10930)
    monkeypatch.setattr(
        tuner, "_probe_supported_flags", lambda _: {"--spec-type", "--mmproj"}
    )
    with pytest.raises(ValueError, match="b10930 cannot"):
        tuner.build_command(model, config, profile, draft_model=draft)
    # Both documented alternatives were re-run on b10930 HIP and Vulkan.
    text_only = tuner.build_command(
        model, config, profile, draft_model=draft, enable_speculative=False
    )
    assert "--mmproj" in text_only and "-md" not in text_only
    model.mmproj = None
    assert "-md" in tuner.build_command(model, config, profile, draft_model=draft)
    model.mmproj = tmp_path / "mmproj.gguf"
    draft.metadata = {"general.architecture": "dflash"}  # DFlash1, not DFlash2
    assert "--mmproj" in tuner.build_command(model, config, profile, draft_model=draft)
