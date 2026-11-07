"""b10948 audit: help-text-only CLI change and the still-open vision+DFlash2 gate."""

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


def test_pinned_b10948_manifest_keeps_the_b10930_option_set():
    manifest = _manifest("b10948")
    assert manifest["tag"] == "b10948"
    assert manifest["commit"] == "5f436dddb440a288ee5611d7d1eca564a6aca9f4"
    flags = set(manifest["flags"])
    # PR #28736 (common_schema) only rewrote the -j/--json-schema and
    # -jf/--json-schema-file help sentences; no option was added or removed.
    assert flags == set(_manifest("b10930")["flags"])
    assert {"-j", "--json-schema", "-jf", "--json-schema-file"} <= flags
    assert len(flags) == 415 and sum(f.startswith("--") for f in flags) == 328
    assert "--rpc" not in flags  # local GGML_RPC=OFF, not an upstream removal
    binaries = manifest["binaries"]
    assert set(binaries) == {"vulkan", "hip"}
    hashes = {b["help_sha256"] for b in binaries.values()}
    assert len(hashes) == 1  # one shared help text for both backends
    previous = {b["help_sha256"] for b in _manifest("b10930")["binaries"].values()}
    assert hashes != previous  # the help wording did change, the surface did not
    assert not flags.intersection(tuner._LEGACY_LOAD_MODES)
    for profile in _profiles():
        for token in profile.extra_args:
            if token.startswith("--"):
                assert token.split("=", 1)[0] in flags, (profile.source_file, token)


def test_no_bundled_profile_relies_on_the_removed_grammar_helpers():
    # examples/json_schema_to_grammar.py, regex_to_grammar.py and
    # ts-type-to-grammar.sh were deleted upstream in this range.
    for profile in _profiles():
        joined = " ".join(profile.extra_args)
        assert "json_schema_to_grammar" not in joined, profile.source_file
        assert "--json-schema" not in joined, profile.source_file
        assert "--grammar" not in joined, profile.source_file


def test_vision_dflash2_gate_is_unchanged_and_names_b10948():
    # b10948 contains no speculative/server change for the recurrent
    # DFlash2 draft memory; the actual image + DFlash2 request was re-run and
    # still fails with HTTP 500 on both backends.
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
    # b11319 is now qualified; historical broken builds remain protected.
    [(10895, False), (10896, True), (10930, True), (10948, True), (12000, False)],
)
def test_vision_with_dflash2_stays_gated_through_b10948(
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


def test_documented_alternatives_still_build_on_b10948(tmp_path, monkeypatch):
    model, draft, profile, config = _qwen_vision_dflash2(tmp_path)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 10948)
    monkeypatch.setattr(
        tuner, "_probe_supported_flags", lambda _: {"--spec-type", "--mmproj"}
    )
    text_only = tuner.build_command(
        model, config, profile, draft_model=draft, enable_speculative=False
    )
    assert "--mmproj" in text_only and "-md" not in text_only
    model.mmproj = None
    assert "-md" in tuner.build_command(model, config, profile, draft_model=draft)
