"""b11160 audit: unchanged CLI, Ling-3.0-flash-VL and Gemma 4 DSpark gates."""

import json
from pathlib import Path

import pytest

import tuner
from settings_loader import load_profiles, match_profile
from test_llama_b11063 import _qwen_vision_dflash2
from test_smoke import _fake_model_md

ROOT = Path(__file__).resolve().parent


def _profiles():
    return load_profiles(ROOT / "settings")


def test_b11160_manifest_keeps_flags_and_help_text():
    current = json.loads((ROOT / "docs/llama-b11160-server-flags.json").read_text())
    previous = json.loads((ROOT / "docs/llama-b11105-server-flags.json").read_text())
    assert current["tag"] == "b11160"
    assert current["commit"] == "70c4e1582"
    assert current["flags"] == previous["flags"]
    assert len(current["flags"]) == 415
    assert set(current["binaries"]) == {"vulkan", "hip"}
    for backend, binary in current["binaries"].items():
        # b11105..b11160 changes no server option or help line (0.5.0 bump
        # only alters --version); the timestamped startup line stays stripped.
        assert binary["help_sha256"] == previous["binaries"][backend]["help_sha256"]
        assert binary["sha256"] != previous["binaries"][backend]["sha256"]
        assert current["notes"][backend]["stripped_startup_log_lines"] == 1
    for profile in _profiles():
        for arg in profile.extra_args:
            if arg.startswith("--"):
                assert arg.split("=", 1)[0] in current["flags"]


@pytest.mark.parametrize(
    "name", ["Ling-3.0-flash-VL-Q4_K_M", "ling3.0-flash-vl-BF16-00001-of-00006"]
)
def test_ling3_vl_profile_wins_over_text_profile(monkeypatch, name):
    profile = match_profile(name, _profiles(), "bailingmoe3")
    assert profile.source_file == "ling-3_0-vl.yaml"
    assert profile.min_llama_build == 11156
    assert not profile.arch_fallback
    assert profile.sampling["chat"]["temperature"] == 1.0
    assert profile.sampling["coding"]["temperature"] == 0.6
    for mode in ("chat", "coding"):
        assert profile.sampling[mode]["top_p"] == 0.95
        assert profile.sampling[mode]["top_k"] == 20
    assert {"--jinja", "--reasoning-preserve"} <= set(profile.extra_args)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11155)
    assert not tuner.check_profile_build(profile, "llama-server")[0]
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11156)
    assert tuner.check_profile_build(profile, "llama-server")[0]
    # The text model keeps its own profile and b10749 floor.
    text = match_profile("Ling-3.0-flash-AD-IQ3_M", _profiles(), "bailingmoe3")
    assert text.source_file == "ling-3.yaml"
    assert text.min_llama_build == 10749


def _bailing(tmp_path, name, sections):
    md = {"general.architecture": "bailingmoe3", "bailingmoe3.block_count": 42}
    if sections is not None:
        md["bailingmoe3.rope.dimension_sections"] = sections
    return _fake_model_md(tmp_path, name, 60, md)


@pytest.mark.parametrize(
    ("sections", "build", "allowed"),
    [
        ([8, 12, 12, 0], 11155, False),  # renamed VL file: silently wrong rope
        ([8, 12, 12, 0], 11156, True),
        ([8, 12, 12, 0], None, True),  # unknown wrapper: no false refusal
        ([0, 0, 0, 0], 11063, True),  # text GGUF with zero sections
        (None, 11063, True),  # ordinary text GGUF
    ],
)
def test_ling3_vl_mrope_gate_uses_metadata(
    tmp_path, monkeypatch, sections, build, allowed
):
    model = _bailing(tmp_path, "renamed-bailing", sections)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: build)
    ok, reason, _ = tuner.check_model_build(model, "llama-server")
    assert ok is allowed
    if not allowed:
        assert "b11156" in reason and "#29151" in reason


def _dflash_draft(tmp_path, name, extra):
    md = {"general.architecture": "dflash", "dflash.block_count": 5, **extra}
    return _fake_model_md(tmp_path, name, 1, md)


@pytest.mark.parametrize(
    "extra",
    [
        {"dflash.hidden_activation": "gelu_pytorch_tanh", "__dspark_scan__": "found"},
        {"dflash.embedding_scale": 53.3},
    ],
)
def test_gemma_backbone_dflash_needs_b11132(tmp_path, monkeypatch, extra):
    draft = _dflash_draft(tmp_path, "gemma-4-12b-it-dspark", extra)
    assert tuner.is_gemma_backbone_dflash(draft)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11131)
    ok, reason, detected = tuner.check_draft_model_build(draft, "llama-server")
    assert not ok and detected == 11131
    assert "b11132" in reason and "#29226" in reason
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11132)
    assert tuner.check_draft_model_build(draft, "llama-server")[0]


def test_qwen_style_dflash_and_dspark_keep_old_floor(tmp_path, monkeypatch):
    for extra in (
        {},
        {"dflash.hidden_activation": "silu"},
        {"__dspark_scan__": "found"},
    ):
        draft = _dflash_draft(tmp_path, "qwen3-dspark", extra)
        assert not tuner.is_gemma_backbone_dflash(draft)
        monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 10700)
        assert tuner.check_draft_model_build(draft, "llama-server")[0]


def test_ling_vl_note_and_vision_gate_stay(tmp_path, monkeypatch):
    for pack in sorted((ROOT / "assets/languages").glob("*.json")):
        text = pack.read_text(encoding="utf-8")
        assert "b11160" not in text, pack.name
        notes = json.loads(text)["profile_notes"]
        assert "b11156" in notes["ling-3_0-vl.yaml"], pack.name
    model, draft, profile, config = _qwen_vision_dflash2(tmp_path)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11160)
    with pytest.raises(ValueError, match="since b10896 .verified through b11249"):
        tuner.build_command(model, config, profile, draft_model=draft)
    assert tuner.QWEN35_VISION_DFLASH2_BROKEN_SINCE == 10896
