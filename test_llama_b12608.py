"""b12608 audit (1.0.0-dev, after stable v1.0.0 = b12359):
new server flags (--ple/--shard, --moe-cache-cpu-share, --tts-lang/--tts-voice,
--allow-unauthenticated-tools, --media-allow-private-urls,
--ui-mcp-proxy-allow-private-urls), removed -dt/--defrag-thold, FP8 KV cache
(f8_e4m3), new architecture / projector floors, and Inkling 2 / Domyn Edge /
Zaya1 / Raon-1 profiles."""

import json
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


def test_b12608_manifest_diff_against_b12187():
    current = _manifest("b12608")
    previous = _manifest("b12187")
    assert current["tag"] == "b12608"
    assert current["commit"] == "93cb37ea7"
    assert set(current["flags"]) - set(previous["flags"]) == {
        "--allow-unauthenticated-tools",
        "--media-allow-private-urls",
        "--moe-cache-cpu-share",
        "--ple",
        "--shard",
        "--tts-lang",
        "--tts-voice",
        "--ui-mcp-proxy-allow-private-urls",
    }
    assert set(previous["flags"]) - set(current["flags"]) == {
        "--defrag-thold",
        "-dt",
    }
    assert len(current["flags"]) == 429
    assert len([f for f in current["flags"] if f.startswith("--")]) == 343
    assert set(current["binaries"]) == {"vulkan", "hip"}
    for profile in _profiles():
        for arg in profile.extra_args:
            if arg.startswith("--"):
                assert arg.split("=", 1)[0] in current["flags"], (
                    profile.source_file,
                    arg,
                )


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--moe-cache-cpu-share", "0.25"),
        ("--ple", "ple.gguf"),
        ("--shard", "extra.gguf"),
        ("--tts-lang", "de"),
        ("--tts-voice", "alloy"),
    ],
)
def test_b12608_new_value_flags_are_pruned_with_their_value(flag, value):
    assert flag in tuner._ARG_FLAGS_WITH_VALUES
    cmd = ["llama-server", "-m", "m.gguf", flag, value, "-c", "4096"]
    kept, removed = tuner._filter_command_for_supported_flags(
        cmd, set(_manifest("b12187")["flags"])
    )
    assert kept == ["llama-server", "-m", "m.gguf", "-c", "4096"]
    assert removed == [f"{flag} {value}"]
    assert tuner._filter_command_for_supported_flags(
        cmd, set(_manifest("b12608")["flags"])
    ) == (cmd, [])


def test_ple_and_shard_are_repeatable_aliases():
    assert "--ple" in tuner._REPEATABLE_VALUE_FLAGS
    assert "--shard" in tuner._REPEATABLE_VALUE_FLAGS
    assert {"--ple", "--shard"} in tuner._FLAG_ALIAS_GROUPS


# ------------------------------------------------------------------ FP8 KV


def test_fp8_kv_quant_factor_and_expert_option():
    assert tuner.kv_quant_factor("f8_e4m3") == 0.50
    assert tuner.kv_quant_factor("f8_e4m3") < tuner.kv_quant_factor("q8_0")
    qt_launcher = pytest.importorskip("qt_launcher")
    assert "f8_e4m3" in qt_launcher.ExpertPanel._KV_QUANT_OPTIONS


# ------------------------------------------------------------------ floors


@pytest.mark.parametrize(
    "arch,floor",
    [
        ("inkling2", 12406),
        ("zonos3", 12424),
        ("kokoro", 12494),
        ("tiny-voca", 12497),
        ("zaya1", 12519),
        ("domynedge", 12568),
        ("inkling2_loop", 12600),
    ],
)
def test_b12608_new_architectures_have_build_floors(tmp_path, monkeypatch, arch, floor):
    model = _fake_model_md(
        tmp_path, f"renamed-{arch}-Q4_K_M", 2.0, {"general.architecture": arch}
    )
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: floor - 1)
    ok, msg, _ = tuner.check_model_build(model, "x")
    assert not ok and f"b{floor}+" in msg and arch in msg
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: floor)
    assert tuner.check_model_build(model, "x")[0]


@pytest.mark.parametrize(
    "projector,floor",
    [
        ("magevl", 12192),
        ("kimik31", 12259),
        ("inkling2v", 12406),
        ("zonos3_gen", 12424),
        ("zonos3_spkenc", 12424),
        ("higgs_audio", 12426),
        ("vibevoice_asr", 12426),
        ("fish_s2_gen", 12452),
        ("fish_s2_spkenc", 12452),
        ("yue2_gen", 12458),
        ("minimax_music3_gen", 12479),
        ("kokoro", 12494),
        ("tiny_voca", 12497),
        ("voxcpm2_gen", 12531),
        ("voxcpm2_spkenc", 12531),
    ],
)
def test_b12608_new_projectors_have_build_floors(
    tmp_path, monkeypatch, projector, floor
):
    model = _fake_model_md(
        tmp_path, "mm-model-Q4_K_M", 4.0, {"general.architecture": "qwen35"}
    )
    mmproj = tmp_path / "mmproj-mm-model-F16.gguf"
    mmproj.write_bytes(b"GGUF")
    model.mmproj = mmproj
    monkeypatch.setattr(
        scanner,
        "read_gguf_metadata",
        lambda path: {
            "general.architecture": "clip",
            "clip.vision.projector_type": projector,
        },
    )
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: floor - 1)
    ok, msg, _ = tuner.check_model_build(model, "x")
    assert not ok and f"b{floor}+" in msg and projector in msg
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: floor)
    assert tuner.check_model_build(model, "x")[0]


# ------------------------------------------------------------------ profiles


def test_inkling_2_does_not_fall_into_ling_ring_and_blocks_pre_pr_mini(
    tmp_path, monkeypatch
):
    profiles = _profiles()
    # Previously matched ling-ring.yaml via substring "ling-2".
    assert (
        match_profile("Inkling-2-Mini-UD-Q8_K_XL", profiles, "inkling2").source_file
        == "inkling-2.yaml"
    )
    assert (
        match_profile("Inkling-2-Mini-Q4_K_M", profiles, "inkling2_loop").source_file
        == "inkling-2.yaml"
    )
    assert (
        match_profile("renamed-model", profiles, "inkling2_loop").source_file
        == "inkling-2.yaml"
    )

    # Pre-PR Unsloth conversion (inkling2 + inkling2.num_loops) is refused before launch.
    pre_pr = _fake_model_md(
        tmp_path,
        "Inkling-2-Mini-UD-Q8_K_XL",
        24.0,
        {
            "general.architecture": "inkling2",
            "inkling2.block_count": 48,
            "inkling2.num_loops": 3,
            "inkling2.loop.core_start": 6,
            "inkling2.loop.core_end": 18,
        },
    )
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 12608)
    ok, msg, _ = tuner.check_model_build(pre_pr, "x")
    assert not ok and "inkling2_loop" in msg and "b12600+" in msg

    # Mainline inkling2_loop passes on b12608 and fails before b12600.
    mainline_mini = _fake_model_md(
        tmp_path,
        "Inkling-2-Mini-Q4_K_M",
        16.0,
        {
            "general.architecture": "inkling2_loop",
            "inkling2_loop.block_count": 24,
            "inkling2_loop.loop.count": 3,
        },
    )
    assert tuner.check_model_build(mainline_mini, "x")[0]
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 12599)
    ok, msg, _ = tuner.check_model_build(mainline_mini, "x")
    assert not ok and "b12600+" in msg


@pytest.mark.parametrize(
    "name,arch,profile_file,floor",
    [
        ("Domyn-Edge-4B-Q8_0", "domynedge", "domyn-edge.yaml", 12568),
        ("Zaya1-8B-Q4_K_M", "zaya1", "zaya1.yaml", 12519),
        ("Inkling-2-Small-Q4_K_M", "inkling2", "inkling-2.yaml", 12406),
    ],
)
def test_new_b12608_profiles_match_and_gate(name, arch, profile_file, floor):
    profile = match_profile(name, _profiles(), arch)
    assert profile.source_file == profile_file
    assert profile.min_llama_build == floor
    assert not profile.runtime_block_reason


def test_raon1_matches_and_uses_runtime_marker():
    profiles = _profiles()
    for name, arch, expected_file, expected_marker in [
        ("raon-1-mini-preview-Q6_K", "raon1", "raon-1.yaml", "raon1"),
        ("raon-1-nano-Q8_0_MTP", "raon1nano", "raon-1-nano.yaml", "raon1nano"),
    ]:
        profile = match_profile(name, profiles, arch)
        assert profile.source_file == expected_file
        assert profile.required_runtime_markers == [expected_marker]
        assert profile.sampling["chat"]["temperature"] == 0.1
        assert profile.sampling["chat"]["presence_penalty"] == 1.5
        assert profile.sampling["coding"]["temperature"] == 1.0
        assert profile.sampling["coding"]["presence_penalty"] == 0.0


def test_hybrid_and_moe_tables_cover_new_architectures():
    for arch in ("inkling2", "inkling2_loop", "zaya1", "zonos3", "raon1", "raon1nano"):
        assert arch in scanner._HYBRID_ARCHS
    for arch in ("inkling2", "inkling2_loop", "zaya1", "raon1"):
        assert arch in tuner._KNOWN_MOE_ARCHS


def test_qwen4exp_mtp_gate_reverified_on_b12608():
    assert tuner._QWEN4EXP_MTP_HEAD_VERIFIED_BUILD >= 12608
