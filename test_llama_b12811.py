"""b12811 audit (1.1.0-dev, after stable v1.1.0 = b12774):
new server flag (--ui-agent), new architectures (moshi, hy_v4_flash, mainline
raon1), new projectors (moshi, moshi_gen, hy_v4_flash), Inkling 2 DSpark /
DFlash2 floors, DFlash target_hidden_size multi-stream guard, and new local
model profiles (Ternary-Bonsai 2.5, SenseNova-Loop 9B, split Raon-1 Mini vs
Raon-1 Nano)."""

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


def test_b12811_manifest_diff_against_b12608():
    current = _manifest("b12811")
    previous = _manifest("b12608")
    assert current["tag"] == "b12811"
    assert current["commit"] == "00d5c142c"
    assert set(current["flags"]) - set(previous["flags"]) == {"--ui-agent"}
    assert set(previous["flags"]) - set(current["flags"]) == set()
    assert len(current["flags"]) == 430
    assert len([f for f in current["flags"] if f.startswith("--")]) == 344
    assert set(current["binaries"]) == {"vulkan", "hip"}
    assert "--ui-agent" not in tuner._ARG_FLAGS_WITH_VALUES
    for profile in _profiles():
        for arg in profile.extra_args:
            if arg.startswith("--"):
                assert arg.split("=", 1)[0] in current["flags"], (
                    profile.source_file,
                    arg,
                )


def test_ui_agent_boolean_flag_is_pruned_without_eating_next_token():
    cmd = ["llama-server", "-m", "m.gguf", "--ui-agent", "-c", "4096"]
    kept, removed = tuner._filter_command_for_supported_flags(
        cmd, set(_manifest("b12608")["flags"])
    )
    assert kept == ["llama-server", "-m", "m.gguf", "-c", "4096"]
    assert removed == ["--ui-agent"]
    assert tuner._filter_command_for_supported_flags(
        cmd, set(_manifest("b12811")["flags"])
    ) == (cmd, [])


# ------------------------------------------------------------------ floors


@pytest.mark.parametrize(
    "arch,floor",
    [
        ("moshi", 12619),
        ("hy_v4_flash", 12755),
    ],
)
def test_b12811_new_architectures_have_build_floors(tmp_path, monkeypatch, arch, floor):
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
        ("moshi", 12619),
        ("moshi_gen", 12619),
        ("hy_v4_flash", 12755),
    ],
)
def test_b12811_new_projectors_have_build_floors(
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
            "clip.projector_type": projector,
        },
    )
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: floor - 1)
    ok, msg, _ = tuner.check_model_build(model, "x")
    assert not ok and f"b{floor}+" in msg and projector in msg
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: floor)
    assert tuner.check_model_build(model, "x")[0]


# ------------------------------------------------------------------ profiles


def test_bonsai_2_5_matches_dedicated_mainline_profile_without_breaking_older():
    profiles = _profiles()
    p25 = match_profile("Ternary-Bonsai-2.5-27B-Q2_0-mtp", profiles, "qwen35")
    assert p25.source_file == "bonsai-2_5.yaml"
    assert p25.max_context == 262144
    assert not p25.server_binary
    assert p25.sampling["chat"]["temperature"] == 1.0
    assert p25.sampling["chat"]["top_p"] == 0.95
    assert p25.sampling["chat"]["top_k"] == 20
    assert p25.sampling["chat"]["min_p"] == 0.0

    # Existing Bonsai profiles remain intact.
    assert (
        match_profile("Ternary-Bonsai-2-27B-PTQ1_0", profiles, "qwen35").source_file
        == "bonsai-2-27b.yaml"
    )
    assert (
        match_profile("Ternary-Bonsai-2-9B-Q2_0", profiles, "qwen35").source_file
        == "bonsai-2.yaml"
    )
    assert (
        match_profile("Ternary-Bonsai-8B-Q2_0", profiles, "qwen3").source_file
        == "bonsai-ternary.yaml"
    )


def test_sensenova_loop_matches_and_gates_on_runtime_marker():
    profiles = _profiles()
    for name in ("SenseNova-Loop-9B-BF16", "SenseNova-Loop-9B-Q8_0", "renamed"):
        p = match_profile(name, profiles, "sensenova_loop")
        assert p.source_file == "sensenova-loop.yaml"
        assert p.required_runtime_markers == ["sensenova_loop"]
        assert p.max_context == 524288
        assert p.sampling["chat"]["temperature"] == 1.0
        assert p.sampling["chat"]["top_p"] == 0.95


def test_raon1_mini_and_nano_are_isolated_by_runtime_marker():
    profiles = _profiles()
    mini = match_profile("raon-1-mini-preview-Q6_K", profiles, "raon1")
    nano = match_profile("raon-1-nano-Q8_0_MTP", profiles, "raon1nano")
    assert mini.source_file == "raon-1.yaml"
    assert mini.required_runtime_markers == ["raon1"]
    assert mini.auto_draft is False
    assert nano.source_file == "raon-1-nano.yaml"
    assert nano.required_runtime_markers == ["raon1nano"]
    # Architecture-authoritative fallback for renamed files stays separated.
    assert match_profile("renamed", profiles, "raon1").source_file == "raon-1.yaml"
    assert (
        match_profile("renamed", profiles, "raon1nano").source_file
        == "raon-1-nano.yaml"
    )


def test_hy_v4_flash_matches_hy4_mainline_and_enforces_b12755():
    profiles = _profiles()
    p = match_profile("HY4-Flash-220B-A12B-Q2_K_XL", profiles, "hy_v4_flash")
    assert p.source_file == "hy4-mainline.yaml"
    assert p.max_context == 1048576
    assert (
        match_profile("renamed", profiles, "hy_v4_flash").source_file
        == "hy4-mainline.yaml"
    )
    assert "hy_v4_flash" in scanner._HYBRID_ARCHS
    assert "hy_v4_flash" in tuner._KNOWN_MOE_ARCHS
    assert "sensenova_loop" in scanner._HYBRID_ARCHS


# ------------------------------------------------------------------ draft guards


def test_dflash_target_hidden_size_guard_blocks_mainline_and_allows_unversioned_fork(
    tmp_path, monkeypatch
):
    draft = _fake_model_md(
        tmp_path,
        "dflash-raon-1-mini-preview-Q8_0",
        0.7,
        {
            "general.architecture": "dflash",
            "dflash.embedding_length": 2560,
            "dflash.target_hidden_size": 10240,
            "dflash.conv_kernel_size": 2,
            "dflash.conv_group_size": 16,
            "dflash.selector_rank": 256,
            "dflash.selector_top_k": 16,
        },
    )
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 12811)
    monkeypatch.setattr(tuner, "_probe_binary_version_output", lambda _: "b12811")
    ok, msg, _ = tuner.check_draft_model_build(draft, "x")
    assert not ok and "target_hidden_size=10240" in msg and "fc.weight" in msg
    assert tuner._DFLASH_TARGET_HIDDEN_SIZE_VERIFIED_BUILD >= 12811

    # Unversioned wrapper / custom fork falls through to warning-only behavior.
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: None)
    monkeypatch.setattr(tuner, "_probe_binary_version_output", lambda _: "dflash2")
    ok, _msg, _ = tuner.check_draft_model_build(draft, "x")
    assert ok


def test_inkling2_dspark_and_dflash2_draft_build_floors(tmp_path, monkeypatch):
    dspark = _fake_model_md(
        tmp_path,
        "dspark-Inkling-2-Mini-Q8_0",
        0.5,
        {
            "general.architecture": "dflash",
            "__dspark_scan__": "found",
            "dflash.has_markov_context": True,
        },
    )
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 12719)
    ok, msg, _ = tuner.check_draft_model_build(dspark, "x")
    assert not ok and "b12720+" in msg
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 12720)
    assert tuner.check_draft_model_build(dspark, "x")[0]

    dflash2 = _fake_model_md(
        tmp_path,
        "dflash2-Inkling-2-Mini-Q8_0",
        0.5,
        {
            "general.architecture": "dflash",
            "dflash.conv_kernel_size": 2,
            "dflash.conv_group_size": 16,
            "dflash.selector_rank": 256,
            "dflash.selector_top_k": 16,
            "dflash.selector_softcap_after_correction": True,
        },
    )
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 12784)
    monkeypatch.setattr(tuner, "_probe_binary_version_output", lambda _: "b12784")
    ok, msg, _ = tuner.check_draft_model_build(dflash2, "x")
    assert not ok and "b12785+" in msg
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 12785)
    monkeypatch.setattr(tuner, "_probe_binary_version_output", lambda _: "b12785")
    assert tuner.check_draft_model_build(dflash2, "x")[0]


def test_qwen4exp_mtp_gate_reverified_on_b12811():
    assert tuner._QWEN4EXP_MTP_HEAD_VERIFIED_BUILD >= 12811
