"""b11669 audit (v0.7.0): --spec-draft-n-ubatch, mainline prism.hadamard
(Bonsai 2, PR #29600) next to the Prism-private PTQ1_0/PQ2_0 types, and the
decision-model prompt-cache default (PR #30498)."""

import json
from pathlib import Path

import pytest

import tuner
from settings_loader import load_profiles, match_profile
from test_smoke import _fake_model_md, _fake_system

ROOT = Path(__file__).resolve().parent


def _profiles():
    return load_profiles(ROOT / "settings")


def _manifest(tag: str) -> dict:
    return json.loads(
        (ROOT / f"docs/llama-{tag}-server-flags.json").read_text(encoding="utf-8")
    )


# ------------------------------------------------------------------ manifest


def test_b11669_manifest_only_adds_spec_draft_n_ubatch():
    current = _manifest("b11669")
    previous = _manifest("b11538")
    assert current["tag"] == "b11669"
    assert current["commit"] == "620d2225c"
    # "--n-ubatch" is only quoted in the new option's help text.
    assert set(current["flags"]) - set(previous["flags"]) == {
        "--spec-draft-n-ubatch",
        "--n-ubatch",
    }
    assert set(previous["flags"]) <= set(current["flags"])
    assert len(current["flags"]) == 420
    assert set(current["binaries"]) == {"vulkan", "hip"}
    for profile in _profiles():
        for arg in profile.extra_args:
            if arg.startswith("--"):
                assert arg.split("=", 1)[0] in current["flags"]


def test_spec_draft_n_ubatch_is_a_value_flag_pruned_with_its_value():
    assert "--spec-draft-n-ubatch" in tuner._ARG_FLAGS_WITH_VALUES
    cmd = ["llama-server", "-m", "m.gguf", "--spec-draft-n-ubatch", "64", "-c", "4096"]
    kept, removed = tuner._filter_command_for_supported_flags(
        cmd, set(_manifest("b11538")["flags"])
    )
    assert kept == ["llama-server", "-m", "m.gguf", "-c", "4096"]
    assert removed == ["--spec-draft-n-ubatch 64"]
    assert tuner._filter_command_for_supported_flags(
        cmd, set(_manifest("b11669")["flags"])
    ) == (cmd, [])


# ------------------------------------------------- prism.hadamard on mainline


def _bonsai2(tmp_path, *, types, file_type):
    metadata = {
        "general.architecture": "qwen35",
        "general.file_type": file_type,
        "qwen35.block_count": 64,
        "prism.hadamard.version": 1,
        "prism.hadamard.block_size": 1024,
    }
    if types is not None:
        metadata["__ggml_types__"] = types
    return _fake_model_md(tmp_path, "Ternary-Bonsai-2-27B-X", 6.0, metadata)


def _markers(present):
    seen = []

    def check(binary, required):
        seen.append(tuple(required))
        return all(marker in present for marker in required)

    return check, seen


@pytest.mark.parametrize(
    "types,file_type,fork_marker",
    [
        ([0, 30, 143], 143, "PTQ1_0 - 1.75 bpw"),
        ([0, 30, 142], 141, "PQ2_0 - 2.13 bpw"),
        # Header-only read (shard / unreadable tensor table): file type 141,
        # its pre-rename value 142, or 143.
        (None, 141, "PQ2_0 - 2.13 bpw"),
        (None, 142, "PQ2_0 - 2.13 bpw"),
        (None, 143, "PTQ1_0 - 1.75 bpw"),
    ],
)
def test_private_prism_types_need_the_fork_even_with_mainline_hadamard(
    tmp_path, monkeypatch, types, file_type, fork_marker
):
    model = _bonsai2(tmp_path, types=types, file_type=file_type)
    # b11556+ mainline: prism.hadamard is there, the private types are not.
    check, seen = _markers({"prism.hadamard"})
    monkeypatch.setattr(tuner, "_runtime_has_required_markers", check)
    ok, message, _ = tuner.check_model_build(model, "b11669/llama-server")
    assert not ok
    assert seen == [("prism.hadamard", fork_marker)]
    assert "Prism-private" in message and "#29058" in message
    assert "b11556+" in message and "Q1_0/Q2_0" in message
    # The prism fork carries both.
    check, _ = _markers({"prism.hadamard", fork_marker})
    monkeypatch.setattr(tuner, "_runtime_has_required_markers", check)
    assert tuner.check_model_build(model, "2b_llama/llama-server")[0]


def test_mainline_converted_bonsai2_loads_on_b11556_or_the_fork(tmp_path, monkeypatch):
    # Q1_0 = ggml type 41 / file type 40, Q2_0 = type 42 / file type 41.
    model = _bonsai2(tmp_path, types=[0, 30, 41], file_type=40)
    check, seen = _markers({"prism.hadamard"})
    monkeypatch.setattr(tuner, "_runtime_has_required_markers", check)
    assert tuner.check_model_build(model, "b11669/llama-server") == (True, "", None)
    assert seen == [("prism.hadamard",)]
    check, _ = _markers(set())
    monkeypatch.setattr(tuner, "_runtime_has_required_markers", check)
    ok, message, _ = tuner.check_model_build(model, "b11538/llama-server")
    assert not ok and "b11556" in message and "#29600" in message


# ------------------------------------------- decision models: no prompt cache


def _decision(tmp_path):
    return _fake_model_md(
        tmp_path,
        "d1-3B-Q8_0",
        3.2,
        {
            "general.architecture": "lfm2",
            "lfm2.decision.type": "lfm2-d1",
            "lfm2.block_count": 16,
            "lfm2.context_length": 128000,
            "lfm2.embedding_length": 2048,
            "lfm2.attention.head_count": 32,
            "lfm2.attention.head_count_kv": 8,
        },
    )


def test_decision_models_follow_the_b11642_prompt_cache_default(tmp_path):
    model = _decision(tmp_path)
    profile = match_profile(model.name, _profiles(), model.architecture)
    assert profile.source_file == "decision-systemone.yaml"
    cfg = tuner.compute_config(
        model,
        _fake_system(vram_total=16, vram_free=15),
        profile,
        prompt_cache_ram_mib=8192,
    )
    assert cfg.prompt_cache_ram_mib == 0
    cmd = tuner.build_command(
        model, cfg, profile, enable_speculative=False, enable_ngram=False
    )
    at = cmd.index("--cache-ram")
    assert cmd[at + 1] == "0" and "--cache-reuse" not in cmd


def test_chat_models_keep_the_configured_prompt_cache(tmp_path):
    model = _fake_model_md(
        tmp_path,
        "Qwen3.6-4B-Q8_0",
        4.3,
        {
            "general.architecture": "qwen3",
            "qwen3.block_count": 36,
            "qwen3.context_length": 32768,
            "qwen3.embedding_length": 2560,
            "qwen3.attention.head_count": 32,
            "qwen3.attention.head_count_kv": 8,
        },
    )
    profile = match_profile(model.name, _profiles(), model.architecture)
    cfg = tuner.compute_config(
        model,
        _fake_system(vram_total=16, vram_free=15),
        profile,
        prompt_cache_ram_mib=4096,
    )
    assert cfg.prompt_cache_ram_mib == 4096


# ------------------------------------------------------- wording re-checks


# --------------------------------------------------------- build recipes


def _common_recipe() -> str:
    return (ROOT / "building llama.cpp/windows_llama_build_common.ps1").read_text(
        encoding="utf-8"
    )


def test_stable_version_check_rejects_a_dev_suffix():
    import re

    text = _common_recipe()
    assert r"$([regex]::Escape($ExpectedVersion))(?=\s|\(|$)" in text
    assert r"$([regex]::Escape($ExpectedVersion))\b" not in text
    pattern = r"(?im)^\s*version:\s*0\.7\.0(?=\s|\(|$)"
    assert re.search(pattern, "version: 0.7.0 (build 11669, commit 620d2225c)")
    assert not re.search(pattern, "version: 0.7.0-dev (build 11669, commit 620d2225c)")


def test_hip_recipe_points_rocm_clang_at_windows_openssl():
    text = _common_recipe()
    assert "function Get-WindowsOpenSslArgs" in text
    assert "@(Get-WindowsOpenSslArgs)" in text
    for key in (
        "OPENSSL_ROOT_DIR",
        "OPENSSL_INCLUDE_DIR",
        "SSL_EAY",
        "LIB_EAY",
    ):
        assert f'"-D{key}=' in text
    assert r"lib\VC\x64\MD\libssl.lib" in text
    # A HIP tree without OpenSSL counts as incomplete (rebuilt) when
    # OpenSSL-Win64 is installed, and only warns otherwise.
    assert "SSL_EAY:FILEPATH=.*libssl" in text
    assert "HIP build has no OpenSSL although OpenSSL-Win64 is installed" in text


# ------------------------------------------------ new upstream model support


@pytest.mark.parametrize("dtype,floor", [("vllm-decision2", 11571), ("vllm-d3", 11577)])
def test_new_vllm_decision_types_need_their_build(tmp_path, monkeypatch, dtype, floor):
    arch = "vllm-decision2" if dtype == "vllm-decision2" else "qwen35"
    model = _fake_model_md(
        tmp_path,
        f"{dtype}-Q8_0",
        1.0,
        {"general.architecture": arch, f"{arch}.decision.type": dtype},
    )
    assert (
        match_profile(model.name, _profiles(), arch).source_file
        == "decision-systemone.yaml"
    )
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: floor - 1)
    ok, message, _ = tuner.check_model_build(model, "x")
    assert not ok and f"b{floor}+" in message
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: floor)
    assert tuner.check_model_build(model, "x")[0]


def test_vllm_decision2_reads_its_prompt_in_one_batch(tmp_path):
    model = _fake_model_md(
        tmp_path,
        "Decision-2.0-Kai-mainline",
        1.0,
        {
            "general.architecture": "vllm-decision2",
            "vllm-decision2.decision.type": "vllm-decision2",
        },
    )
    assert tuner.single_batch_prompt_tokens(model) > 0


def test_fork_format_decision2_gguf_names_the_fork(tmp_path, monkeypatch):
    model = _fake_model_md(
        tmp_path,
        "Decision-2.0-Kai-0.6B-Q8_0",
        0.7,
        {"general.architecture": "qwen3", "qwen3.decision.type": "decision2"},
    )
    assert (
        match_profile(model.name, _profiles(), "qwen3").source_file
        == "decision-systemone.yaml"
    )
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 11669)
    ok, note, _ = tuner.check_model_build(model, "x")
    assert (
        ok and "Xunzhuo/llama.cpp" in note and "unsupported decision model type" in note
    )


@pytest.mark.parametrize(
    "projector,floor", [("minicpmv4_7", 11551), ("nemotron_omni", 11668)]
)
def test_new_vision_projectors_need_their_build(
    tmp_path, monkeypatch, projector, floor
):
    import scanner

    model = _fake_model_md(
        tmp_path, "vl-model-Q4_K_M", 4.0, {"general.architecture": "qwen35"}
    )
    mmproj = tmp_path / "mmproj-vl-model-F16.gguf"
    mmproj.write_bytes(b"GGUF")
    model.mmproj = mmproj
    monkeypatch.setattr(
        scanner,
        "read_gguf_metadata",
        lambda path: {"general.architecture": "clip", "clip.projector_type": projector},
    )
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: floor - 1)
    ok, message, _ = tuner.check_model_build(model, "x")
    assert not ok and f"b{floor}+" in message and projector in message
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: floor)
    assert tuner.check_model_build(model, "x")[0]
    # Text-only launches are not gated by the projector.
    model.mmproj = None
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: floor - 1)
    assert tuner.check_model_build(model, "x")[0]


def _lilicorr(tmp_path, name):
    return _fake_model_md(
        tmp_path,
        name,
        0.4,
        {
            "general.architecture": "dflash",
            "dflash.block_size": 8,
            "lilicorr.candidate_topk": 8,
        },
    )


def test_lilicorr_drafts_are_recognised_and_need_b11668(tmp_path, monkeypatch):
    import scanner

    draft = _lilicorr(tmp_path, "lilicorr-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4")
    assert draft.is_lilicorr_drafter and not draft.is_dflash2_drafter
    assert scanner._draft_match_base(
        "lilicorr-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-Q8_0.gguf"
    ) == (scanner._draft_match_base("NVIDIA-Nemotron-3.5-Bolt-74B-A9B-Q8_0.gguf"))
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 11667)
    ok, message, _ = tuner.check_draft_model_build(draft, "x")
    assert not ok and "b11668+" in message
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 11669)
    assert tuner.check_draft_model_build(draft, "x")[0]


@pytest.mark.parametrize(
    "name,profile_file",
    [
        ("NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0", "nemotron-3_5-super-vl.yaml"),
        (
            "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q4_K_XL-00001-of-00003",
            "nemotron-3_5-super-vl.yaml",
        ),
        ("NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB", "nemotron-3_5-bolt.yaml"),
        ("Nemotron-3-Super-120B-A12B-UD-Q4_K_XL", "nemotron-3-super.yaml"),
        ("MiniCPM-V-4_7-35B-A3B-Q4_K_M", "minicpm-v-4_7.yaml"),
        ("MiniCPM-V-4.7-1B-Q8_0", "minicpm-v-4_7.yaml"),
        ("Ternary-Bonsai-2-9B-Q2_0", "bonsai-2.yaml"),
        ("Ternary-Bonsai-2-2B-Q2_0", "bonsai-2.yaml"),
        ("Ternary-Bonsai-2-27B-PQ2_0", "bonsai-2-27b.yaml"),
        ("Ternary-Bonsai-8B-Q2_0", "bonsai-ternary.yaml"),
        ("Kolibri-1-Q4_K_M", "kolibri-1.yaml"),
        ("Laguna-XS-2.1-Q8_0", "laguna-xs-2_1.yaml"),
        ("Laguna-S-2.1-UD-Q4_K_XL", "laguna-s-2_1.yaml"),
        ("vllm-d3-nano-Q8_0", "decision-systemone.yaml"),
        ("OrcaSAQ-2-27B-Uncensored", "qwen3_8.yaml"),
    ],
)
def test_new_model_files_match_their_profiles(name, profile_file):
    arch = "kolibri1" if name.startswith("Kolibri") else "qwen35"
    assert match_profile(name, _profiles(), arch).source_file == profile_file


def test_new_profiles_carry_the_upstream_floors_and_sampling():
    by_file = {p.source_file: p for p in _profiles()}
    assert by_file["nemotron-3_5-super-vl.yaml"].min_llama_build == 11668
    assert by_file["nemotron-3_5-bolt.yaml"].min_llama_build == 11668
    assert by_file["laguna-xs-2_1.yaml"].min_llama_build == 11598
    assert by_file["kolibri-1.yaml"].required_runtime_markers == ["kolibri1"]
    assert by_file["bonsai-2.yaml"].server_binary in ("", None)
    mini = by_file["minicpm-v-4_7.yaml"].sampling["chat"]
    assert (mini["temperature"], mini["top_k"], mini["repeat_penalty"]) == (
        0.7,
        100,
        1.03,
    )


def test_kolibri_is_refused_on_mainline(monkeypatch):
    profile = match_profile("Kolibri-1-Q4_K_M", _profiles(), "kolibri1")
    monkeypatch.setattr(tuner, "_runtime_has_required_markers", lambda *_: False)
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 11669)
    ok, message, _ = tuner.check_profile_build(profile, "b11669/llama-server")
    assert not ok and "kolibri1" in message


# ------------------------------------------- small draft context (b11629+)


def test_dflash_and_mtp_head_drafts_get_a_small_draft_ubatch(tmp_path):
    from test_smoke import _fake_model

    target = _fake_model(tmp_path, "Qwen3.8-27B-Q4_K_M", size_gb=15.0)
    target.metadata = {"general.architecture": "qwen35", "qwen35.block_count": 64}
    dflash2 = _fake_model(tmp_path, "Qwen3.8-27B-DFlash2-Q4_K_M", size_gb=1.1)
    dflash2.metadata = {
        "general.architecture": "dflash",
        "dflash.block_count": 5,
        "dflash.block_size": 8,
        "dflash.conv_kernel_size": 2,
        "dflash.conv_group_size": 16,
        "dflash.selector_rank": 256,
        "dflash.selector_top_k": 16,
    }
    profile = match_profile(target.name, _profiles())
    cfg = tuner.compute_config(target, _fake_system(), profile, draft_model=dflash2)
    cmd = tuner.build_command(target, cfg, profile, draft_model=dflash2)
    at = cmd.index("--spec-draft-n-ubatch")
    assert cmd[at + 1] == str(min(64, cfg.ubatch))
    # A plain sibling drafter reads the whole prompt itself: keep -ub.
    plain = _fake_model(tmp_path, "Qwen3.8-27B-Draft-Q4_K_M", size_gb=1.0)
    plain_cmd = tuner.build_command(target, cfg, profile, draft_model=plain)
    assert "-md" in plain_cmd and "--spec-draft-n-ubatch" not in plain_cmd
    # No draft, no flag.
    bare = tuner.build_command(target, cfg, profile, enable_speculative=False)
    assert "--spec-draft-n-ubatch" not in bare
