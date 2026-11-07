"""b12187 audit (0.9.0-dev, after stable v0.8.0 = b11960 and v0.9.0 = b12110):
--connect / --connect-code / --models-memory-margin, mainline loaders for
Xing 4.0 and DeepSeek-V4.1-Flash, new architecture / decision-type /
projector floors, new profiles and the draft-head size check."""

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


def test_b12187_manifest_adds_connect_and_router_memory_margin():
    current = _manifest("b12187")
    previous = _manifest("b11669")
    assert current["tag"] == "b12187"
    assert current["commit"] == "cf659f240"
    assert set(current["flags"]) - set(previous["flags"]) == {
        "--connect",
        "--connect-code",
        "--models-memory-margin",
    }
    assert set(previous["flags"]) <= set(current["flags"])
    assert len(current["flags"]) == 423
    assert set(current["binaries"]) == {"vulkan", "hip"}
    for profile in _profiles():
        for arg in profile.extra_args:
            if arg.startswith("--"):
                assert arg.split("=", 1)[0] in current["flags"], (
                    profile.source_file,
                    arg,
                )


@pytest.mark.parametrize(
    "flag,value", [("--connect-code", "a" * 40), ("--models-memory-margin", "1024")]
)
def test_new_value_flags_are_pruned_with_their_value(flag, value):
    assert flag in tuner._ARG_FLAGS_WITH_VALUES
    cmd = ["llama-server", "-m", "m.gguf", flag, value, "-c", "4096"]
    kept, removed = tuner._filter_command_for_supported_flags(
        cmd, set(_manifest("b11669")["flags"])
    )
    assert kept == ["llama-server", "-m", "m.gguf", "-c", "4096"]
    assert removed == [f"{flag} {value}"]
    assert tuner._filter_command_for_supported_flags(
        cmd, set(_manifest("b12187")["flags"])
    ) == (cmd, [])


def test_connect_code_is_redacted_everywhere(tmp_path):
    import diagnostics
    import ocr_workflow
    import qt_launcher

    code = "c0ffee" * 6 + "abcd"
    cmd = ["llama-server", "--connect", "--connect-code", code, "--port", "8080"]
    assert code not in qt_launcher._redacted_command(cmd)
    assert code not in qt_launcher._redacted_command(
        ["llama-server", f"--connect-code={code}"]
    )
    assert code not in " ".join(ocr_workflow._redact_command(cmd))
    text = diagnostics._redact_support_text("launch: " + " ".join(cmd), [])
    assert code not in text and "--connect-code" in text
    # The tunnel switch itself is not a secret.
    assert "--connect" in qt_launcher._redacted_command(cmd)


# --------------------------------------------------- Xing 4.0 on mainline


def _xing(tmp_path):
    return _fake_model_md(
        tmp_path,
        "xing4_0-29b-IQ4_NL",
        16.0,
        {
            "general.architecture": "xing4_0",
            "xing4_0.block_count": 41,
            "xing4_0.nextn_predict_layers": 1,
            "xing4_0.expert_count": 64,
            "xing4_0.context_length": 262144,
        },
    )


def test_xing_runs_from_b12156_with_one_mtp_draft_token(tmp_path, monkeypatch):
    model = _xing(tmp_path)
    assert model.has_embedded_mtp
    profile = match_profile(model.name, _profiles(), "xing4_0")
    assert profile.source_file == "xing-4_0.yaml"
    assert profile.min_llama_build == 12156 and not profile.runtime_block_reason
    assert tuner.resolve_draft_n_max(profile) == 1
    chat = profile.sampling["chat"]
    assert (chat["temperature"], chat["top_p"], chat["repeat_penalty"]) == (
        1.0,
        0.95,
        1.05,
    )
    assert profile.sampling["coding"]["temperature"] == 0.8
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: 12187)
    assert tuner.check_model_build(model, "x")[0]
    assert tuner.check_profile_build(profile, "x")[0]
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 12187)
    cfg = tuner.compute_config(model, _fake_system(), profile, user_ctx=8192)
    cmd = tuner.build_command(model, cfg, profile)
    assert "draft-mtp" in cmd[cmd.index("--spec-type") + 1]
    assert cmd[cmd.index("--spec-draft-n-max") + 1] == "1"


# ------------------------------------------------- architecture floors


@pytest.mark.parametrize("arch,floor", sorted(tuner._ARCH_MIN_BUILD.items()))
def test_new_architectures_name_their_first_build(tmp_path, monkeypatch, arch, floor):
    build = floor[0]
    model = _fake_model_md(
        tmp_path, f"renamed-{arch}-Q4_K_M", 2.0, {"general.architecture": arch}
    )
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: build - 1)
    ok, message, _ = tuner.check_model_build(model, "x")
    assert not ok and f"b{build}+" in message and arch in message
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: build)
    assert tuner.check_model_build(model, "x")[0]


def test_kolibri_stays_on_its_runtime_marker():
    # hob-llama.cpp reports b11434 but loads kolibri1; mainline since b11709.
    assert "kolibri1" not in tuner._ARCH_MIN_BUILD
    profile = match_profile("Kolibri-1-Q4_K_M", _profiles(), "kolibri1")
    assert profile.required_runtime_markers == ["kolibri1"]
    assert not profile.min_llama_build


# ------------------------------------------------- decision types


@pytest.mark.parametrize(
    "name,arch,dtype,floor",
    [
        ("granite-switch-4.1-8b-Q4_K_M", "granitehybrid", "granite-switch", 11707),
        ("h2o-lightning-4b-Q8_0", "qwen35", "h2o-lightning", 11773),
        ("MiniCPM-OCR-Jev-Q8_0", "qwen35", "openbmb-minicpm-ocr-jev", 11928),
        ("d1-2B-Q8_0", "lfm3", "lfm3-d1", 11945),
        ("d1-omni-2B-Q8_0", "lfm3", "lfm3-d1-omni", 11945),
    ],
)
def test_new_decision_types_need_their_build(
    tmp_path, monkeypatch, name, arch, dtype, floor
):
    model = _fake_model_md(
        tmp_path,
        name,
        2.0,
        {"general.architecture": arch, f"{arch}.decision.type": dtype},
    )
    expected = (
        "granite-switch-4_1.yaml"
        if dtype == "granite-switch"
        else "decision-systemone.yaml"
    )
    assert match_profile(name, _profiles(), arch).source_file == expected
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: floor - 1)
    ok, message, _ = tuner.check_model_build(model, "x")
    assert not ok and f"b{floor}+" in message
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: max(floor, 11945))
    assert tuner.check_model_build(model, "x")[0]


# ------------------------------------------------- projector floors


@pytest.mark.parametrize(
    "projector,floor",
    [
        ("muse-spark-audio", 11671),
        ("voxtral_rt", 11675),
        ("sensenova_u1", 11801),
        ("minicpm_ocr_1_0", 11826),
        ("deepseek41v", 11870),
        ("lfm3vl", 11938),
        ("lfm3a", 11941),
        ("nemotron_transcribe", 12087),
        ("drama3_gen", 12116),
    ],
)
def test_new_projectors_need_their_build(tmp_path, monkeypatch, projector, floor):
    import scanner

    model = _fake_model_md(
        tmp_path, "mm-model-Q4_K_M", 4.0, {"general.architecture": "qwen35"}
    )
    mmproj = tmp_path / "mmproj-mm-model-F16.gguf"
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
    assert "multimodal projector" in message
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: floor)
    assert tuner.check_model_build(model, "x")[0]


# ------------------------------------------------- profiles


@pytest.mark.parametrize(
    "name,arch,profile_file,floor",
    [
        ("Kimi-K3.1-Pro-Q2_K-00001-of-00012", "kimi-k31", "kimi-k3_1.yaml", 12180),
        ("LFM3-1.7B-Q8_0", "lfm3", "lfm3.yaml", 11938),
        ("LFM3-Omni-2B-Q8_0", "lfm3", "lfm3.yaml", 11938),
        (
            "Muse-Spark-Open-816B-A42B-Q4_K_S-00001-of-00010",
            "muse-spark",
            "muse-spark-open.yaml",
            11671,
        ),
        (
            "Mistral-Large-4-1T-A52B-Q4_K_M",
            "mistral-large4",
            "mistral-large-4.yaml",
            11998,
        ),
        ("Clover-35B-A3B-Preview-Q4_K_M", "clover_moe", "clover.yaml", 12060),
        ("Ripple-65B-Preview-AD-Q4_K_M", "ricursive", "ripple.yaml", 12158),
        ("limite-1b-violetto-Q8_0", "limite", "limite.yaml", 11988),
        ("flint-int4-q4_1", "flint", "flint.yaml", 11803),
        ("MiniCPM-OCR-1.0-Q8_0", "qwen35", "minicpm-ocr-1_0.yaml", 11826),
        (
            "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-Q8_0",
            "nemotron_h_moe",
            "nemotron-3_5-transcribe.yaml",
            12087,
        ),
        ("Nemotron-3.5-Embed-1B-Q8_0", "mistral3", "nemotron-embed.yaml", 11833),
        (
            "DeepSeek-V4.1-Flash-Q2_K-00001-of-00008",
            "deepseek41",
            "deepseek-v4_1.yaml",
            11870,
        ),
        ("xing4_0-29b-IQ4_NL", "xing4_0", "xing-4_0.yaml", 12156),
    ],
)
def test_new_model_files_match_their_profiles(name, arch, profile_file, floor):
    profile = match_profile(name, _profiles(), arch)
    assert profile.source_file == profile_file
    assert profile.min_llama_build == floor
    assert not profile.runtime_block_reason


def test_renamed_files_follow_their_architecture():
    for name, arch, expected in [
        ("renamed", "kimi-k31", "kimi-k3_1.yaml"),
        ("renamed", "lfm3", "lfm3.yaml"),
        ("renamed", "flint", "flint.yaml"),
        ("renamed", "xing4_0", "xing-4_0.yaml"),
        # authoritative architectures beat a misleading filename
        ("Kimi-K3-repack", "kimi-k31", "kimi-k3_1.yaml"),
        ("DeepSeek-V4-Flash-custom", "deepseek41", "deepseek-v4_1.yaml"),
    ]:
        assert match_profile(name, _profiles(), arch).source_file == expected
    # ...but only those: a plain K3 file keeps the K3 profile.
    assert match_profile("Kimi-K3-Q2_K", _profiles(), "kimi-k3").source_file == (
        "kimi-k3.yaml"
    )


def test_ocr_and_transcribe_profiles_turn_reasoning_off():
    by_file = {p.source_file: p for p in _profiles()}
    for name in ("minicpm-ocr-1_0.yaml", "nemotron-3_5-transcribe.yaml"):
        args = by_file[name].extra_args
        assert args[args.index("--reasoning") + 1] == "off", name
        assert by_file[name].sampling["chat"]["temperature"] == 0.0
    assert "--embeddings" in by_file["nemotron-embed.yaml"].extra_args


def test_new_profile_sampling_follows_the_model_cards():
    by_file = {p.source_file: p for p in _profiles()}

    def chat(name):
        s = by_file[name].sampling["chat"]
        return (s["temperature"], s["top_k"], s["top_p"], s["repeat_penalty"])

    assert chat("lfm3.yaml") == (0.1, 50, 1.0, 1.1)
    assert chat("muse-spark-open.yaml") == (1.0, 64, 1.0, 1.0)
    assert chat("mistral-large-4.yaml")[0::2] == (1.0, 0.95)
    assert chat("kimi-k3_1.yaml")[0::2] == (1.0, 0.95)
    assert chat("flint.yaml") == (1.0, 20, 0.95, 1.0)
    assert chat("limite.yaml")[0::2] == (0.6, 0.95)
    assert by_file["clover.yaml"].sampling["chat"]["presence_penalty"] == 0.9
    assert chat("deepseek-v4_1.yaml")[0::2] == (1.0, 0.95)


# ------------------------------------------------- draft heads and quant tags


def test_draft_head_for_another_model_size_is_not_paired(tmp_path):
    import scanner

    head = tmp_path / "dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4.gguf"
    nano = tmp_path / "NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD_Q6_K.gguf"
    bolt = tmp_path / "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-Q3_K_XL.gguf"
    for path in (head, nano, bolt):
        path.write_bytes(b"GGUF")
    assert scanner._find_draft(nano, [head]) is None
    assert scanner._find_draft(bolt, [head]) == head
    assert not scanner.is_draft_compatible(nano, head)
    assert scanner.is_draft_compatible(bolt, head)
    # Unlabelled names are not judged by size.
    assert not scanner._size_labels_conflict("Qwen3.8-27B-Q4_K_M.gguf", "mtp-head.gguf")


@pytest.mark.parametrize("tag", ["NVFP4", "F8_E4M3", "FP8"])
def test_nvfp4_and_fp8_quant_tags_are_stripped_for_pairing(tag):
    import scanner

    assert scanner._strip_quant_keep_variant(f"Flint-40B-{tag}.gguf").lower() == (
        "flint-40b"
    )


# ------------------------------------------------- Kolibri 1 hob GGUFs


def _kolibri(tmp_path, pre, gating):
    return _fake_model_md(
        tmp_path,
        "Kolibri-1-Q4_K_M",
        44.0,
        {
            "general.architecture": "kolibri1",
            "tokenizer.ggml.pre": pre,
            "kolibri1.expert_gating_func": gating,
            "kolibri1.block_count": 50,
            "kolibri1.expert_count": 384,
        },
    )


def test_hob_kolibri_gguf_gets_mainline_overrides(tmp_path, monkeypatch):
    hob = _kolibri(tmp_path, "kolibri1", 5)
    assert tuner.kolibri_mainline_overrides(hob, 12187) == [
        "--override-kv",
        "tokenizer.ggml.pre=str:qwen2,kolibri1.expert_gating_func=int:2",
    ]
    # The hob runtime itself (b11434) and unknown builds keep the file as is.
    assert tuner.kolibri_mainline_overrides(hob, 11434) == []
    assert tuner.kolibri_mainline_overrides(hob, None) == []
    # Mainline conversions need nothing.
    assert tuner.kolibri_mainline_overrides(_kolibri(tmp_path, "qwen2", 2), 12187) == []
    profile = match_profile(hob.name, _profiles(), "kolibri1")
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 12187)
    cfg = tuner.compute_config(hob, _fake_system(), profile, user_ctx=8192)
    cmd = tuner.build_command(hob, cfg, profile, server_binary="llama-server")
    assert cmd.count("--override-kv") == 1
    assert cmd[cmd.index("--override-kv") + 1].startswith(
        "tokenizer.ggml.pre=str:qwen2"
    )
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11434)
    cmd = tuner.build_command(hob, cfg, profile, server_binary="llama-server")
    assert "--override-kv" not in cmd


# ------------------------------------------------- drafts by architecture


def test_lilicorr_draft_with_extra_blocks_runs_as_dflash(tmp_path):
    """NVIDIA's LiLiCorr drafter stores blk.5..7 beyond dflash.block_count;
    the tensor scan reports that as an MTP block. It must stay a DFlash draft
    (draft-mtp aborted with "failed to create MTP context")."""
    from test_smoke import _fake_model

    draft = _fake_model_md(
        tmp_path,
        "dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4",
        0.9,
        {
            "general.architecture": "dflash",
            "dflash.block_count": 5,
            "dflash.block_size": 8,
            "lilicorr.candidate_topk": 8,
            "__mtp_scan__": "found",
            "__max_block_index__": 7,
        },
    )
    assert not draft.has_embedded_mtp
    assert draft.drafter_spec_type == "dflash"
    target = _fake_model(tmp_path, "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-Q4_K_M", 20.0)
    target.metadata = {"general.architecture": "nemotron_h_moe"}
    profile = match_profile(target.name, _profiles(), "nemotron_h_moe")
    cfg = tuner.compute_config(target, _fake_system(), profile, user_ctx=8192)
    cmd = tuner.build_command(target, cfg, profile, draft_model=draft)
    assert cmd[cmd.index("--spec-type") + 1] == "draft-dflash"


# ------------------------------------------------- auto_draft profiles


def test_measured_slower_drafters_are_not_selected_by_default():
    by_file = {p.source_file: p for p in _profiles()}
    assert by_file["minicpm-v-4_7.yaml"].auto_draft is False
    assert by_file["minicpm-v-4_7.yaml"].draft_max == 7
    assert by_file["nemotron-3_5-bolt.yaml"].auto_draft is False
    assert by_file["xing-4_0.yaml"].auto_draft is True
    assert by_file["xing-4_0.yaml"].draft_p_min == 0.0
    assert by_file["_default.yaml"].auto_draft is True


def test_cli_keeps_an_external_draft_off_when_the_profile_says_so(tmp_path):
    import auto_tuner
    from test_smoke import _fake_model

    model = _fake_model(tmp_path, "MiniCPM-V-4_7-35B-A3B-Q6_K", 28.0)
    model.metadata = {"general.architecture": "qwen35moe"}
    draft = _fake_model(tmp_path, "dspark-MiniCPM-V-4_7-35B-A3B-F16", 0.8)
    args = dict(non_interactive=True, draft_available=True, thinking_available=False)
    out = auto_tuner._ask_interactive_features(
        model, draft, tmp_path / "s.json", draft_default=False, **args
    )
    assert out[1] is False
    out = auto_tuner._ask_interactive_features(
        model, draft, tmp_path / "s.json", draft_default=True, **args
    )
    assert out[1] is True
    # Embedded MTP (no external file) is not affected.
    out = auto_tuner._ask_interactive_features(
        model, None, tmp_path / "s.json", draft_default=False, **args
    )
    assert out[1] is True


def test_gui_lists_but_does_not_preselect_a_slower_drafter(tmp_path, monkeypatch):
    import test_smoke

    qt_launcher = pytest.importorskip("qt_launcher")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    test_smoke._QT_TEST_APP = (
        qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    )
    monkeypatch.setattr(
        qt_launcher.app_settings,
        "_settings_file",
        lambda: tmp_path / "autotuner_settings.json",
    )
    from scanner import ModelEntry

    def entry(name, draft_name):
        model_path = tmp_path / f"{name}.gguf"
        draft_path = tmp_path / f"{draft_name}.gguf"
        test_smoke._write_minimal_gguf(model_path)
        test_smoke._write_minimal_gguf(draft_path)
        return ModelEntry(
            path=model_path,
            name=model_path.stem,
            group=".",
            size_bytes=model_path.stat().st_size,
            draft=draft_path,
            folder_drafts=[draft_path],
            metadata={"general.architecture": "qwen35moe"},
        )

    window = qt_launcher.MainWindow(
        tmp_path, test_smoke.SETTINGS_DIR, start_background=False
    )
    window._profiles = _profiles()  # normally loaded by the background scan
    mini = entry("MiniCPM-V-4_7-35B-A3B-Q6_K", "dspark-MiniCPM-V-4_7-35B-A3B-F16")
    window._populate_draft_combo(mini, {})
    labels = [window._cb_draft.itemText(i) for i in range(window._cb_draft.count())]
    assert any("dspark-MiniCPM" in label for label in labels)
    assert window._draft_enabled() is False
    qwen = entry("Qwen3.8-27B-Q4_K_M", "Qwen3.8-27B-DFlash2-Q4_K_M")
    window._populate_draft_combo(qwen, {})
    assert window._draft_enabled() is True
    window.close()
