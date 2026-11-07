"""b11249 audit: --rpc always advertised, Muse Glimmer json_schema and the
Nemotron 3 Puzzle HIP scan notes, re-checked blockers and fork pins."""

import json
from pathlib import Path

import pytest

import tuner
from settings_loader import load_profiles, match_profile
from test_llama_b11063 import _qwen_vision_dflash2
from test_smoke import _fake_model_md, _fake_system

ROOT = Path(__file__).resolve().parent


def _profiles():
    return load_profiles(ROOT / "settings")


def _manifest(tag: str) -> dict:
    return json.loads(
        (ROOT / f"docs/llama-{tag}-server-flags.json").read_text(encoding="utf-8")
    )


def _pack_notes():
    for pack in sorted((ROOT / "assets/languages").glob("*.json")):
        yield pack.name, json.loads(pack.read_text(encoding="utf-8"))["profile_notes"]


def test_b11249_manifest_only_adds_the_always_listed_rpc_flag():
    current = _manifest("b11249")
    previous = _manifest("b11195")
    assert current["tag"] == "b11249"
    assert current["commit"] == "6d78fb072"
    # PR #29537 registers --rpc unconditionally; builds without GGML_RPC now
    # list it and reject it at parse time ("RPC not supported in this build").
    assert set(current["flags"]) - set(previous["flags"]) == {"--rpc"}
    assert set(previous["flags"]) <= set(current["flags"])
    assert len(current["flags"]) == 416
    assert set(current["binaries"]) == {"vulkan", "hip"}
    for backend, binary in current["binaries"].items():
        assert binary["help_sha256"] != previous["binaries"][backend]["help_sha256"]
        # The server's "initializing ..." log line now follows argument
        # parsing, so --help no longer prints a timestamped startup line.
        assert current["notes"][backend]["stripped_startup_log_lines"] == 0
    for profile in _profiles():
        for arg in profile.extra_args:
            if arg.startswith("--"):
                assert arg.split("=", 1)[0] in current["flags"]


def test_rpc_is_never_emitted_and_only_pruned_where_unadvertised():
    old = set(_manifest("b11195")["flags"])
    new = set(_manifest("b11249")["flags"])
    cmd = [
        "llama-server",
        "-m",
        "model.gguf",
        "--rpc",
        "192.168.0.2:50052",
        "-c",
        "4096",
    ]
    kept, removed = tuner._filter_command_for_supported_flags(cmd, old)
    assert kept == ["llama-server", "-m", "model.gguf", "-c", "4096"]
    assert removed == ["--rpc 192.168.0.2:50052"]
    # b11249 advertises --rpc everywhere: a user extra is passed through and a
    # build without RPC fails fast with llama.cpp's own message.
    assert tuner._filter_command_for_supported_flags(cmd, new) == (cmd, [])
    # AutoTuner itself never plans RPC offload.
    for profile in _profiles():
        assert not any(arg.startswith("--rpc") for arg in profile.extra_args)


def test_muse_glimmer_note_names_the_json_schema_fix():
    profile = match_profile("Muse-Glimmer-30B-UD-Q5_K_XL", _profiles(), "muse-glimmer")
    assert profile.source_file == "muse-glimmer.yaml"
    assert profile.min_llama_build == 10353  # loader floor unchanged
    assert "--jinja" in profile.extra_args
    assert "b11249" in profile.notes and "#29615" in profile.notes
    assert "json_schema" in profile.notes
    for name, notes in _pack_notes():
        note = notes["muse-glimmer.yaml"]
        assert "b11249" in note and "#29615" in note and "json_schema" in note, name
        assert "#29242" in note, name  # the b11100 tool-call note stays


def test_nemotron_puzzle_note_recommends_hip_b11249():
    profile = match_profile("Nemotron-Labs-3-Puzzle-75B-A9B-Q4_K_M", _profiles())
    assert profile.source_file == "nemotron-3-puzzle.yaml"
    assert profile.min_llama_build == 10786  # loader floor unchanged
    assert "b11249" in profile.notes and "#28717" in profile.notes
    for name, notes in _pack_notes():
        note = notes["nemotron-3-puzzle.yaml"]
        assert "b11249" in note and "#28717" in note and "96" in note, name


# ------------------------------------------ MiMo-V2.6-Distill-Qwen-9B ----
# Xiaomi's Qwen3.5-9B SFT used to match mimo-v2_6.yaml ("mimo-v2.6" in the
# filename): MoE notes, temp 1.0 / top_k 0 instead of its own 0.6 / 20 / 0.95.


@pytest.mark.parametrize(
    ("name", "arch", "expected"),
    [
        ("MiMo-V2.6-Distill-Qwen-9B-Q8_0", "qwen35", "mimo-v2_6-distill.yaml"),
        ("mimo_v2.6_distill_qwen_9b-IQ4_XS", "qwen35", "mimo-v2_6-distill.yaml"),
        ("MiMo-V2.6-Flash-RL-Q4_K_M", "mimo2", "mimo-v2_6.yaml"),
        ("Qwen3.5-9B-Q8_0", "qwen35", "qwen3_5-3_6.yaml"),
        ("some-opaque-requant", "qwen35", "qwen3_5-3_6.yaml"),
    ],
)
def test_mimo_distill_gets_its_own_profile(name, arch, expected):
    assert match_profile(name, _profiles(), arch).source_file == expected


def test_mimo_distill_profile_follows_the_official_generation_config(tmp_path):
    profile = match_profile("MiMo-V2.6-Distill-Qwen-9B-Q8_0", _profiles(), "qwen35")
    assert profile.arch_fallback == []  # qwen35 stays with the Qwen profile
    assert profile.max_context == 262144
    for mode in ("chat", "coding"):
        sampling = profile.sampling[mode]
        assert (sampling["temperature"], sampling["top_k"], sampling["top_p"]) == (
            0.6,
            20,
            0.95,
        )
        assert sampling["presence_penalty"] == 0.0
    assert profile.extra_args == ["--jinja", "--reasoning-preserve"]
    assert profile.min_llama_build == 0  # plain qwen35 loader
    model = _fake_model_md(
        tmp_path,
        "MiMo-V2.6-Distill-Qwen-9B-Q8_0",
        9,
        {"general.architecture": "qwen35", "qwen35.context_length": 262144},
    )
    config = tuner.compute_config(
        model, _fake_system(), profile, user_ctx=32768, prompt_cache_ram_mib=0
    )
    cmd = tuner.build_command(model, config, profile)
    assert cmd[cmd.index("--temp") + 1] == "0.6"
    assert cmd[cmd.index("--top-k") + 1] == "20"
    assert cmd[cmd.index("--top-p") + 1] == "0.95"
    assert "--jinja" in cmd and "--reasoning-preserve" in cmd
    for name, notes in _pack_notes():
        note = notes["mimo-v2_6-distill.yaml"]
        assert "Qwen3.5-9B" in note and "0.6" in note and "Qwen3-Coder" in note, name


@pytest.mark.parametrize("build", [11195, 11248, 11249])
def test_vision_dflash2_gate_stays_and_names_b11249(tmp_path, monkeypatch, build):
    # Re-run on b11249 (HIP and Vulkan): #29385 migrated speculative decoding
    # to batch_ext but DFlash still skips pinned M-RoPE image batches, so the
    # draft cache gap and HTTP 500 remain (upstream #27408 open).
    model, draft, profile, config = _qwen_vision_dflash2(tmp_path)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: build)
    monkeypatch.setattr(
        tuner, "_probe_supported_flags", lambda _: {"--spec-type", "--mmproj"}
    )
    with pytest.raises(ValueError, match=rf"b{build}.*verified through b11249"):
        tuner.build_command(model, config, profile, draft_model=draft)
    assert tuner.QWEN35_VISION_DFLASH2_BROKEN_SINCE == 10896


def test_mimo_mtp_floor_stays_a_fixed_fact():
    # The MiMo-V2 MTP-head requirement is a fixed floor, not an audit marker.
    assert tuner._MIN_MIMO2_MTP_SIDECAR_BUILD == 11195
    for name, notes in _pack_notes():
        assert "b11195" not in notes["deepseek-v4_1.yaml"] + notes["xing-4_0.yaml"]
        assert notes["mimo-v2_6.yaml"].count("b11195") == 2, name


def test_fork_recipes_keep_their_validated_pins():
    recipes = ROOT / "building llama.cpp"
    # ROCmFPX main 721db4193 (fork build 11898) does not compile with MSVC:
    # qwen4exp.cpp calls llama_lazy_reader::prefetch, which the _WIN32 reader
    # does not have. The RDNA4 MMQ issue (ROCmFPX #26) is still open, so the
    # validated pin and the local patch stay.
    for name in ("rocmfpx_vulkan_llama_build.ps1", "rocmfpx_hip_llama_build.ps1"):
        text = (recipes / name).read_text(encoding="utf-8")
        assert '"aed0d5fd9620ee96a10cb4e6b16c18514ea370e1"' in text, name
        assert '-FixedIdentity "b11544"' in text, name
        assert "rocmfpx-rdna4-mmq-fallback.patch" in text, name
    # No Prism release after prism-b10743.
    for name in (
        "ternary_bonsai_vulkan_llama_build.ps1",
        "ternary_bonsai_hip_llama_build.ps1",
    ):
        text = (recipes / name).read_text(encoding="utf-8")
        assert '"adfffbe41b2cabcd51fff326ab045662265062bb"' in text, name


# ------------------------------------------------ default build selection ----
# The display order groups builds by family name, so the Ternary/Bonsai '2b_'
# fork sorts before plain mainline builds. Index 0 used to be the fallback for
# a missing saved build (e.g. b11224 replaced by b11249), which silently moved
# every model onto the special-purpose Prism fork.

_LOCAL_TREES = [
    "0.5.0_hip_llama.cpp",
    "0.5.0_vulkan_llama.cpp",
    "2b_b10743_hip_llama.cpp",
    "2b_b10743_vulkan_llama.cpp",
    "b11195_hip_llama.cpp",
    "b11195_vulkan_llama.cpp",
    "b11249_hip_llama.cpp",
    "b11249_vulkan_llama.cpp",
    "d_b9781_vulkan_llama.cpp",
    "fpx_b11544_hip_llama.cpp",
    "fpx_b11544_vulkan_llama.cpp",
    "ocr_b17400_vulkan_llama.cpp",
    "tq_b10298_vulkan_llama.cpp",
]


def _listed_forks(root, names=_LOCAL_TREES):
    from auto_tuner import _fork_name_sort_key

    forks = [(name, root / name) for name in names]
    forks.sort(
        key=lambda item: (item[0].lower() != "llama.cpp", *_fork_name_sort_key(item[0]))
    )
    return forks


def test_display_order_is_unchanged(tmp_path):
    forks = _listed_forks(tmp_path)
    assert forks[0][0] == "2b_b10743_vulkan_llama.cpp"
    assert [name for name, _ in forks].index("b11249_vulkan_llama.cpp") > 0


@pytest.mark.parametrize(
    ("preferred", "expected"),
    [
        (None, "b11249_vulkan_llama.cpp"),
        ("b11224_vulkan_llama.cpp", "b11249_vulkan_llama.cpp"),  # replaced build
        ("b11224_hip_llama.cpp", "b11249_hip_llama.cpp"),
        ("b11195_hip_llama.cpp", "b11195_hip_llama.cpp"),  # existing choice wins
        ("2b_b10743_vulkan_llama.cpp", "2b_b10743_vulkan_llama.cpp"),
        ("2b_b10687_hip_llama.cpp", "2b_b10743_hip_llama.cpp"),  # family kept
        ("fpx_b11000_vulkan_llama.cpp", "fpx_b11544_vulkan_llama.cpp"),
        ("gone_special_llama.cpp", "b11249_vulkan_llama.cpp"),
    ],
)
def test_default_fork_index_prefers_the_newest_build_of_the_saved_family(
    tmp_path, preferred, expected
):
    from auto_tuner import _default_fork_index

    forks = _listed_forks(tmp_path)
    saved = tmp_path / preferred if preferred else None
    assert forks[_default_fork_index(forks, saved)][0] == expected


def test_default_fork_index_keeps_legacy_and_single_fork_defaults(tmp_path):
    from auto_tuner import _default_fork_index

    legacy = _listed_forks(
        tmp_path, ["llama.cpp", "2b_b10743_vulkan_llama.cpp", "b11249_vulkan_llama.cpp"]
    )
    assert legacy[_default_fork_index(legacy)][0] == "llama.cpp"
    # A replaced numbered build still moves on to the newest numbered build.
    stale = tmp_path / "b11224_vulkan_llama.cpp"
    assert legacy[_default_fork_index(legacy, stale)][0] == "b11249_vulkan_llama.cpp"
    only_fork = _listed_forks(tmp_path, ["2b_b10743_vulkan_llama.cpp"])
    assert _default_fork_index(only_fork) == 0
    assert _default_fork_index([]) == 0


def test_cli_picker_defaults_to_newest_mainline(tmp_path, monkeypatch, capsys):
    from auto_tuner import _pick_fork

    forks = _listed_forks(tmp_path)
    newest = tmp_path / "b11249_vulkan_llama.cpp"
    assert _pick_fork(forks, non_interactive=True) == newest
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "")
    assert _pick_fork(forks) == newest
    listed = [name for name, _path in forks]
    assert f"(default {listed.index('b11249_vulkan_llama.cpp') + 1})" in prompts[0]
    assert "Using fork: b11249_vulkan_llama.cpp" in capsys.readouterr().out

    def eof(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert _pick_fork(forks) == newest
    monkeypatch.setattr("builtins.input", lambda _prompt: "1")
    assert _pick_fork(forks) == forks[0][1]  # an explicit number still wins


def test_gui_combo_falls_back_to_newest_build_of_the_saved_family(tmp_path):
    import types

    from qt_launcher import MainWindow

    class Combo:
        def __init__(self):
            self.items, self.index = [], -1

        def blockSignals(self, _blocked):
            pass

        def clear(self):
            self.items = []

        def addItem(self, name, userData=None):
            self.items.append((name, userData))

        def setCurrentIndex(self, index):
            self.index = index

    forks = _listed_forks(tmp_path)
    for saved, expected in (
        (tmp_path / "b11224_vulkan_llama.cpp", "b11249_vulkan_llama.cpp"),
        (tmp_path / "b11195_hip_llama.cpp", "b11195_hip_llama.cpp"),
        (None, "b11249_vulkan_llama.cpp"),
    ):
        applied = []
        window = types.SimpleNamespace(
            _fork_combo=Combo(),
            _fork_path=None,
            _apply_fork=applied.append,
            _refresh_control_runtimes=lambda: None,
            _refresh_fork_combo_width=lambda: None,
        )
        MainWindow._populate_fork_combo(window, forks, saved)
        assert window._fork_combo.items[window._fork_combo.index][0] == expected
        assert window._fork_path == tmp_path / expected
        assert applied == [window._fork_combo.index]
