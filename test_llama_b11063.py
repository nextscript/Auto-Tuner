"""b11063 audit: unchanged CLI surface (the only new --help byte is a timestamped
startup log line), the dedicated Ling 3.0 (Bailing V3) chat parser note, the
re-verified vision + DFlash2 gate and V4.1 block (now naming b11063), and the
still-fork-only Prism / ROCmFPX tensor types on mainline."""

import json
from pathlib import Path

import pytest

import tuner
from settings_loader import load_profiles, match_profile
from test_kv_policy import _model
from test_smoke import _fake_model_md, _fake_system

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


def test_pinned_b11063_manifest_keeps_the_b11042_option_set():
    manifest = _manifest("b11063")
    assert manifest["tag"] == "b11063"
    assert manifest["commit"] == "3d82ef62d"
    flags = set(manifest["flags"])
    # b11042..b11063 (21 commits) touches neither common/arg.cpp nor the
    # server option table: same option set as b10948 … b11042. PR #29125
    # ("server : improve startup log messages") prints a timestamped
    # "llama_server: initializing ..." line even for --help; the manifest
    # strips that one line and the remaining help bytes hash the same.
    assert flags == set(_manifest("b11042")["flags"])
    assert len(flags) == 415 and sum(f.startswith("--") for f in flags) == 328
    assert "--rpc" not in flags  # local GGML_RPC=OFF, not an upstream removal
    binaries = manifest["binaries"]
    assert set(binaries) == {"vulkan", "hip"}
    hashes = {b["help_sha256"] for b in binaries.values()}
    assert len(hashes) == 1
    assert hashes == {
        b["help_sha256"] for b in _manifest("b11042")["binaries"].values()
    }
    assert manifest["notes"] == {
        "vulkan": {"stripped_startup_log_lines": 1},
        "hip": {"stripped_startup_log_lines": 1},
    }
    assert not flags.intersection(tuner._LEGACY_LOAD_MODES)
    for profile in _profiles():
        for token in profile.extra_args:
            if token.startswith("--"):
                assert token.split("=", 1)[0] in flags, (profile.source_file, token)


# ------------------------------------------------------------ Ling 3.0 ----


def _ling3(tmp_path):
    """The local Ling-3.0-flash-AD-IQ3_M header (bailingmoe3, 42 blocks, 512
    experts, one NextN/MTP block, the <role>/<arg_key> template markers)."""
    template = (
        "<role>SYSTEM</role>{{ system }}<role>HUMAN</role>{{ user }}"
        "<role>ASSISTANT</role><think>\n"
        "<tool_call><function=name><arg_key>k</arg_key><arg_value>v</arg_value>"
        "</function></tool_call>"
    )
    return _fake_model_md(
        tmp_path,
        "Ling-3.0-flash-AD-IQ3_M-00001-of-00002",
        62.2,
        {
            "general.architecture": "bailingmoe3",
            "general.name": "Ling 3.0 Flash",
            "general.file_type": 27,
            "bailingmoe3.block_count": 42,
            "bailingmoe3.expert_count": 512,
            "bailingmoe3.context_length": 131072,
            "bailingmoe3.nextn_predict_layers": 1,
            "tokenizer.chat_template": template,
        },
    )


def test_ling3_profile_documents_the_b11063_parser_but_keeps_the_b10749_gate():
    profiles = _profiles()
    profile = match_profile(
        "Ling-3.0-flash-AD-IQ3_M-00001-of-00002", profiles, "bailingmoe3"
    )
    assert profile.source_file == "ling-3.yaml"
    # The loader contract (corrected no-scan SSM tensors) is unchanged; the
    # dedicated parser only affects tool calls emitted inside the pre-opened
    # think block, so it is documented, not gated.
    assert profile.min_llama_build == 10749
    assert profile.extra_args == ["--jinja", "--reasoning-preserve"]
    assert profile.draft_max == 3
    assert "b11063" in profile.notes and "#28682" in profile.notes
    assert "tool_calls" in profile.notes and "reasoning_content" in profile.notes
    for pack in sorted((ROOT / "assets/languages").glob("*.json")):
        notes = json.loads(pack.read_text(encoding="utf-8"))["profile_notes"]
        note = notes["ling-3.yaml"]
        assert "b11063" in note and "#28682" in note, pack.name
        assert "tool_calls" in note and "reasoning_content" in note, pack.name
        assert "<role>" in note and "--jinja" in note, pack.name
        assert "b10749" in note, pack.name


def test_ling3_command_keeps_jinja_reasoning_preserve_and_the_mtp_head(
    tmp_path, monkeypatch
):
    model = _ling3(tmp_path)
    assert model.has_embedded_mtp
    profile = match_profile(model.name, _profiles(), "bailingmoe3")
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 11063)
    monkeypatch.setattr(
        tuner,
        "_probe_supported_flags",
        lambda _: {"--jinja", "--reasoning-preserve", "--spec-type", "--mmproj"},
    )
    config = tuner.compute_config(
        model, _fake_system(vram_total=48, vram_free=46), profile, user_ctx=16384
    )
    cmd = tuner.build_command(
        model, config, profile, server_binary="llama-server", enable_speculative=True
    )
    # The dedicated parser is chosen by llama-server from the template with
    # --jinja; AutoTuner adds no reasoning-format override that would bypass it.
    assert "--jinja" in cmd and "--reasoning-preserve" in cmd
    assert "--reasoning-format" not in cmd and "--no-reasoning-preserve" not in cmd
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert cmd[cmd.index("--spec-draft-n-max") + 1] == "3"
    assert cmd[cmd.index("--temp") + 1] == "0.6"
    assert cmd[cmd.index("--top-p") + 1] == "0.95"
    assert cmd[cmd.index("--top-k") + 1] == "20"
    assert "-md" not in cmd


# ------------------------------------------- still fork-only on mainline ----


def test_mainline_b11063_still_lacks_the_prism_and_rocmfpx_types(tmp_path, monkeypatch):
    # ggml.h at b11063 still ends at GGML_TYPE_COUNT = 43: Prism PQ2_0/PTQ1_0
    # (142/143, issue #29058 open) and ROCmFPX (100..111, PR #24185 open)
    # stay fork-only, so both metadata gates keep refusing mainline runtimes.
    monkeypatch.setattr(tuner, "_runtime_has_required_markers", lambda *_: False)
    bonsai = _fake_model_md(
        tmp_path,
        "Ternary-Bonsai-2-27B-PQ2_0",
        6.71,
        {
            "general.architecture": "qwen35",
            "general.file_type": 141,
            "qwen35.block_count": 64,
            "prism.hadamard.version": 1,
        },
    )
    ok, message, detected = tuner.check_model_build(bonsai, "llama-server")
    assert not ok and detected is None
    assert "ggml type 142" in message and "#29058" in message
    agnes = _fake_model_md(
        tmp_path,
        "Agnes-3.0-Flash-Preview-MTP-imatrix-Q4_0-ROCmFP4-STRIX_LEAN",
        18.06,
        {
            "general.architecture": "qwen35",
            "general.file_type": 106,
            "qwen35.block_count": 73,
            "__ggml_types__": [0, 1, 100, 101],
        },
    )
    ok, message, detected = tuner.check_model_build(agnes, "llama-server")
    assert not ok and detected is None
    assert "[0, 43)" in message and "#24185" in message


# ------------------------------------------------------------- carry-over ----


def test_vision_dflash2_gate_is_unchanged_and_names_b11063():
    # b11042..b11063 touches neither tools/server's speculative path nor the
    # recurrent memory (server.cpp: startup log wording only); the actual
    # image + DFlash2 request was re-run on b11063 and still fails with
    # HTTP 500 on HIP and Vulkan (upstream issue #27408 open).
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
    [(10895, False), (10896, True), (11042, True), (11063, True), (11100, True)],
)
def test_vision_with_dflash2_stays_gated_through_b11063(
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
