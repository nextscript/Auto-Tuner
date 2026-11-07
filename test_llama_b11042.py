"""b11042 audit: unchanged CLI surface, the new Ternary-Bonsai 2 27B profile
(PrismML fork prism-b10687, PTQ1_0 / PQ2_0) and the metadata-driven
prism.hadamard loader gate that keeps those GGUFs off mainline runtimes."""

import json
from pathlib import Path

import pytest

import tuner
from settings_loader import load_profiles, match_profile
from test_kv_policy import _model
from test_smoke import _fake_model_md, _fake_system

ROOT = Path(__file__).resolve().parent
# v5.5.7 moved the recipe pin to prism-b10743 (Vulkan PQ2_0 path); the
# profile floor stays at b10687, the first fork build with both packings.
PRISM_PIN = "adfffbe41b2cabcd51fff326ab045662265062bb"


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


def test_pinned_b11042_manifest_keeps_the_b11030_option_set():
    manifest = _manifest("b11042")
    assert manifest["tag"] == "b11042"
    assert manifest["commit"] == "ec9281505"
    flags = set(manifest["flags"])
    # b11030..b11042 (12 commits) touches neither common/arg.cpp nor the
    # server option table: same option set and the same --help bytes as
    # b10948 / b10977 / b11030.
    assert flags == set(_manifest("b11030")["flags"])
    assert len(flags) == 415 and sum(f.startswith("--") for f in flags) == 328
    assert "--rpc" not in flags  # local GGML_RPC=OFF, not an upstream removal
    binaries = manifest["binaries"]
    assert set(binaries) == {"vulkan", "hip"}
    hashes = {b["help_sha256"] for b in binaries.values()}
    assert len(hashes) == 1
    assert hashes == {
        b["help_sha256"] for b in _manifest("b11030")["binaries"].values()
    }
    assert not flags.intersection(tuner._LEGACY_LOAD_MODES)
    for profile in _profiles():
        for token in profile.extra_args:
            if token.startswith("--"):
                assert token.split("=", 1)[0] in flags, (profile.source_file, token)


# ------------------------------------------------------ Ternary-Bonsai 2 ----


def _bonsai2(tmp_path, packing: str):
    """A Ternary-Bonsai 2 27B header as the real GGUFs carry it (qwen35,
    64 blocks, prism.hadamard.* fold, file_type 143 = PTQ1_0 / 141 = PQ2_0)."""
    file_type = {"PTQ1_0": 143, "PQ2_0": 141}[packing]
    size_gb = {"PTQ1_0": 5.54, "PQ2_0": 6.71}[packing]
    metadata = {
        "general.architecture": "qwen35",
        "general.file_type": file_type,
        "qwen35.block_count": 64,
        "qwen35.context_length": 262144,
        "qwen35.embedding_length": 5120,
        "qwen35.attention.head_count": 24,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "qwen35.ssm.group_count": 16,
        "prism.hadamard.version": 1,
        "prism.hadamard.block_size": 1024,
        "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
        "prism.hadamard.sign_mode": "explicit",
        "prism.hadamard.gdn_v_grouped": True,
    }
    return _fake_model_md(
        tmp_path, f"Ternary-Bonsai-2-27B-{packing}", size_gb, metadata
    )


def test_bonsai2_profile_matches_both_packings_and_leaves_the_old_bonsai_alone():
    profiles = _profiles()
    for name in (
        "Ternary-Bonsai-2-27B-PTQ1_0",
        "Ternary-Bonsai-2-27B-PQ2_0",
        "Bonsai-2-27B-PTQ1_0",
    ):
        profile = match_profile(name, profiles, "qwen35")
        assert profile.source_file == "bonsai-2-27b.yaml", name
    profile = match_profile("Ternary-Bonsai-2-27B-PTQ1_0", profiles)
    assert profile.server_binary == "2b_llama/llama-server"
    assert profile.min_llama_build == 10687  # prism-b10687: first pin with PTQ1_0
    assert profile.required_runtime_markers == []  # see the profile comment
    assert profile.max_context == 262144
    assert profile.extra_args == ["--jinja"]
    assert profile.runtime_block_reason == ""
    # Official thinking-mode values (also the GGUF general.sampling.* defaults).
    for mode in ("chat", "coding"):
        assert profile.sampling[mode]["temperature"] == 1.0
        assert profile.sampling[mode]["top_p"] == 0.95
        assert profile.sampling[mode]["top_k"] == 20
        assert profile.sampling[mode]["min_p"] == 0.0
    # First-generation Bonsai, the 8B ternary family and Qwen3.6 itself keep
    # their profiles: the new patterns are longer, not broader.
    assert (
        match_profile("Ternary-Bonsai-27B-Q2_0", profiles, "qwen35").source_file
        == "bonsai-ternary-27b.yaml"
    )
    assert (
        match_profile("Bonsai-27B-Q1_0", profiles, "qwen35").source_file
        == "bonsai-27b.yaml"
    )
    assert (
        match_profile("Ternary-Bonsai-8B-Q2_0", profiles).source_file
        == "bonsai-ternary.yaml"
    )
    assert (
        match_profile("Qwen3.6-27B-Q4_K_M", profiles, "qwen35").source_file
        == "qwen3_5-3_6.yaml"
    )
    for pack in sorted((ROOT / "assets/languages").glob("*.json")):
        notes = json.loads(pack.read_text(encoding="utf-8"))["profile_notes"]
        note = notes["bonsai-2-27b.yaml"]
        assert "PTQ1_0" in note and "PQ2_0" in note, pack.name
        assert "prism-b10687" in note and "Vulkan" in note, pack.name


@pytest.mark.parametrize(
    "build,allowed",
    [(10660, False), (10686, False), (10687, True), (10700, True), (None, True)],
)
def test_bonsai2_profile_requires_the_prism_b10687_fork_build(
    monkeypatch, build, allowed
):
    profile = match_profile("Ternary-Bonsai-2-27B-PQ2_0", _profiles())
    monkeypatch.setattr(tuner, "probe_binary_build_number", lambda _: build)
    ok, message, detected = tuner.check_profile_build(profile, "llama-server")
    assert ok is allowed
    assert detected == build
    if not allowed:
        # The old recipe pin (prism-b10660, e311ed3) predates PTQ1_0 (type 143).
        assert "b10687+" in message and f"b{build}" in message
    elif build is None:
        assert "b10687+" in message


@pytest.mark.parametrize("packing", ["PTQ1_0", "PQ2_0"])
def test_hadamard_folded_gguf_is_refused_without_the_prism_loader(
    tmp_path, monkeypatch, packing
):
    model = _bonsai2(tmp_path, packing)
    assert not model.has_embedded_mtp
    seen = []

    def markers(binary, required):
        seen.append((binary, tuple(required)))
        return False

    monkeypatch.setattr(tuner, "_runtime_has_required_markers", markers)
    ok, message, detected = tuner.check_model_build(model, "llama-server")
    assert not ok and detected is None
    # b11669 (v5.6.3): the private tensor type also needs the fork's own
    # file-type name, because mainline b11556+ carries prism.hadamard too.
    fork_marker = {"PTQ1_0": "PTQ1_0 - 1.75 bpw", "PQ2_0": "PQ2_0 - 2.13 bpw"}[packing]
    assert seen == [("llama-server", ("prism.hadamard", fork_marker))]
    # Mainline b11042 (Vulkan and HIP) really answers
    # "tensor 'output.weight' has invalid ggml type 143. should be in [0, 43)".
    assert model.name in message and packing in message
    assert "prism.hadamard" in message and "prism-b10687" in message
    type_id = {"PTQ1_0": 143, "PQ2_0": 142}[packing]
    assert f"ggml type {type_id}" in message and "#29058" in message
    # check_model_build is the shared GUI / TUI / OCR pre-launch gate; the
    # same wording reaches the user before llama-server is even started.


def test_hadamard_folded_gguf_loads_on_the_prism_fork(tmp_path, monkeypatch):
    model = _bonsai2(tmp_path, "PTQ1_0")
    monkeypatch.setattr(tuner, "_runtime_has_required_markers", lambda *_: True)
    assert tuner.check_model_build(model, "2b_llama/llama-server") == (True, "", None)


def test_gguf_without_prism_metadata_never_scans_for_the_marker(tmp_path, monkeypatch):
    def explode(*_):
        raise AssertionError("marker scan must not run for ordinary GGUFs")

    monkeypatch.setattr(tuner, "_runtime_has_required_markers", explode)
    plain = _model(tmp_path)
    assert tuner.check_model_build(plain, "llama-server")[0]
    ternary_v1 = _fake_model_md(
        tmp_path,
        "Ternary-Bonsai-27B-Q2_0",
        6.7,
        {
            "general.architecture": "qwen35",
            "general.file_type": 42,
            "qwen35.block_count": 64,
        },
    )
    assert tuner.check_model_build(ternary_v1, "llama-server")[0]


def test_bonsai2_command_uses_the_fork_the_projector_and_the_thinking_sampling(
    tmp_path, monkeypatch
):
    model = _bonsai2(tmp_path, "PQ2_0")
    model.mmproj = tmp_path / "Ternary-Bonsai-2-27B-mmproj-BF16.gguf"
    profile = match_profile(model.name, _profiles(), "qwen35")
    monkeypatch.setattr(tuner, "_runtime_has_required_markers", lambda *_: True)
    monkeypatch.setattr(tuner, "_probe_binary_build_number", lambda _: 10687)
    monkeypatch.setattr(
        tuner,
        "_probe_supported_flags",
        lambda _: {"--jinja", "--mmproj", "--spec-type"},
    )
    config = tuner.compute_config(
        model, _fake_system(vram_total=32, vram_free=31), profile, user_ctx=32768
    )
    assert config.ctx == 32768
    cmd = tuner.build_command(
        model, config, profile, server_binary="2b_llama/llama-server"
    )
    assert cmd[0] == "2b_llama/llama-server"
    assert "--jinja" in cmd and "--mmproj" in cmd
    assert cmd[cmd.index("--temp") + 1] == "1.0"
    assert cmd[cmd.index("--top-p") + 1] == "0.95"
    assert "-md" not in cmd


def test_ternary_bonsai_recipes_pin_the_prism_fork_with_its_build_number():
    build_dir = ROOT / "building llama.cpp"
    for name in (
        "ternary_bonsai_vulkan_llama_build.ps1",
        "ternary_bonsai_hip_llama_build.ps1",
    ):
        text = (build_dir / name).read_text(encoding="utf-8")
        assert f'-ExpectedCommit "{PRISM_PIN}"' in text, name
        # The mainline merge-base count (b10616) would collide with the old
        # tree; the folder carries the fork's own `--version` build number.
        assert '-FixedIdentity "b10743"' in text, name
        assert "prism-b10743-adfffbe" in text, name
        assert 'FolderPrefix "2b_"' in text, name


# ------------------------------------------------------------- carry-over ----


def test_vision_dflash2_gate_range_is_unchanged():
    # b11030..b11042 does not touch the server, the speculative helpers or
    # the recurrent memory; the image + DFlash2 range gate stays as it was.
    assert tuner.QWEN35_VISION_DFLASH2_BROKEN_SINCE == 10896
