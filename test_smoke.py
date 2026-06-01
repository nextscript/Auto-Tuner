"""Smoke tests for AutoTuner.

These tests don't need real GGUF models and run on any GitHub Actions
runner. They cover:
  - profile loading + pattern matching against real-world model names
  - mmproj pairing (longest-prefix) on a synthetic models tree
  - compute_config produces sensible values across hardware shapes
  - hardware detection doesn't crash on a runner without GPUs
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

# Make the project root importable when tests are run from the repo root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hardware import GPUInfo, SystemInfo, detect_system, format_system  # noqa: E402
from scanner import group_entries, scan_models, metadata_has_embedded_mtp  # noqa: E402
from settings_loader import load_profiles, match_profile  # noqa: E402
from tuner import build_command, compute_config, extract_params_billion  # noqa: E402


SETTINGS_DIR = ROOT / "settings"


# ---------------------------------------------------------------------------
# Hardware detection


def test_detect_system_does_not_raise() -> None:
    info = detect_system()
    assert info.total_ram_gb > 0
    assert info.cpu_cores_logical >= 1
    # GitHub runners may have no detected GPUs — that's fine
    assert isinstance(info.gpus, list)


def test_format_system_produces_text() -> None:
    info = detect_system()
    out = format_system(info)
    assert "CPU:" in out and "RAM:" in out


# ---------------------------------------------------------------------------
# Profile loading + pattern matching


def test_all_profiles_load() -> None:
    profiles = load_profiles(SETTINGS_DIR)
    assert len(profiles) >= 8, "expected the 8 shipped YAML profiles"
    files = {p.source_file for p in profiles}
    assert "_default.yaml" in files
    assert "qwen2_5-3.yaml" in files
    assert "gemma-4.yaml" in files
    assert "devstral.yaml" in files


@pytest.mark.parametrize(
    "filename, expected_display",
    [
        ("Qwen3.5-9B-Q8_0", "Qwen3.5 / Qwen3.6 (Alibaba)"),
        ("Qwen3.6-27B-UD-Q3_K_XL", "Qwen3.5 / Qwen3.6 (Alibaba)"),
        ("Qwen3.6-35B-A3B-UD-IQ3_S", "Qwen3.5 / Qwen3.6 (Alibaba)"),
        ("Gemma-4-26B-A4B-IQ3_M", "Gemma 4 (Google)"),
        ("gemma-4-E2B-it-BF16", "Gemma 4 (Google)"),
        ("Devstral-Small-2-24B-Instruct-2512-Q3_K_L", "Devstral (Mistral, code)"),
        ("Ministral-3-14B-Reasoning-2512-Q6_K", "Ministral 3 (Mistral, reasoning)"),
        ("Mistral-Medium-3.5-128B-UD-IQ3_XXS", "Mistral Medium 3.x"),
        ("m51Lab-MiniMax-M2.7-REAP-139B-A10B.i1-IQ3_M", "MiniMax-M2 (MiniMaxAI)"),
        ("Bonsai-8B", "Bonsai 8B (PrismML, 1-bit)"),
        ("Ternary-Bonsai-8B-Q2_0", "Ternary-Bonsai (PrismML, 1.58-bit)"),
        ("Archon-14B.Q6_K", "Frankenmerger / community merge"),
        ("voldemort-10b-dpo.Q8_0", "Frankenmerger / community merge"),
        ("Some-Random-LLM.gguf", "Generic / fallback"),
    ],
)
def test_pattern_matching(filename, expected_display) -> None:
    profiles = load_profiles(SETTINGS_DIR)
    p = match_profile(filename, profiles)
    assert p.display_name == expected_display, (
        f"{filename!r} matched {p.display_name!r}, expected {expected_display!r}"
    )


def test_ministral_does_not_collide_with_mistral_medium() -> None:
    """The ministral pattern has no overlap with mistral-medium."""
    profiles = load_profiles(SETTINGS_DIR)
    assert (
        match_profile("Ministral-3-14B", profiles).display_name
        == "Ministral 3 (Mistral, reasoning)"
    )
    assert (
        match_profile("Mistral-Medium-3.5-128B", profiles).display_name
        == "Mistral Medium 3.x"
    )


# ---------------------------------------------------------------------------
# Param extraction


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Qwen3.5-9B-Q8_0", 9.0),
        ("Qwen3.6-35B-A3B-UD-IQ3_S", 35.0),  # MoE: total params, not active
        ("Mistral-Medium-3.5-128B-UD-IQ3_XXS", 128.0),
        ("gemma-4-E2B-it-BF16", 2.0),  # Gemma "effective" size
        ("gemma-4-E4B-it-Q8_0", 4.0),
        ("Qwen3.5-0.8B-Q8_0", 0.8),
    ],
)
def test_extract_params_billion(name, expected) -> None:
    assert extract_params_billion(name) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Scanner + mmproj pairing


def _write_minimal_gguf(path: Path) -> None:
    """Write a valid empty-metadata GGUF header so the scanner accepts it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 0))  # kv count


def test_scanner_pairs_mmproj_by_size(tmp_path) -> None:
    """Each Qwen3.5 sub-model must get its own size-matched mmproj."""
    folder = tmp_path / "Alibaba" / "Qwen3.5"
    files = [
        "mmproj-Qwen3.5-0.8B-BF16.gguf",
        "Qwen3.5-0.8B-Q8_0.gguf",
        "mmproj-Qwen3.5-2B-BF16.gguf",
        "Qwen3.5-2B-Q8_0.gguf",
        "mmproj-Qwen3.5-9B-BF16.gguf",
        "Qwen3.5-9B-Q8_0.gguf",
    ]
    for f in files:
        _write_minimal_gguf(folder / f)

    entries = scan_models(tmp_path)
    by_name = {e.name: e for e in entries}

    expected = [
        ("Qwen3.5-0.8B-Q8_0", "mmproj-Qwen3.5-0.8B-BF16.gguf"),
        ("Qwen3.5-2B-Q8_0", "mmproj-Qwen3.5-2B-BF16.gguf"),
        ("Qwen3.5-9B-Q8_0", "mmproj-Qwen3.5-9B-BF16.gguf"),
    ]
    for stem, expected_name in expected:
        mmproj = by_name[stem].mmproj
        assert mmproj is not None, f"{stem} should have been paired with an mmproj"
        assert mmproj.name == expected_name


def test_scanner_skips_mmproj_from_main_list(tmp_path) -> None:
    _write_minimal_gguf(tmp_path / "MyModel-Q8_0.gguf")
    _write_minimal_gguf(tmp_path / "mmproj-MyModel-F16.gguf")
    entries = scan_models(tmp_path)
    assert len(entries) == 1
    assert entries[0].name == "MyModel-Q8_0"
    assert entries[0].has_vision


def test_scanner_handles_empty_folder(tmp_path) -> None:
    assert scan_models(tmp_path) == []


def test_group_entries_buckets_by_folder(tmp_path) -> None:
    _write_minimal_gguf(tmp_path / "Vendor1" / "ModelA.gguf")
    _write_minimal_gguf(tmp_path / "Vendor2" / "ModelB.gguf")
    entries = scan_models(tmp_path)
    groups = group_entries(entries)
    assert "Vendor1" in groups
    assert "Vendor2" in groups


# ---------------------------------------------------------------------------
# Tuner


def _fake_system(
    ram_total: float = 64,
    ram_free: float = 48,
    vram_total: float = 24,
    vram_free: float = 22,
):
    """Build a synthetic SystemInfo for tuner tests."""
    return SystemInfo(
        os_name="Linux test",
        cpu_name="Test CPU",
        cpu_cores_physical=16,
        cpu_cores_logical=32,
        total_ram_gb=ram_total,
        free_ram_gb=ram_free,
        gpus=[
            GPUInfo(
                index=0,
                name="Test GPU",
                vendor="amd",
                total_vram_mb=int(vram_total * 1024),
                free_vram_mb=int(vram_free * 1024),
            )
        ]
        if vram_total > 0
        else [],
    )


def _fake_model(tmp_path, name, size_gb):
    p = tmp_path / f"{name}.gguf"
    _write_minimal_gguf(p)
    from scanner import ModelEntry

    return ModelEntry(
        path=p,
        name=name,
        group=".",
        size_bytes=int(size_gb * 1024**3),
    )


def test_small_model_full_offload(tmp_path) -> None:
    """A small model on a big GPU → full offload, generous context."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    assert cfg.full_offload is True
    assert cfg.ngl == 999
    assert cfg.ctx >= 32768


def test_huge_model_falls_back_to_partial_or_cpu(tmp_path) -> None:
    """A 50 GB model on a 24 GB GPU → not a full offload."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Mistral-Medium-3.5-128B-UD-IQ3_XXS", size_gb=50.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(ram_total=96, ram_free=80), profile)
    assert cfg.full_offload is False
    assert cfg.ctx >= 2048  # always at least the floor


def test_no_gpu_falls_back_to_cpu(tmp_path) -> None:
    """No GPU → ngl=0, no full_offload."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Bonsai-8B", size_gb=4.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(vram_total=0, vram_free=0), profile)
    assert cfg.ngl == 0
    assert cfg.full_offload is False


def test_user_ctx_override_wins(tmp_path) -> None:
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile, user_ctx=8192)
    assert cfg.ctx == 8192


def test_devstral_uses_high_context_when_ram_is_plenty(tmp_path) -> None:
    """Regression test for the v1 bug: Devstral was capped at 16k context
    even with tons of free RAM. With a roomy system it must now reach far
    above that."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(
        tmp_path, "Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL", size_gb=13.5
    )
    profile = match_profile(model.name, profiles)
    cfg = compute_config(
        model,
        _fake_system(ram_total=96, ram_free=71, vram_total=24, vram_free=22.8),
        profile,
    )
    assert cfg.ctx > 16384, f"v1 bug regression — got only {cfg.ctx}"


# ---------------------------------------------------------------------------
# Command builder


def test_build_command_includes_essentials(tmp_path) -> None:
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(model, cfg, profile, port=12345)
    assert "-m" in cmd and str(model.path) in cmd
    assert "-c" in cmd and str(cfg.ctx) in cmd
    assert "-ngl" in cmd
    assert "--port" in cmd and "12345" in cmd
    assert "-ctk" in cmd and "-ctv" in cmd


def test_build_command_passes_extra_args(tmp_path) -> None:
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Bonsai-8B", size_gb=4.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(model, cfg, profile, extra_args=["--metrics", "--log-disable"])
    assert "--metrics" in cmd
    assert "--log-disable" in cmd


# ---------------------------------------------------------------------------
# MTP detection (scanner.metadata_has_embedded_mtp) — tri-state scan logic


def _mtp_meta(**kw):
    return dict(kw)


def test_mtp_single_file_key_and_tensor() -> None:
    """Standard single-file MTP GGUF: metadata key + scan found → detected."""
    md = _mtp_meta(
        **{
            "general.architecture": "qwen3moe",
            "qwen3moe.nextn_predict_layers": 1,
            "__mtp_scan__": "found",
        }
    )
    assert metadata_has_embedded_mtp(md) is True


def test_mtp_sharded_inconclusive_trusts_key() -> None:
    """Sharded MTP model read from shard 1: the nextn block lives in the LAST
    shard, so the scan is 'inconclusive'. The metadata key must still be
    trusted — this was the 'sometimes detected, sometimes not' bug."""
    md = _mtp_meta(
        **{
            "general.architecture": "deepseek2",
            "deepseek2.nextn_predict_layers": 1,
            "__mtp_scan__": "inconclusive",
        }
    )
    assert metadata_has_embedded_mtp(md) is True


def test_mtp_ud_quant_absent_suppresses_false_positive() -> None:
    """UD/unsloth quant keeps the metadata key but strips the MTP weights. A
    complete non-sharded scan reports 'absent' → key is correctly vetoed."""
    md = _mtp_meta(
        **{
            "general.architecture": "qwen2",
            "qwen2.nextn_predict_layers": 1,
            "__mtp_scan__": "absent",
        }
    )
    assert metadata_has_embedded_mtp(md) is False


def test_mtp_name_based_without_key() -> None:
    """No metadata key at all, but the tensor scan saw a nextn-named tensor
    (scan='found'). Detection must succeed without 'MTP' in the filename."""
    md = _mtp_meta(
        **{"general.architecture": "bailingmoe2", "__mtp_scan__": "found"}
    )
    assert metadata_has_embedded_mtp(md) is True


def test_mtp_external_metadata_no_scan_trusts_key() -> None:
    """Metadata from a source that ran no tensor scan (no __mtp_scan__ key):
    the metadata key is trusted."""
    md = _mtp_meta(
        **{"general.architecture": "qwen3moe", "qwen3moe.nextn_predict_layers": 2}
    )
    assert metadata_has_embedded_mtp(md) is True


def test_mtp_negative_clean_model() -> None:
    md = _mtp_meta(**{"general.architecture": "llama", "__mtp_scan__": "absent"})
    assert metadata_has_embedded_mtp(md) is False
    assert metadata_has_embedded_mtp({}) is False


# ---------------------------------------------------------------------------
# n-gram (ngram-mod) speculative decoding flags


def _spec_tokens(cmd):
    """Return the --spec-type value, or '' if none was emitted."""
    if "--spec-type" in cmd:
        return cmd[cmd.index("--spec-type") + 1]
    return ""


def test_ngram_disabled_by_default(tmp_path) -> None:
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Llama-3-8B", size_gb=8.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(model, cfg, profile)
    assert "--spec-type" not in cmd
    assert "--spec-ngram-mod-n-match" not in cmd


def test_ngram_standalone_on_any_model(tmp_path) -> None:
    """ngram-mod needs no draft model — it must work on a plain model."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Llama-3-8B", size_gb=8.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(model, cfg, profile, enable_ngram=True)
    assert _spec_tokens(cmd) == "ngram-mod"
    assert "--spec-ngram-mod-n-match" in cmd
    assert "--spec-ngram-mod-n-min" in cmd
    assert "--spec-ngram-mod-n-max" in cmd


def test_ngram_suppressed_when_embedded_mtp(tmp_path) -> None:
    """MTP + ngram → ONLY draft-mtp; ngram-mod is dropped.

    Combining draft-mtp,ngram-mod in one --spec-type list crashes
    mid-generation on MTP models (llama.cpp #23154, open as of b9305). The
    trained MTP head supersedes a generic ngram hash anyway, so draft-mtp wins
    and ngram-mod (plus its --spec-ngram-mod-* flags) is suppressed.
    """
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.6-MoE-A3B", size_gb=8.0)
    model.metadata = {
        "general.architecture": "qwen3moe",
        "qwen3moe.nextn_predict_layers": 1,
        "qwen3moe.block_count": 48,
        "__mtp_scan__": "found",
    }
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(model, cfg, profile, enable_ngram=True)
    types = _spec_tokens(cmd).split(",")
    assert "draft-mtp" in types
    assert "ngram-mod" not in types
    assert "--spec-ngram-mod-n-match" not in cmd
    assert "--spec-ngram-mod-n-min" not in cmd
    assert "--spec-ngram-mod-n-max" not in cmd


def test_ngram_survives_speculative_disabled(tmp_path) -> None:
    """Unchecking Draft (enable_speculative=False) suppresses MTP but ngram
    is independent and must still be emitted."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.6-MoE-A3B", size_gb=8.0)
    model.metadata = {
        "general.architecture": "qwen3moe",
        "qwen3moe.nextn_predict_layers": 1,
        "qwen3moe.block_count": 48,
        "__mtp_scan__": "found",
    }
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(
        model, cfg, profile, enable_speculative=False, enable_ngram=True
    )
    assert _spec_tokens(cmd) == "ngram-mod"


def test_ngram_map_k4v_coexists_with_mtp(tmp_path) -> None:
    """b9334: ngram-map-k4v is the MTP-compatible draftless method.

    Unlike ngram-mod (suppressed next to draft-mtp because of #23154), a
    profile that selects ngram_method=ngram-map-k4v must emit BOTH draft-mtp
    and ngram-map-k4v in --spec-type, plus the k4v parameter flags (and never
    the ngram-mod flags). This is ggerganov's MTP clean-up combo (PR #23269).
    """
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.6-MoE-A3B", size_gb=8.0)
    model.metadata = {
        "general.architecture": "qwen3moe",
        "qwen3moe.nextn_predict_layers": 1,
        "qwen3moe.block_count": 48,
        "__mtp_scan__": "found",
    }
    profile = match_profile(model.name, profiles)
    profile.ngram_method = "ngram-map-k4v"
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(model, cfg, profile, enable_ngram=True)
    types = _spec_tokens(cmd).split(",")
    assert "draft-mtp" in types
    assert "ngram-map-k4v" in types
    assert "ngram-mod" not in types
    assert "--spec-ngram-map-k4v-size-n" in cmd
    assert "--spec-ngram-map-k4v-size-m" in cmd
    assert "--spec-ngram-map-k4v-min-hits" in cmd
    assert "--spec-ngram-mod-n-match" not in cmd


def test_ngram_map_k_emits_type_only(tmp_path) -> None:
    """ngram-map-k / ngram-simple / ngram-cache: emit only the --spec-type
    token and rely on llama.cpp defaults (no guessed sub-param flags)."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Llama-3-8B", size_gb=8.0)
    profile = match_profile(model.name, profiles)
    profile.ngram_method = "ngram-map-k"
    cfg = compute_config(model, _fake_system(), profile)
    cmd = build_command(model, cfg, profile, enable_ngram=True)
    assert _spec_tokens(cmd) == "ngram-map-k"
    assert "--spec-ngram-mod-n-match" not in cmd
    assert "--spec-ngram-map-k4v-size-n" not in cmd


def test_ngram_method_invalid_falls_back(tmp_path) -> None:
    """An unknown ngram_method in YAML must fall back to ngram-mod at load."""
    from settings_loader import _validate_ngram_method

    assert _validate_ngram_method("totally-bogus", "x.yaml") == "ngram-mod"
    assert _validate_ngram_method("NGRAM-MAP-K4V", "x.yaml") == "ngram-map-k4v"


# ---------------------------------------------------------------------------
# GPU detection / iGPU filtering


def test_filter_drops_igpu_next_to_dgpu() -> None:
    """User's exact scenario: Intel iGPU + AMD RX 9070 XT.
    The iGPU must be ignored so the tuner doesn't underuse the dGPU."""
    from hardware import _filter_inference_gpus

    gpus = [
        GPUInfo(
            index=0,
            name="Intel(R) Graphics",
            vendor="intel",
            total_vram_mb=2048,
            free_vram_mb=1900,
        ),
        GPUInfo(
            index=1,
            name="AMD Radeon RX 9070 XT",
            vendor="amd",
            total_vram_mb=16 * 1024,
            free_vram_mb=int(15.2 * 1024),
        ),
    ]
    used, ignored = _filter_inference_gpus(gpus)
    assert len(used) == 1
    assert used[0].vendor == "amd"
    assert len(ignored) == 1
    assert ignored[0].vendor == "intel"


def test_filter_keeps_matched_dual_gpus() -> None:
    """Two equal dGPUs (e.g. 2x RTX 4090) must both be kept for tensor-split."""
    from hardware import _filter_inference_gpus

    gpus = [
        GPUInfo(
            index=0,
            name="RTX 4090",
            vendor="nvidia",
            total_vram_mb=24 * 1024,
            free_vram_mb=23 * 1024,
        ),
        GPUInfo(
            index=1,
            name="RTX 4090",
            vendor="nvidia",
            total_vram_mb=24 * 1024,
            free_vram_mb=23 * 1024,
        ),
    ]
    used, ignored = _filter_inference_gpus(gpus)
    assert len(used) == 2 and not ignored


def test_vendor_inference() -> None:
    from hardware import _vendor_from_name

    assert _vendor_from_name("AMD Radeon RX 9070 XT") == "amd"
    assert _vendor_from_name("NVIDIA GeForce RTX 4090") == "nvidia"
    assert _vendor_from_name("Intel(R) UHD Graphics 770") == "intel"
    assert _vendor_from_name("Intel(R) Arc(TM) A770") == "intel"
    assert _vendor_from_name("some-mystery-card") == "unknown"


def test_filter_3gpu_workstation_setup() -> None:
    """Basti's workstation: 9700 Pro 32GB + 9070 XT 16GB + Intel iGPU.

    Both Radeons must be kept (16 is exactly half of 32 → still a peer
    under the >= half-of-largest rule); the iGPU must be ignored; the
    9700 Pro must end up as the first entry (largest), so main_gpu
    selection later picks it.
    """
    from hardware import _filter_inference_gpus

    gpus = [
        GPUInfo(
            index=0,
            name="Intel(R) Graphics",
            vendor="intel",
            total_vram_mb=2048,
            free_vram_mb=1900,
        ),
        GPUInfo(
            index=1,
            name="AMD Radeon RX 9070 XT",
            vendor="amd",
            total_vram_mb=16 * 1024,
            free_vram_mb=14 * 1024,
        ),
        GPUInfo(
            index=2,
            name="AMD Radeon AI PRO R9700",
            vendor="amd",
            total_vram_mb=32 * 1024,
            free_vram_mb=31 * 1024,
        ),
    ]
    used, ignored = _filter_inference_gpus(gpus)
    assert len(used) == 2
    assert used[0].total_vram_mb == 32 * 1024  # 9700 Pro first (largest)
    assert used[1].total_vram_mb == 16 * 1024  # 9070 XT kept as peer
    assert len(ignored) == 1
    assert ignored[0].vendor == "intel"


def test_vendor_inference_r9700_workstation_card() -> None:
    """The new Radeon AI PRO R9700 must register as AMD (was untested)."""
    from hardware import _vendor_from_name

    assert _vendor_from_name("AMD Radeon AI PRO R9700") == "amd"
    assert _vendor_from_name("Radeon AI PRO R9700 AI TOP") == "amd"


def _fake_dual_gpu_system(
    large_total: float = 32,
    large_free: float = 31,
    small_total: float = 16,
    small_free: float = 14,
    ram_total: float = 96,
    ram_free: float = 80,
) -> SystemInfo:
    """Synthetic two-GPU SystemInfo (large + small) for placement tests."""
    return SystemInfo(
        os_name="Linux test",
        cpu_name="Test CPU",
        cpu_cores_physical=24,
        cpu_cores_logical=24,
        total_ram_gb=ram_total,
        free_ram_gb=ram_free,
        gpus=[
            GPUInfo(
                index=0,
                name="AMD Radeon AI PRO R9700",
                vendor="amd",
                total_vram_mb=int(large_total * 1024),
                free_vram_mb=int(large_free * 1024),
            ),
            GPUInfo(
                index=1,
                name="AMD Radeon RX 9070 XT",
                vendor="amd",
                total_vram_mb=int(small_total * 1024),
                free_vram_mb=int(small_free * 1024),
            ),
        ],
    )


def test_multi_gpu_pins_to_largest_when_model_fits(tmp_path) -> None:
    """Smart placement: a model that fits on the 9700 Pro (32 GB) alone
    must NOT be tensor-split across the 9070 XT. Tensor-split should be
    "1.000,0.000" so the 9070 XT stays free for OBS/desktop work."""
    profiles = load_profiles(SETTINGS_DIR)
    # 9B model, easily fits on a 32GB GPU even with full KV cache.
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_dual_gpu_system(), profile)

    assert cfg.tensor_split is not None, "Expected tensor_split to be set"
    assert cfg.main_gpu == 0, f"Expected main_gpu=0 (largest), got {cfg.main_gpu}"

    parts = [float(x) for x in cfg.tensor_split.split(",")]
    assert len(parts) == 2
    assert parts[0] > 0.99, f"Expected ~1.0 on GPU 0, got {parts[0]}"
    assert parts[1] < 0.01, f"Expected ~0.0 on GPU 1, got {parts[1]}"


def test_multi_gpu_spreads_when_model_too_large_for_single_gpu(tmp_path) -> None:
    """Smart placement: a model too big for the 9700 Pro alone (with KV +
    safety) must spread across both GPUs proportionally to total VRAM."""
    profiles = load_profiles(SETTINGS_DIR)
    # 40 GB model — well over the 32 GB single-GPU budget once KV/safety
    # are added; must use both GPUs.
    model = _fake_model(tmp_path, "Mistral-Medium-3.5-128B-UD-IQ3_XXS", size_gb=40.0)
    profile = match_profile(model.name, profiles)
    cfg = compute_config(model, _fake_dual_gpu_system(), profile)

    assert cfg.tensor_split is not None
    assert cfg.main_gpu == 0  # still the largest

    parts = [float(x) for x in cfg.tensor_split.split(",")]
    assert len(parts) == 2
    # Proportional split: 32/48 ≈ 0.667 and 16/48 ≈ 0.333. Either both
    # are non-trivial (true spread), OR the model didn't fit at all and
    # we offloaded to RAM — but in that case the split is still set,
    # so we only assert it's not the pinned form.
    pinned = parts[0] > 0.99 and parts[1] < 0.01
    assert not pinned, f"Expected proportional split, got pinned: {cfg.tensor_split}"


def _fake_dual_gpu_system_with_vk_order(
    large_total: float = 32,
    large_free: float = 31,
    small_total: float = 16,
    small_free: float = 14,
    ram_total: float = 96,
    ram_free: float = 80,
    large_vk_idx: int = 1,
    small_vk_idx: int = 0,
) -> SystemInfo:
    """Two-GPU system with hip_index set (simulating vulkaninfo resolution).

    Default: small GPU is Vulkan device 0, large is device 1 — the
    typical case on Windows where the primary display adapter (the
    gaming GPU) enumerates first in Vulkan.
    """
    return SystemInfo(
        os_name="Windows 11 test",
        cpu_name="Test CPU",
        cpu_cores_physical=24,
        cpu_cores_logical=24,
        total_ram_gb=ram_total,
        free_ram_gb=ram_free,
        gpus=[
            GPUInfo(
                index=0,
                name="AMD Radeon AI PRO R9700",
                vendor="amd",
                total_vram_mb=int(large_total * 1024),
                free_vram_mb=int(large_free * 1024),
                hip_index=large_vk_idx,
            ),
            GPUInfo(
                index=1,
                name="AMD Radeon RX 9070 XT",
                vendor="amd",
                total_vram_mb=int(small_total * 1024),
                free_vram_mb=int(small_free * 1024),
                hip_index=small_vk_idx,
            ),
        ],
    )


def test_vulkan_env_var_emitted_for_single_gpu_pinning(tmp_path) -> None:
    """When hip_index is known and the model fits on one GPU, both
    HIP_VISIBLE_DEVICES and GGML_VK_VISIBLE_DEVICES must be emitted
    so the config works on both ROCm and Vulkan backends."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    profile = match_profile(model.name, profiles)
    sys_info = _fake_dual_gpu_system_with_vk_order()
    cfg = compute_config(model, sys_info, profile)

    assert cfg.full_offload is True
    assert cfg.main_gpu == 0  # remapped: only one device visible
    # Both env vars must point to the R9700's Vulkan index (1).
    assert "HIP_VISIBLE_DEVICES" in cfg.env_overrides
    assert "GGML_VK_VISIBLE_DEVICES" in cfg.env_overrides
    assert cfg.env_overrides["HIP_VISIBLE_DEVICES"] == "1"
    assert cfg.env_overrides["GGML_VK_VISIBLE_DEVICES"] == "1"


def test_vulkan_env_var_emitted_for_multi_gpu_spread(tmp_path) -> None:
    """When a large model must spread across both GPUs, both
    HIP_VISIBLE_DEVICES and GGML_VK_VISIBLE_DEVICES must list
    all devices in Vulkan order."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Mistral-Medium-3.5-128B-UD-IQ3_XXS", size_gb=40.0)
    profile = match_profile(model.name, profiles)
    sys_info = _fake_dual_gpu_system_with_vk_order()
    cfg = compute_config(model, sys_info, profile)

    assert "GGML_VK_VISIBLE_DEVICES" in cfg.env_overrides
    # Both env vars must list devices in ascending Vulkan order.
    expected = cfg.env_overrides["HIP_VISIBLE_DEVICES"]
    assert cfg.env_overrides["GGML_VK_VISIBLE_DEVICES"] == expected


def test_priority_weighted_tensor_split(tmp_path) -> None:
    """With gpu_priorities R9700=2 / 9070 XT=1, the R9700 must receive
    a significantly larger share of the tensor_split than a pure
    VRAM-proportional split would give. This keeps the gaming GPU
    (9070 XT) mostly free for OBS/desktop while the AI GPU (R9700)
    does the heavy lifting."""
    profiles = load_profiles(SETTINGS_DIR)
    model = _fake_model(tmp_path, "Mistral-Medium-3.5-128B-UD-IQ3_XXS", size_gb=40.0)
    profile = match_profile(model.name, profiles)
    sys_info = _fake_dual_gpu_system_with_vk_order(
        large_free=31, small_free=14,
    )
    prio = {
        "AMD Radeon AI PRO R9700": 2,
        "AMD Radeon RX 9070 XT": 1,
    }
    cfg = compute_config(model, sys_info, profile, gpu_priorities=prio)

    assert cfg.tensor_split is not None
    parts = [float(x) for x in cfg.tensor_split.split(",")]
    assert len(parts) == 2
    # Vulkan order: device 0 = 9070 XT (small_vk_idx=0), device 1 = R9700 (large_vk_idx=1)
    xt_share = parts[0]   # 9070 XT
    r97_share = parts[1]  # R9700
    # R9700 must get the lion's share — at least 75% (it has 2× priority
    # AND 2× the free VRAM). Pure VRAM-proportional would give only 67%.
    assert r97_share > 0.75, (
        f"R9700 should dominate with priority=2; got share={r97_share:.3f}"
    )
    assert xt_share < 0.25, (
        f"9070 XT should be ≤25% with priority=1; got share={xt_share:.3f}"
    )


# ---------------------------------------------------------------------------
# llama-server resolver


def test_resolver_returns_input_when_nothing_matches(tmp_path, monkeypatch) -> None:
    """When the binary can't be found anywhere, the resolver echoes the
    original input — `launch()` then prints a clean 'not found' error
    rather than us silently swallowing the failure."""
    from auto_tuner import _resolve_server_binary

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    result = _resolve_server_binary("definitely-not-a-binary-anywhere")
    assert result == "definitely-not-a-binary-anywhere"


def test_resolver_finds_binary_in_sibling_llama_cpp(tmp_path, monkeypatch) -> None:
    """Simulate the user's layout: auto_tuner sits beside an ai-local/
    folder that contains the llama.cpp checkout."""
    from auto_tuner import _resolve_server_binary

    # Build the tree: tmp/Auto Tuner/, tmp/ai-local/llama.cpp/build/...
    auto_dir = tmp_path / "Auto Tuner"
    auto_dir.mkdir()
    server = (
        tmp_path
        / "ai-local"
        / "llama.cpp"
        / "build"
        / "bin"
        / "Release"
        / "llama-server.exe"
    )
    server.parent.mkdir(parents=True)
    server.write_text("")

    monkeypatch.chdir(auto_dir)
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)

    resolved = _resolve_server_binary("llama-server")
    assert Path(resolved).resolve() == server.resolve(), (
        f"expected {server}, got {resolved}"
    )


def test_resolver_distinguishes_between_llama_and_1b_llama(tmp_path, monkeypatch) -> None:
    """The Bonsai-Ternary profile uses a relative path starting with the
    fork's directory name. The resolver must respect that and pick the
    1b_llama.cpp checkout, not the regular one sitting next to it."""
    from auto_tuner import _resolve_server_binary

    auto_dir = tmp_path / "Auto Tuner"
    auto_dir.mkdir()
    regular = (
        tmp_path
        / "ai-local"
        / "llama.cpp"
        / "build"
        / "bin"
        / "Release"
        / "llama-server.exe"
    )
    bitnet = (
        tmp_path
        / "ai-local"
        / "1b_llama.cpp"
        / "build"
        / "bin"
        / "Release"
        / "llama-server.exe"
    )
    for s in (regular, bitnet):
        s.parent.mkdir(parents=True)
        s.write_text("")

    monkeypatch.chdir(auto_dir)
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)

    # Default resolves to the regular fork
    res1 = _resolve_server_binary("llama-server")
    assert Path(res1).resolve() == regular.resolve()

    # Profile-style relative path must hit the BitNet fork
    res2 = _resolve_server_binary("1b_llama.cpp/build/bin/Release/llama-server.exe")
    assert Path(res2).resolve() == bitnet.resolve()


def test_turbo_quant_mode_selection(tmp_path, monkeypatch) -> None:
    """Test that the turbo mode selection logic works (mocking input)."""
    from auto_tuner import main
    import sys
    from unittest.mock import patch

    # Mocking sys.argv to avoid command line arguments
    sys.argv = ["auto_tuner.py"]

    # We can't easily test the interactive input in a pure unit test
    # without complex mocking, but we can verify the logic if we
    # were to refactor main. For now, we ensure no crashes occur
    # when simulating different inputs.
    with patch("builtins.input", side_effect=["2", KeyboardInterrupt]):
        try:
            main([])
        except (KeyboardInterrupt, SystemExit):
            pass  # Expected behavior for testing exit


# ---------------------------------------------------------------------------
# Profile schema (server_binary field)


def test_profile_supports_server_binary_field() -> None:
    """Bonsai-Ternary should declare its preferred server binary."""
    profiles = load_profiles(SETTINGS_DIR)
    by_name = {p.source_file: p for p in profiles}
    assert "bonsai-ternary.yaml" in by_name
    p = by_name["bonsai-ternary.yaml"]
    assert p.server_binary, "Bonsai-Ternary profile must set server_binary"
    assert "1b_llama" in p.server_binary.lower()


def test_ternary_bonsai_pattern_beats_regular_bonsai() -> None:
    """Longest-pattern-wins: a Ternary-Bonsai filename must match the
    BitNet profile, not the generic Bonsai one."""
    profiles = load_profiles(SETTINGS_DIR)
    p = match_profile("Ternary-Bonsai-8B-Q2_0", profiles)
    assert p.source_file is not None, "matched profile must come from a YAML file"
    assert "ternary" in (p.display_name + " " + p.source_file).lower()
    assert p.server_binary, "Ternary profile must override the server"


# ---------------------------------------------------------------------------
# Performance targets


def test_performance_target_registry_has_three_tiers() -> None:
    """Sanity: the three documented tiers exist and are well-ordered."""
    from performance_target import PERFORMANCE_TARGETS, list_target_names

    names = list_target_names()
    assert names == ["safe", "balanced", "throughput"]
    safe = PERFORMANCE_TARGETS["safe"]
    bal = PERFORMANCE_TARGETS["balanced"]
    thr = PERFORMANCE_TARGETS["throughput"]
    # KV reservation should shrink monotonically: safe ≥ balanced ≥ throughput
    assert safe.moe_placement_ctx_target >= bal.moe_placement_ctx_target
    assert bal.moe_placement_ctx_target >= thr.moe_placement_ctx_target
    # Same for VRAM safety bands
    assert safe.moe_vram_safety_gb >= bal.moe_vram_safety_gb
    assert bal.moe_vram_safety_gb >= thr.moe_vram_safety_gb


def test_resolve_performance_target_priority() -> None:
    """CLI choice beats profile choice beats default."""
    from performance_target import resolve_performance_target

    # CLI wins
    assert resolve_performance_target("safe", "throughput").name == "safe"
    # Profile wins when CLI is empty
    assert resolve_performance_target(None, "throughput").name == "throughput"
    # Default kicks in when both empty
    assert resolve_performance_target(None, None).name == "balanced"
    # Unknown values are silently skipped
    assert resolve_performance_target("typo", "alsotypo").name == "balanced"
    # Mixed case + whitespace tolerated
    assert resolve_performance_target("  Throughput ", None).name == "throughput"


def test_yaml_performance_target_is_loaded() -> None:
    """qwen3_5-3_6.yaml ships with performance_target: throughput."""
    profiles = load_profiles(SETTINGS_DIR)
    by_file = {p.source_file: p for p in profiles}
    assert "qwen3_5-3_6.yaml" in by_file
    assert by_file["qwen3_5-3_6.yaml"].performance_target == "throughput"


def test_throughput_places_more_moe_layers_on_gpu_than_safe(tmp_path) -> None:
    """The whole point of the perf_target switch: on a constrained MoE
    setup, 'throughput' should reserve less KV and therefore put MORE
    expert layers on the GPU than 'safe'.

    We simulate Basti's actual situation: Qwen3.6-A3B (~28 GB Q6_K),
    16 GB VRAM, 48 GB free RAM. Different ctx targets in the two tiers
    must produce different cpu_moe counts.
    """
    from performance_target import PERFORMANCE_TARGETS

    # Use a real MoE profile — Qwen3.6 ships expert_count via GGUF in the
    # wild, but our fake_model has empty metadata. So we hand-craft a
    # ModelEntry with metadata that tells the tuner "this is MoE".
    from scanner import ModelEntry

    p = tmp_path / "Qwen3.6-35B-A3B-UD-Q6_K.gguf"
    _write_minimal_gguf(p)
    model = ModelEntry(
        path=p,
        name="Qwen3.6-35B-A3B-UD-Q6_K",
        group=".",
        size_bytes=int(28.0 * 1024**3),
        metadata={
            "general.architecture": "qwen3moe",
            "qwen3moe.expert_count": 128,
            "qwen3moe.block_count": 64,
            "qwen3moe.context_length": 262144,
        },
    )

    profiles = load_profiles(SETTINGS_DIR)
    profile = match_profile(model.name, profiles)
    sys_info = _fake_system(ram_total=64, ram_free=48, vram_total=16, vram_free=14)

    cfg_safe = compute_config(
        model, sys_info, profile, perf_target=PERFORMANCE_TARGETS["safe"]
    )
    cfg_thr = compute_config(
        model, sys_info, profile, perf_target=PERFORMANCE_TARGETS["throughput"]
    )

    # Both should detect the MoE
    assert cfg_safe.is_moe and cfg_thr.is_moe

    # Throughput places fewer experts on CPU (= more on GPU). Allow equal
    # for edge cases where the model fully fits or fully doesn't fit, but
    # in this constrained scenario throughput must beat safe strictly.
    safe_cpu = cfg_safe.n_cpu_moe or 0
    thr_cpu = cfg_thr.n_cpu_moe or 0
    assert thr_cpu < safe_cpu, (
        f"throughput should keep more experts on GPU than safe; "
        f"got safe={safe_cpu} CPU experts, throughput={thr_cpu}"
    )

    # And the metadata field round-trips into TunedConfig
    assert cfg_safe.performance_target == "safe"
    assert cfg_thr.performance_target == "throughput"


def test_perf_target_default_balanced_when_profile_has_none(tmp_path) -> None:
    """A profile without performance_target: falls back to balanced."""
    profiles = load_profiles(SETTINGS_DIR)
    # Pick a profile that we know has no perf_target set
    p_default = next((pr for pr in profiles if pr.source_file == "_default.yaml"), None)
    assert p_default is not None
    assert p_default.performance_target == ""

    model = _fake_model(tmp_path, "Qwen3.5-9B-Q8_0", size_gb=9.0)
    cfg = compute_config(model, _fake_system(), p_default)
    assert cfg.performance_target == "balanced"


# ---------------------------------------------------------------------------
# Hybrid Mamba/Transformer detection (Punkt 3)


def test_hybrid_architecture_detection_by_name() -> None:
    """Architecture name matches a known hybrid → True."""
    from scanner import metadata_is_hybrid_architecture

    assert (
        metadata_is_hybrid_architecture({"general.architecture": "nemotron_h"}) is True
    )
    assert metadata_is_hybrid_architecture({"general.architecture": "jamba"}) is True
    # Pure Transformer
    assert (
        metadata_is_hybrid_architecture({"general.architecture": "qwen3moe"}) is False
    )
    assert metadata_is_hybrid_architecture({"general.architecture": "llama"}) is False


def test_hybrid_detection_via_ssm_keys() -> None:
    """Generic SSM key catches new hybrid archs not on the allow-list."""
    from scanner import metadata_is_hybrid_architecture

    md = {
        "general.architecture": "future_hybrid_arch",
        "future_hybrid_arch.ssm.state_size": 16,
    }
    assert metadata_is_hybrid_architecture(md) is True


def test_attention_layer_count_pure_transformer() -> None:
    """For pure Transformer, attention count == block count."""
    from scanner import metadata_attention_layer_count

    md = {
        "general.architecture": "qwen3moe",
        "qwen3moe.block_count": 64,
    }
    assert metadata_attention_layer_count(md) == 64


def test_attention_layer_count_hybrid_with_explicit_metadata() -> None:
    """Hybrid with explicit attention count uses that, not the heuristic."""
    from scanner import metadata_attention_layer_count

    md = {
        "general.architecture": "nemotron_h",
        "nemotron_h.block_count": 50,
        "nemotron_h.attention.block_count": 12,
    }
    assert metadata_attention_layer_count(md) == 12


def test_attention_layer_count_hybrid_with_heuristic() -> None:
    """Hybrid without explicit count falls back to per-arch ratio."""
    from scanner import metadata_attention_layer_count

    md = {
        "general.architecture": "nemotron_h",
        "nemotron_h.block_count": 50,
    }
    # Nemotron heuristic: ~25%
    n = metadata_attention_layer_count(md)
    assert 10 <= n <= 16, f"expected ~25% of 50, got {n}"


def test_kv_per_token_estimate_smaller_for_hybrid(tmp_path) -> None:
    """A hybrid model should produce a smaller per-token KV estimate
    than the same-shaped pure Transformer would."""
    from tuner import kv_per_token_mb_from_metadata

    common = {
        "attention.head_count": 32,
        "attention.head_count_kv": 8,
        "embedding_length": 4096,
        "block_count": 50,
    }
    pure = {
        "general.architecture": "llama",
        **{f"llama.{k}": v for k, v in common.items()},
    }
    hybrid = {
        "general.architecture": "nemotron_h",
        **{f"nemotron_h.{k}": v for k, v in common.items()},
    }

    pure_kv = kv_per_token_mb_from_metadata(pure)
    hybrid_kv = kv_per_token_mb_from_metadata(hybrid)
    assert pure_kv > 0 and hybrid_kv > 0
    # Hybrid should be roughly 25% of pure (Nemotron heuristic).
    ratio = hybrid_kv / pure_kv
    assert 0.15 <= ratio <= 0.35, f"expected hybrid KV ≈ 25% of pure, got {ratio:.2%}"


def test_two_pass_placement_recovers_when_first_pass_dumps_to_cpu(tmp_path) -> None:
    """Defensive net: if pass 1 forces all experts to CPU but VRAM is
    free, pass 2 with a halved target_ctx must recover at least one
    layer onto the GPU."""
    from performance_target import PERFORMANCE_TARGETS
    from scanner import ModelEntry

    # Construct a tight scenario: 28 GB MoE on 14 GB VRAM, but we lie
    # about KV-per-token by NOT supplying metadata, so the params-based
    # heuristic kicks in (which over-estimates for MoE) and pass 1 dumps
    # everything to CPU. The Two-Pass should retry and pull at least
    # one layer back onto the GPU once the target_ctx is halved.
    p = tmp_path / "FakeMoE-35B-A3B.gguf"
    _write_minimal_gguf(p)
    model = ModelEntry(
        path=p,
        name="FakeMoE-35B-A3B",
        group=".",
        size_bytes=int(28.0 * 1024**3),
        metadata={
            "general.architecture": "qwen3moe",
            "qwen3moe.expert_count": 128,
            "qwen3moe.block_count": 64,
            "qwen3moe.context_length": 262144,
            # Deliberately no attention.* keys → forces params heuristic.
        },
    )
    profiles = load_profiles(SETTINGS_DIR)
    profile = match_profile(model.name, profiles)
    sys_info = _fake_system(ram_total=64, ram_free=48, vram_total=16, vram_free=14)

    cfg_safe = compute_config(
        model, sys_info, profile, perf_target=PERFORMANCE_TARGETS["safe"]
    )
    # Even safe (128k target) should not dump all 64 layers to CPU when
    # the two-pass kicks in.
    assert (cfg_safe.n_cpu_moe or 0) < 64, (
        f"two-pass should have rescued at least one layer; "
        f"got n_cpu_moe={cfg_safe.n_cpu_moe}"
    )


# ---------------------------------------------------------------------------
# Reasoning / thinking detection (Punkt 2)


def test_thinking_detection_qwen3_coder_is_false() -> None:
    """Qwen3-Coder must NOT be flagged as a thinking model just because
    its filename contains 'qwen3' — that's the bug we're fixing."""
    from scanner import metadata_supports_thinking

    # No metadata at all — pure filename heuristic.
    assert metadata_supports_thinking({}, "Qwen3-Coder-30B-A3B-Q6_K") is False


def test_thinking_detection_uses_chat_template() -> None:
    """When the chat template contains thinking markers, return True."""
    from scanner import metadata_supports_thinking

    md_with_think = {
        "tokenizer.chat_template": "{% if enable_thinking %}<think>\n{% endif %}"
    }
    assert metadata_supports_thinking(md_with_think, "SomeModel") is True


def test_thinking_detection_template_without_marker_is_false() -> None:
    """A model that ships a chat template without <think> markers is not
    a thinking model — even if its name contains 'qwen3'."""
    from scanner import metadata_supports_thinking

    md_no_think = {
        "tokenizer.chat_template": "<|im_start|>user\n{{ messages }}<|im_end|>"
    }
    # Filename hint says "qwen3" → would have been True under old heuristic.
    assert metadata_supports_thinking(md_no_think, "Qwen3.6-35B-Q6") is False


def test_thinking_detection_falls_back_to_filename_when_no_template() -> None:
    """No metadata at all → use filename keywords."""
    from scanner import metadata_supports_thinking

    assert metadata_supports_thinking({}, "Gemma-3-27B-it") is True
    assert metadata_supports_thinking({}, "DeepSeek-R1-Distill-Llama-70B") is True
    assert metadata_supports_thinking({}, "QwQ-32B-Preview") is True
    # Pure non-thinking
    assert metadata_supports_thinking({}, "Mistral-7B-Instruct") is False


def test_thinking_detection_excludes_qwen3_2507_instruct() -> None:
    """Qwen3-2507-Instruct is the explicitly non-thinking branch — even
    if it inherits a generic template that mentions <think>, the filename
    exclusion must win."""
    from scanner import metadata_supports_thinking

    md_with_think = {"tokenizer.chat_template": "<think>{{ content }}</think>"}
    assert (
        metadata_supports_thinking(md_with_think, "Qwen3-7B-Instruct-2507-Q6_K")
        is False
    )


# ---------------------------------------------------------------------------
# mmproj detection — token-anywhere + .mmproj extension (regression guards)


def test_mmproj_token_detected_mid_name(tmp_path) -> None:
    """A projector whose 'mmproj' marker sits mid-name (after the quant
    label) must still be classified as a projector and paired with its
    model. Regression for qwen3.6-…-mxfp4-moe-mmproj-f16.gguf, which the
    old prefix-only check ('mmproj-' / 'mmproj_') missed entirely."""
    folder = tmp_path / "Alibaba" / "Abliterated"
    _write_minimal_gguf(folder / "qwen3.6-35b-a3b-mxfp4_moe.gguf")
    _write_minimal_gguf(folder / "qwen3.6-35b-a3b-mxfp4-moe-mmproj-f16.gguf")

    entries = scan_models(tmp_path)
    # Only the model is a choosable entry; the projector is attached.
    assert len(entries) == 1, [e.name for e in entries]
    model = entries[0]
    assert model.name == "qwen3.6-35b-a3b-mxfp4_moe"
    assert model.has_vision, "mid-name mmproj projector was not paired"
    assert model.mmproj is not None
    assert model.mmproj.name == "qwen3.6-35b-a3b-mxfp4-moe-mmproj-f16.gguf"


def test_mmproj_extension_file_is_paired(tmp_path) -> None:
    """A projector saved with a literal '.mmproj' extension (e.g. some
    audio projectors) is not matched by the '*.gguf' glob; the scanner
    must pick it up via the dedicated '*.mmproj' pass and attach it."""
    folder = tmp_path / "Liquid AI"
    _write_minimal_gguf(folder / "LFM2.5-Audio-1.5B-bf16.gguf")
    # ".mmproj" extension projector (content irrelevant — never parsed as model)
    (folder).mkdir(parents=True, exist_ok=True)
    (folder / "LFM2.5-Audio-1.5B-f32.mmproj").write_bytes(b"GGUF")

    entries = scan_models(tmp_path)
    assert len(entries) == 1, [e.name for e in entries]
    model = entries[0]
    assert model.has_vision, ".mmproj-extension projector was not paired"
    assert model.mmproj is not None
    assert model.mmproj.name.endswith(".mmproj")


def test_mmproj_pairing_still_size_specific(tmp_path) -> None:
    """The looser (bidirectional, separator-tolerant) matcher must NOT
    let a model grab a projector for a different size in the same dir."""
    folder = tmp_path / "X"
    _write_minimal_gguf(folder / "Qwen3.5-2B-Q8_0.gguf")
    _write_minimal_gguf(folder / "mmproj-Qwen3.5-0.8B-BF16.gguf")
    entries = scan_models(tmp_path)
    by_name = {e.name: e for e in entries}
    # The 2B model must NOT be paired with the 0.8B projector.
    assert by_name["Qwen3.5-2B-Q8_0"].mmproj is None


# ---------------------------------------------------------------------------
# GGUF sampling metadata fallback (loops / tool-call fix)


def _fake_model_md(tmp_path, name, size_gb, metadata):
    """Like _fake_model but attaches GGUF metadata."""
    from scanner import ModelEntry

    p = tmp_path / f"{name}.gguf"
    _write_minimal_gguf(p)
    return ModelEntry(
        path=p,
        name=name,
        group=".",
        size_bytes=int(size_gb * 1024**3),
        metadata=metadata,
    )


def test_metadata_sampling_reader() -> None:
    """general.sampling.* keys are read and typed correctly."""
    from scanner import metadata_sampling

    md = {
        "general.sampling.temp": 1.0,
        "general.sampling.top_k": 20,
        "general.sampling.top_p": 0.95,
    }
    s = metadata_sampling(md)
    assert s["temperature"] == 1.0
    assert s["top_k"] == 20.0
    assert s["top_p"] == 0.95
    assert "min_p" not in s  # absent key stays absent
    # Empty / missing metadata yields an empty dict, never raises.
    assert metadata_sampling({}) == {}


def test_gguf_sampling_used_when_profile_is_fallback(tmp_path) -> None:
    """A model that matches only the generic fallback profile must adopt
    the author-recommended GGUF samplers instead of the generic defaults.
    This is the loop / broken-tool-call fix for unprofiled models: any
    GGUF that ships general.sampling.* but has no tailored YAML profile."""
    profiles = load_profiles(SETTINGS_DIR)
    md = {
        "general.architecture": "llama",
        "llama.block_count": 32,
        "general.sampling.temp": 1.0,
        "general.sampling.top_k": 40,
        "general.sampling.top_p": 0.95,
    }
    # A name no shipped profile pattern matches → fallback profile.
    model = _fake_model_md(tmp_path, "SomeUnprofiledFinetune-13B-Q6_K", 13.0, md)
    profile = match_profile(model.name, profiles)
    assert not profile.patterns, "expected the fallback profile for this name"
    cfg = compute_config(model, _fake_system(), profile)
    assert cfg.sampling["temperature"] == 1.0
    assert cfg.sampling["top_p"] == 0.95


def test_matched_profile_wins_over_gguf_sampling(tmp_path) -> None:
    """When a real family profile matches, its explicit sampler values
    take precedence over the GGUF recommendation (Basti hand-tuned these)."""
    profiles = load_profiles(SETTINGS_DIR)
    md = {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": 40,
        "qwen35moe.expert_count": 256,
        # Deliberately absurd GGUF value that must be overridden.
        "general.sampling.top_k": 99,
    }
    model = _fake_model_md(tmp_path, "Qwen3.6-35B-A3B-UD-Q6_K", 30.0, md)
    profile = match_profile(model.name, profiles)
    assert profile.patterns, "expected a matched (non-fallback) profile"
    cfg = compute_config(model, _fake_system(), profile)
    # qwen3_5-3_6.yaml sets top_k 20 for both chat and coding.
    assert cfg.sampling["top_k"] == 20


# ---------------------------------------------------------------------------
# MoE multi-GPU capacity-fill (don't strand VRAM on the secondary card)


def test_moe_spread_fills_both_gpus_by_capacity(tmp_path) -> None:
    """An MoE too large for the primary alone must spread proportionally to
    each card's usable capacity, NOT priority-weighted — so the secondary
    GPU is filled to roughly the same utilisation as the primary. Stranding
    several GB on the 9070 XT (the old priority-weighted behaviour) forced
    extra expert layers onto the CPU and slowed the model down."""
    profiles = load_profiles(SETTINGS_DIR)
    md = {
        "general.architecture": "qwen3moe",
        "qwen3moe.block_count": 48,
        "qwen3moe.expert_count": 128,
        "qwen3moe.attention.head_count": 40,
        "qwen3moe.attention.head_count_kv": 8,
        "qwen3moe.attention.key_length": 128,
        "qwen3moe.attention.value_length": 128,
        "qwen3moe.embedding_length": 5120,
    }
    model = _fake_model_md(tmp_path, "BigMoE-40B-A10B-Q4", 40.0, md)
    profile = match_profile(model.name, profiles)
    prio = {
        "AMD Radeon AI PRO R9700": 2,
        "AMD Radeon RX 9070 XT": 1,
    }
    cfg = compute_config(
        model, _fake_dual_gpu_system_with_vk_order(), profile, gpu_priorities=prio
    )
    assert cfg.tensor_split is not None, "expected a spread, got pin/None"
    parts = [float(x) for x in cfg.tensor_split.split(",")]
    assert len(parts) == 2
    # Vulkan order: device 0 = 9070 XT (cap ~13 GB), device 1 = R9700 (cap ~30 GB).
    xt_share, r97_share = parts[0], parts[1]
    # Capacity-proportional shares: 13/43 ≈ 0.30 and 30/43 ≈ 0.70.
    # The 9070 XT must get a MEANINGFUL share — well above the ~0.20 the
    # old priority-weighting produced.
    assert xt_share > 0.25, (
        f"9070 XT under-filled (capacity-fill regressed): share={xt_share:.3f}"
    )
    # Sanity: the larger card still carries the larger share.
    assert r97_share > xt_share


def test_dense_spread_stays_priority_weighted(tmp_path) -> None:
    """Dense models keep the priority-weighted split: the high-priority AI
    card carries the bulk so the gaming GPU stays as free as possible. This
    must NOT change to capacity-fill (that's MoE-only)."""
    profiles = load_profiles(SETTINGS_DIR)
    # Dense (no expert_count) 40 GB model — too big for the 32 GB primary.
    md = {
        "general.architecture": "llama",
        "llama.block_count": 80,
        "llama.attention.head_count": 64,
        "llama.attention.head_count_kv": 8,
        "llama.embedding_length": 8192,
    }
    model = _fake_model_md(tmp_path, "Dense-70B-Q4", 40.0, md)
    profile = match_profile(model.name, profiles)
    prio = {
        "AMD Radeon AI PRO R9700": 2,
        "AMD Radeon RX 9070 XT": 1,
    }
    cfg = compute_config(
        model, _fake_dual_gpu_system_with_vk_order(), profile, gpu_priorities=prio
    )
    assert cfg.tensor_split is not None
    parts = [float(x) for x in cfg.tensor_split.split(",")]
    xt_share, r97_share = parts[0], parts[1]
    # Priority×VRAM (2×32 vs 1×16) → R9700 ≥ ~0.75; 9070 XT ≤ ~0.25.
    assert r97_share > 0.70, f"dense split not priority-weighted: r97={r97_share:.3f}"
    assert xt_share < 0.30, f"dense split puts too much on the 9070 XT: {xt_share:.3f}"
    