import os
from pathlib import Path
import shutil
import subprocess
import textwrap
import time

import pytest


ROOT = Path(__file__).resolve().parent
COMMON_RECIPE = ROOT / "building llama.cpp" / "windows_llama_build_common.ps1"


def _common_recipe() -> str:
    return COMMON_RECIPE.read_text(encoding="utf-8")


def _pwsh() -> str:
    executable = shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell 7 is not installed on this runner")
    return executable


def _run_pwsh(
    script: str, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    process_env["AUTOTUNER_TEST_COMMON_RECIPE"] = str(COMMON_RECIPE)
    if env:
        process_env.update(env)
    return subprocess.run(
        [_pwsh(), "-NoProfile", "-Command", textwrap.dedent(script)],
        cwd=ROOT,
        env=process_env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )


def test_windows_hip_recipe_blocks_peer_copy_and_checks_decoded_text() -> None:
    common = _common_recipe()

    assert '"-DGGML_CUDA_NO_PEER_COPY=ON"' in common
    assert "GGML_CUDA_NO_PEER_COPY:BOOL=ON" in common
    assert "Test-HipMultiGpuSemanticOutput" in common
    assert "Resolve-HipSemanticValidationModel" in common
    assert "AUTOTUNER_HIP_VERIFY_MODEL" in common
    assert '"HIP_VISIBLE_DEVICES"] = "0,1"' in common
    assert '"--tensor-split", "0.667,0.333"' in common
    assert 'expected = "HIP MULTI GPU OK"' in common
    assert "assistantContent.Trim() -ne $expected" in common
    assert "Stop-ProcessTreeBounded" in common
    assert "Get-TaskTextBounded" in common


def test_windows_hip_recipe_keeps_actionable_warnings_and_consumed_flags() -> None:
    common = _common_recipe()

    # llama.cpp's Windows HIP path uses ROCm clang as C/C++ compilers and
    # GGML_HIP as the backend switch. CMAKE_HIP_COMPILER is ignored upstream
    # (CMake caches it as UNINITIALIZED), so do not pass the noisy dead option.
    assert '"-DCMAKE_C_COMPILER=$clang"' in common
    assert '"-DCMAKE_CXX_COMPILER=$clangxx"' in common
    assert "-DCMAKE_HIP_COMPILER=" not in common

    for contract in (
        '"-DGGML_HIP=ON"',
        '"-DGGML_VULKAN=OFF"',
        '"-DGPU_TARGETS=gfx1201"',
        '"-DGGML_HIP_GRAPHS=ON"',
        '"-DGGML_HIP_NO_VMM=ON"',
        '"-DGGML_HIP_RCCL=OFF"',
        '"-DGGML_CUDA_NO_PEER_COPY=ON"',
        '"-DGGML_CUDA_FA=ON"',
        '"-DGGML_CUDA_FA_ALL_QUANTS=ON"',
    ):
        assert contract in common

    # Keep upstream/compiler diagnostics visible. Third-party code must not
    # become release-blocking through a blanket Werror policy either.
    assert "LLAMA_ALL_WARNINGS=OFF" not in common
    assert "GGML_ALL_WARNINGS=OFF" not in common
    assert "-Werror" not in common
    assert "/WX" not in common


def test_common_recipe_parses_in_powershell() -> None:
    result = _run_pwsh(
        """
        $null = [scriptblock]::Create(
            (Get-Content -LiteralPath $env:AUTOTUNER_TEST_COMMON_RECIPE -Raw)
        )
        "PowerShell parse OK"
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PowerShell parse OK" in result.stdout


def test_process_cleanup_helper_terminates_a_nonterminating_child() -> None:
    started = time.monotonic()
    result = _run_pwsh(
        """
        . $env:AUTOTUNER_TEST_COMMON_RECIPE
        $startInfo = [Diagnostics.ProcessStartInfo]::new()
        $startInfo.FileName = (Get-Process -Id $PID).Path
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        foreach ($argument in @("-NoProfile", "-Command", "Start-Sleep -Seconds 300")) {
            [void]$startInfo.ArgumentList.Add($argument)
        }
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $startInfo
        if (-not $process.Start()) { throw "child did not start" }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        try {
            Stop-ProcessTreeBounded -Process $process -TimeoutMilliseconds 5000
            [void](Get-TaskTextBounded -Task $stdoutTask -TimeoutMilliseconds 5000)
            [void](Get-TaskTextBounded -Task $stderrTask -TimeoutMilliseconds 5000)
            if (-not $process.HasExited) { throw "child is still running" }
        } finally {
            if (-not $process.HasExited) { $process.Kill($true) }
            $process.Dispose()
        }
        "bounded cleanup OK"
        """
    )
    elapsed = time.monotonic() - started

    assert result.returncode == 0, result.stdout + result.stderr
    assert "bounded cleanup OK" in result.stdout
    assert elapsed < 20


def test_failed_clone_cleans_every_pre_promotion_staging_tree(tmp_path: Path) -> None:
    result = _run_pwsh(
        """
        . $env:AUTOTUNER_TEST_COMMON_RECIPE
        function global:git {
            $rows = @($args)
            if ($rows -contains "ls-remote") {
                $global:LASTEXITCODE = 0
                return "1111111111111111111111111111111111111111`trefs/tags/v0.3.0"
            }
            if ($rows -contains "clone") {
                $destination = [string]$rows[-1]
                New-Item -ItemType Directory -Force -Path $destination | Out-Null
                $global:LASTEXITCODE = 23
                return
            }
            throw "unexpected mocked git call: $($rows -join ' ')"
        }

        $workspace = $env:AUTOTUNER_TEST_WORKSPACE
        $cases = @(
            { Invoke-LlamaPrereleaseBuild -Backend HIP -Tag b1234 -Workspace $workspace },
            { Invoke-LlamaStableBuild -Backend HIP -Tag v0.3.0 -Workspace $workspace },
            {
                Invoke-LlamaPinnedForkBuild `
                    -Backend HIP `
                    -Name "synthetic fork" `
                    -RemoteUrl "https://invalid.example/repo.git" `
                    -ExpectedCommit "2222222222222222222222222222222222222222" `
                    -FolderPrefix "synthetic_" `
                    -Workspace $workspace
            }
        )
        foreach ($case in $cases) {
            try {
                & $case
                throw "synthetic clone unexpectedly succeeded"
            } catch {
                if ($_.Exception.Message -notlike "*failed with exit code 23*") {
                    throw
                }
            }
            $leftovers = @(Get-ChildItem $workspace -Directory -Filter "_tmp_*")
            if ($leftovers.Count -ne 0) {
                throw "staging tree leaked: $($leftovers.FullName -join ', ')"
            }
        }
        "staging cleanup OK"
        """,
        env={"AUTOTUNER_TEST_WORKSPACE": str(tmp_path)},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "staging cleanup OK" in result.stdout


@pytest.mark.parametrize("modern", [False, True])
def test_flash_attention_options_follow_checked_out_source(tmp_path, modern) -> None:
    ggml = tmp_path / "ggml"
    ggml.mkdir()
    (ggml / "CMakeLists.txt").write_text(
        'set (GGML_CUDA_FA_QUANTS "q8_0-q8_0" CACHE STRING "quants")'
        if modern
        else 'option(GGML_CUDA_FA_ALL_QUANTS "all" OFF)',
        encoding="utf-8",
    )
    result = _run_pwsh(
        """
        . $env:AUTOTUNER_TEST_COMMON_RECIPE
        Get-LlamaFlashAttentionArgs -Repo $env:AUTOTUNER_TEST_WORKSPACE
        """,
        env={"AUTOTUNER_TEST_WORKSPACE": str(tmp_path)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = result.stdout.splitlines()
    assert args == (
        ["-UGGML_CUDA_FA_ALL_QUANTS", "-DGGML_CUDA_FA_QUANTS=all"]
        if modern
        else ["-DGGML_CUDA_FA_ALL_QUANTS=ON"]
    )


def test_stable_recipe_accepts_prefixed_and_unprefixed_exact_tags() -> None:
    common = _common_recipe()

    assert "$remoteTag = $Tag" in common
    assert '$alternateTag = if ($Tag.StartsWith("v"))' in common
    assert "git clone --branch $remoteTag --single-branch" in common
    assert "Checkout is not the requested exact stable tag $remoteTag" in common


def test_all_windows_powershell_recipes_parse_cleanly() -> None:
    recipes_dir = ROOT / "building llama.cpp"
    ps1_files = sorted(recipes_dir.glob("*.ps1"))
    assert len(ps1_files) >= 15
    result = _run_pwsh(
        """
        $dir = Split-Path -Parent $env:AUTOTUNER_TEST_COMMON_RECIPE
        $files = Get-ChildItem -LiteralPath $dir -Filter "*.ps1" | Sort-Object Name
        foreach ($f in $files) {
            $null = [scriptblock]::Create((Get-Content -LiteralPath $f.FullName -Raw))
        }
        "Parsed $($files.Count) scripts OK"
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Parsed {len(ps1_files)} scripts OK" in result.stdout


def test_ui_build_detects_b12451_llama_chat_cmake_layout(tmp_path: Path) -> None:
    (tmp_path / "ui" / "apps" / "chat").mkdir(parents=True)
    (tmp_path / "ui" / "apps" / "chat" / "package.json").write_text(
        "{}", encoding="utf-8"
    )
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts" / "ui-assets.cmake").write_text("# cmake", encoding="utf-8")
    result = _run_pwsh(
        """
        . $env:AUTOTUNER_TEST_COMMON_RECIPE
        function global:Assert-CommandAvailable { param([string]$Name) }
        $out = Invoke-LlamaUiBuild -Repo $env:AUTOTUNER_TEST_WORKSPACE -Enabled $true
        "RESULT=$out"
        """,
        env={"AUTOTUNER_TEST_WORKSPACE": str(tmp_path)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT=OFF" in result.stdout
    assert "Letting CMake build Llama Chat" in result.stdout


def test_select_hip_msvc_toolset_probes_newest_first(tmp_path: Path) -> None:
    msvc_root = tmp_path / "VC" / "Tools" / "MSVC"
    for ver in ("14.44.35207", "14.50.35717", "14.53.36930"):
        (msvc_root / ver / "include").mkdir(parents=True)
    result = _run_pwsh(
        """
        . $env:AUTOTUNER_TEST_COMMON_RECIPE
        $vs = $env:AUTOTUNER_TEST_WORKSPACE
        $env:VCToolsInstallDir = "$vs\\VC\\Tools\\MSVC\\14.53.36930\\"
        $env:INCLUDE = "$vs\\VC\\Tools\\MSVC\\14.53.36930\\include;C:\\sdk"
        function global:mock_clangxx {
            if ($env:INCLUDE -like "*14.50.35717*") {
                $global:LASTEXITCODE = 0
            } else {
                $global:LASTEXITCODE = 1
            }
        }
        $picked = Select-HipMsvcToolset -VsPath $vs -ClangXX "mock_clangxx"
        "PICKED=$picked"
        "RESTORED_VC=$env:VCToolsInstallDir"
        """,
        env={"AUTOTUNER_TEST_WORKSPACE": str(tmp_path)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "14.50.35717" in result.stdout
    assert "14.53.36930\\" in result.stdout
