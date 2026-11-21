Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Shared Windows build implementation for the backend-qualified recipes in this
# directory. The public entry points are small, backend-qualified *.ps1 files;
# run them directly with `pwsh -File <recipe>.ps1`.

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )

    # Windows PowerShell 5.1 promotes native stderr records to terminating
    # NativeCommandError exceptions when ErrorActionPreference is Stop. Tools
    # such as git and cmake legitimately write progress to stderr even on exit
    # code 0, so suppress that promotion locally and trust the native exit code.
    $previousErrorActionPreference = $ErrorActionPreference
    $hasNativeErrorPreference = [bool](
        Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue
    )
    if ($hasNativeErrorPreference) {
        $previousNativeErrorPreference = $PSNativeCommandUseErrorActionPreference
    }
    try {
        $ErrorActionPreference = "Continue"
        if ($hasNativeErrorPreference) {
            $PSNativeCommandUseErrorActionPreference = $false
        }
        & $Command 2>&1 | ForEach-Object {
            $line = if ($_ -is [Management.Automation.ErrorRecord]) {
                [string]$_.Exception.Message
            } else {
                [string]$_
            }
            Write-Host $line
        }
        $exitCode = $LASTEXITCODE
    } finally {
        if ($hasNativeErrorPreference) {
            $PSNativeCommandUseErrorActionPreference = $previousNativeErrorPreference
        }
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "$Description failed with exit code $exitCode"
    }
}

function Get-NativeText {
    param(
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $hasNativeErrorPreference = [bool](
        Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue
    )
    if ($hasNativeErrorPreference) {
        $previousNativeErrorPreference = $PSNativeCommandUseErrorActionPreference
    }
    try {
        $ErrorActionPreference = "Continue"
        if ($hasNativeErrorPreference) {
            $PSNativeCommandUseErrorActionPreference = $false
        }
        $output = @(& $Command 2>&1)
        $exitCode = $LASTEXITCODE
        $text = ($output | ForEach-Object {
            if ($_ -is [Management.Automation.ErrorRecord]) {
                [string]$_.Exception.Message
            } else {
                [string]$_
            }
        } | Out-String).Trim()
    } finally {
        if ($hasNativeErrorPreference) {
            $PSNativeCommandUseErrorActionPreference = $previousNativeErrorPreference
        }
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "$Description failed with exit code $exitCode`n$text"
    }
    return $text
}

function Assert-CommandAvailable {
    param([Parameter(Mandatory = $true)][string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found in PATH"
    }
}

function Get-Sha256FileHash {
    param([Parameter(Mandatory = $true)][string]$Path)

    # Use .NET directly: a PowerShell 7 module path can shadow the Windows
    # PowerShell Utility module and make Get-FileHash unavailable in 5.1.
    $stream = [IO.File]::OpenRead($Path)
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha256.ComputeHash($stream))).Replace('-', '')
    } finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

function Select-HipMsvcToolset {
    param(
        [Parameter(Mandatory = $true)][string]$VsPath,
        [Parameter(Mandatory = $true)][string]$ClangXX
    )

    # The MSVC STL hard-fails on unsupported clang (STL1000 in yvals_core.h):
    # VS 2026's newest toolsets (14.52+) require Clang 22, while ROCm 7.2
    # ships Clang 21. Probe the installed toolsets newest-first with the
    # actual ROCm clang and return the newest one its STL still accepts.
    $msvcRoot = Join-Path $VsPath "VC\Tools\MSVC"
    if (-not (Test-Path $msvcRoot -PathType Container)) {
        throw "No MSVC toolset directories found under $msvcRoot"
    }
    $toolsets = @(
        Get-ChildItem $msvcRoot -Directory |
            Where-Object { $_.Name -match '^\d+(\.\d+)+$' } |
            Sort-Object { [version]$_.Name } -Descending
    )
    if ($toolsets.Count -eq 0) {
        throw "No versioned MSVC toolset directories found under $msvcRoot"
    }

    $probeFile = [IO.Path]::GetTempFileName()
    $currentName = Split-Path ($env:VCToolsInstallDir.TrimEnd('\', '/')) -Leaf
    $savedVcDir = $env:VCToolsInstallDir
    $savedInclude = if ($null -ne $env:INCLUDE) { $env:INCLUDE } else { "" }
    $oldTokenPattern = [regex]::Escape("MSVC\$currentName\")
    try {
        [IO.File]::WriteAllText($probeFile, "#include <array>`nint main(){return 0;}`n")
        foreach ($toolset in $toolsets) {
            $env:VCToolsInstallDir = $toolset.FullName
            $env:INCLUDE = [regex]::Replace(
                $savedInclude,
                $oldTokenPattern,
                "MSVC\$($toolset.Name)\",
                [Text.RegularExpressions.RegexOptions]::IgnoreCase
            )
            $previousErrorActionPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                & $ClangXX -fsyntax-only -x c++ $probeFile 2>&1 | Out-Null
                $exitCode = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $previousErrorActionPreference
            }
            if ($exitCode -eq 0) {
                return $toolset.FullName
            }
        }
    } finally {
        $env:VCToolsInstallDir = $savedVcDir
        $env:INCLUDE = $savedInclude
        Remove-Item $probeFile -Force -ErrorAction SilentlyContinue
    }
    throw "No installed MSVC toolset accepts this ROCm clang (MSVC STL error STL1000). Install an older C++ toolset (e.g. 14.50.x or 14.44.x) via the Visual Studio installer."
}

function Import-VisualStudioBuildEnvironment {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere -PathType Leaf)) {
        throw "vswhere.exe was not found: $vswhere"
    }
    $instanceJson = Get-NativeText "Visual Studio discovery" {
        & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json
    }
    $instances = $instanceJson | ConvertFrom-Json
    $instance = @($instances)[0]
    if (-not $instance) {
        throw "A Visual Studio C++ x64 toolchain was not found"
    }
    $vsPath = [string]$instance.installationPath
    $vsVersion = [string]$instance.installationVersion
    if (-not $vsPath -or $vsVersion -notmatch '^(\d+)\.') {
        throw "Visual Studio discovery returned an incomplete installation record"
    }
    $vsGenerator = switch ([int]$Matches[1]) {
        18 { "Visual Studio 18 2026" }
        17 { "Visual Studio 17 2022" }
        16 { "Visual Studio 16 2019" }
        15 { "Visual Studio 15 2017" }
        default { throw "Unsupported Visual Studio version: $vsVersion" }
    }
    $vsDevCmd = Join-Path $vsPath "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $vsDevCmd -PathType Leaf)) {
        throw "VsDevCmd.bat was not found: $vsDevCmd"
    }

    # Ninja + ROCm clang still link against the Microsoft x64 ABI. Import the
    # exact VS environment instead of requiring a special shell shortcut.
    $commandLine = "`"$vsDevCmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
    $envRows = & $env:ComSpec /d /s /c $commandLine
    if ($LASTEXITCODE -ne 0) {
        throw "Visual Studio x64 build environment initialization failed"
    }
    foreach ($row in $envRows) {
        if ($row -match '^([^=]+)=(.*)$') {
            Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
        }
    }
    return [PSCustomObject]@{
        Path = $vsPath
        Version = $vsVersion
        Generator = $vsGenerator
    }
}

function Resolve-NinjaPath {
    param([Parameter(Mandatory = $true)][string]$VisualStudioPath)

    $found = Get-Command ninja -ErrorAction SilentlyContinue
    if ($found) {
        return $found.Source
    }
    $candidate = Join-Path $VisualStudioPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
    if (-not (Test-Path $candidate -PathType Leaf)) {
        throw "ninja.exe was not found in PATH or the Visual Studio CMake tools"
    }
    $env:PATH = "$(Split-Path -Parent $candidate);$env:PATH"
    return $candidate
}

function Get-HipCompatibilityResourceDir {
    param(
        [Parameter(Mandatory = $true)][string]$Clang,
        [Parameter(Mandatory = $true)][string]$RocmPath,
        [Parameter(Mandatory = $true)][string]$Workspace
    )

    $resourceDir = Get-NativeText "ROCm clang resource-dir probe" {
        & $Clang -print-resource-dir
    }
    $wrapper = Join-Path $resourceDir "include\__clang_hip_runtime_wrapper.h"
    if (-not (Test-Path $wrapper -PathType Leaf)) {
        throw "ROCm clang HIP wrapper was not found: $wrapper"
    }
    $forwardInclude = "#include <__clang_cuda_math_forward_declares.h>"
    $cmathInclude = "#include <cmath>"
    $sourceText = Get-Content $wrapper -Raw
    $forwardIndex = $sourceText.IndexOf($forwardInclude, [StringComparison]::Ordinal)
    $cmathIndex = $sourceText.IndexOf($cmathInclude, [StringComparison]::Ordinal)
    if ($forwardIndex -ge 0 -and $forwardIndex -lt $cmathIndex) {
        return ""
    }

    # HIP SDK 7.2 still lacks LLVM PR #201563. With MSVC 14.51, <cmath>
    # creates implicit host+device constexpr overloads before clang's device
    # declarations, so every HIP TU fails. Keep Program Files pristine: copy
    # clang's resource tree into the workspace and apply the exact upstream
    # include reorder there, then pass that private tree via -resource-dir.
    $rocmVersion = Split-Path $RocmPath -Leaf
    $clangVersion = Split-Path $resourceDir -Leaf
    $patchedDir = Join-Path $Workspace "toolchains\rocm-${rocmVersion}-clang-${clangVersion}-llvm-pr201563"
    $patchedWrapper = Join-Path $patchedDir "include\__clang_hip_runtime_wrapper.h"
    $sourceHash = Get-Sha256FileHash $wrapper
    $marker = Join-Path $patchedDir ".autotuner-source-sha256"
    $reuse = $false
    if ((Test-Path $patchedWrapper -PathType Leaf) -and (Test-Path $marker -PathType Leaf)) {
        $markerHash = (Get-Content $marker -Raw).Trim()
        $candidateText = Get-Content $patchedWrapper -Raw
        $candidateForward = $candidateText.IndexOf($forwardInclude, [StringComparison]::Ordinal)
        $candidateCmath = $candidateText.IndexOf($cmathInclude, [StringComparison]::Ordinal)
        $reuse = (
            $markerHash -eq $sourceHash -and
            $candidateForward -ge 0 -and
            $candidateForward -lt $candidateCmath
        )
    }
    if (-not $reuse) {
        if (Test-Path $patchedDir) {
            Remove-Item $patchedDir -Recurse -Force
        }
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $patchedDir) | Out-Null
        Copy-Item -LiteralPath $resourceDir -Destination $patchedDir -Recurse -Force

        $newline = if ($sourceText.Contains("`r`n")) { "`r`n" } else { "`n" }
        $withoutLateInclude = [regex]::Replace(
            $sourceText,
            '(?m)^#include <__clang_cuda_math_forward_declares\.h>\r?\n',
            ''
        )
        $patchedText = [regex]::Replace(
            $withoutLateInclude,
            '(?m)^(#if !defined\(__HIPCC_RTC__\)\r?\n)',
            ('$1' + $forwardInclude + $newline),
            1
        )
        [IO.File]::WriteAllText(
            $patchedWrapper,
            $patchedText,
            [Text.UTF8Encoding]::new($false)
        )
        Set-Content -LiteralPath $marker -Value $sourceHash -NoNewline -Encoding ascii
    }

    Write-Host "==> Applied workspace-local LLVM PR #201563 HIP cmath fix: $patchedDir"
    return $patchedDir.Replace('\', '/')
}

function Initialize-LlamaBuildEnvironment {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("Vulkan", "HIP")][string]$Backend,
        [Parameter(Mandatory = $true)][string]$Workspace,
        [string]$RocmPath = ""
    )

    if ($env:OS -ne "Windows_NT") {
        throw "These recipes are for native Windows builds"
    }
    foreach ($tool in ("git", "cmake")) {
        Assert-CommandAvailable $tool
    }
    if (-not (Test-Path $Workspace -PathType Container)) {
        New-Item -ItemType Directory -Force -Path $Workspace | Out-Null
    }

    $vs = Import-VisualStudioBuildEnvironment
    if ($Backend -eq "Vulkan") {
        Assert-CommandAvailable "glslc"
        return [PSCustomObject]@{
            Backend = "Vulkan"
            VisualStudioPath = $vs.Path
            VisualStudioGenerator = $vs.Generator
            RocmPath = $null
            NinjaPath = $null
        }
    }

    if (-not $RocmPath) {
        $RocmPath = $env:ROCM_PATH
    }
    if (-not $RocmPath -or -not (Test-Path $RocmPath -PathType Container)) {
        throw "A pinned ROCm HIP SDK path is required (for example -RocmPath 'C:\Program Files\AMD\ROCm\7.2')"
    }
    $RocmPath = (Resolve-Path $RocmPath).Path
    $bin = Join-Path $RocmPath "bin"
    $clang = Join-Path $bin "clang.exe"
    $clangxx = Join-Path $bin "clang++.exe"
    $hipcc = Join-Path $bin "hipcc.exe"
    $hipInfo = Join-Path $bin "hipInfo.exe"
    foreach ($tool in ($clang, $clangxx, $hipcc, $hipInfo)) {
        if (-not (Test-Path $tool -PathType Leaf)) {
            throw "Incomplete HIP SDK: required tool not found: $tool"
        }
    }

    $env:ROCM_PATH = $RocmPath
    $env:HIP_PATH = "$RocmPath\"
    $env:PATH = "$bin;$env:PATH"
    # VsDevCmd (18) defaults to the newest installed MSVC toolset and cannot
    # be told to use another one, so swap the toolset paths in the imported
    # environment when the default's STL rejects the ROCm clang (STL1000).
    $hipToolset = (Select-HipMsvcToolset -VsPath $vs.Path -ClangXX $clangxx).TrimEnd('\', '/')
    $currentToolsetDir = (Resolve-Path $env:VCToolsInstallDir).Path.TrimEnd('\', '/')
    if (-not $hipToolset.Equals($currentToolsetDir, [StringComparison]::OrdinalIgnoreCase)) {
        $tokenOld = [regex]::Escape("MSVC\$(Split-Path $currentToolsetDir -Leaf)\")
        $tokenNew = "MSVC\$(Split-Path $hipToolset -Leaf)\"
        foreach ($var in "VCToolsInstallDir", "INCLUDE", "LIB", "LIBPATH", "PATH") {
            if (Test-Path "Env:$var") {
                $replaced = [regex]::Replace(
                    [string](Get-Item "Env:$var").Value,
                    $tokenOld,
                    $tokenNew,
                    [Text.RegularExpressions.RegexOptions]::IgnoreCase
                )
                Set-Item "Env:$var" $replaced
            }
        }
        $env:VCToolsVersion = Split-Path $hipToolset -Leaf
        Write-Host "==> MSVC toolset for ROCm clang: $env:VCToolsVersion (newest STL this clang accepts)"
    }
    $ninja = Resolve-NinjaPath -VisualStudioPath $vs.Path
    $compatResourceDir = Get-HipCompatibilityResourceDir `
        -Clang $clang -RocmPath $RocmPath -Workspace $Workspace

    $hipVersion = Get-NativeText "hipcc version probe" { & $hipcc --version }
    $deviceInfo = Get-NativeText "hipInfo device probe" { & $hipInfo }
    foreach ($needle in ("AMD Radeon AI PRO R9700", "AMD Radeon RX 9070 XT", "gfx1201")) {
        if ($deviceInfo -notmatch [regex]::Escape($needle)) {
            throw "HIP SDK at '$RocmPath' did not expose required device/target '$needle'"
        }
    }
    Write-Host "==> HIP SDK: $RocmPath"
    Write-Host ($hipVersion -split "`r?`n" | Select-Object -First 1)
    Write-Host "==> HIP target verified: gfx1201 (R9700 + RX 9070 XT)"

    return [PSCustomObject]@{
        Backend = "HIP"
        VisualStudioPath = $vs.Path
        VisualStudioGenerator = $vs.Generator
        RocmPath = $RocmPath
        NinjaPath = $ninja
        Clang = $clang
        ClangXX = $clangxx
        CompatibilityResourceDir = $compatResourceDir
    }
}

function Ensure-SpirvHeaders {
    param(
        [Parameter(Mandatory = $true)][string]$Workspace,
        [Parameter(Mandatory = $true)][string]$Generator,
        [ValidateRange(1, 256)][int]$Parallel = 20
    )

    $spirv = Join-Path $Workspace "SPIRV-Headers"
    if (-not (Test-Path "$spirv\.git" -PathType Container)) {
        if (Test-Path $spirv) {
            throw "SPIRV-Headers exists but is not a Git checkout: $spirv"
        }
        Invoke-NativeChecked "SPIRV-Headers clone" {
            git clone https://github.com/KhronosGroup/SPIRV-Headers.git $spirv
        }
    } else {
        $dirty = (& git -C $spirv status --porcelain --untracked-files=no | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) {
            throw "Could not inspect SPIRV-Headers checkout"
        }
        if ($dirty) {
            throw "SPIRV-Headers has tracked local changes; refusing to update: $spirv"
        }
        Invoke-NativeChecked "SPIRV-Headers update" { git -C $spirv pull --ff-only }
    }

    $install = (Join-Path $spirv "install").Replace('\', '/')
    Invoke-NativeChecked "SPIRV-Headers configure" {
        cmake -S $spirv -B "$spirv\build" -G $Generator -A x64 "-DCMAKE_INSTALL_PREFIX=$install"
    }
    Invoke-NativeChecked "SPIRV-Headers build" {
        cmake --build "$spirv\build" --config Release --parallel $Parallel
    }
    Invoke-NativeChecked "SPIRV-Headers install" {
        cmake --install "$spirv\build" --config Release
    }
    return $install
}

function Get-LatestLlamaPrereleaseTag {
    $rows = git ls-remote --refs --tags https://github.com/ggml-org/llama.cpp.git "refs/tags/b*" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not query llama.cpp pre-release tags"
    }
    $tags = @(
        foreach ($row in $rows) {
            if ($row -match 'refs/tags/(b(\d+))$') {
                [PSCustomObject]@{ Tag = $Matches[1]; Number = [int]$Matches[2] }
            }
        }
    )
    if ($tags.Count -eq 0) {
        throw "No official llama.cpp bNNNN tag was found"
    }
    return ($tags | Sort-Object Number -Descending | Select-Object -First 1).Tag
}

function Get-LatestStableSemanticTag {
    $headers = @{
        "Accept" = "application/vnd.github+json"
        "User-Agent" = "AutoTuner-llama-build"
        "X-GitHub-Api-Version" = "2022-11-28"
    }
    try {
        $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=100" -Headers $headers
        $stable = @(
            $releases |
                Where-Object {
                    -not $_.draft -and -not $_.prerelease -and
                    $_.tag_name -match '^v?\d+\.\d+\.\d+$'
                } |
                ForEach-Object {
                    [PSCustomObject]@{
                        Tag = [string]$_.tag_name
                        Version = [version](([string]$_.tag_name) -replace '^v', '')
                    }
                } |
                Sort-Object Version -Descending
        )
        if ($stable.Count -gt 0) {
            return $stable[0].Tag
        }
    } catch {
        Write-Warning "GitHub release API lookup failed: $($_.Exception.Message)"
    }

    $rows = git ls-remote --refs --tags https://github.com/ggml-org/llama.cpp.git 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not query llama.cpp stable tags"
    }
    $tags = @(
        foreach ($row in $rows) {
            if ($row -match 'refs/tags/(v?\d+\.\d+\.\d+)$') {
                [PSCustomObject]@{
                    Tag = $Matches[1]
                    Version = [version]($Matches[1] -replace '^v', '')
                }
            }
        }
    )
    if ($tags.Count -eq 0) {
        throw "No official stable llama.cpp X.Y.Z tag was found"
    }
    return ($tags | Sort-Object Version -Descending | Select-Object -First 1).Tag
}

function Get-LlamaBuildCount {
    param(
        [Parameter(Mandatory = $true)][string]$Repo,
        [string]$Commit = "HEAD"
    )

    $text = (& git -C $Repo rev-list --count $Commit | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $text -notmatch '^\d+$') {
        throw "Could not determine the llama.cpp compatibility build for $Commit"
    }
    $number = [int]$text
    if ($number -lt 1000) {
        throw "History is incomplete; refusing suspicious build number $number"
    }
    return $number
}

function Get-ExactRemotePrereleaseTag {
    param(
        [Parameter(Mandatory = $true)][string]$Repo,
        [string]$Commit = "HEAD",
        [string]$Remote = "origin"
    )

    $sha = (& git -C $Repo rev-parse $Commit 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $sha) {
        throw "Could not resolve commit '$Commit'"
    }
    $rows = git -C $Repo ls-remote --refs --tags $Remote "refs/tags/b*" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not query $Remote b-tags"
    }
    $exactTags = @(
        foreach ($row in $rows) {
            if ($row -match "^$([regex]::Escape($sha))\s+refs/tags/(b\d+)$") {
                $Matches[1]
            }
        }
    )
    if ($exactTags.Count -eq 0) {
        return $null
    }
    return ($exactTags | Sort-Object { [int]($_ -replace '^b', '') } -Descending | Select-Object -First 1)
}

function Get-LlamaRuntimeSemanticVersion {
    param([Parameter(Mandatory = $true)][string]$Repo)

    $cmakeText = Get-Content (Join-Path $Repo "CMakeLists.txt") -Raw
    $major = [regex]::Match($cmakeText, '(?m)^set\(LLAMA_VERSION_MAJOR\s+(\d+)\)')
    $minor = [regex]::Match($cmakeText, '(?m)^set\(LLAMA_VERSION_MINOR\s+(\d+)\)')
    $patch = [regex]::Match($cmakeText, '(?m)^set\(LLAMA_VERSION_PATCH\s+(\d+)\)')
    if (-not ($major.Success -and $minor.Success -and $patch.Success)) {
        throw "Could not determine llama.cpp semantic version from CMakeLists.txt"
    }
    return "$($major.Groups[1].Value).$($minor.Groups[1].Value).$($patch.Groups[1].Value)"
}

function Invoke-LlamaUiBuild {
    param(
        [Parameter(Mandatory = $true)][string]$Repo,
        [bool]$Enabled = $true
    )

    if (-not $Enabled) {
        return "OFF"
    }
    # Since b12451 (PR #32032), Llama Chat lives in ui/apps/chat alongside
    # ui/lib/{ui,voice}, and scripts/ui-assets.cmake stages and builds it out
    # of tree when LLAMA_BUILD_UI=ON and LLAMA_USE_PREBUILT_UI=OFF.
    $cmakeUiSource = Join-Path $Repo "ui\apps\chat\package.json"
    $cmakeUiScript = Join-Path $Repo "scripts\ui-assets.cmake"
    if ((Test-Path $cmakeUiSource -PathType Leaf) -and (Test-Path $cmakeUiScript -PathType Leaf)) {
        Assert-CommandAvailable "npm"
        Write-Host "==> Letting CMake build Llama Chat from ui/apps/chat"
        return "OFF"
    }

    $uiSource = $null
    foreach ($relative in ("tools/ui", "tools/server/webui")) {
        $candidate = Join-Path $Repo $relative
        if (Test-Path (Join-Path $candidate "package.json") -PathType Leaf) {
            $uiSource = $candidate
            break
        }
    }
    if (-not $uiSource) {
        Write-Host "==> No UI source in $(Split-Path -Leaf $Repo); use the prebuilt UI"
        return "ON"
    }

    Assert-CommandAvailable "npm"
    Write-Host "==> Building Web UI from $uiSource"
    Push-Location $uiSource
    try {
        if (Test-Path "package-lock.json" -PathType Leaf) {
            Invoke-NativeChecked "Web UI dependency install" { npm ci }
        } else {
            Invoke-NativeChecked "Web UI dependency install" { npm install }
        }
        Invoke-NativeChecked "Web UI build" { npm run build }
    } finally {
        Pop-Location
    }
    return "OFF"
}

function Find-LlamaServerBinary {
    param([Parameter(Mandatory = $true)][string]$Repo)

    foreach ($relative in (
        "build\bin\Release\llama-server.exe",
        "build\bin\llama-server.exe",
        "build\Release\llama-server.exe",
        "llama-server.exe"
    )) {
        $candidate = Join-Path $Repo $relative
        if (Test-Path $candidate -PathType Leaf) {
            return $candidate
        }
    }
    throw "llama-server.exe was not produced under $Repo"
}

function Stop-ProcessTreeBounded {
    param(
        [Parameter(Mandatory = $true)][Diagnostics.Process]$Process,
        [ValidateRange(1, 60000)][int]$TimeoutMilliseconds = 10000
    )

    if ($Process.HasExited) {
        return
    }
    try {
        $treeKillMethod = $Process.GetType().GetMethod("Kill", [type[]]@([bool]))
        if ($treeKillMethod) {
            $Process.Kill($true)
        } else {
            # .NET Framework (Windows PowerShell 5.1) has no Kill(Boolean)
            # overload. taskkill provides equivalent bounded tree termination.
            $previousErrorActionPreference = $ErrorActionPreference
            try {
                $ErrorActionPreference = "Continue"
                & taskkill.exe /PID $Process.Id /T /F 2>&1 | Out-Null
                $taskkillExit = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $previousErrorActionPreference
            }
            $Process.Refresh()
            if ($taskkillExit -ne 0 -and -not $Process.HasExited) {
                throw "taskkill failed with exit code $taskkillExit"
            }
        }
    } catch {
        # The process can exit between HasExited and Kill. Ignore only that
        # benign race; every other termination failure must fail the gate.
        $Process.Refresh()
        if (-not $Process.HasExited) {
            throw
        }
    }
    if (-not $Process.HasExited -and -not $Process.WaitForExit($TimeoutMilliseconds)) {
        throw "process $($Process.Id) did not exit within $TimeoutMilliseconds ms"
    }
}

function Get-TaskTextBounded {
    param(
        [object]$Task,
        [ValidateRange(1, 60000)][int]$TimeoutMilliseconds = 5000
    )

    if ($null -eq $Task) {
        return ""
    }
    if (-not $Task.Wait($TimeoutMilliseconds)) {
        throw "redirected process stream did not close within $TimeoutMilliseconds ms"
    }
    return $Task.GetAwaiter().GetResult()
}

function Remove-FailedLlamaStagingDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if (-not (Test-Path $Path)) {
        return
    }
    try {
        Remove-Item $Path -Recurse -Force
        Write-Host "==> Removed failed $Description staging tree: $Path"
    } catch {
        # Cleanup must never hide the original clone/identity error.
        Write-Warning "Could not remove failed $Description staging tree '$Path': $($_.Exception.Message)"
    }
}

function Get-FreeLoopbackPort {
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    try {
        $listener.Start()
        return ([Net.IPEndPoint]$listener.LocalEndpoint).Port
    } finally {
        $listener.Stop()
    }
}

function Resolve-HipSemanticValidationModel {
    $override = [Environment]::GetEnvironmentVariable("AUTOTUNER_HIP_VERIFY_MODEL")
    if ($override) {
        if (-not (Test-Path $override -PathType Leaf)) {
            throw "AUTOTUNER_HIP_VERIFY_MODEL does not point to a GGUF file: $override"
        }
        return (Resolve-Path $override).Path
    }

    # The Windows HIP recipes are intentionally pinned to Pandaking's dual-RDNA4
    # workstation. Keep one small, deterministic Qwen3 model as a correctness
    # oracle so a fast but numerically corrupt multi-GPU build can never pass.
    $defaultModel = "I:\models\Microsoft\fastcontext-1.0-4b-rl-q8_0.gguf"
    if (-not (Test-Path $defaultModel -PathType Leaf)) {
        throw "HIP semantic verification model not found: $defaultModel. Set AUTOTUNER_HIP_VERIFY_MODEL to a compatible Qwen3 GGUF."
    }
    return $defaultModel
}

function ConvertTo-WindowsCommandLineArgument {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Argument
    )

    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }

    # ProcessStartInfo.ArgumentList is unavailable on .NET Framework. Apply
    # the documented CommandLineToArgvW escaping rules for its Arguments fallback.
    $quoted = [Text.StringBuilder]::new()
    [void]$quoted.Append('"')
    $backslashes = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ($character -eq [char]'\') {
            $backslashes++
            continue
        }
        if ($character -eq [char]'"') {
            [void]$quoted.Append(('\' * (2 * $backslashes + 1)))
            [void]$quoted.Append('"')
            $backslashes = 0
            continue
        }
        if ($backslashes -gt 0) {
            [void]$quoted.Append(('\' * $backslashes))
            $backslashes = 0
        }
        [void]$quoted.Append($character)
    }
    if ($backslashes -gt 0) {
        [void]$quoted.Append(('\' * (2 * $backslashes)))
    }
    [void]$quoted.Append('"')
    return $quoted.ToString()
}

function Test-HipMultiGpuSemanticOutput {
    param(
        [Parameter(Mandatory = $true)][string]$Server,
        [Parameter(Mandatory = $true)][string]$Model
    )

    $expected = "HIP MULTI GPU OK"
    $alias = "autotuner-hip-verify"
    $port = Get-FreeLoopbackPort
    $arguments = @(
        "-m", $Model,
        "-c", "4096",
        "-ngl", "999",
        "-t", "4",
        "-tb", "4",
        "-b", "512",
        "-ub", "256",
        "--host", "127.0.0.1",
        "--port", "$port",
        "--tensor-split", "0.667,0.333",
        "--main-gpu", "0",
        "--parallel", "1",
        "-fa", "on",
        "-a", $alias
    )

    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Server
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.EnvironmentVariables["HIP_VISIBLE_DEVICES"] = "0,1"
    if ($startInfo.PSObject.Properties.Name -contains "ArgumentList") {
        foreach ($argument in $arguments) {
            [void]$startInfo.ArgumentList.Add($argument)
        }
    } else {
        $startInfo.Arguments = (
            $arguments | ForEach-Object { ConvertTo-WindowsCommandLineArgument $_ }
        ) -join " "
    }

    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $stdoutTask = $null
    $stderrTask = $null
    $failure = $null
    $cleanupErrors = [Collections.Generic.List[string]]::new()
    $assistantContent = ""
    $started = $false
    try {
        if (-not $process.Start()) {
            throw "llama-server process did not start"
        }
        $started = $true
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()

        $ready = $false
        $deadline = [DateTime]::UtcNow.AddSeconds(180)
        while ([DateTime]::UtcNow -lt $deadline) {
            if ($process.HasExited) {
                throw "llama-server exited during semantic-check startup (code $($process.ExitCode))"
            }
            try {
                $health = Invoke-RestMethod `
                    -Uri "http://127.0.0.1:$port/health" `
                    -TimeoutSec 2 `
                    -ErrorAction Stop
                if ([string]$health.status -eq "ok") {
                    $ready = $true
                    break
                }
            } catch {
                # The server is still loading the bounded 4B validation model.
            }
            Start-Sleep -Milliseconds 250
        }
        if (-not $ready) {
            throw "llama-server did not become ready within 180 seconds"
        }

        $request = @{
            model = $alias
            messages = @(@{
                role = "user"
                content = "Reply with exactly: $expected"
            })
            seed = 424242
            temperature = 0.0
            top_k = 1
            max_tokens = 32
            stream = $false
            chat_template_kwargs = @{ enable_thinking = $false }
        } | ConvertTo-Json -Depth 8 -Compress
        $response = Invoke-RestMethod `
            -Method Post `
            -Uri "http://127.0.0.1:$port/v1/chat/completions" `
            -ContentType "application/json" `
            -Body $request `
            -TimeoutSec 120 `
            -ErrorAction Stop
        $assistantContent = [string]$response.choices[0].message.content
        if ($assistantContent.Trim() -ne $expected) {
            throw "expected '$expected', got '$($assistantContent.Trim())'"
        }
    } catch {
        $failure = $_.Exception
    } finally {
        if ($started) {
            try {
                Stop-ProcessTreeBounded -Process $process -TimeoutMilliseconds 10000
            } catch {
                $cleanupErrors.Add($_.Exception.Message)
            }
        }
    }

    $stdout = ""
    $stderr = ""
    foreach ($stream in @(
        @{ Name = "stdout"; Task = $stdoutTask },
        @{ Name = "stderr"; Task = $stderrTask }
    )) {
        try {
            $text = Get-TaskTextBounded -Task $stream.Task -TimeoutMilliseconds 5000
            if ($stream.Name -eq "stdout") { $stdout = $text } else { $stderr = $text }
        } catch {
            $cleanupErrors.Add("$($stream.Name): $($_.Exception.Message)")
        }
    }
    try {
        $process.Dispose()
    } catch {
        $cleanupErrors.Add("process dispose: $($_.Exception.Message)")
    }

    if ($failure -or $cleanupErrors.Count -gt 0) {
        $primary = if ($failure) { $failure.Message } else { "process cleanup failed" }
        $cleanup = if ($cleanupErrors.Count -gt 0) {
            "`nCleanup: $($cleanupErrors -join '; ')"
        } else { "" }
        $tail = (($stdout + "`n" + $stderr) -split "`r?`n" | Select-Object -Last 30) -join "`n"
        throw "HIP multi-GPU semantic verification failed: $primary$cleanup`n$tail"
    }

    Write-Host "==> HIP multi-GPU semantic output verified: $assistantContent"
}

function Test-LlamaBuildOutput {
    param(
        [Parameter(Mandatory = $true)][string]$Repo,
        [Parameter(Mandatory = $true)][ValidateSet("Vulkan", "HIP")][string]$Backend,
        [int]$ExpectedBuild = -1,
        [string]$ExpectedVersion = "",
        [string]$ExpectedCommit = ""
    )

    if ($ExpectedCommit) {
        $actualCommit = (& git -C $Repo rev-parse HEAD | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $ExpectedCommit) {
            throw "Source commit mismatch: expected $ExpectedCommit, found $actualCommit"
        }
    }

    $server = Find-LlamaServerBinary -Repo $Repo
    $versionOutput = (& $env:ComSpec /d /s /c "`"`"$server`" --version 2>&1`"" | Out-String).Trim()
    $versionExit = $LASTEXITCODE
    if ($versionExit -ne 0) {
        throw "llama-server --version failed with exit code $versionExit`n$versionOutput"
    }
    Write-Host $versionOutput

    if ($ExpectedBuild -ge 0) {
        $buildMatch = [regex]::Match($versionOutput, '(?im)\bbuild\s+(\d+)\b')
        if (-not $buildMatch.Success -or [int]$buildMatch.Groups[1].Value -ne $ExpectedBuild) {
            throw "Expected runtime build $ExpectedBuild, got:`n$versionOutput"
        }
    }
    # The version must end at whitespace or "(": a plain \b would also accept
    # "0.7.0-dev" for a stable 0.7.0 build that was configured as dev.
    if ($ExpectedVersion -and $versionOutput -notmatch "(?im)^\s*version:\s*$([regex]::Escape($ExpectedVersion))(?=\s|\(|$)") {
        throw "Expected runtime version $ExpectedVersion, got:`n$versionOutput"
    }

    $deviceOutput = (& $env:ComSpec /d /s /c "`"`"$server`" --list-devices 2>&1`"" | Out-String).Trim()
    $deviceExit = $LASTEXITCODE
    if ($deviceExit -ne 0) {
        throw "llama-server --list-devices failed with exit code $deviceExit`n$deviceOutput"
    }
    Write-Host $deviceOutput

    $expectedPrefix = if ($Backend -eq "HIP") { "ROCm" } else { "Vulkan" }
    $wrongPrefix = if ($Backend -eq "HIP") { "Vulkan" } else { "ROCm" }
    foreach ($gpuName in ("AMD Radeon AI PRO R9700", "AMD Radeon RX 9070 XT")) {
        if ($deviceOutput -notmatch "(?im)^\s*$expectedPrefix\d+:.*$([regex]::Escape($gpuName))") {
            throw "$Backend build did not expose '$gpuName' as an $expectedPrefix device"
        }
    }
    if ($deviceOutput -match "(?im)^\s*$wrongPrefix\d+:") {
        throw "$Backend build unexpectedly also exposes the $wrongPrefix backend"
    }

    if ($Backend -eq "HIP") {
        $cache = Join-Path $Repo "build\CMakeCache.txt"
        if (-not (Test-Path $cache -PathType Leaf)) {
            throw "HIP CMake cache was not found: $cache"
        }
        $cacheText = Get-Content $cache -Raw
        if ($cacheText -notmatch '(?m)^GGML_CUDA_NO_PEER_COPY:BOOL=ON\s*$') {
            throw "Unsafe HIP build: GGML_CUDA_NO_PEER_COPY must be ON to prevent silent Windows multi-GPU output corruption"
        }
        if ($cacheText -notmatch '(?m)^SSL_EAY:FILEPATH=.*libssl\.lib\s*$') {
            if (@(Get-WindowsOpenSslArgs 3>$null).Count -gt 0) {
                # OpenSSL is installed, so this tree predates the recipe fix
                # (or the fix failed): treat it as incomplete so an existing
                # output is reconfigured and rebuilt with HTTPS.
                throw "HIP build has no OpenSSL although OpenSSL-Win64 is installed: HTTPS (-hf downloads, https MCP servers) would be disabled"
            }
            Write-Warning "HIP build has no OpenSSL: HTTPS (-hf downloads, https MCP servers) is disabled. Install OpenSSL-Win64 or set OPENSSL_ROOT_DIR, then rebuild."
        }
        $validationModel = Resolve-HipSemanticValidationModel
        Test-HipMultiGpuSemanticOutput -Server $server -Model $validationModel
    }

    Write-Host "==> Verified $Backend output: $server"
    return $server
}

function Copy-HipRuntimeDependencies {
    param(
        [Parameter(Mandatory = $true)][string]$Repo,
        [Parameter(Mandatory = $true)][string]$RocmPath
    )

    $server = Find-LlamaServerBinary -Repo $Repo
    $destination = Split-Path -Parent $server
    $bin = Join-Path $RocmPath "bin"
    foreach ($name in ("amdhip64_7.dll", "amd_comgr_3.dll", "hipblas.dll", "rocblas.dll", "hipblaslt.dll")) {
        $source = Join-Path $bin $name
        if (-not (Test-Path $source -PathType Leaf)) {
            throw "Required HIP runtime dependency was not found: $source"
        }
        Copy-Item $source -Destination $destination -Force
    }
    # TheRock distributions use this direct amdhip dependency; AMD's HIP SDK
    # 7.2 currently does not ship it. Bundle it when present.
    $kpack = Join-Path $bin "rocm_kpack.dll"
    if (Test-Path $kpack -PathType Leaf) {
        Copy-Item $kpack -Destination $destination -Force
    }

    # rocblas/hipblaslt resolve their Tensile kernels relative to the copied
    # DLL, not ROCM_PATH. Link the exact SDK data trees beside the executable
    # without duplicating ~1 GiB into every local build.
    foreach ($directoryName in ("rocblas", "hipblaslt")) {
        $sourceDirectory = Join-Path $bin $directoryName
        $destinationDirectory = Join-Path $destination $directoryName
        if (-not (Test-Path $sourceDirectory -PathType Container)) {
            throw "Required HIP kernel library was not found: $sourceDirectory"
        }
        if (Test-Path $destinationDirectory) {
            $item = Get-Item $destinationDirectory -Force
            $isLink = [bool]($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
            $linkTarget = if ($item.PSObject.Properties.Name -contains "LinkTarget") {
                [string]$item.LinkTarget
            } else { "" }
            if (-not $isLink -or $linkTarget -and $linkTarget -ne $sourceDirectory) {
                throw "Refusing to replace existing HIP kernel path: $destinationDirectory"
            }
        } else {
            New-Item -ItemType Junction -Path $destinationDirectory -Target $sourceDirectory | Out-Null
        }
    }
    Write-Host "==> Bundled matching ROCm 7 runtime DLLs and linked gfx1201 kernel libraries"
}

function Invoke-LlamaCMakeBuild {
    param(
        [Parameter(Mandatory = $true)][string]$Repo,
        [Parameter(Mandatory = $true)][ValidateSet("Vulkan", "HIP")][string]$Backend,
        [Parameter(Mandatory = $true)][string]$Workspace,
        [string]$RocmPath = "",
        [string]$BuildIsDev = "",
        [bool]$BuildUi = $true,
        [string[]]$Targets = @(),
        [string[]]$ExtraCMakeArgs = @(),
        [ValidateRange(1, 256)][int]$Parallel = 20
    )

    $environment = Initialize-LlamaBuildEnvironment -Backend $Backend -Workspace $Workspace -RocmPath $RocmPath
    $uiPrebuilt = Invoke-LlamaUiBuild -Repo $Repo -Enabled $BuildUi
    $buildDir = Join-Path $Repo "build"

    $commonArgs = @(
        "-DGGML_NATIVE=OFF",
        "-DGGML_AVX2=ON",
        "-DGGML_AVX_VNNI=ON",
        "-DGGML_BMI2=ON",
        "-DGGML_AVX512=OFF",
        "-DGGML_AVX512_VBMI=OFF",
        "-DGGML_AVX512_VNNI=OFF",
        "-DGGML_AVX512_BF16=OFF",
        "-DGGML_LTO=OFF",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_EXAMPLES=ON",
        "-DLLAMA_BUILD_UI=$(if ($BuildUi) { 'ON' } else { 'OFF' })",
        "-DLLAMA_USE_PREBUILT_UI=$uiPrebuilt",
        "-DGGML_CCACHE=OFF"
    )
    if ($BuildIsDev -in ("ON", "OFF")) {
        $commonArgs += "-DLLAMA_BUILD_IS_DEV=$BuildIsDev"
    }

    if ($Backend -eq "Vulkan") {
        $generator = $environment.VisualStudioGenerator
        $spirvInstall = Ensure-SpirvHeaders -Workspace $Workspace -Generator $generator -Parallel $Parallel
        $cmakeArgs = @(
            "-G", $generator,
            "-A", "x64",
            "-DGGML_VULKAN=ON",
            "-DGGML_HIP=OFF",
            "-DGGML_VULKAN_CHECK_RESULTS=OFF",
            "-DGGML_VULKAN_DEBUG=OFF",
            "-DGGML_VULKAN_MEMORY_DEBUG=OFF",
            "-DGGML_VULKAN_SHADER_DEBUG_INFO=OFF",
            "-DGGML_VULKAN_VALIDATE=OFF",
            "-DGGML_VULKAN_RUN_TESTS=OFF",
            "-DCMAKE_PREFIX_PATH=$spirvInstall"
        ) + $commonArgs + $ExtraCMakeArgs
        Invoke-NativeChecked "$Backend configure" {
            cmake -S $Repo -B $buildDir @cmakeArgs
        }
        $buildArgs = @("--build", $buildDir, "--config", "Release", "--parallel", $Parallel)
    } else {
        $bin = Join-Path $environment.RocmPath "bin"
        $clang = Join-Path $bin "clang.exe"
        $clangxx = Join-Path $bin "clang++.exe"
        $resourceFlags = @()
        if ($environment.CompatibilityResourceDir) {
            $resourceFlag = "-resource-dir=$($environment.CompatibilityResourceDir)"
            $resourceFlags = @(
                "-DCMAKE_C_FLAGS=$resourceFlag",
                "-DCMAKE_CXX_FLAGS=$resourceFlag"
            )
        }
        $cmakeArgs = @(
            "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_C_COMPILER=$clang",
            "-DCMAKE_CXX_COMPILER=$clangxx",
            # Upstream enables HIP through GGML_HIP and compiles HIP sources
            # with the selected C/C++ ROCm clang drivers. CMAKE_HIP_COMPILER
            # is not consumed by llama.cpp's Windows HIP configuration and
            # only creates CMake's "unused variable" warning.
            "-DGGML_HIP=ON",
            "-DGGML_VULKAN=OFF",
            "-DGPU_TARGETS=gfx1201",
            "-DGGML_HIP_GRAPHS=ON",
            "-DGGML_HIP_NO_VMM=ON",
            "-DGGML_HIP_RCCL=OFF",
            # Direct HIP peer copies silently corrupt tensors on this Windows
            # PCIe pair (R9700 + RX 9070 XT). Force the scheduler's safe staged
            # fallback; Test-LlamaBuildOutput verifies both the cache bit and
            # deterministic decoded text on an actual two-GPU layer split.
            "-DGGML_CUDA_NO_PEER_COPY=ON",
            "-DGGML_CUDA_FA=ON",
            # Pins only: both are already ON with GGML_NATIVE=OFF, and MSVC
            # (the Vulkan tree) has no such options at all.
            "-DGGML_FMA=ON",
            "-DGGML_F16C=ON"
        ) + $resourceFlags + $commonArgs + @(Get-LlamaFlashAttentionArgs -Repo $Repo) + @(Get-WindowsOpenSslArgs) + $ExtraCMakeArgs
        Invoke-NativeChecked "$Backend configure" {
            cmake -S $Repo -B $buildDir @cmakeArgs
        }
        $buildArgs = @("--build", $buildDir, "--parallel", ([Math]::Min($Parallel, 12)))
    }

    if ($Targets.Count -gt 0) {
        $buildArgs += "--target"
        $buildArgs += $Targets
    }
    Invoke-NativeChecked "$Backend build" {
        cmake @buildArgs
    }
    if ($Backend -eq "HIP") {
        Copy-HipRuntimeDependencies -Repo $Repo -RocmPath $environment.RocmPath
    }
}

function Get-WindowsOpenSslArgs {
    # ROCm clang uses the GNU driver (CMake sees MSVC=FALSE), so FindOpenSSL
    # never searches the Windows installer layout (lib/VC/x64/MD) and the HIP
    # server silently built without HTTPS through b11669. Point it at the same
    # OpenSSL the Visual Studio (Vulkan) tree finds on its own.
    $root = if ($env:OPENSSL_ROOT_DIR) { $env:OPENSSL_ROOT_DIR } else { "C:\Program Files\OpenSSL-Win64" }
    $include = Join-Path $root "include"
    $ssl = Join-Path $root "lib\VC\x64\MD\libssl.lib"
    $crypto = Join-Path $root "lib\VC\x64\MD\libcrypto.lib"
    if (-not ((Test-Path (Join-Path $include "openssl\ssl.h") -PathType Leaf) -and
              (Test-Path $ssl -PathType Leaf) -and (Test-Path $crypto -PathType Leaf))) {
        Write-Warning "OpenSSL-Win64 not found under '$root'; the HIP build will have no HTTPS support"
        return
    }
    # FindOpenSSL's non-MSVC, non-MinGW Windows branch derives
    # OPENSSL_{SSL,CRYPTO}_LIBRARY from the cached SSL_EAY / LIB_EAY.
    "-DOPENSSL_ROOT_DIR=$($root.Replace('\', '/'))"
    "-DOPENSSL_INCLUDE_DIR=$($include.Replace('\', '/'))"
    "-DSSL_EAY=$($ssl.Replace('\', '/'))"
    "-DLIB_EAY=$($crypto.Replace('\', '/'))"
}

function Get-LlamaFlashAttentionArgs {
    param([Parameter(Mandatory = $true)][string]$Repo)

    # b10876 replaces the boolean with a K/V combination list. Inspect the
    # checked-out source, not tag numbers: stable and pinned forks may differ.
    $options = Get-Content -LiteralPath (Join-Path $Repo "ggml/CMakeLists.txt") -Raw -ErrorAction Stop
    if ($options -match '(?m)^\s*set\s*\(\s*GGML_CUDA_FA_QUANTS\b') {
        # Clear a legacy ON cache entry too, or upstream still warns/overrides.
        "-UGGML_CUDA_FA_ALL_QUANTS"
        "-DGGML_CUDA_FA_QUANTS=all"
    } else {
        "-DGGML_CUDA_FA_ALL_QUANTS=ON"
    }
}

function Invoke-LlamaRdna4MoePatch {
    param(
        [Parameter(Mandatory = $true)][string]$Repo,
        [Parameter(Mandatory = $true)][ValidateSet("apply", "record", "verify")][string]$Mode
    )
    Assert-CommandAvailable "python"
    $patcher = Join-Path $PSScriptRoot "patch_rdna4_moe.py"
    Invoke-NativeChecked "RDNA4 MoE workaround $Mode" {
        python $patcher $Mode $Repo
    }
}

function Invoke-LlamaPrereleaseBuild {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("Vulkan", "HIP")][string]$Backend,
        [string]$Tag = "latest",
        [string]$Workspace = "L:\LAB\ai-local",
        [string]$RocmPath = "",
        [ValidateRange(1, 256)][int]$Parallel = 20,
        [switch]$Rdna4MoeWorkaround
    )

    if ($Tag -eq "latest") {
        $Tag = Get-LatestLlamaPrereleaseTag
    }
    if ($Tag -notmatch '^b\d+$' -and $Tag -ne "master") {
        throw "Tag must be 'latest', 'master', or an exact bNNNN tag"
    }
    if ($Rdna4MoeWorkaround -and ($Backend -ne "Vulkan" -or $Tag -notin @("b11319", "b11371"))) {
        throw "RDNA4 MoE workaround is qualified only for Vulkan -Tag b11319 or b11371; no unqualified source will be patched"
    }
    $backendToken = $Backend.ToLowerInvariant()
    $tmp = Join-Path $Workspace "_tmp_${backendToken}_prerelease_llama_$PID"
    if (Test-Path $tmp) {
        throw "Staging directory already exists: $tmp"
    }

    $stagingHandled = $false
    try {
    if ($Tag -eq "master") {
        Invoke-NativeChecked "llama.cpp master clone" {
            git clone https://github.com/ggml-org/llama.cpp.git $tmp
        }
    } else {
        Invoke-NativeChecked "llama.cpp $Tag clone" {
            git clone --branch $Tag --single-branch https://github.com/ggml-org/llama.cpp.git $tmp
        }
    }

    $commit = (& git -C $tmp rev-parse HEAD | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $commit) {
        throw "Could not determine llama.cpp commit"
    }
    $shortCommit = $commit.Substring(0, 9)
    $build = Get-LlamaBuildCount -Repo $tmp
    $buildTag = "b$build"
    $exactTag = Get-ExactRemotePrereleaseTag -Repo $tmp
    if ($Tag -ne "master" -and $exactTag -ne $Tag) {
        throw "Checkout is not the requested exact pre-release $Tag"
    }
    if ($exactTag -and $exactTag -ne $buildTag) {
        throw "Tag/history mismatch: exact tag $exactTag, commit count $buildTag"
    }
    $isExact = $exactTag -eq $buildTag
    $folderVersion = if ($isExact) { $buildTag } else { "${buildTag}_dev_${shortCommit}" }
    $dir = if ($Rdna4MoeWorkaround) {
        "rdna4moe_${folderVersion}_${backendToken}_llama.cpp"
    } else {
        "${folderVersion}_${backendToken}_llama.cpp"
    }
    $repo = Join-Path $Workspace $dir
    $semantic = Get-LlamaRuntimeSemanticVersion -Repo $tmp
    $expectedRuntime = "$semantic-dev"
    if (Test-Path $repo) {
        Write-Host "==> Existing output found; verify without replacing: $repo"
        Remove-Item $tmp -Recurse -Force
        $stagingHandled = $true
        try {
            if ($Rdna4MoeWorkaround) {
                Invoke-LlamaRdna4MoePatch -Repo $repo -Mode verify
            }
            Test-LlamaBuildOutput -Repo $repo -Backend $Backend -ExpectedBuild $build -ExpectedVersion $expectedRuntime -ExpectedCommit $commit | Out-Null
            Write-Host "Success (existing): $repo ($expectedRuntime, $buildTag, $Backend)"
            return
        } catch {
            if ($_.Exception.Message -like "Source commit mismatch:*") { throw }
            if ($Rdna4MoeWorkaround -and (Test-Path (Join-Path $repo "autotuner-rdna4-moe-workaround.json"))) {
                # A qualified receipt mismatch is not an incomplete first build.
                # Preserve changed user source/binaries instead of restamping them.
                throw
            }
            Write-Warning "Existing output is incomplete; resume its clean source/build tree: $($_.Exception.Message)"
        }
    } else {
        Rename-Item $tmp $dir
        $stagingHandled = $true
    }
    } finally {
        if (-not $stagingHandled) {
            Remove-FailedLlamaStagingDirectory -Path $tmp -Description "pre-release"
        }
    }

    Write-Host "==> Build directory: $repo ($buildTag, $Backend, commit $commit)"
    if ($Rdna4MoeWorkaround) {
        Invoke-LlamaRdna4MoePatch -Repo $repo -Mode apply
        if (Test-Path (Join-Path $repo "build\CMakeCache.txt")) {
            Invoke-NativeChecked "Clean incomplete RDNA4 variant before receipt creation" {
                cmake --build (Join-Path $repo "build") --config Release --target clean
            }
        }
    }
    Invoke-LlamaCMakeBuild -Repo $repo -Backend $Backend -Workspace $Workspace -RocmPath $RocmPath -BuildIsDev "ON" -Parallel $Parallel
    Test-LlamaBuildOutput -Repo $repo -Backend $Backend -ExpectedBuild $build -ExpectedVersion $expectedRuntime -ExpectedCommit $commit | Out-Null
    if ($Rdna4MoeWorkaround) {
        Invoke-LlamaRdna4MoePatch -Repo $repo -Mode record
        Invoke-LlamaRdna4MoePatch -Repo $repo -Mode verify
    }
    Write-Host "Success: $repo ($expectedRuntime, $buildTag, $Backend)"
}

function Invoke-LlamaStableBuild {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("Vulkan", "HIP")][string]$Backend,
        [string]$Tag = "latest",
        [string]$Workspace = "L:\LAB\ai-local",
        [string]$RocmPath = "",
        [ValidateRange(1, 256)][int]$Parallel = 20
    )

    if ($Tag -eq "latest") {
        $Tag = Get-LatestStableSemanticTag
    }
    if ($Tag -notmatch '^v?\d+\.\d+\.\d+$') {
        throw "Tag must be 'latest' or an exact X.Y.Z/vX.Y.Z tag"
    }
    $version = $Tag -replace '^v', ''
    $remoteTag = $Tag
    $remoteRef = git ls-remote --refs --tags https://github.com/ggml-org/llama.cpp.git "refs/tags/$remoteTag" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not query llama.cpp stable tag $remoteTag"
    }
    if (-not $remoteRef) {
        $alternateTag = if ($Tag.StartsWith("v")) { $version } else { "v$version" }
        $alternateRef = git ls-remote --refs --tags https://github.com/ggml-org/llama.cpp.git "refs/tags/$alternateTag" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $alternateRef) {
            throw "Stable tag was not found as '$Tag' or '$alternateTag'"
        }
        $remoteTag = $alternateTag
    }
    $backendToken = $Backend.ToLowerInvariant()
    $dir = "${version}_${backendToken}_llama.cpp"
    $repo = Join-Path $Workspace $dir
    $tmp = Join-Path $Workspace "_tmp_${backendToken}_${version}_stable_llama_$PID"
    if (Test-Path $tmp) {
        throw "Staging directory already exists: $tmp"
    }

    $stagingHandled = $false
    try {
    Invoke-NativeChecked "llama.cpp $remoteTag clone" {
        git clone --branch $remoteTag --single-branch https://github.com/ggml-org/llama.cpp.git $tmp
    }
    $resolvedTag = (& git -C $tmp describe --tags --exact-match HEAD 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $resolvedTag -ne $remoteTag) {
        throw "Checkout is not the requested exact stable tag $remoteTag"
    }
    $commit = (& git -C $tmp rev-parse HEAD | Out-String).Trim()
    $build = Get-LlamaBuildCount -Repo $tmp
    $buildTag = "b$build"
    $releaseRef = git ls-remote --refs --tags https://github.com/ggml-org/llama.cpp.git "refs/tags/$buildTag" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $releaseRef) {
        throw "Stable release $Tag has no corresponding pre-release tag $buildTag"
    }
    $releaseSha = (($releaseRef | Select-Object -First 1) -split '\s+')[0]
    if ($releaseSha -ne $commit) {
        throw "Stable/pre-release mismatch: $Tag is $commit but $buildTag is $releaseSha"
    }
    if (Test-Path $repo) {
        Write-Host "==> Existing output found; verify without replacing: $repo"
        Remove-Item $tmp -Recurse -Force
        $stagingHandled = $true
        try {
            Test-LlamaBuildOutput -Repo $repo -Backend $Backend -ExpectedBuild $build -ExpectedVersion $version -ExpectedCommit $commit | Out-Null
            Write-Host "Success (existing): $repo ($version stable / $buildTag, $Backend)"
            return
        } catch {
            if ($_.Exception.Message -like "Source commit mismatch:*") { throw }
            Write-Warning "Existing output is incomplete; resume its clean source/build tree: $($_.Exception.Message)"
        }
    } else {
        Rename-Item $tmp $dir
        $stagingHandled = $true
    }
    } finally {
        if (-not $stagingHandled) {
            Remove-FailedLlamaStagingDirectory -Path $tmp -Description "stable"
        }
    }

    Write-Host "==> Build directory: $repo ($version stable / $buildTag, $Backend, commit $commit)"
    Invoke-LlamaCMakeBuild -Repo $repo -Backend $Backend -Workspace $Workspace -RocmPath $RocmPath -BuildIsDev "OFF" -Parallel $Parallel
    Test-LlamaBuildOutput -Repo $repo -Backend $Backend -ExpectedBuild $build -ExpectedVersion $version -ExpectedCommit $commit | Out-Null
    Write-Host "Success: $repo ($version stable / $buildTag, $Backend)"
}

function Invoke-LlamaPinnedForkBuild {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("Vulkan", "HIP")][string]$Backend,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$RemoteUrl,
        [Parameter(Mandatory = $true)][string]$ExpectedCommit,
        [Parameter(Mandatory = $true)][string]$FolderPrefix,
        [string]$FetchRef = "",
        [string]$FixedIdentity = "",
        [string]$Workspace = "L:\LAB\ai-local",
        [string]$RocmPath = "",
        [bool]$BuildUi = $true,
        [string[]]$Targets = @(),
        [string[]]$ExtraCMakeArgs = @(),
        # Unified diffs applied with `git apply` on top of the pinned commit
        # (downstream-only fixes the fork has not merged yet). HEAD stays the
        # pinned commit, so the commit verification is unchanged; an existing
        # output tree that lacks a listed patch is patched and rebuilt.
        [string[]]$PatchFiles = @(),
        [ValidateRange(1, 256)][int]$Parallel = 20
    )

    foreach ($patch in $PatchFiles) {
        if (-not (Test-Path $patch -PathType Leaf)) {
            throw "$Name patch file not found: $patch"
        }
    }
    $backendToken = $Backend.ToLowerInvariant()
    $tmp = Join-Path $Workspace "_tmp_${FolderPrefix}${backendToken}_llama_$PID"
    if (Test-Path $tmp) {
        throw "Staging directory already exists: $tmp"
    }
    $stagingHandled = $false
    try {
    Invoke-NativeChecked "$Name clone" { git clone $RemoteUrl $tmp }
    if ($FetchRef) {
        Invoke-NativeChecked "$Name pinned ref fetch" {
            git -C $tmp fetch origin $FetchRef --force
        }
    }
    Invoke-NativeChecked "$Name pinned checkout" {
        git -C $tmp checkout --detach $ExpectedCommit
    }
    $actualCommit = (& git -C $tmp rev-parse HEAD | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $ExpectedCommit) {
        throw "$Name checkout mismatch: expected $ExpectedCommit, found $actualCommit"
    }
    foreach ($patch in $PatchFiles) {
        Invoke-NativeChecked "$Name patch $(Split-Path $patch -Leaf)" {
            git -C $tmp apply --whitespace=nowarn $patch
        }
    }

    $identity = $FixedIdentity
    if (-not $identity) {
        $hasUpstream = (& git -C $tmp remote | Where-Object { $_ -eq "upstream" })
        if (-not $hasUpstream) {
            Invoke-NativeChecked "$Name upstream remote" {
                git -C $tmp remote add upstream https://github.com/ggml-org/llama.cpp.git
            }
        }
        Invoke-NativeChecked "$Name upstream history" {
            git -C $tmp fetch upstream master --tags --force
        }
        $base = (& git -C $tmp merge-base HEAD upstream/master | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $base) {
            throw "Could not determine $Name mainline base"
        }
        $identity = "b$(Get-LlamaBuildCount -Repo $tmp -Commit $base)"
    }

    $dir = "${FolderPrefix}${identity}_${backendToken}_llama.cpp"
    $repo = Join-Path $Workspace $dir
    if (Test-Path $repo) {
        Write-Host "==> Existing output found; verify without replacing: $repo"
        Remove-Item $tmp -Recurse -Force
        $stagingHandled = $true
        $pending = @()
        foreach ($patch in $PatchFiles) {
            & git -C $repo apply --reverse --check --whitespace=nowarn $patch 2>$null
            if ($LASTEXITCODE -ne 0) { $pending += $patch }
        }
        if ($pending.Count -gt 0) {
            foreach ($patch in $pending) {
                Invoke-NativeChecked "$Name patch $(Split-Path $patch -Leaf)" {
                    git -C $repo apply --whitespace=nowarn $patch
                }
            }
            Write-Host "==> Applied $($pending.Count) pending patch file(s); rebuilding the existing tree"
        } else {
            try {
                Test-LlamaBuildOutput -Repo $repo -Backend $Backend -ExpectedCommit $ExpectedCommit | Out-Null
                Write-Host "Success (existing): $repo ($Name, $Backend, pinned $ExpectedCommit)"
                return
            } catch {
                if ($_.Exception.Message -like "Source commit mismatch:*") { throw }
                Write-Warning "Existing output is incomplete; resume its clean source/build tree: $($_.Exception.Message)"
            }
        }
    } else {
        Rename-Item $tmp $dir
        $stagingHandled = $true
    }
    } finally {
        if (-not $stagingHandled) {
            Remove-FailedLlamaStagingDirectory -Path $tmp -Description $Name
        }
    }
    Write-Host "==> Build directory: $repo ($Name, $Backend, pinned $ExpectedCommit)"

    Invoke-LlamaCMakeBuild -Repo $repo -Backend $Backend -Workspace $Workspace -RocmPath $RocmPath -BuildUi $BuildUi -Targets $Targets -ExtraCMakeArgs $ExtraCMakeArgs -Parallel $Parallel
    Test-LlamaBuildOutput -Repo $repo -Backend $Backend -ExpectedCommit $ExpectedCommit | Out-Null
    Write-Host "Success: $repo ($Name, $Backend, pinned $ExpectedCommit)"
}
