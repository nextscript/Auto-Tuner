#Requires -RunAsAdministrator
# Vollautomatisches Setup: llama.cpp mit CUDA im aktuellen Script-Ordner
# Ausfuehren als Admin:
#   Set-ExecutionPolicy Bypass -Scope Process -Force
#   .\setup_llamacpp_cuda.ps1

# Windows PowerShell 5.1 turns harmless native stderr progress into terminating
# NativeCommandError records. Relaunch this installer under PowerShell 7, whose
# native-process semantics the package-manager flow below is written for.
if ($PSVersionTable.PSVersion.Major -lt 7) {
    $pwsh = Get-Command pwsh.exe -ErrorAction SilentlyContinue
    if (-not $pwsh) {
        throw "PowerShell 7 is required. Install 'pwsh' and run this script again."
    }
    & $pwsh.Source -NoProfile -File $PSCommandPath
    $pwshExit = $LASTEXITCODE
    if ($pwshExit -ne 0) {
        throw "PowerShell 7 setup process failed with exit code $pwshExit"
    }
    return
}

Set-StrictMode -Off
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

# --- KONFIGURATION ---
$INSTALL_DIR   = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$PARALLEL_JOBS = 12
$CUDA_BASE     = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"

# --- HILFSFUNKTIONEN ---
function Log($msg)  { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function OK($msg)   { Write-Host "    [OK] $msg" -ForegroundColor Green }
function WARN($msg) { Write-Host "    [!!] $msg" -ForegroundColor Yellow }

function Refresh-Path {
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH","User")
}

function Add-ToPath($p) {
    if ((Test-Path $p) -and ($env:PATH -notlike "*$p*")) {
        $env:PATH = "$p;$env:PATH"
        OK "PATH += $p"
    }
}

function Is-Available($cmd) {
    return [bool](Get-Command $cmd -ErrorAction SilentlyContinue)
}

function Get-ExactLlamaBuildTag {
    param(
        [string]$Commit = "HEAD",
        [string]$Remote = "origin"
    )

    $sha = (git rev-parse $Commit 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $sha) {
        throw "Could not resolve llama.cpp commit '$Commit'"
    }

    # A nearest older b-tag must never name a newer development checkout.
    $escapedSha = [regex]::Escape($sha)
    $remoteTags = git ls-remote --tags $Remote "refs/tags/b*" 2>$null
    foreach ($row in $remoteTags) {
        if ($row -match "^$escapedSha\s+refs/tags/(b\d+)(?:\^\{\})?$") {
            return $Matches[1]
        }
    }

    git fetch $Remote --tags --force 2>$null
    $pointingTags = @(
        git tag --points-at $Commit --list "b[0-9]*" 2>$null |
            Where-Object { $_ -match '^b\d+$' } |
            Sort-Object { [int]($_ -replace '^b', '') } -Descending
    )
    if ($pointingTags.Count -gt 0) { return $pointingTags[0] }
    return $null
}



function Get-WingetPackageState {
    param(
        [Parameter(Mandatory=$true)][string]$WingetId,
        [string]$CheckCommand = ""
    )

    $commandAvailable = $false
    if ($CheckCommand) {
        $commandAvailable = [bool](Get-Command $CheckCommand -ErrorAction SilentlyContinue)
    }

    $listed = $false
    if (Is-Available "winget") {
        & winget list --id $WingetId --exact --accept-source-agreements --disable-interactivity *> $null
        $listed = ($LASTEXITCODE -eq 0)
    }

    [PSCustomObject]@{
        Installed = ($commandAvailable -or $listed)
        CommandAvailable = $commandAvailable
        Listed = $listed
    }
}

function Install-OrUpgrade-LatestPackage {
    param(
        [Parameter(Mandatory=$true)][string]$WingetId,
        [Parameter(Mandatory=$true)][string]$ChocoName,
        [string[]]$WingetOverride = @(),
        [string[]]$ChocoExtraArgs = @(),
        [string]$CheckCommand = "",
        [switch]$Optional
    )

    Refresh-Path

    if (Is-Available "winget") {
        $state = Get-WingetPackageState -WingetId $WingetId -CheckCommand $CheckCommand
        $commonArgs = @(
            "--id", $WingetId, "--exact",
            "--accept-package-agreements", "--accept-source-agreements",
            "--silent", "--disable-interactivity"
        )

        if ($state.Installed) {
            & winget upgrade @commonArgs @WingetOverride
            $upgradeExit = $LASTEXITCODE
            Refresh-Path

            $stateAfter = Get-WingetPackageState -WingetId $WingetId -CheckCommand $CheckCommand
            if ($stateAfter.Installed) {
                if ($upgradeExit -eq 0) {
                    OK "$WingetId ist installiert und auf dem neuesten von winget angebotenen Stand."
                } else {
                    OK "$WingetId ist bereits installiert; kein Upgrade verfuegbar oder winget meldete nur einen nicht-kritischen Status (Exitcode $upgradeExit)."
                }
                return $true
            }
        }

        & winget install @commonArgs @WingetOverride
        $installExit = $LASTEXITCODE
        Refresh-Path

        $stateAfterInstall = Get-WingetPackageState -WingetId $WingetId -CheckCommand $CheckCommand
        if ($stateAfterInstall.Installed) {
            OK "$WingetId ist installiert."
            return $true
        }

        if ($Optional) {
            WARN "$WingetId ist ueber winget nicht verfuegbar oder konnte nicht installiert werden (Exitcode $installExit)."
            return $false
        }

        if (Is-Available "choco") {
            WARN "winget konnte $WingetId nicht bestaetigen. Versuche Chocolatey-Paket $ChocoName."
        } else {
            throw "'$WingetId' ist nicht installiert und konnte nicht installiert werden (Exitcode $installExit)."
        }
    }

    if (Is-Available "choco") {
        & choco upgrade $ChocoName -y --no-progress @ChocoExtraArgs
        $chocoExit = $LASTEXITCODE
        Refresh-Path

        if (($CheckCommand -and (Is-Available $CheckCommand)) -or ($chocoExit -eq 0)) {
            OK "$ChocoName ist installiert bzw. aktuell."
            return $true
        }

        if ($Optional) {
            WARN "Chocolatey konnte $ChocoName nicht bestaetigen (Exitcode $chocoExit)."
            return $false
        }

        throw "Chocolatey konnte '$ChocoName' nicht aktualisieren/installieren (Exitcode $chocoExit)."
    }

    if ($Optional) { return $false }
    throw "Weder winget noch ein funktionsfaehiges Chocolatey ist verfuegbar."
}

function Get-LatestCudaInstall {
    if (-not (Test-Path $CUDA_BASE)) { return $null }

    return Get-ChildItem -Path $CUDA_BASE -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^v(?<version>\d+(?:\.\d+){1,2})$' } |
        ForEach-Object {
            [PSCustomObject]@{
                Path    = $_.FullName
                Version = [version]$Matches.version
            }
        } |
        Sort-Object Version -Descending |
        Select-Object -First 1
}

function Install-LatestCudaToolkit {
    Log "Installiere/aktualisiere die aktuellste verfuegbare CUDA-Toolkit-Version..."
    Install-OrUpgrade-LatestPackage -WingetId "Nvidia.CUDA" -ChocoName "cuda"
}


function Deploy-CudaRuntimeDlls {
    param(
        [Parameter(Mandatory=$true)][string]$CudaBin,
        [Parameter(Mandatory=$true)][string]$Destination
    )

    if (-not (Test-Path $CudaBin)) {
        throw "CUDA-bin-Verzeichnis nicht gefunden: $CudaBin"
    }
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null

    # Die EXE wird oft direkt aus dem Build-Ordner/Explorer gestartet. Dann ist der
    # CUDA-bin-Pfad nicht zwingend im Prozess-PATH. Deshalb die benoetigten Runtime-
    # DLLs neben llama-server.exe ablegen (Windows DLL-Suchreihenfolge).
    $patterns = @(
        "cudart64_*.dll",
        "cublas64_*.dll",
        "cublasLt64_*.dll",
        "nvJitLink_*.dll",
        "nvrtc64_*.dll",
        "nvrtc-builtins64_*.dll"
    )

    $copied = @()
    foreach ($pattern in $patterns) {
        $dlls = Get-ChildItem -Path $CudaBin -Filter $pattern -File -ErrorAction SilentlyContinue
        foreach ($dll in $dlls) {
            Copy-Item -LiteralPath $dll.FullName -Destination (Join-Path $Destination $dll.Name) -Force
            $copied += $dll.Name
        }
    }

    $required = @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")
    foreach ($pattern in $required) {
        if (-not (Get-ChildItem -Path $Destination -Filter $pattern -File -ErrorAction SilentlyContinue)) {
            throw "Erforderliche CUDA Runtime-DLL fehlt nach dem Kopieren: $pattern"
        }
    }

    OK "CUDA Runtime-DLLs bereitgestellt: $($copied.Count) Datei(en)"
}

function Get-SystemCmake {
    # Gibt den cmake-Pfad zurueck, der NICHT der VS-eingebettete ist
    $candidates = Get-Command cmake -All -ErrorAction SilentlyContinue
    foreach ($c in $candidates) {
        if ($c.Source -notlike "*Visual Studio*") {
            return $c.Source
        }
    }
    # Fallback: irgendeinen nehmen
    return (Get-Command cmake -ErrorAction SilentlyContinue).Source
}

# --- 0. INSTALL-VERZEICHNIS ---
Log "Erstelle Installationsverzeichnis $INSTALL_DIR"
if (-not (Test-Path $INSTALL_DIR)) {
    New-Item -ItemType Directory -Path $INSTALL_DIR | Out-Null
}
OK $INSTALL_DIR

# --- 1. PAKETMANAGER ---
Log "Pruefe Paketmanager"
$chocoCandidates = @(
    (Join-Path $env:ProgramData "chocolatey\bin\choco.exe"),
    (Join-Path $env:ALLUSERSPROFILE "chocolatey\bin\choco.exe")
) | Select-Object -Unique
foreach ($candidate in $chocoCandidates) {
    if (Test-Path $candidate) { Add-ToPath (Split-Path $candidate -Parent) }
}
Refresh-Path

if (Is-Available "winget") {
    OK "winget ist verfuegbar."
} elseif (Is-Available "choco") {
    OK "Chocolatey: $(choco --version)"
} else {
    $chocoRoot = Join-Path $env:ProgramData "chocolatey"
    if (Test-Path $chocoRoot) {
        throw "Chocolatey-Ordner ist vorhanden, aber choco.exe fehlt oder ist defekt: $chocoRoot. Repariere Chocolatey oder installiere winget."
    }

    Log "Installiere Chocolatey, weil winget nicht verfuegbar ist..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    Refresh-Path
    Add-ToPath (Join-Path $env:ProgramData "chocolatey\bin")
    if (-not (Is-Available "choco")) { throw "Chocolatey wurde installiert, aber choco.exe ist nicht verfuegbar." }
    OK "Chocolatey: $(choco --version)"
}


# --- 2. GIT ---
Log "Installiere/aktualisiere Git auf die neueste stabile Version"
Install-OrUpgrade-LatestPackage -WingetId "Git.Git" -ChocoName "git" -CheckCommand "git"
Refresh-Path
Add-ToPath "C:\Program Files\Git\cmd"
OK "Git: $(git --version)"

# --- 3. CMAKE (System, nicht VS-eingebettet) ---
Log "Installiere/aktualisiere CMake auf die neueste stabile Version"
Install-OrUpgrade-LatestPackage -WingetId "Kitware.CMake" -ChocoName "cmake" -ChocoExtraArgs @("--installargs", "ADD_CMAKE_TO_PATH=System") -CheckCommand "cmake"
Refresh-Path
Add-ToPath "C:\Program Files\CMake\bin"
$CMAKE_EXE = Get-SystemCmake
if (-not $CMAKE_EXE) { throw "CMake wurde installiert, aber nicht gefunden." }
OK "CMake: $CMAKE_EXE ($(& $CMAKE_EXE --version | Select-Object -First 1))"

# --- 4. NODE.JS ---
$NODE_VERSION = "24.19.0"
Log "Installiere/aktualisiere Node.js auf v$NODE_VERSION"
Install-OrUpgrade-LatestPackage `
    -WingetId "OpenJS.NodeJS.LTS" `
    -ChocoName "nodejs-lts" `
    -WingetOverride @("--version", $NODE_VERSION) `
    -ChocoExtraArgs @("--version=$NODE_VERSION", "--allow-downgrade") `
    -CheckCommand "node"
Refresh-Path
Add-ToPath "C:\Program Files\nodejs"
$installedNodeVersion = (node --version).TrimStart("v")
if ($installedNodeVersion -ne $NODE_VERSION) {
    throw "Node.js v$NODE_VERSION wurde angefordert, gefunden wurde aber v$installedNodeVersion."
}
OK "Node.js: v$installedNodeVersion"
OK "npm: $(npm --version)"

# --- 5. VISUAL STUDIO BUILD TOOLS ---
Log "Pruefe/aktualisiere Visual Studio Build Tools"
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

function Get-VsCppInstallation {
    if (-not (Test-Path $vsWhere)) { return $null }
    $json = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json 2>$null
    if (-not $json) { return $null }
    $items = $json | ConvertFrom-Json
    if ($items -is [array]) { return $items | Select-Object -First 1 }
    return $items
}

$vsInstall = Get-VsCppInstallation
$vsOverride = '--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'

if ($vsInstall) {
    OK "Visual Studio C++ Build Tools bereits installiert: $($vsInstall.displayName) $($vsInstall.catalog.productDisplayVersion)"

    foreach ($candidateId in @("Microsoft.VisualStudio.2026.BuildTools", "Microsoft.VisualStudio.2022.BuildTools")) {
        if (Is-Available "winget") {
            $installedCandidate = Get-WingetPackageState -WingetId $candidateId
            if ($installedCandidate.Installed) {
                [void](Install-OrUpgrade-LatestPackage -WingetId $candidateId -ChocoName "visualstudio2022buildtools" -WingetOverride @("--override", $vsOverride) -Optional)
                break
            }
        }
    }
} else {
    $installedVs = $false
    if (Is-Available "winget") {
        foreach ($candidateId in @("Microsoft.VisualStudio.2026.BuildTools", "Microsoft.VisualStudio.2022.BuildTools")) {
            $ok = Install-OrUpgrade-LatestPackage -WingetId $candidateId -ChocoName "visualstudio2022buildtools" -WingetOverride @("--override", $vsOverride) -Optional
            Refresh-Path
            $vsInstall = Get-VsCppInstallation
            if ($ok -and $vsInstall) { $installedVs = $true; break }
        }
    }

    if (-not $installedVs -and (Is-Available "choco")) {
        & choco upgrade visualstudio2022buildtools visualstudio2022-workload-vctools -y --no-progress --package-parameters "--includeRecommended --passive --locale en-US"
        Refresh-Path
        $vsInstall = Get-VsCppInstallation
        $installedVs = [bool]$vsInstall
    }

    if (-not $installedVs) {
        throw "Keine Visual-Studio-Installation mit C++ x64/x86 Build Tools gefunden und die Installation ist fehlgeschlagen."
    }
}

$vsInstall = Get-VsCppInstallation
if (-not $vsInstall) { throw "Keine Visual-Studio-Instanz mit C++ Build Tools gefunden." }
OK "Visual Studio bereit: $($vsInstall.displayName) $($vsInstall.catalog.productDisplayVersion)"


# --- 6. CUDA TOOLKIT ---
Log "Installiere/aktualisiere CUDA Toolkit auf die neueste verfuegbare Version"
Install-LatestCudaToolkit
Refresh-Path

# Installierte CUDA-Versionen numerisch sortieren und die neueste priorisieren.
$latestCuda = Get-LatestCudaInstall
if ($latestCuda) {
    $cudaInstallDir = $latestCuda.Path
    $cudaVersion    = $latestCuda.Version.ToString()
    Add-ToPath (Join-Path $cudaInstallDir "bin")
    $env:CUDA_PATH = $cudaInstallDir
    $env:CUDAToolkit_ROOT = $cudaInstallDir
    OK "Neueste installierte CUDA-Version: $cudaVersion ($cudaInstallDir)"
} else {
    Install-LatestCudaToolkit
    Refresh-Path

    $latestCuda = Get-LatestCudaInstall
    if (-not $latestCuda) {
        throw "CUDA wurde installiert, aber unter '$CUDA_BASE' nicht gefunden. Windows neu starten und das Script erneut ausfuehren."
    }

    $cudaInstallDir = $latestCuda.Path
    $cudaVersion    = $latestCuda.Version.ToString()
    Add-ToPath (Join-Path $cudaInstallDir "bin")
    $env:CUDA_PATH = $cudaInstallDir
    $env:CUDAToolkit_ROOT = $cudaInstallDir
    OK "CUDA installiert: $cudaVersion ($cudaInstallDir)"
}

# Sicherstellen, dass nvcc exakt aus der neuesten CUDA-Installation kommt.
$nvccExe = Join-Path $cudaInstallDir "bin\nvcc.exe"
if (-not (Test-Path $nvccExe)) {
    throw "nvcc.exe wurde in der neuesten CUDA-Installation nicht gefunden: $nvccExe"
}
$cudaVerText = & $nvccExe --version 2>&1 | Select-String "release"
OK "CUDA Compiler: $cudaVerText"

# CUDA_PATH dauerhaft fuer nachfolgende Prozesse aktualisieren.
[System.Environment]::SetEnvironmentVariable("CUDA_PATH", $cudaInstallDir, "Machine")

# --- 6b. CUDA MSBuild-Props in alle VS-Instanzen kopieren ---
Log "Kopiere CUDA MSBuild-Props nach Visual Studio"
$cudaVsIntSrc = Join-Path $cudaInstallDir "extras\visual_studio_integration\MSBuildExtensions"

if (-not (Test-Path $cudaVsIntSrc)) {
    WARN "Visual-Studio-Integration fehlt in CUDA $cudaVersion. Repariere/aktualisiere das aktuelle CUDA-Paket..."
    Install-LatestCudaToolkit
    Refresh-Path
    $latestCuda = Get-LatestCudaInstall
    $cudaInstallDir = $latestCuda.Path
    $cudaVersion = $latestCuda.Version.ToString()
    $cudaVsIntSrc = Join-Path $cudaInstallDir "extras\visual_studio_integration\MSBuildExtensions"
}

if (Test-Path $cudaVsIntSrc) {
    $vsInstallationPaths = & $vsWhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    $vsTargetDirs = foreach ($vsInstallPath in $vsInstallationPaths) {
        $vcMsbuildRoot = Join-Path $vsInstallPath "MSBuild\Microsoft\VC"
        if (Test-Path $vcMsbuildRoot) {
            Get-ChildItem $vcMsbuildRoot -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match '^v\d+$' } |
                ForEach-Object { Join-Path $_.FullName "BuildCustomizations" }
        }
    }
    $copied = 0
    foreach ($target in ($vsTargetDirs | Sort-Object -Unique)) {
        New-Item -ItemType Directory -Path $target -Force | Out-Null
        Copy-Item "$cudaVsIntSrc\*" $target -Force
        OK "CUDA Props -> $target"
        $copied++
    }
    if ($copied -eq 0) { WARN "Keine BuildCustomizations-Verzeichnisse der installierten Visual-Studio-Versionen gefunden." }
} else {
    throw "CUDA MSBuildExtensions wurden nicht gefunden: $cudaVsIntSrc"
}

# --- 7. ABHAENGIGKEITEN FINAL CHECK ---
Log "Finale Pruefung"
$allOk = $true
foreach ($cmd in @("git","node","npm","nvcc")) {
    if (Is-Available $cmd) {
        OK "$cmd OK"
    } else {
        WARN "$cmd FEHLT - bitte neu starten und erneut ausfuehren"
        $allOk = $false
    }
}
if (-not $allOk) { exit 1 }
OK "cmake OK: $CMAKE_EXE"

# --- 8. LLAMA.CPP FRISCH UND VERSIONSRICHTIG KLONEN ---
Log "Pruefe llama.cpp Nightly/Main"
Set-Location $INSTALL_DIR

# Nie einen alten bNNNN-Ordner in-place auf einen neueren Commit ziehen: sonst
# sagt der Ordner z.B. b10549, waehrend die EXE bereits build 10572 meldet. Ein
# prozessspezifisches Staging bewahrt den letzten funktionierenden Build.
$tmpDir = Join-Path $INSTALL_DIR "_tmp_llama_$PID"
if (Test-Path $tmpDir) {
    throw "Staging-Verzeichnis existiert bereits: $tmpDir"
}
$stagingHandled = $false
try {
    git clone https://github.com/ggml-org/llama.cpp.git $tmpDir
    if ($LASTEXITCODE -ne 0) { throw "llama.cpp konnte nicht geklont werden." }

    Push-Location $tmpDir
    try {
        $commit = (git rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $commit) {
            throw "llama.cpp Commit konnte nicht bestimmt werden."
        }
        $shortCommit = $commit.Substring(0, 9)
        $buildText = (git rev-list --count HEAD).Trim()
        if ($LASTEXITCODE -ne 0 -or $buildText -notmatch '^\d+$') {
            throw "llama.cpp Buildnummer konnte nicht bestimmt werden."
        }
        $expectedBuild = [int]$buildText
        if ($expectedBuild -lt 1000) {
            throw "Unvollstaendige Git-History; verdaechtige Buildnummer $expectedBuild."
        }
        $buildTag = "b$expectedBuild"
        $exactTag = Get-ExactLlamaBuildTag -Commit HEAD -Remote origin
        if ($exactTag -and $exactTag -ne $buildTag) {
            throw "Tag/History-Widerspruch: $exactTag vs. $buildTag"
        }
        $isExactTag = $exactTag -eq $buildTag

        $cmakeText = Get-Content (Join-Path $tmpDir "CMakeLists.txt") -Raw
        $majorMatch = [regex]::Match($cmakeText, '(?m)^set\(LLAMA_VERSION_MAJOR\s+(\d+)\)')
        $minorMatch = [regex]::Match($cmakeText, '(?m)^set\(LLAMA_VERSION_MINOR\s+(\d+)\)')
        $patchMatch = [regex]::Match($cmakeText, '(?m)^set\(LLAMA_VERSION_PATCH\s+(\d+)\)')
        if (-not ($majorMatch.Success -and $minorMatch.Success -and $patchMatch.Success)) {
            throw "Semantische llama.cpp Version konnte nicht gelesen werden."
        }
        $semanticVersion = "$($majorMatch.Groups[1].Value).$($minorMatch.Groups[1].Value).$($patchMatch.Groups[1].Value)"
        $expectedRuntimeVersion = "$semanticVersion-dev"
    } finally {
        Pop-Location
    }

    $folderVersion = if ($isExactTag) { $buildTag } else { "${buildTag}_dev_${shortCommit}" }
    $dir = Join-Path $INSTALL_DIR "${folderVersion}_llama.cpp"
    if (Test-Path $dir) {
        $existingCommit = ""
        if (Test-Path (Join-Path $dir ".git")) {
            $existingCommit = (git -C $dir rev-parse HEAD 2>$null).Trim()
        }
        if ($existingCommit -ne $commit) {
            throw "Zielordner existiert mit anderem Commit; nichts geloescht: $dir"
        }
        Remove-Item $tmpDir -Recurse -Force
        $stagingHandled = $true
        OK "Exakter Quellordner bereits vorhanden: $dir"
    } else {
        Rename-Item $tmpDir $dir
        $stagingHandled = $true
        OK "Verzeichnis: $dir"
    }
} finally {
    if (-not $stagingHandled -and (Test-Path $tmpDir)) {
        try {
            Remove-Item $tmpDir -Recurse -Force
            WARN "Fehlgeschlagenes llama.cpp Staging-Verzeichnis entfernt: $tmpDir"
        } catch {
            WARN "Staging-Verzeichnis konnte nicht entfernt werden: $tmpDir ($($_.Exception.Message))"
        }
    }
}
OK "Nightly $buildTag; Runtime-Version $expectedRuntimeVersion; Commit $commit"

# --- 9. UI BAUEN (layout-tolerant; sonst Prebuilt-UI) ---
Log "Pruefe Web-UI"
$uiPrebuilt = "ON"
if ((Test-Path (Join-Path $dir "ui\apps\chat\package.json")) -and (Test-Path (Join-Path $dir "scripts\ui-assets.cmake"))) {
    OK "Llama Chat (ui\apps\chat) wird direkt waehrend des CMake-Builds gebaut"
    $uiPrebuilt = "OFF"
} else {
    $uiDir = $null
    foreach ($relativeUiDir in @("tools\ui", "tools\server\webui")) {
        $candidateUiDir = Join-Path $dir $relativeUiDir
        if (Test-Path (Join-Path $candidateUiDir "package.json")) {
            $uiDir = $candidateUiDir
            break
        }
    }
    if ($uiDir) {
        $distDir = Join-Path $uiDir "dist"
        if (Test-Path $distDir) {
            OK "UI bereits gebaut - uebersprungen"
        } else {
            Log "Baue Web-UI aus $uiDir..."
            Push-Location $uiDir
            try {
                if (Test-Path "package-lock.json") { npm ci } else { npm install }
                if ($LASTEXITCODE -ne 0) { throw "npm dependency install failed" }
                npm run build
                if ($LASTEXITCODE -ne 0) { throw "npm UI build failed" }
            } finally {
                Pop-Location
            }
            OK "UI gebaut"
        }
        $uiPrebuilt = "OFF"
    } else {
        WARN "Keine UI-Quellen gefunden - CMake verwendet die Prebuilt-UI"
    }
}

# --- 10. CMAKE CONFIGURE ---
Log "CMake Konfiguration (CUDA)"
$buildDir = Join-Path $dir "build"

# Altes build-Verzeichnis loeschen (sauberer Neustart)
if (Test-Path $buildDir) {
    Log "Loesche altes build-Verzeichnis..."
    Remove-Item $buildDir -Recurse -Force
}

# VS-Instanz per vswhere ermitteln
$vsWhere2 = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vsWhere2)) {
    WARN "vswhere.exe nicht gefunden!"
    exit 1
}

$vsPath2   = & $vsWhere2 -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
$vsVersion = & $vsWhere2 -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion 2>$null

if (-not ($vsPath2 -and $vsVersion)) {
    WARN "Keine VS-Instanz mit C++-Tools gefunden!"
    exit 1
}

$vsMajor = [int]($vsVersion.Split(".")[0])
$vsGenerator = switch ($vsMajor) {
    18 { "Visual Studio 18 2026" }
    17 { "Visual Studio 17 2022" }
    16 { "Visual Studio 16 2019" }
    15 { "Visual Studio 15 2017" }
    default { throw "Nicht unterstuetzte Visual-Studio-Hauptversion: $vsMajor" }
}
OK "VS: $vsPath2 (v$vsVersion -> $vsGenerator)"

$cmakeArgs = @(
    "-S", $dir, "-B", $buildDir,
    "-G", $vsGenerator, "-A", "x64",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DLLAMA_BUILD_IS_DEV=ON",
    "-DGGML_CUDA=ON", "-DGGML_VULKAN=OFF",
    "-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON",
    "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_UI=ON", "-DLLAMA_USE_PREBUILT_UI=$uiPrebuilt",
    "-DLLAMA_CURL=OFF", "-DGGML_CCACHE=OFF"
)
Log "Starte: $CMAKE_EXE $($cmakeArgs -join ' ')"
& $CMAKE_EXE @cmakeArgs

if ($LASTEXITCODE -ne 0) {
    WARN "CMake Konfiguration fehlgeschlagen! Code: $LASTEXITCODE"
    exit 1
}
OK "CMake Konfiguration erfolgreich"

# --- 11. BUILD ---
Log "Kompiliere llama.cpp mit $PARALLEL_JOBS Jobs..."
& $CMAKE_EXE --build $buildDir --config Release --parallel $PARALLEL_JOBS

if ($LASTEXITCODE -ne 0) {
    WARN "Build fehlgeschlagen! Code: $LASTEXITCODE"
    exit 1
}

# --- CUDA RUNTIME-DLLS BEREITSTELLEN ---
$binPath = Join-Path $buildDir "bin\Release"
Log "Kopiere CUDA Runtime-DLLs neben die EXE-Dateien"
Deploy-CudaRuntimeDlls -CudaBin (Join-Path $cudaInstallDir "bin") -Destination $binPath

# Runtime-Identitaet gegen Quellversion + volle Git-History pruefen.
$server = Join-Path $binPath "llama-server.exe"
# llama-server schreibt --version auf stderr. Beide Streams via cmd.exe
# zusammenfuehren, damit ErrorActionPreference=Stop gueltige Ausgabe nicht als
# terminierenden NativeCommandError behandelt.
$versionOutput = (& $env:ComSpec /d /s /c "`"$server`" --version 2>&1" | Out-String).Trim()
$versionExit = $LASTEXITCODE
if ($versionExit -ne 0) {
    throw "llama-server Smoke-Test fehlgeschlagen (Exitcode $versionExit)."
}
Write-Host $versionOutput
$versionMatch = [regex]::Match(
    $versionOutput,
    '(?im)^\s*version:\s*(\d+\.\d+\.\d+-dev)\s+\(\s*build\s+(\d+)\b'
)
if (-not $versionMatch.Success) {
    throw "llama-server meldet keine semantische Dev-Version plus Buildnummer."
}
$reportedVersion = $versionMatch.Groups[1].Value
$reportedBuild = [int]$versionMatch.Groups[2].Value
if ($reportedVersion -ne $expectedRuntimeVersion -or $reportedBuild -ne $expectedBuild) {
    throw "Source/Runtime-Widerspruch: erwartet $expectedRuntimeVersion ($buildTag), gemeldet $reportedVersion (b$reportedBuild)."
}

# CUDA-bin auch dauerhaft in den System-PATH aufnehmen. Das ist nur ein Fallback;
# die lokale DLL-Kopie macht den Build direkt portabel innerhalb dieses Ordners.
$machinePath = [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
$cudaBinPath = Join-Path $cudaInstallDir "bin"
if (($machinePath -split ';') -notcontains $cudaBinPath) {
    [System.Environment]::SetEnvironmentVariable("PATH", ($machinePath.TrimEnd(';') + ';' + $cudaBinPath), "Machine")
    OK "CUDA bin dauerhaft zum System-PATH hinzugefuegt"
}

# --- FERTIG ---
Log "BUILD ERFOLGREICH!"
OK "Binaries: $binPath"
OK "Version: $reportedVersion (Nightly $buildTag, Commit $shortCommit)"
$exes = Get-ChildItem $binPath -Filter "*.exe" -ErrorAction SilentlyContinue
if ($exes) { $exes | ForEach-Object { OK "  $($_.Name)" } }
Write-Host "`nServer starten:" -ForegroundColor Green
Write-Host "  $binPath\llama-server.exe -m <model.gguf> --host 0.0.0.0 --port 8080" -ForegroundColor Green
