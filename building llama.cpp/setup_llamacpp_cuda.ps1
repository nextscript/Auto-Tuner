#Requires -RunAsAdministrator
# Vollautomatisches Setup: llama.cpp mit CUDA im aktuellen Script-Ordner
# Ausfuehren als Admin:
#   Set-ExecutionPolicy Bypass -Scope Process -Force
#   .\setup_llamacpp_cuda.ps1

Set-StrictMode -Off
$ErrorActionPreference = "Stop"

# --- KONFIGURATION ---
$INSTALL_DIR   = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$PARALLEL_JOBS = 12
$CUDA_VERSION  = "12.6"
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

# --- 1. CHOCOLATEY ---
Log "Pruefe Chocolatey"
if (-not (Is-Available "choco")) {
    Log "Installiere Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    Refresh-Path
    Add-ToPath "$env:ALLUSERSPROFILE\chocolatey\bin"
} else {
    OK "Chocolatey: $(choco --version)"
}

# --- 2. GIT ---
Log "Pruefe Git"
if (-not (Is-Available "git")) {
    choco install git -y --no-progress
    Refresh-Path
    Add-ToPath "C:\Program Files\Git\cmd"
} else {
    OK "Git: $(git --version)"
}

# --- 3. CMAKE (System, nicht VS-eingebettet) ---
Log "Pruefe CMake"
Add-ToPath "C:\Program Files\CMake\bin"
if (-not (Is-Available "cmake")) {
    choco install cmake --installargs 'ADD_CMAKE_TO_PATH=System' -y --no-progress
    Refresh-Path
    Add-ToPath "C:\Program Files\CMake\bin"
}
$CMAKE_EXE = Get-SystemCmake
OK "CMake: $CMAKE_EXE ($(& $CMAKE_EXE --version | Select-Object -First 1))"

# --- 4. NODE.JS ---
Log "Pruefe Node.js"
if (-not (Is-Available "node")) {
    choco install nodejs-lts -y --no-progress
    Refresh-Path
    Add-ToPath "C:\Program Files\nodejs"
} else {
    OK "Node.js: $(node --version)"
}

# --- 5. VISUAL STUDIO BUILD TOOLS ---
Log "Pruefe Visual Studio Build Tools"
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsFound = $false

if (Test-Path $vsWhere) {
    $vsJson = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json 2>$null
    if ($vsJson) {
        $vsInstalls = $vsJson | ConvertFrom-Json
        if ($vsInstalls) {
            $vsFound = $true
            OK "Visual Studio gefunden: $($vsInstalls.displayName)"
        }
    }
}

if (-not $vsFound) {
    Log "Installiere Visual Studio Build Tools 2022..."
    $vsBootstrap = "$env:TEMP\vs_buildtools.exe"
    Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vs_buildtools.exe" -OutFile $vsBootstrap -UseBasicParsing
    Start-Process -FilePath $vsBootstrap -ArgumentList @(
        "--quiet","--wait","--norestart","--nocache",
        "--add","Microsoft.VisualStudio.Workload.VCTools",
        "--add","Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "--add","Microsoft.VisualStudio.Component.Windows11SDK.22621",
        "--add","Microsoft.VisualStudio.Component.VC.CMake.Project"
    ) -Wait -NoNewWindow
    Refresh-Path
    OK "Build Tools installiert"
}

# --- 6. CUDA TOOLKIT ---
Log "Pruefe CUDA Toolkit"
$cudaFound = $false
$nvccPaths = @(
    "$CUDA_BASE\v12.9\bin",
    "$CUDA_BASE\v12.8\bin",
    "$CUDA_BASE\v12.6\bin",
    "$CUDA_BASE\v12.4\bin",
    "$CUDA_BASE\v12.2\bin",
    "$CUDA_BASE\v12.0\bin",
    "$CUDA_BASE\v11.8\bin"
)
foreach ($p in $nvccPaths) { Add-ToPath $p }

if (Is-Available "nvcc") {
    $cudaVer = nvcc --version 2>&1 | Select-String "release"
    OK "CUDA vorhanden: $cudaVer"
    $cudaFound = $true
}

if (-not $cudaFound) {
    Log "Installiere CUDA Toolkit $CUDA_VERSION..."
    $cudaInstaller = "$env:TEMP\cuda_installer.exe"
    $cudaUrl = "https://developer.download.nvidia.com/compute/cuda/12.6.0/network_installers/cuda_12.6.0_windows_network.exe"
    WARN "Lade CUDA Network Installer (~2GB Download)..."
    Invoke-WebRequest -Uri $cudaUrl -OutFile $cudaInstaller -UseBasicParsing
    Start-Process -FilePath $cudaInstaller -ArgumentList @(
        "-s",
        "cuda_profiler_api_12.6",
        "cudart_12.6",
        "nvcc_12.6",
        "cublas_12.6",
        "cublas_dev_12.6",
        "curand_12.6",
        "curand_dev_12.6",
        "visual_studio_integration_12.6"
    ) -Wait -NoNewWindow
    Refresh-Path
    foreach ($p in $nvccPaths) { Add-ToPath $p }
    if (-not (Is-Available "nvcc")) {
        WARN "nvcc nicht gefunden. Bitte neu starten und Script erneut ausfuehren!"
        exit 1
    }
}

# CUDA_PATH setzen
$cudaInstallDir = $null
if ($env:CUDA_PATH -and (Test-Path $env:CUDA_PATH)) {
    $cudaInstallDir = $env:CUDA_PATH
    OK "CUDA_PATH bereits gesetzt: $cudaInstallDir"
} else {
    if (Test-Path $CUDA_BASE) {
        $latest = Get-ChildItem $CUDA_BASE | Sort-Object Name -Descending | Select-Object -First 1
        if ($latest) {
            $cudaInstallDir = $latest.FullName
            $env:CUDA_PATH = $cudaInstallDir
            OK "CUDA_PATH = $cudaInstallDir"
        }
    }
}

# --- 6b. CUDA MSBuild-Props in alle VS-Instanzen kopieren ---
Log "Kopiere CUDA MSBuild-Props nach Visual Studio"
$cudaVsIntSrc = "$cudaInstallDir\extras\visual_studio_integration\MSBuildExtensions"

# Falls MSBuildExtensions nicht existiert: visual_studio_integration nachinstallieren
if (-not (Test-Path $cudaVsIntSrc)) {
    Log "Installiere CUDA visual_studio_integration..."
    $cudaInstaller2 = "$env:TEMP\cuda_vsi.exe"
    Invoke-WebRequest -Uri "https://developer.download.nvidia.com/compute/cuda/12.6.0/network_installers/cuda_12.6.0_windows_network.exe" -OutFile $cudaInstaller2 -UseBasicParsing
    Start-Process -FilePath $cudaInstaller2 -ArgumentList "-s visual_studio_integration_12.6" -Wait -NoNewWindow
}

if (Test-Path $cudaVsIntSrc) {
    # Alle bekannten VS MSBuild-Zielordner
    $vsTargetDirs = @(
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Microsoft\VC\v160\BuildCustomizations",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\MSBuild\Microsoft\VC\v160\BuildCustomizations",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Microsoft\VC\v170\BuildCustomizations"
    )
    $copied = 0
    foreach ($target in $vsTargetDirs) {
        # Nur kopieren wenn VS-Basisordner existiert
        $vsBase = Split-Path (Split-Path (Split-Path (Split-Path $target)))
        if (Test-Path $vsBase) {
            New-Item -ItemType Directory -Path $target -Force | Out-Null
            Copy-Item "$cudaVsIntSrc\*" $target -Force
            OK "CUDA Props -> $target"
            $copied++
        }
    }
    if ($copied -eq 0) {
        WARN "Kein VS-Installationsordner gefunden fuer Props-Copy!"
    }
} else {
    WARN "CUDA MSBuildExtensions nicht gefunden unter $cudaVsIntSrc"
    WARN "CMake wird fehlschlagen. Bitte CUDA manuell neu installieren."
    exit 1
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

# --- 8. LLAMA.CPP KLONEN (ueberspringen falls vorhanden) ---
Log "Pruefe llama.cpp"
Set-Location $INSTALL_DIR

# Vorhandenes b*_llama.cpp Verzeichnis suchen
$existingDir = Get-ChildItem $INSTALL_DIR -Directory | Where-Object { $_.Name -match "^b\d+_llama\.cpp$" } | Sort-Object Name -Descending | Select-Object -First 1

if ($existingDir) {
    $dir = $existingDir.FullName
    OK "Vorhandenes Verzeichnis gefunden: $dir (ueberspringe Clone)"
} else {
    $tmpDir = Join-Path $INSTALL_DIR "_tmp_llama"
    if (Test-Path $tmpDir) { Remove-Item $tmpDir -Recurse -Force }
    git clone https://github.com/ggml-org/llama.cpp.git $tmpDir
    Push-Location $tmpDir
    $desc = git describe --tags --always 2>$null
    $ver  = [regex]::Match($desc, 'b\d+').Value
    if (-not $ver) { $ver = "bUNKNOWN" }
    Pop-Location
    $dir = Join-Path $INSTALL_DIR "${ver}_llama.cpp"
    if (Test-Path $dir) { Remove-Item $dir -Recurse -Force }
    Rename-Item $tmpDir $dir
    OK "Verzeichnis: $dir"
}

# --- 9. UI BAUEN (ueberspringen falls dist vorhanden) ---
Log "Pruefe Web-UI"
$uiDir  = Join-Path $dir "tools\ui"
$distDir = Join-Path $uiDir "dist"
if (Test-Path $distDir) {
    OK "UI bereits gebaut - uebersprungen"
} elseif (Test-Path $uiDir) {
    Log "Baue Web-UI..."
    Push-Location $uiDir
    npm ci
    npm run build
    Pop-Location
    OK "UI gebaut"
} else {
    WARN "tools\ui nicht gefunden - uebersprungen"
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
    17 { "Visual Studio 17 2022" }
    16 { "Visual Studio 16 2019" }
    15 { "Visual Studio 15 2017" }
    default { "Visual Studio 16 2019" }
}
OK "VS: $vsPath2 (v$vsVersion -> $vsGenerator)"

$cmakeArgs = @(
    "-S", $dir, "-B", $buildDir,
    "-G", $vsGenerator, "-A", "x64",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_CUDA=ON", "-DGGML_VULKAN=OFF",
    "-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON",
    "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_UI=ON", "-DLLAMA_USE_PREBUILT_UI=OFF",
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

# --- FERTIG ---
Log "BUILD ERFOLGREICH!"
$binPath = Join-Path $buildDir "bin\Release"
OK "Binaries: $binPath"
$exes = Get-ChildItem $binPath -Filter "*.exe" -ErrorAction SilentlyContinue
if ($exes) { $exes | ForEach-Object { OK "  $($_.Name)" } }
Write-Host "`nServer starten:" -ForegroundColor Green
Write-Host "  $binPath\llama-server.exe -m <model.gguf> --host 0.0.0.0 --port 8080" -ForegroundColor Green
