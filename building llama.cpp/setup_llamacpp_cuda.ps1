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
Log "Installiere/aktualisiere Node.js auf die neueste LTS-Version"
Install-OrUpgrade-LatestPackage -WingetId "OpenJS.NodeJS.LTS" -ChocoName "nodejs-lts" -CheckCommand "node"
Refresh-Path
Add-ToPath "C:\Program Files\nodejs"
OK "Node.js: $(node --version)"
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

# --- 8. LLAMA.CPP KLONEN (ueberspringen falls vorhanden) ---
Log "Pruefe llama.cpp"
Set-Location $INSTALL_DIR

# Vorhandenes b*_llama.cpp Verzeichnis suchen
$existingDir = Get-ChildItem $INSTALL_DIR -Directory | Where-Object { $_.Name -match "^b\d+_llama\.cpp$" } | Sort-Object Name -Descending | Select-Object -First 1

if ($existingDir) {
    $dir = $existingDir.FullName
    OK "Vorhandenes Verzeichnis gefunden: $dir"
    if (Test-Path (Join-Path $dir ".git")) {
        Log "Aktualisiere llama.cpp auf den neuesten Stand"
        Push-Location $dir
        git fetch --prune origin
        $currentBranch = (git branch --show-current).Trim()
        if (-not $currentBranch) { $currentBranch = "master" }
        git pull --ff-only origin $currentBranch
        if ($LASTEXITCODE -ne 0) { Pop-Location; throw "llama.cpp-Repository konnte nicht aktualisiert werden." }
        Pop-Location
        OK "llama.cpp-Quellcode aktualisiert"
    }
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

# --- CUDA RUNTIME-DLLS BEREITSTELLEN ---
$binPath = Join-Path $buildDir "bin\Release"
Log "Kopiere CUDA Runtime-DLLs neben die EXE-Dateien"
Deploy-CudaRuntimeDlls -CudaBin (Join-Path $cudaInstallDir "bin") -Destination $binPath

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
$exes = Get-ChildItem $binPath -Filter "*.exe" -ErrorAction SilentlyContinue
if ($exes) { $exes | ForEach-Object { OK "  $($_.Name)" } }
Write-Host "`nServer starten:" -ForegroundColor Green
Write-Host "  $binPath\llama-server.exe -m <model.gguf> --host 0.0.0.0 --port 8080" -ForegroundColor Green
