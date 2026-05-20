"""Hardware detection: CPU, RAM, and GPU(s) across vendors.

Supports NVIDIA (nvidia-smi), AMD (rocm-smi), Intel (lspci/WMI),
Apple Silicon (sysctl), and Windows Registry-based VRAM detection.
Multi-GPU aware. Uses subprocess, winreg, and vendor SDKs for accurate
free VRAM reporting on all platforms.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import psutil
import os

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Windows Registry-Helper f\u00fcr 64-bit VRAM
# ---------------------------------------------------------------------------


def _get_vram_from_registry() -> Dict[str, int]:
    """Lese DedicatedVRAM aus der Windows Registry (64-bit sicher).

    Liest HardwareInformation.qwMemorySize aus dem Registry-Key
    des GPU-Drivers. Vermeidet den 32-Bit-Overflow von
    Win32_VideoController.AdapterRAM.
    """
    result: Dict[str, int] = {}
    try:
        import winreg
    except ImportError:
        return result

    reg_path_base = (
        r"SYSTEM\CurrentControlSet\Control\Class"
        r"\{4d36e968-e325-11ce-bfc1-08002be10318}"
    )

    for i in range(100):
        key_name = f"000{i}"
        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                reg_path_base + "\\" + key_name,
                0,
                winreg.KEY_READ,
            )
        except FileNotFoundError:
            break
        except OSError:
            continue

        driver_desc = ""
        vram_qw = 0
        try:
            driver_desc, _ = winreg.QueryValueEx(key, "DriverDesc")
            qw_mem, _ = winreg.QueryValueEx(key, "HardwareInformation.qwMemorySize")
            vram_qw = int(qw_mem)
        except (FileNotFoundError, OSError, ValueError):
            pass
        finally:
            try:
                winreg.CloseKey(key)
            except Exception:
                pass

        if not driver_desc or vram_qw <= 0:
            continue

        desc_lower = driver_desc.lower()
        if any(
            skip in desc_lower
            for skip in (
                "basic render",
                "remote display",
                "hyper-v",
                "rdp",
                "microsoft",
                "mirror",
            )
        ):
            continue

        # AMD RX 9000 Series (RDNA 5) und andere echte GPUs nicht filtern
        result[driver_desc] = vram_qw

    return result


def _get_gpu_vram_used_via_wmi() -> Dict[str, float]:
    """Echte VRAM-Nutzung (DedicatedUsage) über WMI win32com auslesen.

    Returns mapping of GPU name (lowercased) -> used VRAM in MB.
    Uses Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory
    which reports DedicatedUsage in bytes.

    Die Verbindung zwischen LUID-Counter und GPU-Namen wird über das
    VideoProcessor-Feld im Win32_VideoController hergestellt, welches die
    DeviceId als Hex-Wert enthält (z.B. "AMD Radeon Graphics Processor (0x7550)").

    Returns empty dict if WMI is unavailable or GPU performance counters
    are not registered (common on AMD RX 9000 series under Windows).
    """
    result: Dict[str, float] = {}
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        return result

    try:
        pythoncom.CoInitialize()
        wmi = win32com.client.GetObject("winmgmts:\\\\root\\\\cimv2")

        # Schritt 1: DeviceId (hex) -> GPU-Name Mapping über Win32_VideoController
        # VideoProcessor enthält z.B. "AMD Radeon Graphics Processor (0x7550)"
        devid_to_name: Dict[str, str] = {}
        try:
            for vc in wmi.ExecQuery(
                "SELECT Name, VideoProcessor FROM Win32_VideoController"
            ):
                name = (vc.Name or "").strip()
                processor = (vc.VideoProcessor or "").strip()
                if not name or not processor:
                    continue
                # Filter out virtual/auxiliary adapters
                lower = name.lower()
                if any(
                    skip in lower
                    for skip in (
                        "basic render",
                        "remote display",
                        "hyper-v",
                        "rdp",
                        "microsoft",
                        "mirror",
                    )
                ):
                    continue
                # Extrahiere DeviceId aus Klammern, z.B. "(0x7550)" -> "7550"
                import re

                m = re.search(r"\(0x([0-9a-fA-F]+)\)", processor)
                if m:
                    dev_id = m.group(1).lower()
                    devid_to_name[dev_id] = name
        except Exception:
            pass

        # Schritt 2: VRAM-Nutzung auslesen und mit GPU-Namen verknüpfen
        for obj in wmi.ExecQuery(
            "SELECT Name, DedicatedUsage "
            "FROM Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory"
        ):
            counter_name = str(obj.Name or "").split("_phys")[0].strip()
            if not counter_name:
                continue

            # LUID extrahieren (z.B. "luid_0x00000000_0x00015915" -> "00015915")
            luid_parts = counter_name.split("_")
            luid_dev = ""
            for part in luid_parts:
                if part.startswith("0x"):
                    luid_dev = part[2:].lower()  # "0x" entfernen, lowercase
                    break

            used_bytes = float(obj.DedicatedUsage or 0)
            used_mb = used_bytes / (1024 * 1024)

            if not luid_dev:
                continue

            # Versuche, über DeviceId den GPU-Namen zu finden
            gpu_name = None

            # Methode 1: Direkter Match (LUID == DeviceId)
            if luid_dev in devid_to_name:
                gpu_name = devid_to_name[luid_dev]
            else:
                # Methode 2: Vergleiche hex-Werte (LUID und DEV können unterschiedlich formatiert sein)
                # z.B. LUID "15915" vs DeviceId "7550" – beide als int vergleichen
                try:
                    luid_int = int(luid_dev, 16)
                    for known_dev_id, known_name in devid_to_name.items():
                        dev_int = int(known_dev_id, 16)
                        if luid_int == dev_int:
                            gpu_name = known_name
                            break
                except ValueError:
                    pass

            if gpu_name:
                result[gpu_name.lower()] = used_mb
            else:
                # Fallback: Verwende Counter-Name als Key (für Debugging)
                result[counter_name.lower()] = used_mb

    except Exception:
        # WMI nicht verfügbar, RPC-Server fehlerhaft, oder keine GPU-Counter
        pass
    finally:
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass

    return result


def _get_gpu_vram_free_via_wmi() -> Dict[str, float]:
    """Echten freien VRAM (AvailableVideoMemory) über WMI auslesen.

    Returns mapping of GPU name (lowercased) -> free VRAM in MB.
    Uses Win32_VideoController.AvailableVideoMemory — this attribute reports
    the amount of video memory available to applications, in bytes.

    This is a fallback for GPUs where DedicatedUsage counters are not
    registered (common on AMD RX 9000 series / RDNA 5 under Windows).
    """
    result: Dict[str, float] = {}
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        return result

    try:
        pythoncom.CoInitialize()
        wmi = win32com.client.GetObject("winmgmts:\\\\root\\\\cimv2")
        for obj in wmi.ExecQuery(
            "SELECT Name, AvailableVideoMemory FROM Win32_VideoController"
        ):
            name = str(obj.Name or "").strip()
            if not name:
                continue
            # Filter out virtual/auxiliary adapters
            lower = name.lower()
            if any(
                skip in lower
                for skip in (
                    "basic render",
                    "remote display",
                    "hyper-v",
                    "rdp",
                    "microsoft",
                    "mirror",
                )
            ):
                continue
            free_bytes = int(obj.AvailableVideoMemory or 0)
            if free_bytes < 0:
                free_bytes = 0
            free_mb = free_bytes / (1024 * 1024)
            result[name.lower()] = free_mb
    except Exception:
        pass
    finally:
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass

    return result


def _get_gpu_vram_via_dxgi_powershell() -> Dict[str, float]:
    """PowerShell-Fallback für VRAM-Usage-Erkennung (AMD RX 9000 Series).

    Returns mapping of GPU name (lowercased) -> used_vram_mb.

    Bei AMD RX 9000 Series (RDNA 5) sind DedicatedVideoMemory und
    AvailableVideoMemory in Win32_VideoController leer. Daher wird eine
    kombinierte PowerShell-Abfrage verwendet die:

    1. Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory
       für DedicatedUsage (VRAM-Nutzung) ausliest
    2. Über Win32_VideoController.VideoProcessor die DeviceId mit dem
       GPU-Namen verknüpft
    3. Bei fehlendem LUID-Match: GPU mit der höchsten VRAM-Nutzung als
       diskrete GPU verwendet (Fallback für AMD RX 9000)

    WICHTIG: Diese Funktion gibt NUR die genutzte VRAM-Menge zurück.
    Total VRAM muss über _get_vram_from_registry() bezogen werden!

    Returns empty dict if PowerShell is unavailable or fails.
    """
    result: Dict[str, float] = {}

    # Registry-VRAM für DeviceId-Mapping holen
    registry_vram = _get_vram_from_registry()

    ps_script = r"""
$ErrorActionPreference = 'SilentlyContinue'

# Schritt 1: GPU-Namen und DeviceId aus Win32_VideoController sammeln
# VideoProcessor enthält z.B. "AMD Radeon Graphics Processor (0x7550)"
$devidToName = @{}
$controllers = Get-CimInstance Win32_VideoController |
    Where-Object {
        $_.PNPDeviceID -like 'PCI*' -and
        $_.Name -notmatch 'Basic Render|Remote Display|Hyper-V|RDP|Mirror'
    }
foreach ($ctrl in $controllers) {
    $name = $ctrl.Name.Trim()
    $processor = $ctrl.VideoProcessor
    if (-not $processor) { $processor = "" }
    $processor = $processor.Trim()
    if (-not $name -or -not $processor) { continue }
    
    # Extrahiere DeviceId aus Klammern, z.B. "(0x7550)" -> "7550"
    if ($processor -match '\(0x([0-9a-fA-F]+)\)') {
        $devId = $matches[1].ToLower()
        $devidToName[$devId] = $name
    }
}

# Schritt 2: GPU-Performance-Counter auslesen
$counters = Get-CimInstance -Namespace root\cimv2 `
    -Query "SELECT Name, DedicatedUsage FROM Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory"

# Schritt 3: Usage über LUID-to-DeviceId matching mit Fallback
$results = @()
foreach ($cnt in $counters) {
    $counterName = ($cnt.Name -split '_phys')[0].Trim()
    $luidParts = $counterName -split '_'
    if ($luidParts.Count -lt 3) { continue }
    
    # LUID extrahieren – WICHTIG: den LETZTEN 0x-Teil nehmen!
    # Format: luid_0xXXXXXXXX_0xYYYYYYYY -> wir brauchen 0xYYYYYYYY
    $luidDev = ""
    for ($i = $luidParts.Count - 1; $i -ge 0; $i--) {
        $part = $luidParts[$i]
        if ($part -match '^0x([0-9a-fA-F]+)$') {
            $luidDev = $matches[1]
            break
        }
    }
    
    if (-not $luidDev) { continue }
    $luidDevLower = $luidDev.ToLower()
    
    # Versuche, GPU-Namen zu finden
    $ctrlName = $null
    
    # Methode 1: Direkter Match mit DeviceId
    if ($devidToName.ContainsKey($luidDevLower)) {
        $ctrlName = $devidToName[$luidDevLower]
    } else {
        # Methode 2: Vergleiche hex-Werte (LUID und DEV können unterschiedlich sein)
        try {
            $luidInt = [Convert]::ToInt64($luidDev, 16)
            foreach ($mapKey in $devidToName.Keys) {
                $devInt = [Convert]::ToInt64($mapKey, 16)
                if ($luidInt -eq $devInt) {
                    $ctrlName = $devidToName[$mapKey]
                    break
                }
            }
        } catch {}
    }
    
    # Fallback: Wenn kein LUID-Match, aber GPU mit hoher Usage vorhanden -> nimm sie
    if (-not $ctrlName) {
        $usageBytes = [int64]($cnt.DedicatedUsage -as [int64])
        if ($usageBytes -lt 0) { $usageBytes = 0 }
        $usedMB = $usageBytes / (1024 * 1024)
        # Wenn Usage > 100 MB, handelt es sich wahrscheinlich um die diskrete GPU
        if ($usedMB -gt 100) {
            foreach ($name in $devidToName.Values) {
                if ($name -like "*Radeon*" -or $name -like "*GeForce*" -or $name -like "*RTX*" -or $name -like "*GTX*") {
                    $ctrlName = $name
                    break
                }
            }
            # Wenn keine diskrete GPU im Match, nimm die erste nicht-virtuelle GPU
            if (-not $ctrlName -and $devidToName.Count -eq 1) {
                $ctrlName = $devidToName.Values[0]
            }
        }
    }
    
    if (-not $ctrlName) { continue }
    
    $usageBytes = [int64]($cnt.DedicatedUsage -as [int64])
    if ($usageBytes -lt 0) { $usageBytes = 0 }
    $usedMB = $usageBytes / (1024 * 1024)
    
    $results += [PSCustomObject]@{
        Name     = $ctrlName
        UsedMB   = [int64]$usedMB
    }
}

if ($results) {
    $results | ConvertTo-Json -Compress -Depth 3
} else {
    Write-Output "[]"
}
"""

    try:
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                ps_script,
            ],
            timeout=15,
        )
        if not out:
            return result

        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]

        # Mapping: controller_name_lower -> used_mb
        wmi_used_map: Dict[str, float] = {}
        for d in data:
            name = (d.get("Name") or "").strip()
            if not name:
                continue
            lower = name.lower()
            if any(
                skip in lower
                for skip in (
                    "basic render",
                    "remote display",
                    "hyper-v",
                    "rdp",
                    "microsoft",
                    "mirror",
                )
            ):
                continue
            try:
                used_mb = float(d.get("UsedMB") or 0)
                wmi_used_map[lower] = used_mb
            except (TypeError, ValueError):
                continue

        # Jetzt mit Registry-VRAM free_mb berechnen
        for reg_name, vram_bytes in registry_vram.items():
            reg_lower = reg_name.lower()
            if any(
                skip in reg_lower
                for skip in (
                    "basic render",
                    "remote display",
                    "hyper-v",
                    "rdp",
                    "microsoft",
                    "mirror",
                )
            ):
                continue
            total_mb = vram_bytes / (1024 * 1024)
            used_mb = wmi_used_map.get(reg_lower, 0)
            if total_mb > 0:
                result[reg_lower] = used_mb
    except Exception:
        pass

    return result


@dataclass
class GPUInfo:
    index: int
    name: str
    vendor: str  # "nvidia" | "amd" | "intel" | "apple" | "unknown"
    total_vram_mb: int
    free_vram_mb: int
    gpu_util_percent: float = 0.0  # GPU-Auslastung in %

    @property
    def total_vram_gb(self) -> float:
        return self.total_vram_mb / 1024

    @property
    def free_vram_gb(self) -> float:
        return self.free_vram_mb / 1024


@dataclass
class SystemInfo:
    os_name: str
    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    total_ram_gb: float
    free_ram_gb: float
    gpus: List[GPUInfo] = field(default_factory=list)
    # GPUs that were detected but considered too small/auxiliary to use for
    # inference (typically integrated GPUs alongside a discrete card). Kept
    # for transparency in the menu header.
    ignored_gpus: List[GPUInfo] = field(default_factory=list)

    @property
    def total_vram_gb(self) -> float:
        return sum(g.total_vram_gb for g in self.gpus)

    @property
    def free_vram_gb(self) -> float:
        return sum(g.free_vram_gb for g in self.gpus)

    @property
    def primary_vendor(self) -> str:
        if not self.gpus:
            return "cpu"
        return max(self.gpus, key=lambda g: g.total_vram_mb).vendor

    @property
    def is_multi_gpu(self) -> bool:
        return len(self.gpus) > 1


# ---------------------------------------------------------------------------
# Helpers


def _run(cmd: List[str], timeout: float = 5) -> Optional[str]:
    """Run a command and return stdout, or None on any failure.

    On Windows, CREATE_NO_WINDOW suppresses the brief console-window flash
    that would otherwise appear for every powershell / wmic call.
    """
    try:
        kwargs: dict = {}
        if os.name == "nt":
            # Prevent subprocess from creating a visible console window.
            # Without this flag, every powershell call briefly flashes a
            # black terminal on screen.
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="ignore",
            **kwargs,
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


# ---------------------------------------------------------------------------
# GPU detection per vendor


def _detect_nvidia() -> List[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    gpus: List[GPUInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            try:
                gpu_util = float(parts[4].replace("%", "").strip())
            except (ValueError, IndexError):
                gpu_util = 0.0
            try:
                gpus.append(
                    GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        vendor="nvidia",
                        total_vram_mb=int(parts[2]),
                        free_vram_mb=int(parts[3]),
                        gpu_util_percent=gpu_util,
                    )
                )
            except ValueError:
                continue
    return gpus


def _get_nvidia_gpu_utilization() -> Dict[str, float]:
    """Ermittle GPU-Auslastung über nvidia-smi.

    Returns mapping of GPU name (lowercased) -> utilization %.
    """
    result: Dict[str, float] = {}
    if not shutil.which("nvidia-smi"):
        return result
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return result
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpu_name = parts[1]
                gpu_util = float(parts[2].replace("%", "").strip())
                result[gpu_name.lower()] = gpu_util
            except (ValueError, IndexError):
                continue
    return result


def _detect_amd_rocm() -> List[GPUInfo]:
    if not shutil.which("rocm-smi"):
        return []

    # Try JSON first - more reliable across rocm-smi versions
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--json"])
    if out:
        try:
            data = json.loads(out)
            gpus: List[GPUInfo] = []
            for key, info in data.items():
                m = re.match(r"card(\d+)", key, re.IGNORECASE)
                if not m:
                    continue
                idx = int(m.group(1))
                total_b = 0
                used_b = 0
                gpu_pct = 0.0
                name = (
                    info.get("Card Series")
                    or info.get("Card model")
                    or info.get("Card SKU")
                    or f"AMD GPU {idx}"
                )
                for k, v in info.items():
                    if "VRAM Total Memory" in k or k == "Total Memory (B)":
                        try:
                            total_b = int(str(v).strip().split()[0])
                        except (ValueError, IndexError):
                            pass
                    elif "VRAM Total Used" in k or k == "Used Memory (B)":
                        try:
                            used_b = int(str(v).strip().split()[0])
                        except (ValueError, IndexError):
                            pass
                    # GPU-Utilization aus verschiedenen möglichen Feldnamen
                    elif (
                        "GPU Item" in k
                        or "System Total" in k
                        or "GPU utilization" in k.lower()
                    ):
                        try:
                            val_str = str(v).strip().replace("%", "")
                            gpu_pct = float(val_str)
                        except (ValueError, IndexError):
                            pass
                total_mb = total_b // (1024 * 1024)
                used_mb = used_b // (1024 * 1024)
                gpus.append(
                    GPUInfo(
                        index=idx,
                        name=name,
                        vendor="amd",
                        total_vram_mb=total_mb,
                        free_vram_mb=max(0, total_mb - used_mb),
                        gpu_util_percent=gpu_pct,
                    )
                )
            if gpus:
                return gpus
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Text fallback
    out = _run(["rocm-smi", "--showmeminfo", "vram"])
    if not out:
        return []
    by_idx: dict = {}
    for line in out.splitlines():
        m = re.match(r"GPU\[(\d+)\].*?Total\s+Memory.*?(\d+)\s*$", line, re.IGNORECASE)
        if m:
            by_idx.setdefault(int(m.group(1)), {})["total"] = int(m.group(2))
    return [
        GPUInfo(
            index=i,
            name=f"AMD GPU {i}",
            vendor="amd",
            total_vram_mb=info.get("total", 0) // (1024 * 1024),
            free_vram_mb=info.get("total", 0) // (1024 * 1024),
            gpu_util_percent=0.0,
        )
        for i, info in by_idx.items()
    ]


def _vendor_from_name(name: str) -> str:
    """Best-effort vendor inference from a GPU's display name."""
    n = name.lower()
    if any(s in n for s in ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla")):
        return "nvidia"
    if any(s in n for s in ("amd", "radeon", "rx ", "rdna")):
        return "amd"
    if "intel" in n or "arc" in n:
        return "intel"
    return "unknown"


# PowerShell snippet: enumerates every PCI video adapter, reads VRAM from the
# 64-bit registry value (HardwareInformation.qwMemorySize) so 16 GB+ cards
# are reported correctly. Win32_VideoController.AdapterRAM is signed 32-bit
# and overflows at 4 GB, so it's only used as a last-resort fallback.
#
# VRAM free detection: Uses DedicatedUsage from GPUAdapterMemory to calculate
# free = total - used (more reliable than AvailableVideoMemory on AMD RX 9000).
# Matching über VideoProcessor.DeviceId statt Win32_PnPEntity.
_WIN_GPU_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'

# Schritt 1: GPU-Namen aus Win32_VideoController sammeln
$adapters = Get-CimInstance Win32_VideoController |
    Where-Object {
        $_.PNPDeviceID -like 'PCI*' -and
        $_.Name -notmatch 'Basic Render|Remote Display|Hyper-V|RDP|Mirror'
    }

# Schritt 2: Registry VRAM holen (64-bit sicher)
$regBase = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
$regKeys = Get-ChildItem $regBase -ErrorAction SilentlyContinue
$regVram = @{}  # DriverDesc -> VRAM bytes
foreach ($k in $regKeys) {
    $p = Get-ItemProperty $k.PSPath -ErrorAction SilentlyContinue
    if ($null -ne $p -and $null -ne $p.DriverDesc -and $null -ne $p.'HardwareInformation.qwMemorySize') {
        $regVram[$p.DriverDesc] = [int64]$p.'HardwareInformation.qwMemorySize'
    }
}

# Schritt 3: DeviceId -> GPU-Namen Mapping über VideoProcessor erstellen
# VideoProcessor enthält z.B. "AMD Radeon Graphics Processor (0x7550)"
$devidToName = @{}
foreach ($a in $adapters) {
    $processor = ($a.VideoProcessor or "").Trim()
    if ($processor -match '\(0x([0-9a-fA-F]+)\)') {
        $devId = $matches[1].ToLower()
        $devidToName[$devId] = $a.Name.Trim()
    }
}

# Schritt 4: DedicatedUsage aus GPUAdapterPerformanceCounters holen mit Fallback
$counters = Get-CimInstance -Namespace root\cimv2 `
    -Query "SELECT Name, DedicatedUsage FROM Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory"

$results = @()
foreach ($cnt in $counters) {
    $counterName = ($cnt.Name -split '_phys')[0].Trim()
    $luidParts = $counterName -split '_'
    if ($luidParts.Count -lt 3) { continue }
    
    # LUID extrahieren – WICHTIG: den LETZTEN 0x-Teil nehmen!
    # Format: luid_0xXXXXXXXX_0xYYYYYYYY -> wir brauchen 0xYYYYYYYY
    $luidDev = ""
    for ($i = $luidParts.Count - 1; $i -ge 0; $i--) {
        $part = $luidParts[$i]
        if ($part -match '^0x([0-9a-fA-F]+)$') {
            $luidDev = $matches[1]
            break
        }
    }
    
    if (-not $luidDev) { continue }
    $luidDevLower = $luidDev.ToLower()
    
    # Versuche, GPU-Namen über LUID/DeviceId zu finden
    $ctrlName = $null
    
    # Methode 1: Direkter Match mit DeviceId
    if ($devidToName.ContainsKey($luidDevLower)) {
        $ctrlName = $devidToName[$luidDevLower]
    } else {
        # Methode 2: Vergleiche hex-Werte (LUID und DEV können unterschiedlich sein)
        try {
            $luidInt = [Convert]::ToInt64($luidDev, 16)
            foreach ($mapKey in $devidToName.Keys) {
                $devInt = [Convert]::ToInt64($mapKey, 16)
                if ($luidInt -eq $devInt) {
                    $ctrlName = $devidToName[$mapKey]
                    break
                }
            }
        } catch {}
    }
    
    # Fallback: Wenn kein LUID-Match, aber GPU mit hoher Usage vorhanden -> nimm sie
    if (-not $ctrlName) {
        $usageBytes = [int64]($cnt.DedicatedUsage -as [int64])
        if ($usageBytes -lt 0) { $usageBytes = 0 }
        $usedMB = $usageBytes / (1024 * 1024)
        # Wenn Usage > 100 MB, handelt es sich wahrscheinlich um die diskrete GPU
        if ($usedMB -gt 100) {
            foreach ($name in $devidToName.Values) {
                if ($name -like "*Radeon*" -or $name -like "*GeForce*" -or $name -like "*RTX*" -or $name -like "*GTX*") {
                    $ctrlName = $name
                    break
                }
            }
            # Wenn keine diskrete GPU im Match, nimm die erste nicht-virtuelle GPU
            if (-not $ctrlName -and $devidToName.Count -eq 1) {
                $ctrlName = $devidToName.Values[0]
            }
        }
    }
    
    if (-not $ctrlName) { continue }
    
    # Total VRAM aus Registry
    $vram = [int64]0
    if ($regVram.ContainsKey($ctrlName)) {
        $vram = $regVram[$ctrlName]
    }
    
    $usageBytes = [int64]($cnt.DedicatedUsage -as [int64])
    if ($usageBytes -lt 0) { $usageBytes = 0 }
    $totalMb = $vram / (1024 * 1024)
    $usedMb = $usageBytes / (1024 * 1024)
    $freeMb = [math]::Max(0, $totalMb - $usedMb)
    
    $results += [PSCustomObject]@{
        Name     = $ctrlName
        VRAM     = $vram
        FreeMB   = [int64]$freeMb
        UsedMB   = [int64]$usedMb
        PNP      = ""
    }
}

if ($results) {
    $results | ConvertTo-Json -Compress -Depth 3
} else {
    Write-Output "[]"
}
"""


def _detect_windows_gpus(skip_names: Optional[set] = None) -> List[GPUInfo]:
    """Enumerate every PCI video adapter on Windows via WMI + registry.

    Key fix vs the old code: reads ``HardwareInformation.qwMemorySize`` from
    the registry, which is 64-bit.  WMI's ``AdapterRAM`` is signed 32-bit and
    wraps on cards with > 4 GB of VRAM, so a 16 GB Radeon shows as 0 or
    garbage.

    Uses the native ``winreg``-based ``_get_vram_from_registry()`` helper when
    available (faster, no PowerShell overhead) and falls back to the PowerShell
    snippet for compatibility.

    VRAM free detection priority (updated for AMD RX 9000 / RDNA 5 compatibility):
      1. DedicatedUsage (WMI GPUAdapterMemory) → free = total - used
      2. DXGI/PowerShell (AMD RX 9000 Series fallback) → free = total - used
      3. AvailableVideoMemory (WMI VideoController) → direct free value
         (NOT used for AMD RX 9000 as it reports incorrectly ~4GB)
      4. Both unavailable → free_mb = 0 (unknown)

    ``skip_names`` is for de-duplicating against vendor-specific detectors
    (e.g. an RTX card already found via nvidia-smi shouldn't be re-added).
    """
    if platform.system() != "Windows":
        return []
    skip = {n.lower() for n in (skip_names or set())}

    # Echten belegten VRAM über WMI win32com auslesen (DedicatedUsage)
    vram_used_map: Dict[str, float] = _get_gpu_vram_used_via_wmi()
    # DXGI/PowerShell-Fallback für AMD RX 9000 Series (return: used_mb)
    dxgi_used_map: Dict[str, float] = _get_gpu_vram_via_dxgi_powershell()
    # Echten freien VRAM über WMI win32com auslesen (AvailableVideoMemory)
    # Wird nur als letzte Option verwendet, da bei AMD RX 9000 ungenau
    vram_free_map: Dict[str, float] = _get_gpu_vram_free_via_wmi()
    # GPU-Auslastung (nur NVIDIA über nvidia-smi, schnell)
    # WMI-basierte Utilization-Erkennung kann langsam sein und wird übersprungen
    gpu_util_map: Dict[str, float] = _get_nvidia_gpu_utilization()

    # Try native winreg helper first (faster, no PowerShell overhead)
    registry_vram = _get_vram_from_registry()
    gpus: List[GPUInfo] = []

    if registry_vram:
        for name, vram_bytes in registry_vram.items():
            if name.lower() in skip:
                continue
            total_mb = vram_bytes // (1024 * 1024)
            # VRAM-Berechnung mit Priorität (angepasst für AMD RX 9000):
            # 1. DedicatedUsage (WMI GPUAdapterMemory) → free = total - used
            # 2. DXGI/PowerShell-Fallback (AMD RX 9000 Series) → free = total - used
            # 3. AvailableVideoMemory (WMI VideoController) → direkte freie Menge
            # 4. Beides nicht verfügbar → free_mb = 0 (unbekannt)
            if name.lower() in vram_used_map:
                used_mb = vram_used_map[name.lower()]
                free_mb = max(0, total_mb - int(used_mb))
            elif name.lower() in dxgi_used_map:
                # DXGI/PowerShell-Fallback (AMD RX 9000 Series)
                used_mb = dxgi_used_map[name.lower()]
                free_mb = max(0, total_mb - int(used_mb))
            elif name.lower() in vram_free_map:
                # AvailableVideoMemory als letzte Option
                free_mb = int(min(vram_free_map[name.lower()], total_mb))
            else:
                free_mb = 0
            # GPU-Auslastung holen
            gpu_util = gpu_util_map.get(name.lower(), 0.0)
            gpus.append(
                GPUInfo(
                    index=len(gpus),
                    name=name,
                    vendor=_vendor_from_name(name),
                    total_vram_mb=total_mb,
                    free_vram_mb=free_mb,
                    gpu_util_percent=gpu_util,
                )
            )
        return gpus

    # Fallback: PowerShell-WMI-Ansatz
    out = _run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            _WIN_GPU_PS,
        ],
        timeout=12,
    )
    if not out:
        return []

    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]

    for d in data:
        name = (d.get("Name") or "").strip()
        if not name or name.lower() in skip:
            continue
        try:
            vram = int(d.get("VRAM") or 0)
        except (TypeError, ValueError):
            vram = 0
        if vram < 0:  # paranoia: 32-bit overflow
            vram = 0
        total_mb = vram // (1024 * 1024)

        # Verwende FreeMB aus PowerShell (berechnet als total - used) direkt,
        # da der PowerShell-Skript bereits DedicatedUsage korrekt verwendet
        ps_free = d.get("FreeMB")
        if ps_free is not None:
            try:
                free_mb = min(int(ps_free), total_mb)
            except (TypeError, ValueError):
                free_mb = 0
        # VRAM-Berechnung mit Priorität (wie oben)
        elif name.lower() in vram_used_map:
            used_mb = vram_used_map[name.lower()]
            free_mb = max(0, total_mb - int(used_mb))
        elif name.lower() in dxgi_used_map:
            used_mb = dxgi_used_map[name.lower()]
            free_mb = max(0, total_mb - int(used_mb))
        elif name.lower() in vram_free_map:
            free_mb = int(min(vram_free_map[name.lower()], total_mb))
        else:
            free_mb = 0
        gpus.append(
            GPUInfo(
                index=len(gpus),
                name=name,
                vendor=_vendor_from_name(name),
                total_vram_mb=total_mb,
                free_vram_mb=free_mb,
                gpu_util_percent=0.0,
            )
        )
    return gpus


def _detect_linux_other_gpus(skip_names: Optional[set] = None) -> List[GPUInfo]:
    """Linux: catch GPUs that nvidia-smi/rocm-smi missed (mainly Intel iGPUs).

    Uses lspci for naming. VRAM is unknown without vendor SDKs, so it stays 0
    and these GPUs end up filtered out when a real dGPU is also present.
    """
    if platform.system() != "Linux":
        return []
    skip = {n.lower() for n in (skip_names or set())}
    out = _run(["lspci", "-mm"])
    if not out:
        return []
    gpus: List[GPUInfo] = []
    for line in out.splitlines():
        if not re.search(r'"(VGA|3D|Display)', line):
            continue
        parts = [p.strip('"') for p in re.findall(r'"[^"]*"', line)]
        if len(parts) < 4:
            continue
        vendor_str = parts[1]
        name = parts[2]
        if name.lower() in skip:
            continue
        gpus.append(
            GPUInfo(
                index=len(gpus),
                name=f"{vendor_str} {name}".strip(),
                vendor=_vendor_from_name(f"{vendor_str} {name}"),
                total_vram_mb=0,
                free_vram_mb=0,
                gpu_util_percent=0.0,
            )
        )
    return gpus


def _detect_apple() -> List[GPUInfo]:
    if platform.system() != "Darwin":
        return []
    # Apple Silicon = unified memory; treat the whole RAM as addressable VRAM
    out = _run(["sysctl", "-n", "hw.memsize"])
    if not out:
        return []
    try:
        mem_b = int(out.strip())
    except ValueError:
        return []
    mem_mb = mem_b // (1024 * 1024)
    name_out = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or ""
    label = f"Apple Silicon ({name_out.strip()})" if name_out else "Apple Silicon"
    return [
        GPUInfo(
            index=0,
            name=label,
            vendor="apple",
            total_vram_mb=mem_mb,
            free_vram_mb=mem_mb,
            gpu_util_percent=0.0,
        )
    ]


def _detect_cpu_name() -> str:
    """Detect CPU name — best-effort, never raises."""
    try:
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except OSError:
                pass
        elif platform.system() == "Darwin":
            out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            if out:
                return out.strip()
        elif platform.system() == "Windows":
            # Fallback: zuerst environment variable prüfen (schnell, kein subprocess)
            env_cpu = os.environ.get("AUTOTUNER_CPU_NAME", "")
            if env_cpu:
                return env_cpu
            # PowerShell-Command mit Timeout-Schutz
            try:
                out = _run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        "(Get-CimInstance Win32_Processor).Name",
                    ],
                    timeout=5,
                )
                if out:
                    return out.strip()
            except Exception:
                pass
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


# ---------------------------------------------------------------------------
# Public API


def _filter_inference_gpus(
    gpus: List[GPUInfo],
) -> Tuple[List[GPUInfo], List[GPUInfo]]:
    """Split detected GPUs into (used for inference, ignored).

    Two-stage filter:

    Stage 1 — vendor gate:
      Intel iGPUs are moved to ignored whenever at least one non-Intel GPU
      exists. Intel iGPUs on Windows can report large shared-memory VRAM
      values (e.g. 27 GB on a system with 32 GB RAM) that would fool the
      VRAM-ratio check below and cause real discrete GPUs (e.g. a 16 GB
      RX 9070 XT next to a 32 GB R9700) to be wrongly excluded.

    Stage 2 — VRAM-ratio gate:
      Among the remaining (non-Intel) GPUs drop any that have less than
      one-third the VRAM of the largest card. This still catches tiny
      integrated or MXM GPUs (e.g. 2 GB) while correctly keeping a 16 GB
      card alongside a 32 GB card (16 × 3 = 48 ≥ 32 ✓).
      The old ½ threshold was too aggressive: 15.9 GB × 2 = 31.8 GB which
      is just below 32 GB, so the 9070 XT was incorrectly ignored.

    Also drops GPUs with 0 reported VRAM when at least one GPU has measured
    VRAM (those are usually iGPUs whose memory we couldn't read).
    """
    if len(gpus) < 2:
        return gpus, []

    measured = [g for g in gpus if g.total_vram_mb > 0]
    unmeasured = [g for g in gpus if g.total_vram_mb <= 0]

    # If we have at least one measured GPU, drop the unmeasured ones —
    # almost always iGPUs without registry VRAM info.
    if measured and unmeasured:
        kept_pool = measured
        ignored = list(unmeasured)
    else:
        kept_pool = list(gpus)
        ignored = []

    # Stage 1: always ignore Intel iGPUs when real discrete GPUs exist.
    # Must happen before the VRAM-ratio sort, because iGPUs report shared
    # system-RAM as "VRAM" and can appear larger than actual dGPUs.
    non_intel = [g for g in kept_pool if g.vendor != "intel"]
    intel_igpus = [g for g in kept_pool if g.vendor == "intel"]
    if non_intel:
        ignored.extend(intel_igpus)
        kept_pool = non_intel

    if len(kept_pool) < 2:
        return kept_pool, ignored

    # Stage 2: drop GPUs with less than 1/3 of the largest card's VRAM.
    sorted_g = sorted(kept_pool, key=lambda g: g.total_vram_mb, reverse=True)
    largest = sorted_g[0]
    used: List[GPUInfo] = [largest]
    for g in sorted_g[1:]:
        # Keep as a peer if it's at least one-third of the largest's VRAM.
        # Example: 9070 XT ~16 GB next to R9700 32 GB → 16×3=48 ≥ 32 ✓
        if g.total_vram_mb * 3 >= largest.total_vram_mb:
            used.append(g)
        else:
            ignored.append(g)
    return used, ignored


def detect_system() -> SystemInfo:
    """Detect everything in one call. Best-effort; never raises.

    Every sub-detection step is wrapped in try/except so that a failure
    in one component (e.g. nvidia-smi timeout) does not break the entire
    detection pipeline.
    """
    try:
        vm = psutil.virtual_memory()
    except Exception:
        vm = None  # will be handled below

    # --- GPU detection (each vendor independently protected) ---
    raw: List[GPUInfo] = []
    for detector in (_detect_nvidia, _detect_amd_rocm, _detect_apple):
        try:
            raw.extend(detector())
        except Exception:
            pass

    # OS-specific catch-all detectors fill in whatever the vendor-specific
    # ones missed (Windows: AMD without ROCm, Intel Arc; Linux: Intel iGPUs).
    found_names = {g.name.lower() for g in raw}
    for detector in (_detect_windows_gpus, _detect_linux_other_gpus):
        try:
            raw.extend(detector(skip_names=found_names))
            found_names = {g.name.lower() for g in raw}
        except Exception:
            pass

    # Re-index in detection order for stable display
    for i, g in enumerate(raw):
        g.index = i

    used, ignored = _filter_inference_gpus(raw)

    # --- CPU / RAM fallbacks ---
    os_name = f"{platform.system()} {platform.release()}"

    try:
        cpu_name = _detect_cpu_name()
    except Exception:
        cpu_name = "Unknown CPU"

    try:
        cpu_cores_physical = psutil.cpu_count(logical=False) or 1
    except Exception:
        cpu_cores_physical = 1

    try:
        cpu_cores_logical = psutil.cpu_count(logical=True) or 1
    except Exception:
        cpu_cores_logical = 1

    if vm is not None:
        total_ram_gb = vm.total / (1024**3)
        free_ram_gb = vm.available / (1024**3)
    else:
        total_ram_gb = 0.0
        free_ram_gb = 0.0

    return SystemInfo(
        os_name=os_name,
        cpu_name=cpu_name,
        cpu_cores_physical=cpu_cores_physical,
        cpu_cores_logical=cpu_cores_logical,
        total_ram_gb=total_ram_gb,
        free_ram_gb=free_ram_gb,
        gpus=used,
        ignored_gpus=ignored,
    )


def format_system(info: SystemInfo) -> str:
    """Human-readable summary, used for the menu header."""
    lines = [
        f"OS:   {info.os_name}",
        f"CPU:  {info.cpu_name} ({info.cpu_cores_physical}C/{info.cpu_cores_logical}T)",
        f"RAM:  {info.total_ram_gb:.1f} GB total, {info.free_ram_gb:.1f} GB free",
    ]
    if info.gpus:
        for g in info.gpus:
            tag = f"[{g.vendor}]"
            if g.total_vram_mb > 0:
                lines.append(
                    f"GPU{g.index}: {tag} {g.name} "
                    f"({g.total_vram_gb:.1f} GB total, "
                    f"{g.free_vram_gb:.1f} GB free)"
                )
            else:
                lines.append(f"GPU{g.index}: {tag} {g.name} (VRAM unknown)")
    else:
        lines.append("GPU:  none detected (CPU-only inference)")

    for g in info.ignored_gpus:
        size = f"{g.total_vram_gb:.1f} GB" if g.total_vram_mb > 0 else "VRAM unknown"
        lines.append(
            f"      (ignored: [{g.vendor}] {g.name}, {size} — too small or auxiliary)"
        )
    return "\n".join(lines)
