#!/usr/bin/env python3
"""Build a standalone, noconsole AutoTuner binary with PyInstaller.

Produces a **one-file** artifact that beginners can run without a Python
install or a console window:

  * Windows  → ``dist/AutoTuner.exe``      (``--windowed`` = no console)
  * Linux    → ``dist/AutoTuner-Linux``     (ELF, no terminal)

PyInstaller cannot cross-compile — build the Windows ``.exe`` ON Windows
and the Linux binary ON Linux. Publish BOTH as assets of the same GitHub
Release (tag ``v<VERSION>`` from ``autotuner_version.py``); the in-app
updater picks the matching asset for the host OS at runtime
(``_BinaryUpdateWorker``).

Usage
-----
    # one-time, in the build environment:
    python -m pip install pyinstaller
    python -m pip install -r requirements.txt

    # then:
    python build_exe.py

The script cleans ``build/`` and ``dist/`` first and refuses to run if
PyInstaller is missing. Settings profiles (``settings/*.yaml``) are
bundled as read-only data; user state stays in ``app_settings.app_data_dir``
(next to the binary) and is preserved across updates.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Single source of truth for the version that the updater compares against.
from autotuner_version import VERSION

REPO_ROOT = Path(__file__).resolve().parent
ENTRY = REPO_ROOT / "qt_launcher.py"
SETTINGS_DIR = REPO_ROOT / "settings"
DIST = REPO_ROOT / "dist"
BUILD = REPO_ROOT / "build"


def _exe_name() -> str:
    # Windows → AutoTuner.exe ; Linux → AutoTuner-Linux (no extension).
    return "AutoTuner" if os.name == "nt" else "AutoTuner-Linux"


def _ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print(
            "[build] PyInstaller is not installed.\n"
            "        Run:  python -m pip install pyinstaller",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _clean() -> None:
    for d in (BUILD, DIST):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    spec = REPO_ROOT / f"{_exe_name()}.spec"
    if spec.exists():
        try:
            spec.unlink()
        except OSError:
            pass


def main() -> int:
    _ensure_pyinstaller()

    if not SETTINGS_DIR.is_dir():
        print(f"[build] settings/ not found at {SETTINGS_DIR}", file=sys.stderr)
        return 1

    print(f"[build] AutoTuner v{VERSION} on {platform.system()} ({platform.machine()})")
    _clean()

    # PyInstaller's --add-data separator is OS-specific (`;` on Windows,
    # `:` on POSIX). Bundling settings/ makes the read-only YAML profiles
    # available inside the frozen _MEIPASS folder.
    sep = ";" if os.name == "nt" else ":"
    add_data_settings = f"{SETTINGS_DIR}{sep}settings"

    # Local modules that are only imported lazily (inside functions) and
    # that PyInstaller's static analysis can occasionally miss. Listing
    # them as hidden imports is cheap insurance.
    hidden_imports = [
        "auto_tuner",
        "launcher",
        "server_process",
        "diagnostics",
        "get_metadata",
        "qt_log_viewer",
        "performance_target",
        "settings_loader",
        "scanner",
        "hardware",
        "tuner",
        "app_settings",
        "autotuner_version",
    ]

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--windowed",          # noconsole: no terminal on Windows, none on Linux
        "--name",
        _exe_name(),
        "--noconfirm",
        "--clean",
        "--distpath",
        str(DIST),
        "--workpath",
        str(BUILD),
        "--add-data",
        add_data_settings,
        "--paths",
        str(REPO_ROOT),
    ]
    for mod in hidden_imports:
        cmd += ["--hidden-import", mod]
    cmd.append(str(ENTRY))

    print("[build] $ " + " ".join(cmd))
    try:
        subprocess.check_call(cmd, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError as exc:
        print(f"[build] PyInstaller failed: {exc}", file=sys.stderr)
        return exc.returncode or 1

    out = DIST / (_exe_name() + (".exe" if os.name == "nt" else ""))
    if out.exists():
        size_mb = out.stat().st_size / (1024 * 1024)
        print(
            f"\n[build] OK — {out} ({size_mb:.1f} MB)\n"
            f"[build] Publish this as a GitHub Release asset tagged v{VERSION}.\n"
            "[build] The in-app updater will offer it to users on "
            f"{platform.system()}."
        )
        return 0

    print(f"[build] Expected output not found: {out}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
