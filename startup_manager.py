"""Cross-platform login autostart integration for the AutoTuner GUI.

The registration is user-local and disabled unless the user explicitly enables
it in the GUI:

* Windows: ``HKCU\\...\\Run``
* Linux: ``~/.config/autostart/AutoTuner.desktop``
* macOS: ``~/Library/LaunchAgents/com.dawasteh.autotuner.plist``
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path
from typing import List

APP_NAME = "AutoTuner"
_WINDOWS_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_MACOS_LABEL = "com.dawasteh.autotuner"


class AutostartError(RuntimeError):
    """Raised when the host's login-autostart registration cannot be changed."""


def platform_name() -> str:
    """Return the friendly name of the current supported desktop platform."""
    if sys.platform == "win32":
        return "Windows"
    if sys.platform == "linux":
        return "Linux"
    if sys.platform == "darwin":
        return "macOS"
    return sys.platform or "Unknown OS"


def launch_arguments() -> List[str]:
    """Return the executable and arguments needed to reopen this installation."""
    if getattr(sys, "frozen", False):
        return [str(Path(sys.executable).resolve())]
    launcher = Path(__file__).resolve().parent / "qt_launcher.py"
    return [str(Path(sys.executable).resolve()), str(launcher)]


def is_autostart_enabled() -> bool:
    """Return whether AutoTuner is registered to start at user login."""
    try:
        if sys.platform == "win32":
            return _windows_is_enabled()
        if sys.platform == "linux":
            return _linux_autostart_path().is_file()
        if sys.platform == "darwin":
            return _macos_launch_agent_path().is_file()
    except OSError:
        return False
    return False


def set_autostart_enabled(enabled: bool) -> None:
    """Enable or disable user-login autostart on the current platform."""
    try:
        if sys.platform == "win32":
            _set_windows_autostart(enabled)
        elif sys.platform == "linux":
            _set_linux_autostart(enabled)
        elif sys.platform == "darwin":
            _set_macos_autostart(enabled)
        else:
            raise AutostartError(f"Autostart is not supported on {platform_name()}.")
    except AutostartError:
        raise
    except (OSError, ValueError) as exc:
        action = "enable" if enabled else "disable"
        raise AutostartError(
            f"Could not {action} autostart on {platform_name()}: {exc}"
        ) from exc


def _windows_is_enabled() -> bool:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _WINDOWS_RUN_KEY) as key:
            winreg.QueryValueEx(key, APP_NAME)
            return True
    except FileNotFoundError:
        return False


def _set_windows_autostart(enabled: bool) -> None:
    import winreg

    if enabled:
        command = subprocess.list2cmdline(launch_arguments())
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _WINDOWS_RUN_KEY) as key:
            winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, command)
        return

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            _WINDOWS_RUN_KEY,
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            winreg.DeleteValue(key, APP_NAME)
    except FileNotFoundError:
        pass


def _linux_autostart_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "autostart" / f"{APP_NAME}.desktop"


def _desktop_exec_quote(value: str) -> str:
    """Quote one freedesktop Exec argument without invoking a shell."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("`", "\\`")
        .replace("$", "\\$")
        .replace("%", "%%")
    )
    return f'"{escaped}"'


def _set_linux_autostart(enabled: bool) -> None:
    path = _linux_autostart_path()
    if not enabled:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    command = " ".join(_desktop_exec_quote(arg) for arg in launch_arguments())
    content = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Version=1.0\n"
        f"Name={APP_NAME}\n"
        "Comment=Start AutoTuner automatically after login\n"
        f"Exec={command}\n"
        "Terminal=false\n"
        "X-GNOME-Autostart-enabled=true\n"
    )
    path.write_text(content, encoding="utf-8")


def _macos_launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_MACOS_LABEL}.plist"


def _set_macos_autostart(enabled: bool) -> None:
    path = _macos_launch_agent_path()
    if not enabled:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Label": _MACOS_LABEL,
        "ProgramArguments": launch_arguments(),
        "RunAtLoad": True,
    }
    with path.open("wb") as fh:
        plistlib.dump(payload, fh, sort_keys=True)
