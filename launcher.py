"""Launch llama-server as a child process with proper Ctrl+C handling.

Two platform-specific tricks are needed to make Ctrl+C reliably stop
llama-server:

1. **Process-group isolation** so the child doesn't get our Ctrl+C from
   the console — we want to handle the signal ourselves and shut the
   child down in an orderly way.
   - Windows: ``CREATE_NEW_PROCESS_GROUP`` + ``CTRL_BREAK_EVENT``.
   - Unix:    ``start_new_session`` + ``SIGTERM`` to the process group.

2. **Polling wait instead of blocking wait.** This is the fix for the
   bug "Ctrl+C does nothing, I have to force-kill the window":

   On Windows, ``Popen.wait()`` with no timeout calls
   ``WaitForSingleObjectEx(handle, INFINITE, FALSE)``. The third
   argument ``FALSE`` means the wait is *not alertable*, so Python's
   SIGINT handler cannot run while we're inside it. The handler only
   queues the interrupt; it's only delivered between bytecodes — and
   no bytecode runs while we're blocked in that C-level call. Result:
   Ctrl+C appears to be ignored, and the user has to force-kill the
   terminal window. Linux/macOS aren't affected because there
   ``SIGINT`` interrupts the blocking syscall with ``EINTR``.

   The fix: wait with a short timeout in a loop. Every tick, control
   returns to the interpreter, the SIGINT flag is checked, and
   ``KeyboardInterrupt`` fires immediately.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import List, Optional

# How long to wait between SIGTERM/CTRL_BREAK and SIGKILL/TerminateProcess.
_GRACEFUL_TIMEOUT_SECONDS = 10

# Polling interval for proc.wait(). Must be short enough that Ctrl+C feels
# instant to the user, long enough to keep CPU usage negligible.
_POLL_INTERVAL_SECONDS = 0.25


def _is_windows() -> bool:
    return os.name == "nt"


def _spawn(cmd: List[str], env: Optional[dict] = None) -> subprocess.Popen:
    """Start the child process detached enough that we can signal its group."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    if _is_windows():
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        return subprocess.Popen(cmd, creationflags=flags, env=merged_env)
    return subprocess.Popen(cmd, start_new_session=True, env=merged_env)


def _terminate(proc: subprocess.Popen) -> None:
    """Politely ask the child (and its descendants) to exit."""
    if proc.poll() is not None:
        return
    try:
        if _is_windows():
            # CTRL_BREAK_EVENT goes to the new process group we created with
            # CREATE_NEW_PROCESS_GROUP. (CTRL_C_EVENT can't be targeted at a
            # specific group via GenerateConsoleCtrlEvent — it would also hit
            # us. Break-event hits only the child's group, which is what we
            # want. llama-server has no custom CTRL_BREAK handler, so the
            # Windows default handler terminates the process cleanly enough
            # that the OS reclaims VRAM/RAM.)
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            # Negative pid → send SIGTERM to the whole process group so any
            # children spawned by llama-server (rare, but possible) also die.
            os.kill(-proc.pid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass


def _force_kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if _is_windows():
            proc.kill()
        else:
            os.kill(-proc.pid, getattr(signal, "SIGKILL", 9))
    except (ProcessLookupError, OSError):
        pass


def _quote(arg: str) -> str:
    """Best-effort shell quoting for the command echo (display only)."""
    if not arg:
        return '""'
    if any(c in arg for c in (" ", "\t", '"', "'")):
        return '"' + arg.replace('"', '\\"') + '"'
    return arg


def _poll_wait(proc: subprocess.Popen, deadline: float) -> bool:
    """Wait for the child until ``deadline`` (monotonic time), polling.

    Returns True if the child exited, False on timeout. Re-raises
    ``KeyboardInterrupt`` so the caller can decide whether a second
    Ctrl+C means "escalate".
    """
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            proc.wait(timeout=min(_POLL_INTERVAL_SECONDS, max(0.01, remaining)))
            return True
        except subprocess.TimeoutExpired:
            continue
    return proc.poll() is not None


def _install_terminal_signal_handlers() -> dict:
    """Route SIGTERM (and SIGHUP on Unix) into KeyboardInterrupt.

    The child runs in its own session/process group so it deliberately does
    NOT receive the terminal's signals — WE are responsible for forwarding
    a shutdown. Without this, closing the terminal window (SIGHUP) or a
    ``kill <pid>`` (SIGTERM) killed only the Python wrapper and left
    llama-server running with the VRAM still allocated. Converting both
    into KeyboardInterrupt funnels every "please stop" into the one
    graceful-shutdown path in :func:`launch`.

    Returns the previous handlers so the caller can restore them.
    """

    def _raise_kbint(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    previous: dict = {}
    for name in ("SIGTERM", "SIGHUP"):  # SIGHUP doesn't exist on Windows
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            previous[sig] = signal.signal(sig, _raise_kbint)
        except (ValueError, OSError):
            pass  # non-main thread / unsupported — keep default behaviour
    return previous


def _restore_signal_handlers(previous: dict) -> None:
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def launch(cmd: List[str], env_overrides: Optional[dict] = None) -> int:
    """Run llama-server until it exits or until the user presses Ctrl+C.

    Returns the child's exit code (or 130 if it had to be killed).

    Guarantee: on EVERY exit path — normal child exit, Ctrl+C, SIGTERM,
    SIGHUP (terminal closed), or an unexpected exception — the child is
    either already dead or gets terminated (SIGTERM/CTRL_BREAK, escalating
    to SIGKILL). This function never leaves an orphaned llama-server
    holding VRAM.
    """
    print(
        f"\n[AutoTuner] Starting:\n  {' '.join(_quote(c) for c in cmd)}\n", flush=True
    )
    if env_overrides:
        for k, v in env_overrides.items():
            print(f"[AutoTuner] Env: {k}={v}")
    try:
        proc = _spawn(cmd, env=env_overrides)
    except FileNotFoundError:
        print(
            f"[AutoTuner] ERROR: server binary '{cmd[0]}' not found in PATH.\n"
            "  Install llama.cpp or pass --server /path/to/llama-server",
            file=sys.stderr,
        )
        return 127
    except OSError as exc:
        print(
            f"[AutoTuner] ERROR: server binary '{cmd[0]}' could not be started: {exc}\n"
            "  Check that the selected llama.cpp build matches this OS/CPU and is executable.",
            file=sys.stderr,
        )
        return 126

    print(f"[AutoTuner] llama-server PID: {proc.pid}")
    print(
        "[AutoTuner] Press Ctrl+C to stop the server (twice to force-kill).\n",
        flush=True,
    )

    prev_handlers = _install_terminal_signal_handlers()
    try:
        # ---- Main wait loop -----------------------------------------------
        # Polling — see the module docstring for why we can't use plain
        # proc.wait() on Windows.
        try:
            while True:
                try:
                    return proc.wait(timeout=_POLL_INTERVAL_SECONDS)
                except subprocess.TimeoutExpired:
                    continue
        except KeyboardInterrupt:
            pass

        # ---- Graceful shutdown ---------------------------------------------
        print("\n[AutoTuner] Stopping llama-server...", flush=True)
        _terminate(proc)

        deadline = time.monotonic() + _GRACEFUL_TIMEOUT_SECONDS
        try:
            if _poll_wait(proc, deadline):
                print("[AutoTuner] Stopped cleanly.")
                return proc.returncode if proc.returncode is not None else 0
        except KeyboardInterrupt:
            # Second Ctrl+C during the grace period — user wants out *now*.
            print("[AutoTuner] Second Ctrl+C — forcing kill.")
            _force_kill(proc)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            return 130

        print(
            f"[AutoTuner] Did not exit within "
            f"{_GRACEFUL_TIMEOUT_SECONDS}s — forcing kill."
        )
        _force_kill(proc)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        return 130
    finally:
        # Safety net for ANY exit path not covered above (unexpected
        # exception, a KeyboardInterrupt landing between statements, …):
        # if the child is somehow still alive, take it down before the
        # wrapper exits so no orphan keeps the VRAM allocated.
        if proc.poll() is None:
            _terminate(proc)
            try:
                if not _poll_wait(proc, time.monotonic() + 3):
                    _force_kill(proc)
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        pass
            except KeyboardInterrupt:
                _force_kill(proc)
        _restore_signal_handlers(prev_handlers)
        