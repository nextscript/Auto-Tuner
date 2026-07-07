"""ServerProcess – wrapper around subprocess.Popen with log capture and graceful control."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import queue
import time
from typing import List, Optional


class ServerProcess:
    """Manage a llama-server subprocess, capture stdout/stderr, and provide clean start/stop."""

    def __init__(self, cmd: List[str], env_overrides: Optional[dict] = None) -> None:
        self.cmd = cmd
        self.env_overrides: dict = env_overrides or {}
        self.proc: Optional[subprocess.Popen] = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _reader(self) -> None:
        """Read stdout and stderr line‑by‑line and put them into the log queue."""
        assert self.proc is not None
        assert not self._stop_event.is_set()
        try:
            # stdout includes stderr (Popen uses stderr=STDOUT). Reading a
            # single pipe avoids the classic deadlock where stderr fills while
            # stdout is still open.
            if self.proc.stdout is not None:
                for line in self.proc.stdout:  # type: ignore[attr-defined]
                    self.log_queue.put(line)
        except Exception:
            pass
        finally:
            self._stop_event.set()

    def start(self) -> None:
        """Spawn the subprocess and start the log‑reading thread."""
        if self.proc is not None:
            return  # already running

        env = os.environ.copy()
        if self.env_overrides:
            env.update(self.env_overrides)

        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP
            self.proc = subprocess.Popen(
                self.cmd,
                creationflags=flags,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        else:
            self.proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
                env=env,
            )

        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()

    def stop(self) -> None:
        """Gracefully ask the process to exit, then force kill if needed."""
        if self.proc is None:
            return

        # Send termination signal appropriate for platform
        try:
            if os.name == "nt":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # Send SIGTERM to the whole process group (negative pid)
                os.kill(-self.proc.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

        # Wait a bit for graceful exit
        deadline = time.monotonic() + 10  # 10 s grace period
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return
            time.sleep(0.1)

        # Force kill if still alive
        try:
            if os.name == "nt":
                self.proc.kill()
            else:
                os.kill(-self.proc.pid, 9)  # SIGKILL
        except (ProcessLookupError, OSError):
            pass

        # Ensure the reader thread stops
        self._stop_event.set()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=3)

        self.proc = None

    def wait(self, timeout: Optional[float] = None) -> int:
        """Block until the process exits (or timeout) and return its exit code."""
        if self.proc is None:
            return 0
        return self.proc.wait(timeout=timeout)

    def get_logs(self) -> List[str]:
        """Retrieve all log lines that have been queued so far."""
        logs: List[str] = []
        while not self.log_queue.empty():
            logs.append(self.log_queue.get_nowait())
        return logs
