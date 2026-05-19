"""Qt log viewer for llama-server output.

Opens alongside a running ServerProcess and streams stdout/stderr into a
read-only text widget.  A system-tray icon provides Stop/Quit actions.
"""

from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QCloseEvent, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QSystemTrayIcon,
    QTextEdit,
)

from server_process import ServerProcess


class LogViewerWindow(QMainWindow):
    def __init__(self, server_process: ServerProcess) -> None:
        super().__init__()
        self.server_process = server_process

        self.setWindowTitle("AutoTuner — llama-server log")
        self.resize(900, 600)

        # ── Central log widget ──────────────────────────────────────────
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.setCentralWidget(self.log_edit)

        # ── System tray ─────────────────────────────────────────────────
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon())  # replace with real icon path if available

        self.tray_menu = QMenu()

        stop_action = QAction("Stop Server", self)
        stop_action.triggered.connect(self._stop_server)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self._quit)

        self.tray_menu.addAction(stop_action)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()  # <-- was missing in Nemotron's version

        # ── Polling timer ────────────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_logs)
        self._timer.start(200)  # 200 ms

    # ------------------------------------------------------------------
    def _poll_logs(self) -> None:
        """Append new log lines to the text widget and auto-scroll."""
        for line in self.server_process.get_logs():
            self.log_edit.append(line.rstrip("\n"))
        # verticalScrollBar() is typed as QScrollBar | None in PyQt6 stubs
        sb = self.log_edit.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def _stop_server(self) -> None:
        """Gracefully stop the server (window stays open to read the log)."""
        self._timer.stop()
        self.server_process.stop()
        self.log_edit.append("\n[AutoTuner] Server stopped.")

    def _quit(self) -> None:
        """Stop the server and exit the application."""
        self._stop_server()
        QApplication.quit()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Double-click tray icon → show/raise the window."""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show()
            self.raise_()
            self.activateWindow()

    def closeEvent(self, a0: QCloseEvent | None) -> None:  # noqa: N802
        """Window X button → stop server and quit cleanly."""
        self._stop_server()
        if a0 is not None:
            a0.accept()
        QApplication.quit()
