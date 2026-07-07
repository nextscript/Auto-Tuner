"""Single source of truth for the AutoTuner version + update endpoints.

Imported by:
  * the GUI — for the update check (compare running version to the latest
    GitHub Release tag) and the About dialog;
  * ``build_exe.py`` — so the built artifact and the version-comparison
    logic can never drift apart.

Bump :data:`VERSION` for every GitHub Release. The release tag MUST be
``v<VERSION>`` (e.g. ``VERSION = "1.2.0"`` → git tag ``v1.2.0``); the
binary updater strips a leading ``v`` before comparing.

The AutoTuner runs on Windows 10/11 (compiled ``.exe``) and on Ubuntu
(source install, or an optional compiled Linux binary). The updater
publishes OS-specific assets and picks the right one at runtime — see
``qt_launcher._BinaryUpdateWorker``.
"""

from __future__ import annotations

#: Semantic version of the running build. Compared against GitHub Release
#: ``tag_name`` (with a leading ``v`` stripped).
VERSION = "4.8.4"

#: GitHub repository used for BOTH the source-ZIP updater (dev installs)
#: and the binary release assets (compiled .exe / Linux binary).
GITHUB_REPO = "DaWasteh/Auto-Tuner"

#: User-Agent / X-GitHub-Api-Version header identity for the update worker.
USER_AGENT = "AutoTuner-updater"
