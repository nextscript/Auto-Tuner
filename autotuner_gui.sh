#!/bin/bash

# Wechselt in das Verzeichnis, in dem dieses Skript liegt
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# GUI-Session robust vorbereiten (Ubuntu/Wayland/X11)
# ---------------------------------------------------------------------------
# Wenn der Launcher aus einem Dateimanager-Terminal kommt, sind Cache- oder
# Wayland-Variablen manchmal unbrauchbar. Qt soll dann nicht an einem kaputten
# Wayland-Socket hängen bleiben, und Fontconfig braucht einen schreibbaren Cache.
CACHE_BASE="${XDG_CACHE_HOME:-$HOME/.cache}"
if ! mkdir -p "$CACHE_BASE/fontconfig" 2>/dev/null || ! touch "$CACHE_BASE/.autotuner-write-test" 2>/dev/null; then
    CACHE_BASE="/tmp/autotuner-${USER:-user}/cache"
    mkdir -p "$CACHE_BASE/fontconfig" 2>/dev/null || true
    export XDG_CACHE_HOME="$CACHE_BASE"
else
    rm -f "$CACHE_BASE/.autotuner-write-test" 2>/dev/null || true
    export XDG_CACHE_HOME="$CACHE_BASE"
fi

if [ -n "${WAYLAND_DISPLAY:-}" ]; then
    if printf '%s' "$WAYLAND_DISPLAY" | grep -q '^/'; then
        WAYLAND_SOCKET="$WAYLAND_DISPLAY"
    else
        WAYLAND_SOCKET="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/$WAYLAND_DISPLAY"
    fi
    if [ ! -S "$WAYLAND_SOCKET" ]; then
        unset WAYLAND_DISPLAY
    fi
fi

if [ -z "${QT_QPA_PLATFORM:-}" ]; then
    if [ -n "${WAYLAND_DISPLAY:-}" ]; then
        export QT_QPA_PLATFORM="wayland;xcb"
    elif [ -n "${DISPLAY:-}" ]; then
        export QT_QPA_PLATFORM="xcb"
    fi
fi

# ---------------------------------------------------------------------------
# LOKALE PFADE (Linux/macOS)
# ---------------------------------------------------------------------------
# AUTOTUNER_MODELS = Ordner mit *.gguf Modellen
# LLAMA_CPP_DIR    = Container-Ordner mit allen *_llama.cpp Builds
# Bereits gesetzte Umgebungsvariablen haben Vorrang. Dadurch bleibt das
# Script portabel und überschreibt keine benutzerspezifischen Pfade.
: "${AUTOTUNER_MODELS:=$SCRIPT_DIR/models}"
export AUTOTUNER_MODELS

if [ -z "${LLAMA_CPP_DIR:-}" ]; then
    for cand in \
        "$SCRIPT_DIR/llama_cpp" \
        "$SCRIPT_DIR/llama.cpp" \
        "$SCRIPT_DIR/../ai-local" \
        "$SCRIPT_DIR/../LAB/ai-local" \
        "$HOME/ai-local" \
        "$HOME/LAB/ai-local"; do
        if [ -d "$cand" ]; then
            export LLAMA_CPP_DIR="$cand"
            break
        fi
    done
fi

# ---------------------------------------------------------------------------
# Python-Interpreter bevorzugen:
#   Linux/macOS: .venv_linux
#   Windows:     .venv (wird von den .bat-Startern genutzt)
# Falls jemand unter Linux bewusst eine POSIX-.venv angelegt hat, akzeptieren
# wir sie als Fallback. Die Windows-.venv in diesem Repo hat nur Scripts/ und
# wird dadurch unter Linux nicht versehentlich verwendet.
# ---------------------------------------------------------------------------
if [ -f "./.venv_linux/bin/python" ]; then
    PY="./.venv_linux/bin/python"
elif [ -f "./.venv/bin/python" ]; then
    PY="./.venv/bin/python"
else
    echo "[WARN] .venv_linux nicht gefunden - nutze globales python3."
    echo "       Ubuntu/Linux-venv anlegen: python3 -m venv .venv_linux && .venv_linux/bin/python -m pip install -r requirements.txt"
    PY="python3"
fi

# Auto-Tuner GUI starten
if [ "${AUTOTUNER_FOREGROUND:-0}" = "1" ]; then
    "$PY" qt_launcher.py
else
    # Wir starten die App direkt. Die .desktop-Datei (Terminal=false)
    # kümmert sich unter Linux darum, dass kein Terminal aufpoppt.
    # Ein nohup/setsid hier würde die GUI-Session-Variablen (XAUTHORITY)
    # zerstören und zum Absturz führen.
    "$PY" qt_launcher.py
fi
