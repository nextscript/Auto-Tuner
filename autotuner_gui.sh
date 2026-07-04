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
# Python-Interpreter bevorzugen (venv), sonst globales python3
# ---------------------------------------------------------------------------
if [ -f "./.venv/bin/python" ]; then
    PY="./.venv/bin/python"
else
    echo "[WARN] .venv nicht gefunden - nutze globales python3."
    echo "       Um venv anzulegen: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    PY="python3"
fi

# Auto-Tuner GUI starten
# Standard: vom ggf. geöffneten Dateimanager-Terminal entkoppeln, damit ein
# versehentlich geschlossenes Terminal die GUI nicht beendet. Für Debugging:
#   AUTOTUNER_FOREGROUND=1 ./autotuner_gui.sh
if [ "${AUTOTUNER_FOREGROUND:-0}" = "1" ]; then
    "$PY" qt_launcher.py
else
    LOG_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/autotuner"
    if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
        LOG_DIR="/tmp/autotuner-${USER:-user}"
        mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR="/tmp"
    fi
    LOG_FILE="$LOG_DIR/gui.log"

    # Keep the normal graphical user session intact. `setsid` can break Qt's
    # Wayland/X11 connection on some Ubuntu desktops; `nohup` is enough to
    # survive a terminal close while preserving the GUI environment.
    PYTHONUNBUFFERED=1 nohup "$PY" qt_launcher.py >"$LOG_FILE" 2>&1 < /dev/null &
    exit 0
fi
