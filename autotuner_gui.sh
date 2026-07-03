#!/bin/bash

# Wechselt in das Verzeichnis, in dem dieses Skript liegt
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# LOKALE PFADE (MUSS UNTER UBUNTU ANGEPASST WERDEN!)
# ---------------------------------------------------------------------------
# AUTOTUNER_MODELS = Ordner mit *.gguf Modellen
# LLAMA_CPP_DIR    = Container-Ordner mit allen *_llama.cpp Builds
export AUTOTUNER_MODELS="/home/sebas/GitHub/Auto Tuner/models"
export LLAMA_CPP_DIR="/home/sebas/GitHub/Auto Tuner/llama_cpp"

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
"$PY" qt_launcher.py
