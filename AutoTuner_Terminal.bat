@echo off
REM Wechselt ins Verzeichnis dieser .bat-Datei (egal von wo aus gestartet)
cd /d "%~dp0"

REM ---------------------------------------------------------------------------
REM Lokale Pfade (nur in diesem Prozess gesetzt, nicht systemweit).
REM Wenn du das Repo bewegst oder die Modelle/llama.cpp woanders liegen,
REM hier anpassen.
REM ---------------------------------------------------------------------------
set "AUTOTUNER_MODELS=C:\LAB\ai-local\models"
set "LLAMA_CPP_DIR=C:\LAB\ai-local\llama.cpp"

REM -- venv-Interpreter bevorzugen, sonst auf globales python zurueckfallen ----
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [WARN] .venv nicht gefunden - nutze globales python.
    echo        venv anlegen:  py -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -r requirements.txt
    set "PY=python"
)

REM Auto-Tuner starten und alle uebergebenen Argumente weiterreichen.
"%PY%" auto_tuner.py %*

REM Fenster offen halten, falls ein Fehler kommt - sonst klappt es zu
pause