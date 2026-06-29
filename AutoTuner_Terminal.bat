@echo off
REM Wechselt ins Verzeichnis dieser .bat-Datei (egal von wo aus gestartet)
cd /d "%~dp0"

REM ---------------------------------------------------------------------------
REM Lokale Pfade (nur in diesem Prozess gesetzt, nicht systemweit).
REM Wenn du das Repo bewegst oder die Modelle/llama.cpp woanders liegen,
REM hier anpassen.
REM AUTOTUNER_MODELS = Ordner mit *.gguf Modellen
REM LLAMA_CPP_DIR    = Container-Ordner mit allen *_llama.cpp Builds (NICHT
REM                    ein einzelner Fork) - der AutoTuner findet darin alle
REM                    Forks automatisch. Auf dieser Maschine: H:\LAB\ai-local
REM                    (C:\LAB ist veraltet und existiert nicht mehr).
REM ---------------------------------------------------------------------------
set "AUTOTUNER_MODELS=H:\LAB\ai-local\models"
set "LLAMA_CPP_DIR=H:\LAB\ai-local"

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