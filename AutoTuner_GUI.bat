@echo off
REM Startet den AutoTuner Qt-Launcher ohne sichtbares Terminal-Fenster.
REM Nutzt das projekteigene .venv (pythonw.exe = kein Konsolenfenster).

cd /d "%~dp0"

REM -- Pfade anpassen (gleiche Einstellungen wie AutoTuner_Terminal.bat) -------
set "AUTOTUNER_MODELS=C:\LAB\ai-local\models"
set "LLAMA_CPP_DIR=C:\LAB\ai-local\llama.cpp"

REM -- venv-Interpreter bevorzugen, sonst auf globales pythonw zurueckfallen --
set "PYW=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%PYW%" (
    echo [WARN] .venv nicht gefunden - nutze globales pythonw.
    echo        venv anlegen:  py -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -r requirements.txt
    set "PYW=pythonw"
)

REM start "" entkoppelt den Prozess vom aktuellen cmd-Fenster.
start "" "%PYW%" "%~dp0qt_launcher.py" %*