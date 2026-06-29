@echo off
REM Startet den AutoTuner Qt-Launcher ohne sichtbares Terminal-Fenster.
REM Nutzt das projekteigene .venv (pythonw.exe = kein Konsolenfenster).

cd /d "%~dp0"

REM -- Pfade anpassen (gleiche Einstellungen wie AutoTuner_Terminal.bat) -------
REM AUTOTUNER_MODELS = Ordner mit *.gguf Modellen
REM LLAMA_CPP_DIR    = Container-Ordner mit allen *_llama.cpp Builds (NICHT
REM                    ein einzelner Fork) - der AutoTuner findet darin alle
REM                    Forks automatisch. Auf dieser Maschine: H:\LAB\ai-local
REM                    (C:\LAB ist veraltet und existiert nicht mehr).
set "AUTOTUNER_MODELS=H:\LAB\ai-local\models"
set "LLAMA_CPP_DIR=H:\LAB\ai-local"

REM -- venv-Interpreter bevorzugen, sonst auf globales pythonw zurueckfallen --
set "PYW=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%PYW%" (
    echo [WARN] .venv nicht gefunden - nutze globales pythonw.
    echo        venv anlegen:  py -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -r requirements.txt
    set "PYW=pythonw"
)

REM start "" entkoppelt den Prozess vom aktuellen cmd-Fenster.
start "" "%PYW%" "%~dp0qt_launcher.py" %*