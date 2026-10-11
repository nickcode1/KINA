@echo off
REM One-click keyboard teleoperation of both arms with demo recording (episodes -> data\demos\*.hdf5).
REM Extra arguments are passed through, e.g.  teleop.bat --pos-speed 0.5
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
REM  ^ falls back to the active python (e.g. `conda activate robo`) when there is no .venv
"%PY%" -c "import pynput" 2>nul || "%PY%" -m pip install pynput==1.8.2
"%PY%" scripts\teleop.py --record %*
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo [teleop] FAILED (exit code %errorlevel%) - see the messages above.
pause
exit /b 1
