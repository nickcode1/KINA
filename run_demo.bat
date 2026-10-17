@echo off
REM One-click demo: scripted fork grasp (saves outputs\scripted_motion.mp4), then the interactive viewer.
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
REM  ^ falls back to the active python (e.g. `conda activate robo`) when there is no .venv
echo [run_demo] 1/2 recording scripted fork grasp video ...
"%PY%" scripts\record_video.py
if errorlevel 1 goto :error
echo [run_demo] video saved in outputs\scripted_motion.mp4
echo [run_demo] 2/2 opening the MuJoCo viewer (close the window to exit) ...
"%PY%" scripts\view.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo [run_demo] FAILED (exit code %errorlevel%) - see the messages above.
pause
exit /b 1
