@echo off
REM WiFeye fallback demo without hardware, with the browser dashboard (http://127.0.0.1:8050/).
REM   run_replay_demo.bat            one recorded trial per activity, real time, with ground truth
REM   run_replay_demo.bat loso 4     honest replay of subject 4 (model retrained without them, ~1 min first)
REM   run_replay_demo.bat trial 1_fall_t06   one specific recorded trial
cd /d "%~dp0"
set PY=D:\Espressif\IDF\python_env\idf4.3_py3.11_env\Scripts\python.exe
if not exist "%PY%" set PY=python
if /i "%~1"=="loso" (
    "%PY%" live_demo_engine.py --loso-subject %2 --ui
) else if /i "%~1"=="trial" (
    "%PY%" live_demo_engine.py --replay %2 --ui
) else (
    "%PY%" live_demo_engine.py --replay-demo --ui
)
pause
