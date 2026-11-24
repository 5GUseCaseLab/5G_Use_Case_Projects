@echo off
REM WiFeye live demo - Laptop A (Master): ESP32 RX1 on USB + RX2 relayed by Laptop B.
REM Network: connect BOTH laptops to the router's Wi-Fi "CSI" (same as during data collection).
REM The worker is found automatically (UDP broadcast); start run_live_worker.bat on Laptop B first.
REM Usage: run_live_master.bat            two ESP32s (needs Laptop B running)
REM        run_live_master.bat single     one ESP32 (RX1 only)
REM The first 12 seconds are a warm-up while the 10 s buffer fills (the room need not be empty).
REM A browser dashboard opens automatically (http://127.0.0.1:8050/); press F11 there for full screen.
cd /d "%~dp0"
set PY=D:\Espressif\IDF\python_env\idf4.3_py3.11_env\Scripts\python.exe
if not exist "%PY%" set PY=python
if /i "%~1"=="single" (
    "%PY%" live_demo_engine.py --role master --port auto --calibrate 12 --ui
) else (
    "%PY%" live_demo_engine.py --role master --port auto --dual --calibrate 12 --ui
)
pause
