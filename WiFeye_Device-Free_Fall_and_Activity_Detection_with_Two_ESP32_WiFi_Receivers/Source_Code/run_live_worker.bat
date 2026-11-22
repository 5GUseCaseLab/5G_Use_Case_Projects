@echo off
REM WiFeye live demo - Laptop B (Worker): reads ESP32 RX2 on USB and relays it to Laptop A.
REM Network: connect BOTH laptops to the router's Wi-Fi "CSI" (same as during data collection).
REM The master's address is learned automatically from its sync messages - no IP needed.
REM Start this BEFORE run_live_master.bat on Laptop A.
cd /d "%~dp0"
REM same Python search order as run_collector_worker.bat (the worker laptop used for data collection)
set "PY=C:\Espressif\tools\python\v5.2.1\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=D:\Espressif\IDF\python_env\idf4.3_py3.11_env\Scripts\python.exe"
if not exist "%PY%" set "PY=C:\Espressif\IDF\python_env\idf4.3_py3.11_env\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" live_demo_engine.py --role worker --port auto
pause
