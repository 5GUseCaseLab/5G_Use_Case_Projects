@echo off
set "PY_BIN=D:\Espressif\IDF\python_env\idf4.3_py3.11_env\Scripts\python.exe"
if not exist "%PY_BIN%" set "PY_BIN=C:\Espressif\IDF\python_env\idf4.3_py3.11_env\Scripts\python.exe"
if not exist "%PY_BIN%" set "PY_BIN=C:\Espressif\tools\python\v5.2.1\venv\Scripts\python.exe"
if not exist "%PY_BIN%" set "PY_BIN=python"

"%PY_BIN%" -c "import serial" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] pyserial is not installed in Python environment.
    echo [INFO] Installing pyserial automatically...
    "%PY_BIN%" -m pip install pyserial
)

"%PY_BIN%" "%~dp0dataset_collector.py" --mode master --port auto %*
pause

