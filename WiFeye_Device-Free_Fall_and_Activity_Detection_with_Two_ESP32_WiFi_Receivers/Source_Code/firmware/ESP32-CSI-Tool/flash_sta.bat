@echo off
if not exist X:\ (
    subst X: "D:\5g project"
)
call "D:\Espressif\IDF\idf_cmd_init.bat" esp-idf-fc678c8c5dd9db319f6ca87eea8be9f4
cd /d X:\ESP32-CSI-Tool\active_sta
if "%~1"=="" (
    idf.py flash monitor
) else (
    idf.py -p %1 flash monitor
)

