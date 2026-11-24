@echo off
if not exist X:\ (
    subst X: "D:\5g project"
)
call "D:\Espressif\IDF\idf_cmd_init.bat" esp-idf-fc678c8c5dd9db319f6ca87eea8be9f4

set "SUBPROJ=%~1"
if "%SUBPROJ%"=="" (
    echo Please specify the subproject:
    echo   menuconfig.bat active_sta
    echo   menuconfig.bat active_ap
    echo   menuconfig.bat passive
    exit /b 1
)

cd /d "X:\ESP32-CSI-Tool\%SUBPROJ%"
idf.py menuconfig

