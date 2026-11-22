@echo off
if not exist X:\ (
    subst X: "D:\5g project"
)
call "D:\Espressif\IDF\idf_cmd_init.bat" esp-idf-fc678c8c5dd9db319f6ca87eea8be9f4

set "SUBPROJ=%~1"
if "%SUBPROJ%"=="" set "SUBPROJ=active_ap"

cd /d "X:\ESP32-CSI-Tool\%SUBPROJ%"
echo ========================================================
echo Starting Real-Time CSI Amplitude Plotting for %SUBPROJ%
echo Press Ctrl+] or close plot window to stop.
echo ========================================================
idf.py monitor | python ../python_utils/serial_plot_csi_live.py

