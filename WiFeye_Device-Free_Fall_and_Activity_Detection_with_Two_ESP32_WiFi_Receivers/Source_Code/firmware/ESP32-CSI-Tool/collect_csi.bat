@echo off
if not exist X:\ (
    subst X: "D:\5g project"
)
call "D:\Espressif\IDF\idf_cmd_init.bat" esp-idf-fc678c8c5dd9db319f6ca87eea8be9f4

set "SUBPROJ=%~1"
if "%SUBPROJ%"=="" set "SUBPROJ=active_ap"

set "OUTFILE=%~2"
if "%OUTFILE%"=="" set "OUTFILE=csi_experiment_%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.csv"
set "OUTFILE=%OUTFILE: =0%"

cd /d "X:\ESP32-CSI-Tool\%SUBPROJ%"
echo ========================================================
echo Collecting CSI data from %SUBPROJ% to %OUTFILE%
echo Press Ctrl+] or Ctrl+C to stop.
echo ========================================================
idf.py monitor | findstr "CSI_DATA" > "%OUTFILE%"

