@echo off
if not exist X:\ (
    subst X: "D:\5g project"
)
call "D:\Espressif\IDF\idf_cmd_init.bat" esp-idf-fc678c8c5dd9db319f6ca87eea8be9f4
cd /d X:\ESP32-CSI-Tool
echo =======================================================
echo ESP32 CSI Tool Environment Activated!
echo Workspace mapped to X:\ESP32-CSI-Tool
echo Subprojects available:
echo   - active_sta (Active CSI collection - Station)
echo   - active_ap  (Active CSI collection - Access Point)
echo   - passive    (Passive CSI collection)
echo =======================================================
cmd.exe /k

