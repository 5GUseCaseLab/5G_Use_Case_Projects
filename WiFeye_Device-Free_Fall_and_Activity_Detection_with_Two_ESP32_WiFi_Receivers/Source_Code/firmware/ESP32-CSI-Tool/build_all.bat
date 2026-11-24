@echo off
if not exist X:\ (
    subst X: "D:\5g project"
)
call "D:\Espressif\IDF\idf_cmd_init.bat" esp-idf-fc678c8c5dd9db319f6ca87eea8be9f4

echo [1/3] Building active_sta...
cd /d X:\ESP32-CSI-Tool\active_sta
call idf.py build

echo [2/3] Building active_ap...
cd /d X:\ESP32-CSI-Tool\active_ap
call idf.py build

echo [3/3] Building passive...
cd /d X:\ESP32-CSI-Tool\passive
call idf.py build

echo =======================================================
echo All ESP32 CSI Tool projects built successfully!
echo =======================================================
cd /d X:\ESP32-CSI-Tool

