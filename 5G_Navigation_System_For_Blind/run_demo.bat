@echo off
cd /d %~dp0
call .venv\Scripts\activate
python edge_server\server.py
pause
