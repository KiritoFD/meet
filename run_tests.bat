@echo off
set PYTHONPATH=%PYTHONPATH%;%CD%
"C:\Users\28961\anaconda3\envs\meet6\python.exe" -m pytest tests/connect/test_socket_manager.py -v 