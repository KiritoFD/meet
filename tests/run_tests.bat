@echo off
call conda activate meet
set PYTHONPATH=%PYTHONPATH%;C:\Users\xy\Github\meet
python -m unittest discover -v
