@echo off
echo Installing prerequisites...

REM Install pre-built dlib wheel
pip install https://github.com/jloh02/dlib/releases/download/v19.24/dlib-19.24.0-cp312-cp312-win_amd64.whl

REM Install other requirements
pip install -r requirements.txt
echo Installation complete.
pause
