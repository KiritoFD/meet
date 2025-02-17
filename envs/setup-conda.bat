@echo off
echo Creating new conda environment...
call conda create -n meet-face python=3.9 -y
call conda activate meet-face
call conda install -c conda-forge dlib -y
pip install face_recognition
pip install -r requirements.txt
echo Installation complete.
pause
