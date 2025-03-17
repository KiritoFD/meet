@ECHO OFF
@SET PYTHONIOENCODING=utf-8
@SET PYTHONUTF8=1
@FOR /F "tokens=2 delims=:." %%A in ('chcp') do for %%B in (%%A) do set "_CONDA_OLD_CHCP=%%B"
@chcp 65001 > NUL
@CALL "C:\Users\28961\anaconda3\condabin\conda.bat" activate "C:\Users\28961\anaconda3\envs\meet1"
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@C:\Users\28961\anaconda3\envs\meet1\python.exe -m pip install -U -r C:\Users\28961\Documents\GitHub\meet\envs\condaenv.7_ii9eep.requirements.txt --exists-action=b
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@chcp %_CONDA_OLD_CHCP%>NUL
