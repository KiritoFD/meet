@echo off

:: 检查是否已安装 Conda
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Conda未安装，正在下载安装程序...
    :: 下载最新版 Miniconda
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    
    :: 静默安装
    start /wait "" Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Miniconda3
    
    :: 删除安装程序
    del Miniconda3-latest-Windows-x86_64.exe
    
    echo Conda安装完成！
)

:: 创建conda环境
call conda env create -f meet.yaml

:: 激活环境
call conda activate meet

echo 环境安装完成！
pause 