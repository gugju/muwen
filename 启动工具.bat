@echo off
rem ============================================================
rem  YOLO数据集工具箱 启动器
rem  自动探测 python（优先 yolo conda 环境），双击即用
rem ============================================================
chcp 65001 >nul
setlocal enabledelayedexpansion

set "PY_EXE="
for %%P in (
    "%USERPROFILE%\anaconda3\envs\yolo\python.exe"
    "%USERPROFILE%\miniconda3\envs\yolo\python.exe"
    "%ProgramData%\anaconda3\envs\yolo\python.exe"
    "%ProgramData%\miniconda3\envs\yolo\python.exe"
    "D:\00software\anaconda\envs\yolo\python.exe"
    "D:\00software\anaconda\python.exe"
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
) do (
    if exist "%%~P" if not defined PY_EXE set "PY_EXE=%%~P"
)

if not defined PY_EXE set "PY_EXE=python"

if not exist "%PY_EXE%" (
    echo [错误] 未找到可用的 Python 环境，请先安装 Python 并：
    echo         1. pip install ultralytics opencv-python
    echo         2. 或修改本 bat 中的候选路径
    pause
    exit /b 1
)

start "" "%PY_EXE%" "%~dp0yolo_tool.py"
