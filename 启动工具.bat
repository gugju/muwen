@echo off
rem YOLO数据集工具箱 启动器 —— 用 yolo conda 环境的 python 拉起 GUI
chcp 65001 >nul
set "PY_EXE=D:\00software\anaconda\envs\yolo\python.exe"

if not exist "%PY_EXE%" (
    echo [错误] 未找到 Python: %PY_EXE%
    echo 请修改本 bat 里的 PY_EXE 变量指向带 ultralytics 的 python.exe
    pause
    exit /b 1
)

start "" "%PY_EXE%" "%~dp0yolo_tool.py"
