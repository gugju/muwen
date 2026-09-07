@echo off
rem ============================================================
rem  YOLO Dataset Toolkit - launcher
rem  Auto-detect python (prefer 'yolo' conda env), double-click to run
rem  NOTE: keep this file ASCII-only. cmd.exe parses .bat in the
rem  system codepage (GBK); UTF-8 Chinese comments corrupt parsing.
rem ============================================================
setlocal enabledelayedexpansion

set "PY_EXE="
for %%P in (
    "%USERPROFILE%\anaconda3\envs\yolo\pythonw.exe"
    "%USERPROFILE%\miniconda3\envs\yolo\pythonw.exe"
    "%ProgramData%\anaconda3\envs\yolo\pythonw.exe"
    "%ProgramData%\miniconda3\envs\yolo\pythonw.exe"
    "D:\00software\anaconda\envs\yolo\pythonw.exe"
    "D:\00software\anaconda\pythonw.exe"
    "%USERPROFILE%\anaconda3\pythonw.exe"
    "%USERPROFILE%\miniconda3\pythonw.exe"
) do (
    if exist "%%~P" if not defined PY_EXE set "PY_EXE=%%~P"
)

rem fallback: pythonw on PATH
if not defined PY_EXE (
    where pythonw >nul 2>nul && set "PY_EXE=pythonw"
)

if not defined PY_EXE (
    echo [ERROR] No Python found. Please install Python with:
    echo         pip install ultralytics opencv-python
    echo     or edit the candidate paths in this bat file.
    pause
    exit /b 1
)

if not exist "%PY_EXE%" (
    echo [ERROR] Python not found: %PY_EXE%
    pause
    exit /b 1
)

start "" "%PY_EXE%" "%~dp0yolo_tool.py"
