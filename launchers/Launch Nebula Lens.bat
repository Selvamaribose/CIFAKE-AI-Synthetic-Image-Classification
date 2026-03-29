@echo off
setlocal

set "ROOT=%~dp0..\\"
cd /d "%ROOT%"

set "VENV_PYTHON=%ROOT%.venv\Scripts\python.exe"
set "VENV_PYTHONW=%ROOT%.venv\Scripts\pythonw.exe"
set "DESKTOP_ENTRY=%ROOT%desktop_app.py"

if not exist "%VENV_PYTHON%" (
  echo Nebula Lens could not find .venv\Scripts\python.exe.
  echo Create the virtual environment and install requirements first.
  pause
  exit /b 1
)

"%VENV_PYTHON%" -c "import streamlit, webview" >nul 2>&1
if errorlevel 1 (
  echo The desktop app dependencies are not installed in .venv yet.
  echo Run pip install -r requirements.txt once, then open the launcher again.
  pause
  exit /b 1
)

if not exist "%DESKTOP_ENTRY%" (
  echo desktop_app.py was not found in the repository root.
  pause
  exit /b 1
)

if exist "%VENV_PYTHONW%" (
  start "" "%VENV_PYTHONW%" "%DESKTOP_ENTRY%"
) else (
  start "" "%VENV_PYTHON%" "%DESKTOP_ENTRY%"
)

exit /b 0
