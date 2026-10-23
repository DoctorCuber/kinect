@echo off
net session >nul 2>&1
if errorlevel 1 (
  powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0'"
  if errorlevel 1 (
    echo Administrator approval was cancelled. The Kinect window will not open.
    pause
  )
  exit /b
)
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\pythonw.exe" (
  start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0start_kinect.py" %*
  exit /b 0
)
where py >nul 2>&1
if %errorlevel%==0 (
  py -3 "%~dp0start_kinect.py" %*
  if errorlevel 1 pause
  exit /b %errorlevel%
)
where python >nul 2>&1
if %errorlevel%==0 (
  python "%~dp0start_kinect.py" %*
  if errorlevel 1 pause
  exit /b %errorlevel%
)
echo Python 3.12 or newer is required.
echo Install it from https://www.python.org/downloads/ and check "Add python.exe to PATH".
pause
exit /b 1
