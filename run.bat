@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\pythonw.exe" (
  start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0sleepwalk_detector.py" %*
  exit /b 0
)
where py >nul 2>&1
if %errorlevel%==0 (
  py -3 "%~dp0sleepwalk_detector.py" %*
  if errorlevel 1 pause
  exit /b %errorlevel%
)
where python >nul 2>&1
if %errorlevel%==0 (
  python "%~dp0sleepwalk_detector.py" %*
  if errorlevel 1 pause
  exit /b %errorlevel%
)
echo Python 3.12 or newer is required.
echo Install it from https://www.python.org/downloads/ and check "Add python.exe to PATH".
pause
exit /b 1
