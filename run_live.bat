@echo off
REM Starts CensorCam for live streaming and restarts it automatically if it ever exits.
REM While it restarts, OBS keeps showing the last (already censored) frame.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
REM Your personal settings live in live.local.args (not committed to git); defaults in live.args.
set ARGS=live.args
if exist live.local.args set ARGS=live.local.args
echo Using settings from %ARGS%
:loop
python -m censorcam @%ARGS% %*
echo.
echo CensorCam exited (code %ERRORLEVEL%). Restarting in 2 seconds - close this window to stop.
timeout /t 2 /nobreak >nul
goto loop
