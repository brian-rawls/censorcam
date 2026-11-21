@echo off
setlocal
cd /d "%~dp0"
REM One-time setup. Needs Python 3.12 (SpoutGL has no ready-made build for Python 3.14).

py -3.12 -c "import sys" >nul 2>&1
if errorlevel 1 (
  echo Python 3.12 not found - installing it with the Python install manager...
  py install 3.12
  py -3.12 -c "import sys" >nul 2>&1
  if errorlevel 1 (
    echo Could not get Python 3.12 automatically.
    echo Install Python 3.12 from https://www.python.org/downloads/ and run this file again.
    goto :fail
  )
)

if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)"
  if errorlevel 1 (
    echo Removing old .venv that was built with a different Python version...
    rmdir /s /q .venv
  )
)
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv || goto :fail
)

call .venv\Scripts\activate.bat
python --version
python -m pip install --upgrade pip || goto :fail
REM PyTorch with CUDA for the RTX 4070 Ti Super. If this index URL is outdated,
REM get the current command from https://pytorch.org/get-started/locally/ (Windows, Pip, CUDA).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :fail
pip install -r requirements.txt || goto :fail
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || goto :fail
python -c "import SpoutGL, cv2, ultralytics; print('SpoutGL, OpenCV and Ultralytics OK')" || goto :fail
python -c "from ultralytics import YOLO; YOLO('yolo11s-seg.pt'); YOLO('yolo11n-seg.pt')" || goto :fail
echo.
echo ===== INSTALL SUCCEEDED =====
echo Next: run tests with   .venv\Scripts\python -m pytest -q
pause
exit /b 0

:fail
echo.
echo ===== INSTALL FAILED =====
echo Copy the error text above and send it to Claude.
pause
exit /b 1
