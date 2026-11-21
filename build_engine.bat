@echo off
setlocal
cd /d "%~dp0"
title CensorCam - building TensorRT engine
call .venv\Scripts\activate.bat || goto :fail
echo Building TensorRT engine (a few minutes). Output is also saved to build_log.txt
echo ==== %date% %time% ==== > build_log.txt
echo Installing TensorRT 10.16 (about 1-2 GB, first time only)...
python -c "import tensorrt" >nul 2>&1 || pip install "tensorrt-cu12==10.16.1.11" --extra-index-url https://pypi.nvidia.com >> build_log.txt 2>&1
python -c "import tensorrt; print('TensorRT', tensorrt.__version__)" >> build_log.txt 2>&1 || (type build_log.txt & goto :fail)
python -u tools\export_tensorrt.py --weights yolo11s-seg.pt >> build_log.txt 2>&1
echo export exit code %ERRORLEVEL% >> build_log.txt
type build_log.txt
if not exist yolo11s-seg.engine goto :fail
REM Write live.local.args (personal settings, starts from live.args) switched to the engine and to 60 fps (TensorRT is fast enough for a 14 ms deadline).
if not exist live.local.args copy /y live.args live.local.args >nul
powershell -NoProfile -Command "$l = Get-Content live.local.args; for ($i=0; $i -lt $l.Count-1; $i++) { if ($l[$i] -eq '--weights') { $l[$i+1]='yolo11s-seg.engine' } elseif ($l[$i] -eq '--fps') { $l[$i+1]='60' } elseif ($l[$i] -eq '--deadline-ms') { $l[$i+1]='14' } }; Set-Content live.local.args $l" || goto :fail
python -u -c "from censorcam.segmenter import YoloSegmenter as Y; s=Y('yolo11s-seg.engine', warmup_shape=None); print('TensorRT warm inference ms:', round(s.warmup(n=30), 1))" >> build_log.txt 2>&1
type build_log.txt
echo.
echo ===== ENGINE READY - live.local.args now uses yolo11s-seg.engine at 60 fps. Starting CensorCam... =====
start "" run_live.bat
timeout /t 10 /nobreak >nul
exit /b 0
:fail
echo.
echo ===== FAILED - see build_log.txt =====
pause
exit /b 1
