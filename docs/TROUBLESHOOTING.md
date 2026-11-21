# Troubleshooting

These are real problems hit while setting CensorCam up on Windows with a PS5, an HD60 X, OBS 33 and TikTok LIVE Studio, and how each was fixed.

## Install

### `Failed building wheel for SpoutGL` … `No such file or directory: 'vendor/SpoutGL/Binaries\x64\Spout.dll'`
The virtual environment was built with **Python 3.14**, and SpoutGL only publishes builds for 3.8–3.13. Because SpoutGL failed, the rest of `requirements.txt` (including `ultralytics`) never installed either.
**Fix:** install Python 3.12 and run `install.bat` again. It deletes a `.venv` built with the wrong version.

### `The 'install' command is unavailable because this is the legacy py.exe command`
The older Python launcher can't install Python versions itself.
**Fix:** run `winget install -e --id Python.Python.3.12` (or use the python.org installer), open a **new** terminal, check `py -3.12 --version`, then run `install.bat` again.

### `CUDA available: False`
PyTorch installed without CUDA, or the driver is too old.
**Fix:** update the NVIDIA driver. Get the current install command from pytorch.org (Windows · Pip · CUDA) and rerun it inside `.venv`.

## Capture

### `Could not open capture device`
Another app owns the capture card: Elgato 4K Capture Utility, Camera Hub, or a video source in OBS.
**Fix:** close those apps and remove the capture card's **video** source from OBS. Keep its audio source.

### Black picture, even in NORMAL
The PS5 has HDCP enabled.
**Fix:** Settings → System → HDMI → Enable HDCP: Off.

### Washed-out colours
The PS5 is sending HDR.
**Fix:** Settings → Screen and Video → HDR: Off.

### `[input] … WARNING: device did not accept the requested resolution`
**Fix:** try `--fourcc NV12` or `--fourcc YUY2`, or `--backend msmf`.

## OBS

### Blue-tinted / wrong colours in OBS
Red and blue were swapped on the way into OBS: some Spout builds ignore the BGR pixel format. CensorCam now sends **RGBA** by default.
**Fix:** if colours are still wrong, add `--spout-format bgra` to `live.local.args` (on its own two lines: `--spout-format`, then `bgra`).

### Image upside down
**Fix:** add `--spout-flip` to `live.local.args`.

### Spout source is black, or "CensorCam" isn't in the sender list
CensorCam isn't running, or it's still loading the model.
**Fix:** start `run_live.bat` and wait for `[run] mode = …`, then reselect the sender.

### Motion looks slightly choppy compared to the capture card directly
You're running at 30 fps while the console outputs 60, and the Spout and OBS clocks are independent.
**Fix:** build the TensorRT engine (`build_engine.bat`), which switches to 60 fps, and set OBS to 60 FPS.

## BLUR PEOPLE

### BLUR PEOPLE blurs the whole picture; console shows `LATCHED full censorship (inference 21.0 ms > deadline 14.0 ms)`
At 60 fps, the PyTorch model (~21 ms) is slower than the 14 ms deadline, so every frame fails closed. That's the safety feature working.
**Fix:** run `build_engine.bat`. With TensorRT, live inference was ~8 ms (p95 ~10 ms) and there were zero fail-safe frames. Alternatives: `--weights yolo11n-seg.pt`, or 30 fps with `--deadline-ms 25`.

### `build_engine.bat`: `No module named 'tensorrt'`
Ultralytics didn't install TensorRT automatically.
**Fix:** the current `build_engine.bat` installs `tensorrt-cu12==10.16.1.11` itself (about 2.2 GB). Check `build_log.txt` for the full output.

### `WARNING 'half' is deprecated … Use 'quantize' instead`
This is harmless. CensorCam detects the newer Ultralytics API and passes `quantize=16`.

### Some characters aren't blurred
That's a detection miss, not a fail-safe problem. Measure and tune it with [ACCURACY.md](ACCURACY.md).

## TikTok LIVE Studio

### The game can only be resized proportionally in TikTok
OBS is set to a vertical canvas, so the Virtual Camera already sends a cropped portrait image.
**Fix:** do the layout in OBS (hold **Shift** while dragging to stretch freely, or use Edit Transform → *Stretch to bounds*). Or switch OBS to a 1920×1080 canvas and lay it out in TikTok as before. See [SETUP.md §6](SETUP.md#6-obs).

### Audio is ahead of the video
See [SETUP.md §9](SETUP.md#9-audio-and-video-sync). OBS's Sync Offset doesn't apply to monitored audio, so use a delay filter.

## Still stuck?

Open an issue and include:
- the `[model]` and `[input]` startup lines;
- one `[stat]` line from each mode;
- the output of `http://127.0.0.1:8765/metrics`;
- `build_log.txt`, if the problem is the engine build.

---
*CensorCam by Aeternum Logic · developed by Brian Rawls*
