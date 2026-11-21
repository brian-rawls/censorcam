# Setup guide

This guide covers a full live setup: **PS5 → Elgato HD60 X → CensorCam → OBS → Virtual Camera → TikTok LIVE Studio**, controlled from a Stream Deck. Other consoles and UVC capture cards work the same way.

- [1. Install](#1-install)
- [2. Speed up the model with TensorRT](#2-speed-up-the-model-with-tensorrt-recommended)
- [3. Test with a recording](#3-test-with-a-recording)
- [4. Console and capture card](#4-console-and-capture-card)
- [5. Start CensorCam live](#5-start-censorcam-live)
- [6. OBS](#6-obs)
- [7. TikTok LIVE Studio](#7-tiktok-live-studio)
- [8. Stream Deck](#8-stream-deck)
- [9. Audio and video sync](#9-audio-and-video-sync)
- [10. Long sessions](#10-long-sessions)

---

## 1. Install

You need:
- Windows 10 or 11
- an NVIDIA RTX GPU with a current driver
- [OBS Studio](https://obsproject.com/) 30 or newer

Steps:

1. **Python 3.12.** If you don't have it, run `winget install -e --id Python.Python.3.12`, or use the python.org installer. Python 3.14 alone won't work, because SpoutGL ships builds only up to 3.13. You can keep 3.14 installed alongside it; `install.bat` picks 3.12 with `py -3.12`.
2. Double-click **`install.bat`**. It:
   - creates `.venv` with Python 3.12 (and replaces one built with another version);
   - installs PyTorch with CUDA (about 3 GB), then everything in `requirements.txt`;
   - checks that CUDA, SpoutGL, OpenCV and Ultralytics import;
   - downloads `yolo11s-seg.pt` and `yolo11n-seg.pt`.

   It ends with **INSTALL SUCCEEDED** or **INSTALL FAILED** and the error. Check that the output includes:
   ```
   CUDA available: True NVIDIA GeForce RTX ...
   ```
3. **Spout2 plugin for OBS:** with OBS closed, run the installer from [Off-World-Live/obs-spout2-plugin releases](https://github.com/Off-World-Live/obs-spout2-plugin/releases).
4. Run the tests:
   ```
   .venv\Scripts\activate
   python -m pytest -q
   ```
   Expect `25 passed`.

## 2. Speed up the model with TensorRT (recommended)

Double-click **`build_engine.bat`**. It:
1. installs `tensorrt-cu12==10.16.1.11` (about 2.2 GB download, first time only);
2. exports `yolo11s-seg.pt` to `yolo11s-seg.engine`, an FP16 TensorRT engine (about 5 minutes);
3. writes **`live.local.args`** (your personal settings, copied from `live.args`) set to use the engine at **60 fps** with a **14 ms** deadline;
4. measures the engine and starts `run_live.bat`.

Everything is logged to `build_log.txt`.

Reference numbers on an RTX 4070 Ti SUPER:

| Model | Warm benchmark | Live inference p95 |
|---|---|---|
| PyTorch `yolo11s-seg.pt` | ~15–21 ms | ~30 ms |
| TensorRT `yolo11s-seg.engine` | **2.4 ms** | ~10 ms |

At 60 fps the PyTorch model is too slow, so BLUR PEOPLE would fail closed to full blur on every frame. Either build the engine or stay at 30 fps (the default `live.args` before the engine build).

Rebuild the engine after upgrading your NVIDIA driver, TensorRT or Ultralytics.

## 3. Test with a recording

Record a few minutes of gameplay first. You can record straight from the capture card in OBS before switching to the CensorCam setup. Then:

```
.venv\Scripts\activate

# Watch it live, switching modes at http://127.0.0.1:8765/
python -m censorcam --video "C:\path\to\gameplay.mp4" --output preview --start-mode people

# Save a censored copy to scrub through frame by frame
python -m censorcam --video "C:\path\to\gameplay.mp4" --output file:censored.mp4 --start-mode people

# Throughput benchmark (no real-time pacing)
python -m censorcam --video "C:\path\to\gameplay.mp4" --no-realtime --output null --start-mode people --metrics-json bench.json
```

Add `--weights yolo11s-seg.engine` to any of these to test the TensorRT engine. Then measure accuracy properly: see [ACCURACY.md](ACCURACY.md).

## 4. Console and capture card

**PS5:**
- **Settings → System → HDMI → Enable HDCP: Off.** With HDCP on, the capture is black.
- **Settings → Screen and Video → Video Output → HDR: Off.** HDR capture looks washed out in an SDR stream.

**Capture card:**
- Close Elgato 4K Capture Utility, Camera Hub and Elgato Studio, including from the system tray.
- **Remove the capture card's *video* source from OBS.** Only one app can own the device, and it must be CensorCam. This is also a safety feature: OBS can't show raw gameplay if it can't see it.
- Keep the capture card's **audio** in OBS: **+ → Audio Input Capture →** the HD60 X audio device.
- Run `python -m censorcam --list-devices` to see the device name. The default settings use `HD60 X`, which matches any device whose name contains that text.

## 5. Start CensorCam live

Double-click **`run_live.bat`**. It runs `python -m censorcam @live.local.args` (or `@live.args` if you have no local file) and **restarts it automatically** if it ever exits. Expect:

```
[sys] process priority HIGH, timer resolution 1 ms
[model] yolo11s-seg.engine@cuda:0 ... loaded + warmed ...
[input] device HD60 X: 1920x1080 @ 60.00 fps
[control] dashboard: http://127.0.0.1:8765/
[run] mode = BLUR ALL. Ctrl+C to quit.
```

It always starts in **BLUR ALL**. Every 5 seconds it prints a status line:

```
[stat] BLUR PEOPLE out  59.9 fps | infer p50 7.2 p95 9.6 ms | pipeline p95 18.5 ms | drops cap 1 | failsafe 0 | watchdog 2
```

What healthy looks like:
- `out` close to your fps;
- `failsafe` not climbing while in BLUR PEOPLE;
- `drops cap` roughly stable.

**Settings files:** `live.args` holds the repo defaults (30 fps, PyTorch model). `live.local.args` holds your own settings; `build_engine.bat` creates it, or you can copy `live.args` yourself. It isn't committed to git, and `run_live.bat` prefers it. Edit it in Notepad, one argument per line. All options are listed in [CONTROL_API.md](CONTROL_API.md#command-line-options).

## 6. OBS

1. **Settings → Video:** FPS **60**, or **30** if you didn't build the engine. Choose the canvas based on where you want to do your layout (see the note below).
2. **Sources → + → Spout2 Capture** → name it `CensorCam` → sender **CensorCam**. Pick the named sender rather than "first available", so no other Spout app can end up on your stream.
3. *Poll time for new senders:* any fast setting is fine. It only affects how quickly OBS reconnects after CensorCam restarts, not latency.
4. Add your capture card's **audio** as an Audio Input Capture.
5. Optional: create a **"BRB"** scene with no game source in it. That's an OBS-level panic button that doesn't depend on CensorCam.
6. Click **Start Virtual Camera**.

**Where to do the vertical layout.** Do it in one place only:

- **In OBS (recommended):** use a vertical canvas (e.g. 1080×1920) and arrange the game in OBS. To stretch the game freely, hold **Shift** while dragging, or use **right-click → Transform → Edit Transform → Bounding Box Type: Stretch to bounds**. Hold **Alt** while dragging to crop. In TikTok, make the Virtual Camera source fill the frame and leave it alone.
- **In TikTok:** use a **1920×1080** OBS canvas with the Spout source **Fit to screen**. TikTok then receives the full widescreen picture, like it did from the capture card, and you stretch or crop there.

If the image in OBS is **upside down**, add `--spout-flip` to `live.local.args`. If the colours are wrong, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md#blue-tinted--wrong-colours-in-obs).

## 7. TikTok LIVE Studio

1. **Add source → Camera → OBS Virtual Camera.**
2. Check that switching modes on the dashboard changes TikTok's preview within a moment.
3. Audio: see [section 9](#9-audio-and-video-sync).

## 8. Stream Deck

CensorCam listens on `http://127.0.0.1:8765` (local machine only).

| Button | URL |
|---|---|
| NORMAL | `http://127.0.0.1:8765/mode/normal` |
| BLUR PEOPLE | `http://127.0.0.1:8765/mode/people` |
| BLUR ALL (panic) | `http://127.0.0.1:8765/mode/all` |
| Toggle BLUR ALL | `http://127.0.0.1:8765/toggle/all` |

Ways to wire the buttons:

- **Built-in System → Website action:** tick **"GET request in background"**. Some users report this action occasionally misfires, so test it before relying on it.
- **"API Request" plugin** (Stream Deck Marketplace): a GET request per key. It can change the key icon based on the response, so pointing a key at `/status/text` can show `NORMAL` / `BLUR PEOPLE` / `BLUR ALL` / `FAILSAFE`.
- **Native OBS scene buttons:** see [CONTROL_API.md → OBS scene bridge](CONTROL_API.md#obs-scene-bridge).

## 9. Audio and video sync

Video takes the longer path (CensorCam → Spout → OBS → Virtual Camera → TikTok), while audio comes straight from the capture card. So **audio arrives early** and needs delaying.

- Virtual Camera carries **no audio**, so TikTok needs the audio from another source.
- OBS's **Advanced Audio Properties → Sync Offset does not apply to audio monitoring**, so it won't delay audio sent to a virtual cable.
- One approach:
  1. Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/).
  2. On the capture-card audio source in OBS, add a delay **filter**, e.g. the free *ReaDelay* VST from ReaPlugs (wet 100 %, dry off, length in ms).
  3. Set the source to **Monitor and Output**.
  4. Set **Settings → Audio → Monitoring Device** to **CABLE Input**.
  5. In TikTok LIVE Studio, choose **CABLE Output** as the microphone. Don't also capture game audio in TikTok, or you'll hear it twice.

  Whether OBS applies filters before monitoring wasn't verified for every OBS version, so confirm with a test.
- **Starting value:** the dashboard's `suggested_obs_audio_offset_ms`, plus the downstream latency measured with `tools/measure_latency.py` ([PERFORMANCE.md](PERFORMANCE.md#measuring-end-to-end-latency)).
- **Fine-tune:** play an A/V sync test video (e.g. in the PS5 YouTube app), watch a private test stream on a second device, and adjust the delay until the beeps line up with the flashes.

## 10. Long sessions

- NVIDIA Control Panel → Manage 3D settings → Program Settings → add `.venv\Scripts\python.exe` → **Power management: Prefer maximum performance**.
- Windows power plan: High performance. Turn off USB selective suspend.
- Hardware-accelerated GPU scheduling: on.
- Before your first real stream, do a 2–3 hour soak test:
  ```
  python -m censorcam --video gameplay.mp4 --loop --start-mode people --output spout
  ```
  Watch `failsafe`, `drops cap` and `watchdog` in the status line.
