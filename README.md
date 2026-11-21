# CensorCam

**Manual, fail-closed, GPU-accelerated censorship for livestreaming console gameplay through OBS Studio and TikTok LIVE Studio.**

You decide when to censor, with a Stream Deck button or a browser dashboard. When you turn on **BLUR PEOPLE**, a local AI model finds human-shaped characters in every frame and pixelates them. **BLUR ALL** blurs the whole picture instantly without waiting for the AI. If the AI ever fails or falls behind while BLUR PEOPLE is on, the frame goes out **fully blurred**, never raw.

Built for, and tested on, this setup: PS5 → Elgato HD60 X → Windows PC with an RTX 4070 Ti SUPER → OBS Studio → Virtual Camera → TikTok LIVE Studio.

| Mode | What viewers see | AI used? |
|---|---|---|
| **NORMAL** | Untouched gameplay | No |
| **BLUR PEOPLE** | Characters pixelated (silhouette + safety margin), everything else clear | Yes, on every frame |
| **BLUR ALL** | Whole picture heavily blurred | No; applied on the next frame (and to the frame already on screen within ~10 ms) |

> CensorCam never decides **whether** something should be censored. That is always your call. The AI's only job is to find characters while you have BLUR PEOPLE turned on.

---

## Highlights

- **Fail-closed by design.** Inference errors, missed deadlines, hung threads, crashes and mid-frame mode changes all result in more blur, never less. → [docs/SAFETY.md](docs/SAFETY.md)
- **Low latency at 1080p60.** With TensorRT the model runs in ~2–3 ms (warm benchmark) and ~8 ms per frame in the live pipeline. NORMAL mode adds under 1 ms. → [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- **Model stays loaded** in GPU memory and kept warm, so switching modes is instant.
- **Stream Deck / HTTP control** on `127.0.0.1`, plus an optional OBS-scene bridge (obs-websocket). → [docs/CONTROL_API.md](docs/CONTROL_API.md)
- **Metrics built in:** inference time, pipeline latency, dropped frames, fail-safe counters, plus a barcode probe for measuring end-to-end latency through OBS and TikTok.
- **Accuracy tooling.** COCO "person" models are *not* guaranteed to recognise game characters, so CensorCam ships a tool to label your own footage and score the model against it. → [docs/ACCURACY.md](docs/ACCURACY.md)
- **Tested.** 25 automated tests, including fail-safe fault injection and a real-model end-to-end run.

## How it works

```
PS5 ─HDMI─► Elgato HD60 X ─USB video─► CensorCam (Python + CUDA/TensorRT) ─Spout (GPU)─► OBS "Spout2 Capture"
                    └──────USB audio──────────────────────────────────────────────────────► OBS audio input
OBS (layout, overlays) ─Virtual Camera─► TikTok LIVE Studio ─► TikTok
Stream Deck / browser ─HTTP 127.0.0.1:8765─► CensorCam
```

CensorCam takes exclusive ownership of the capture card, so **OBS never receives raw gameplay**. Every frame passes through code that enforces the selected mode. → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Quick start (Windows)

Requirements: Windows 10/11, an NVIDIA RTX GPU, **Python 3.12** (not 3.14: the Spout library has no build for it), OBS Studio 30+ with the [Spout2 plugin](https://github.com/Off-World-Live/obs-spout2-plugin/releases), and a UVC capture card (tested: Elgato HD60 X).

1. Clone or download this repo.
2. Double-click **`install.bat`**. It sets up Python 3.12, PyTorch with CUDA, the dependencies and the AI models.
3. Double-click **`build_engine.bat`** (optional but recommended). It installs TensorRT, builds a GPU-optimised model (~5 min), and writes your personal `live.local.args` set to 60 fps.
4. Test on a recording:
   ```
   .venv\Scripts\activate
   python -m censorcam --video "C:\path\to\gameplay.mp4" --output preview --start-mode people
   ```
   Open http://127.0.0.1:8765/ to switch modes.
5. Go live: double-click **`run_live.bat`**. In OBS, add a **Spout2 Capture** source (sender `CensorCam`) and start **Virtual Camera**. In TikTok LIVE Studio, add the **OBS Virtual Camera** as a camera source.

Full walkthrough, including PS5 settings, OBS layout, Stream Deck buttons and audio sync: **[docs/SETUP.md](docs/SETUP.md)**.

## Documentation

| Doc | Contents |
|---|---|
| [SETUP.md](docs/SETUP.md) | Install, PS5/HD60 X prep, OBS, TikTok LIVE Studio, Stream Deck, audio sync |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Design options considered, component overview, threading model |
| [SAFETY.md](docs/SAFETY.md) | The fail-closed rules and how each one is tested |
| [PERFORMANCE.md](docs/PERFORMANCE.md) | TensorRT, 30 vs 60 fps, metrics reference, latency measurement |
| [ACCURACY.md](docs/ACCURACY.md) | Measuring how well the model finds characters in *your* game footage |
| [CONTROL_API.md](docs/CONTROL_API.md) | HTTP endpoints, dashboard, OBS scene bridge, all command-line options |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Real problems hit during setup and their fixes |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

## Repository layout

```
censorcam/        the application
  app.py          command-line entry point and wiring
  pipeline.py     frame processor, fail-safe latch, watchdog
  censor.py       mask building, pixelation/blur, compositing
  segmenter.py    YOLO segmentation backend (+ fake backend for tests)
  capture.py      capture-card and video-file sources ("latest frame wins")
  outputs.py      Spout / virtual cam / preview / file sinks + final safety gate
  control.py      HTTP control API, dashboard, OBS scene bridge
  metrics.py      rolling performance metrics
  probe.py        latency barcode encode/decode
  state.py        modes and operator state
tools/            eval_segmentation.py, measure_latency.py, export_tensorrt.py, make_test_clip.py
tests/            pytest suite
install.bat       one-time setup
build_engine.bat  TensorRT engine build + switch to 60 fps
run_live.bat      live launcher with auto-restart
live.args         default live settings (one argument per line)
live.local.args   your personal settings (git-ignored; created by build_engine.bat, preferred by run_live.bat)
```

## Limitations

- How well it detects characters in **your** game is unknown until you measure it ([ACCURACY.md](docs/ACCURACY.md)). Distant, dark, occluded and in-vehicle characters are the usual misses. Reflections, screens, posters and minimap icons are not handled.
- Windows-only for live use (Spout and DirectShow). The tests and the accuracy tools also run on Linux and macOS.
- The TensorRT engine is tied to your GPU, driver and TensorRT version. Rebuild it with `build_engine.bat` after upgrades.

## License

[AGPL-3.0](LICENSE). CensorCam uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is AGPL-3.0 licensed. Model weights are downloaded from Ultralytics at install time and are not included in this repository.
