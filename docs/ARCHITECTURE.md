# Architecture

## Goals

1. Three operator-selected modes: NORMAL, BLUR PEOPLE, BLUR ALL. **The operator, not the AI, decides when to censor.**
2. While BLUR PEOPLE is active, a frame that hasn't been processed must never reach the broadcast.
3. Low latency at stable 1080p30, and 1080p60 with TensorRT.
4. Instant switching between modes, with the model kept resident in GPU memory.
5. Stability over multi-hour streams.

## Options considered

| Option | Verdict | Why |
|---|---|---|
| **Existing OBS plugin** (obs-backgroundremoval) | Rejected | Built for webcam portraits, not small, full-body, third-person characters. Its CUDA/TensorRT path is Linux-only (Windows uses DirectML). Most importantly, it has no fail-closed behaviour: if the model stalls, raw video keeps flowing. Accuracy is hard to measure. |
| **Native C++ OBS filter with TensorRT** | Deferred | Lowest possible latency, but weeks of work, coupled to OBS versions, and hard to debug. Only worthwhile if the Python design measured too slow, which it didn't. |
| **Python app that owns the capture card and sends to OBS over Spout** | **Chosen** | Every frame passes through code that enforces the mode. CUDA/TensorRT inference. Easy to test with recorded footage. Spout passes GPU textures into OBS without compression. |

## Data flow

```
                   ┌──────────────────────────── CensorCam process ───────────────────────────┐
HD60 X ──USB──►  Capture thread ──latest frame──► Processor thread ──► OutputWorker thread ──Spout──► OBS
(DirectShow)     (DeviceSource/FileSource)       mode → infer → mask    (final gate, sinks)
                                                  → pixelate/blur
                                    ▲                    ▲                     ▲
                   Watchdog thread ─┴─ posts fully blurred frames if output goes stale or a stricter mode is selected
                   KeepWarm thread ─── occasional throw-away inference outside BLUR PEOPLE
                   Control thread ──── HTTP API / dashboard / OBS scene bridge → OperatorState.mode
```

### Components

| Module | Role |
|---|---|
| `state.py` | `Mode` enum (`normal`, `people`, `all`) with a strictness order, plus a thread-safe `OperatorState` with change listeners. |
| `capture.py` | `DeviceSource` (OpenCV DirectShow/MSMF) and `FileSource` (recordings, real-time paced or as fast as possible). A dedicated thread drains the device continuously. Consumers always get the **newest** frame; overwritten frames are counted as `capture.dropped`. |
| `segmenter.py` | `YoloSegmenter` wraps Ultralytics YOLO segmentation, restricted to COCO class 0 (`person`). It loads `.pt` (PyTorch, FP16 on CUDA) or `.engine` (TensorRT) files and returns polygons + boxes. `FakeSegmenter` exists for tests and fault injection. |
| `censor.py` | `MaskBuilder` rasterises polygons, falling back to a box when a polygon is missing. It dilates the mask and ORs in the previous *N* masks (`--hold`), which can only *add* coverage. `obscure()` downsamples heavily, so information is destroyed rather than just smoothed. `composite()` copies the obscured image into the masked area. |
| `pipeline.py` | `Processor` applies the mode to each frame, enforces the inference deadline, latches the fail-safe, and reloads the model after repeated errors. `Watchdog` covers a stuck processor and gives instant visual response to mode changes. |
| `outputs.py` | Sinks: `spout`, `vcam` (pyvirtualcam / Unity Capture), `preview`, `file`, `null`. `OutputWorker` is the **only** thread that touches sinks, because Spout's OpenGL context is bound to the thread that created it. It also hosts the **final gate**. |
| `control.py` | `ControlServer` (HTTP on 127.0.0.1 with an optional API key), the HTML dashboard, and `ObsSceneBridge` (obs-websocket v5). |
| `metrics.py` | Rolling p50/p95/max over the last 300 samples, plus counters. |
| `probe.py` | 24-bit timestamp barcode stamped on outgoing frames for latency measurement. |
| `app.py` | CLI parsing (`@file` argument files), wiring, status printing, and Windows high-priority + 1 ms timer tweaks. |

### Per-frame timing (1080p60, RTX 4070 Ti SUPER, TensorRT)

| Step | NORMAL | BLUR PEOPLE |
|---|---|---|
| Capture read + colour conversion | in capture thread | in capture thread |
| Inference (incl. Ultralytics pre/post-processing) | — | ~7–10 ms |
| Mask + pixelate + composite | — | ~1–3 ms |
| BGR → RGBA + Spout send | ~1 ms | ~1 ms |
| **Pipeline latency (arrival → Spout)** | **~0.4 ms** | **~18 ms p95** |

### Why Spout

- Spout shares a GPU texture with OBS: no encoding, no network stack, and no extra virtual-camera device that OBS's own Virtual Camera would conflict with.
- The fallback `--output vcam` uses pyvirtualcam with Unity Capture as a *second* virtual camera, leaving OBS's own Virtual Camera free for TikTok.
- Frames are sent as **RGBA**, which every receiver handles. Some builds ignore `GL_BGR_EXT` and swap red and blue.

### Why the capture card is owned exclusively

If OBS also had the capture card's video, a misconfigured scene, a crash or a plugin glitch could put raw frames on air. With CensorCam owning the device, the worst failure is a **frozen, already-censored** frame in OBS.

## Known trade-offs

- Spout and OBS run on independent frame clocks, so an occasional frame may be shown twice. Running at 60 fps makes this much less noticeable than at 30.
- In BLUR PEOPLE, pipeline latency (~18 ms) is a little over one 60 fps frame. NORMAL mode adds well under 1 ms.
- Ultralytics pre/post-processing runs in Python, so live inference (~8 ms) is slower than the raw engine benchmark (~2.4 ms). A custom TensorRT pipeline with GPU pre-processing could close that gap.

---
*CensorCam by Aeternum Logic · developed by Brian Rawls*
