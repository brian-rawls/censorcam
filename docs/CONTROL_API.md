# Control API, dashboard and options

## HTTP API

CensorCam serves HTTP on `127.0.0.1:<--port>` (default **8765**). It binds to the local machine only. All endpoints are `GET`, so the Stream Deck "Website" action can call them.

| Endpoint | Effect / response |
|---|---|
| `/mode/normal` | Switch to NORMAL. Returns status JSON. |
| `/mode/people` | Switch to BLUR PEOPLE. |
| `/mode/all` | Switch to BLUR ALL. |
| `/toggle/all` | BLUR ALL on/off. Turning it off returns to the mode you were in before. |
| `/toggle/people` | BLUR PEOPLE ↔ NORMAL. |
| `/status` | `{"mode", "mode_label", "failsafe", "status_text", "latency_probe", "model", "last_error", "output_error"}` |
| `/status/text` | Plain text: `NORMAL`, `BLUR PEOPLE`, `BLUR ALL` or `FAILSAFE`. Useful for key titles and icons. |
| `/metrics` | Full metrics JSON (see [PERFORMANCE.md](PERFORMANCE.md#metrics-reference)). |
| `/probe/on`, `/probe/off` | Latency barcode on/off. It is **visible on stream**, so use it only for testing. |
| `/report/downstream?ms=N` | Used by `tools/measure_latency.py --report`. |
| `/` | Dashboard: three big mode buttons, a FAILSAFE banner, the probe toggle and live metrics. |

Mode names also accept aliases: `off`, `blur_people`, `person`, `blur_all`, `full`, `panic`.

`failsafe` / `FAILSAFE` means BLUR PEOPLE is selected but frames are currently going out fully blurred: the fail-safe is latched, or the last frame came from the fail-safe, watchdog or final gate. `last_error` explains why, e.g. `inference 21.0 ms > deadline 14.0 ms`.

### API key

Start with `--api-key SECRET` to require `?key=SECRET` on every endpoint except the dashboard page. For the dashboard itself, open `http://127.0.0.1:8765/?key=SECRET`. This protects against web pages in your browser triggering local requests.

## OBS scene bridge

With `--obs-scenes`, CensorCam follows OBS's **program** scene through obs-websocket v5, which is built into OBS 28+. Name three scenes:

| Scene name (default) | Mode |
|---|---|
| `CENSOR: NORMAL` | NORMAL |
| `CENSOR: BLUR PEOPLE` | BLUR PEOPLE |
| `CENSOR: BLUR ALL` | BLUR ALL |

Each scene should contain the same Spout2 Capture source and the same layout.

1. Enable OBS → **Tools → WebSocket Server Settings**.
2. Add to `live.local.args`:
   ```
   --obs-scenes
   --obs-password
   YOUR_PASSWORD
   ```
3. Use the Stream Deck OBS plugin's **Scene** keys. They light up to show the active scene.

Notes:
- Set the OBS transition to **Cut**.
- Don't use Studio Mode, because CensorCam follows the program scene, not the preview.
- Use either the HTTP buttons or the scene bridge, not both: a scene switch overrides an HTTP mode change and vice versa.
- Rename scenes with `--scene-normal`, `--scene-people` and `--scene-all`.

## Command-line options

Settings files hold one argument per line and are passed as `python -m censorcam @live.local.args`. `live.args` holds the repo defaults; `live.local.args` holds your personal settings and is git-ignored. Any option can also be given on the command line; the last occurrence wins.

### Input
| Option | Default | Notes |
|---|---|---|
| `--device` | `0` | Capture device index or part of its name (`HD60 X`). Run `--list-devices` to see names. |
| `--video PATH` | — | Use a recording instead of a device. |
| `--width` / `--height` / `--fps` | 1920 / 1080 / 30 | Requested capture format. If the device delivers 60 and you ask for 30, every other frame is dropped. |
| `--fourcc` | auto | Force a pixel format, e.g. `NV12` or `YUY2`. |
| `--backend` | `dshow` | `dshow`, `msmf` or `any`. |
| `--no-realtime` | off | Video files: process as fast as possible (benchmark). |
| `--loop` | off | Video files: loop forever. |
| `--max-frames` | 0 | Stop after N frames (0 = no limit). |

### Output
| Option | Default | Notes |
|---|---|---|
| `--output` | `spout` | Comma-separated list: `spout`, `vcam`, `preview`, `null`, `file:out.mp4`. |
| `--spout-name` | `CensorCam` | Spout sender name to pick in OBS. |
| `--spout-flip` | off | Use if OBS shows the image upside down. |
| `--spout-format` | `rgba` | `bgra`, `bgr` or `rgb`. Change only if colours are wrong. |
| `--vcam-backend` | `unitycapture` | Backend for `--output vcam`. |

### Model
| Option | Default | Notes |
|---|---|---|
| `--weights` | `yolo11s-seg.pt` | `.pt` or TensorRT `.engine`. Also try `yolo11n-seg.pt` (faster), `yolo11m-seg.pt` (more accurate) or `yolo26s-seg.pt`. |
| `--model-device` | `cuda:0` if available | `cpu` works for testing but is far too slow for live use. |
| `--imgsz` | 640 | Model input size. 960 finds more distant characters but costs time. |
| `--conf` | 0.20 | Lower means fewer misses but more over-blur. |
| `--no-half` | off | Disable FP16 for `.pt` models. |
| `--no-model` | off | Don't load a model; BLUR PEOPLE then always fails closed. |

### Censoring
| Option | Default | Notes |
|---|---|---|
| `--shape` | `mask` | `mask` (silhouette), `box` (bounding box) or `both`. |
| `--dilate` | 16 | Grow masks by N px at 1080p. |
| `--hold` | 1 | Also cover where characters were in the previous N frames. |
| `--people-block` / `--people-style` | 32 / `pixelate` | How character regions are obscured. |
| `--all-block` / `--all-style` | 48 / `blur` | How BLUR ALL (and every fail-safe frame) is obscured. |

### Safety
| Option | Default | Notes |
|---|---|---|
| `--start-mode` | `all` | Mode at startup. |
| `--deadline-ms` | 25 | Inference time limit before a frame fails closed. Use 14 at 60 fps, 25 at 30 fps. |
| `--latch-failures` / `--latch-window-s` / `--latch-hold-s` | 3 / 1.0 / 2.0 | Fail-safe latch settings. |
| `--watchdog-ms` | 120 | How long the output may go quiet in a censored mode before the watchdog sends blurred frames. |
| `--keepwarm-s` | 1.0 | Background throw-away inference interval outside BLUR PEOPLE (0 = off). |

### Control and misc
| Option | Default | Notes |
|---|---|---|
| `--port` | 8765 | HTTP port. |
| `--api-key` | — | Require `?key=`. |
| `--obs-scenes`, `--obs-host`, `--obs-port`, `--obs-password` | off, localhost, 4455, — | OBS scene bridge. |
| `--scene-normal` / `--scene-people` / `--scene-all` | `CENSOR: …` | Scene names for the bridge. |
| `--status-every` | 5 | Seconds between console status lines (0 = off). |
| `--metrics-json` | — | Write final metrics to this file on exit. |
| `--list-devices` | — | Print capture devices and exit. |
