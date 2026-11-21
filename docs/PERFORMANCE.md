# Performance and latency

## Budgets

| Frame rate | Time per frame | Recommended `--deadline-ms` | Model |
|---|---|---|---|
| 30 fps | 33.3 ms | 25 | `yolo11s-seg.pt` (PyTorch FP16) is enough |
| 60 fps | 16.7 ms | 14 | `yolo11s-seg.engine` (TensorRT) |

`build_engine.bat` builds the engine and writes `live.local.args` at 60 fps / 14 ms in one step.

## Measured on an RTX 4070 Ti SUPER (PS5 → HD60 X, 1080p60)

| | NORMAL | BLUR PEOPLE (PyTorch) | BLUR PEOPLE (TensorRT) |
|---|---|---|---|
| Output fps | 60.0 | 60 (but every frame failed closed) | **59.9** |
| Inference p50 / p95 | — | 15–21 / 30 ms | **~7.5 / ~9.7 ms** |
| Pipeline latency p95 (arrival → Spout) | **0.4 ms** | — | **~18 ms** |
| Fail-safe frames | 0 | every frame | **0** |

The raw TensorRT engine measures **2.4 ms** warm. The live number is higher because it includes Ultralytics pre/post-processing (letterbox resize, mask decoding) and GPU sharing with OBS and the game capture.

## Tips for smoothness

- **60 fps end to end.** CensorCam, OBS and the capture card should all run at 60. At 30 fps the two independent clocks (Spout ↔ OBS) cause a visible occasional repeated frame.
- CensorCam raises its own process priority to **HIGH** and sets a **1 ms timer resolution** on Windows.
- The keep-warm inference runs on its own thread, so it never delays a video frame.
- NVIDIA Control Panel: *Prefer maximum performance* for `.venv\Scripts\python.exe`.
- Hardware-accelerated GPU scheduling: on.
- If BLUR PEOPLE shows `FAILSAFE`, check `last_error` on the dashboard. It names the inference time versus the deadline.

## Metrics reference

These come from `GET /metrics`. The dashboard and console status lines show a subset.

| Field | Meaning |
|---|---|
| `output_fps` | Frames delivered to the sinks per second. |
| `inference_ms` | Model time per frame in BLUR PEOPLE (n, mean, p50, p95, max over the last 300 frames). |
| `process_ms` | Frame arrival → processed frame ready. |
| `pipeline_latency_ms` | Frame arrival → handed to Spout. **CensorCam's added latency.** |
| `sink_send_ms` | Time spent inside the sinks (Spout upload etc.). |
| `capture.frames_in` / `capture.dropped` | Frames read from the device / frames skipped because a newer one arrived first. A slowly rising `dropped` is normal; a fast rise means processing can't keep up. |
| `counters.failsafe_frames` | BLUR PEOPLE frames sent fully blurred. |
| `counters.deadline_misses` / `inference_errors` / `failsafe_latches` | Why the fail-safe triggered. |
| `counters.watchdog_frames` | Frames the watchdog posted (stalls, plus instant re-blur on mode changes). |
| `counters.gate_frames` | Frames the final gate had to blur. |
| `counters.output_overwritten` | Processed frames replaced before the output thread sent them. |
| `counters.model_reloads` | Background model reloads after repeated errors. |
| `downstream_latency_ms` | Results reported by `tools/measure_latency.py --report`. |
| `censorcam_to_display_ms_p50` | Pipeline p50 + downstream p50. |
| `suggested_obs_audio_offset_ms` | Starting point for the audio delay (see SETUP §9). |

## Measuring end-to-end latency

Glass-to-glass latency = **capture-card latency** + **CensorCam pipeline** + **downstream** (OBS → Virtual Camera → TikTok preview).

1. **Pipeline:** read `pipeline_latency_ms` from `/metrics`.
2. **Downstream:** use the latency probe.
   1. Tick **latency probe** on the dashboard. A small barcode appears in the middle of the picture; it **is visible on stream**.
   2. `python tools/measure_latency.py --screenshot shot.png`, then find the barcode's rectangle (including its black border) in Paint.
   3. `python tools/measure_latency.py --region X,Y,W,H --seconds 20 --report`. Run it once against the OBS preview and once against TikTok LIVE Studio's preview.

   Accuracy is about ± one monitor refresh. The barcode encodes the wall-clock time at which the frame left CensorCam, as 24 bits of milliseconds with a parity bit.
3. **Capture card:** this can't be measured in software. Film the TV and the OBS preview side by side with a phone in slow motion, and count frames between them.

TikTok's own server and viewer delay comes on top of all this and is the same for audio and video.

## Benchmarking on a recording

```
python -m censorcam --video gameplay.mp4 --no-realtime --output null --start-mode people --weights yolo11s-seg.engine --metrics-json bench.json
```

Compare models by changing `--weights`. For accuracy at the same time, use `tools/eval_segmentation.py` ([ACCURACY.md](ACCURACY.md)).
