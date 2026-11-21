# Safety model (fail-closed)

The promise: **while BLUR PEOPLE or BLUR ALL is selected, viewers never see an unprocessed frame.** When anything goes wrong, the result is *more* blur, never less.

## Rules

| # | Situation | What goes out | Where enforced | Test |
|---|---|---|---|---|
| 1 | Normal operation in BLUR PEOPLE | The mask for a frame is computed **from that same frame**. Older masks are only ever *added* (`--hold`), never substituted. | `Processor._people`, `MaskBuilder` | `test_success_blurs_people_only`, `test_hold_only_adds_coverage` |
| 2 | Inference raises an exception (CUDA error, driver reset, …) | Frame fully blurred | `Processor._people` | `test_exception_fails_closed` |
| 3 | No model loaded / model reloading | Frame fully blurred | `Processor._people` | `test_no_model_fails_closed` |
| 4 | Inference slower than `--deadline-ms` | Frame fully blurred | `Processor._people` | `test_deadline_miss_fails_closed` |
| 5 | ≥ `--latch-failures` (3) failures within `--latch-window-s` (1 s) | **Latched**: every frame fully blurred for `--latch-hold-s` (2 s), extended while failures continue. The model is re-tested only every 5th frame, so a slow model can't also cut the frame rate. | `Processor._fail` | `test_latch_holds_after_repeated_failures` |
| 6 | 5 consecutive inference exceptions | Model reloaded in the background; full blur meanwhile | `Processor._reload_async` | — |
| 7 | Processing thread hangs | **Watchdog** posts a fully blurred copy of the newest captured frame every `--watchdog-ms` (120 ms) | `Watchdog` | `test_watchdog_covers_hung_inference` |
| 8 | Operator selects a stricter mode mid-frame | Stricter mode wins for that frame | `Processor.process` (re-check after processing) | `test_switch_to_all_during_inference_wins` |
| 9 | Operator selects a stricter mode while the capture is frozen | Watchdog immediately re-sends the on-screen frame blurred (< 100 ms; typically ~10 ms) | `Watchdog` (mode-change kick) | `test_mode_change_censors_instantly_even_without_new_frames` |
| 10 | Any frame weaker than the current mode reaches the output thread | Blurred at the **final gate** before it reaches any sink | `OutputWorker._run` | `test_final_gate_obscures_weaker_frame` |
| 11 | Mixed errors, slow frames and good frames | In 60 frames of random faults, no output ever matches the raw character region | all of the above | `test_no_raw_region_ever_leaks_under_random_failures` |
| 12 | CensorCam crashes or is closed | OBS shows the last frame it received, which was already censored. `run_live.bat` restarts the app. | architecture | — |
| 13 | Startup | Starts in **BLUR ALL** (`--start-mode all`) | `app.py` | — |

## How strong is the obscuring?

- **BLUR ALL:** downsample to 1/48 (a 40×22 grid at 1080p), slight blur, bilinear upscale. No faces, text or silhouettes survive. Information is destroyed, not just smoothed, so it can't be "un-blurred".
- **BLUR PEOPLE:** the character region is pixelated into 32 px blocks (`--people-block`), and masks are grown by 16 px (`--dilate`) to cover segmentation edge errors. The *outline* of the silhouette is still visible by design; use `--shape box` or `--shape both` to hide shapes as well.

## What the fail-safe does *not* cover

- **Characters the model doesn't detect.** A missed detection isn't an error, so it isn't caught by the rules above. This is an accuracy question; measure it with [ACCURACY.md](ACCURACY.md) and tune `--conf`, `--dilate`, `--shape`, `--imgsz` or the model.
- **NORMAL mode** passes video through untouched, by design.
- **Things outside CensorCam:** other OBS sources, overlays, TikTok-side content. Keep a "BRB" scene in OBS as an independent panic button.

---
*CensorCam by Aeternum Logic · developed by Brian Rawls*
