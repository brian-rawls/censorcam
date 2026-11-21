# Changelog

CensorCam by Aeternum Logic, developed by Brian Rawls.

## Unreleased
- Added author and company credits: README Credits section, AUTHORS.md, NOTICE, source-file copyright headers, startup banner and dashboard footer.

## 0.1.0 (2026-11-20)

First working release, tested live with PS5 → Elgato HD60 X → OBS 33 → TikTok LIVE Studio on an RTX 4070 Ti SUPER.

### Features
- Three operator-controlled modes: NORMAL, BLUR PEOPLE, BLUR ALL.
- YOLO person segmentation (Ultralytics), kept resident on the GPU, with a background keep-warm thread.
- Fail-closed pipeline: per-frame deadline, failure latch, watchdog, mid-frame mode re-check, final output gate, start in BLUR ALL.
- Capture-card ownership, so OBS never sees raw frames. Spout output to OBS (RGBA), plus virtual-camera, preview and file outputs.
- HTTP control API and dashboard for Stream Deck; optional OBS scene bridge.
- Metrics: inference, pipeline latency, drops and fail-safe counters; latency barcode probe and `tools/measure_latency.py`.
- `tools/eval_segmentation.py`: labelme pre-labelling and scoring against your own footage.
- TensorRT engine build (`build_engine.bat`), which also switches to 1080p60.
- Personal settings in git-ignored `live.local.args`; `live.args` keeps the repo defaults.
- Windows realtime tweaks: high process priority and 1 ms timer resolution.
- 25 automated tests, including fault injection and a real-model end-to-end run.

### Fixes made during bring-up
- `install.bat` now requires Python 3.12 (SpoutGL has no 3.14 build) and fails loudly.
- Spout now sends RGBA: BGR was shown blue-tinted by the OBS Spout2 plugin.
- Keep-warm inference moved off the frame path, removing a once-per-second hitch.
- Supports Ultralytics' `quantize` argument, which replaced `half`.
- `build_engine.bat` installs TensorRT itself and logs to `build_log.txt`.
