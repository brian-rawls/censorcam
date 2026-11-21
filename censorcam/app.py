# CensorCam - Copyright (C) 2026 Aeternum Logic. Developed by Brian Rawls.
# Licensed under the GNU Affero General Public License v3.0 - see LICENSE.
"""CensorCam entry point.

    python -m censorcam @live.args           # live: HD60 X -> Spout -> OBS
    python -m censorcam --video clip.mp4 --output preview --start-mode people
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time

from .capture import DeviceSource, FileSource, list_devices
from .censor import MaskBuilder
from .control import ControlServer, ObsSceneBridge
from .metrics import Metrics
from .outputs import (FileSink, NullSink, OutputWorker, PreviewSink, SpoutSink,
                      VirtualCamSink)
from .pipeline import PipelineConfig, Processor, Watchdog
from .state import Mode, OperatorState


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="censorcam", fromfile_prefix_chars="@",
                                description="CensorCam - manual, fail-closed gameplay censorship for OBS. "
                                            "By Aeternum Logic (developer: Brian Rawls).")
    src = p.add_argument_group("input")
    src.add_argument("--device", default=None,
                     help="capture device index or name (e.g. 'HD60 X'). Default input if --video not given.")
    src.add_argument("--video", default=None, help="recorded gameplay file instead of a capture device")
    src.add_argument("--list-devices", action="store_true")
    src.add_argument("--width", type=int, default=1920)
    src.add_argument("--height", type=int, default=1080)
    src.add_argument("--fps", type=float, default=30.0)
    src.add_argument("--fourcc", default="", help="force capture pixel format, e.g. NV12 or YUY2")
    src.add_argument("--backend", default="dshow", choices=["dshow", "msmf", "any"])
    src.add_argument("--no-realtime", action="store_true",
                     help="(video files) process as fast as possible = throughput benchmark")
    src.add_argument("--loop", action="store_true", help="(video files) loop forever")
    src.add_argument("--max-frames", type=int, default=0)

    out = p.add_argument_group("output")
    out.add_argument("--output", default="spout",
                     help="comma list: spout, vcam, preview, null, file:PATH.mp4")
    out.add_argument("--spout-name", default="CensorCam")
    out.add_argument("--spout-flip", action="store_true", help="use if OBS shows the image upside down")
    out.add_argument("--spout-format", default="rgba", choices=["rgba", "bgra", "bgr", "rgb"],
                     help="pixel order sent over Spout; change only if OBS colours look wrong")
    out.add_argument("--vcam-backend", default="unitycapture")

    m = p.add_argument_group("model")
    m.add_argument("--weights", default="yolo11s-seg.pt",
                   help="Ultralytics seg weights (.pt) or exported TensorRT engine (.engine)")
    m.add_argument("--model-device", default=None, help="cuda:0 (default if available) or cpu")
    m.add_argument("--imgsz", type=int, default=640)
    m.add_argument("--conf", type=float, default=0.20,
                   help="detection confidence; lower = fewer misses, more over-blur")
    m.add_argument("--no-half", action="store_true")
    m.add_argument("--no-model", action="store_true",
                   help="don't load a model (BLUR PEOPLE will always fail closed)")

    c = p.add_argument_group("censoring")
    c.add_argument("--shape", default="mask", choices=["mask", "box", "both"],
                   help="obscure silhouettes, bounding boxes, or both")
    c.add_argument("--dilate", type=int, default=16, help="grow masks by N px (at 1080p)")
    c.add_argument("--hold", type=int, default=1, help="OR masks of the previous N frames")
    c.add_argument("--people-block", type=int, default=32)
    c.add_argument("--people-style", default="pixelate", choices=["pixelate", "blur"])
    c.add_argument("--all-block", type=int, default=48)
    c.add_argument("--all-style", default="blur", choices=["pixelate", "blur"])

    s = p.add_argument_group("safety")
    s.add_argument("--start-mode", default="all", help="mode at startup (default: all = safest)")
    s.add_argument("--deadline-ms", type=float, default=25.0)
    s.add_argument("--latch-failures", type=int, default=3)
    s.add_argument("--latch-window-s", type=float, default=1.0)
    s.add_argument("--latch-hold-s", type=float, default=2.0)
    s.add_argument("--watchdog-ms", type=float, default=120.0)
    s.add_argument("--keepwarm-s", type=float, default=1.0)

    ctl = p.add_argument_group("control")
    ctl.add_argument("--port", type=int, default=8765)
    ctl.add_argument("--api-key", default=None)
    ctl.add_argument("--obs-scenes", action="store_true",
                     help="follow OBS program scene changes (obs-websocket)")
    ctl.add_argument("--obs-host", default="localhost")
    ctl.add_argument("--obs-port", type=int, default=4455)
    ctl.add_argument("--obs-password", default="")
    ctl.add_argument("--scene-normal", default="CENSOR: NORMAL")
    ctl.add_argument("--scene-people", default="CENSOR: BLUR PEOPLE")
    ctl.add_argument("--scene-all", default="CENSOR: BLUR ALL")

    misc = p.add_argument_group("misc")
    misc.add_argument("--status-every", type=float, default=5.0)
    misc.add_argument("--metrics-json", default=None, help="write final metrics here on exit")
    return p


def make_sinks(spec: str, args):
    sinks = []
    for part in [x.strip() for x in spec.split(",") if x.strip()]:
        if part == "spout":
            sinks.append(SpoutSink(args.spout_name, args.spout_flip, args.spout_format))
        elif part == "vcam":
            sinks.append(VirtualCamSink(args.vcam_backend))
        elif part == "preview":
            sinks.append(PreviewSink())
        elif part == "null":
            sinks.append(NullSink())
        elif part.startswith("file:"):
            sinks.append(FileSink(part[5:]))
        else:
            raise SystemExit(f"unknown output {part!r}")
    return sinks


def _windows_realtime_tweaks() -> None:
    """High process priority + 1 ms timer resolution for even frame pacing."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000080)  # HIGH_PRIORITY_CLASS
        ctypes.windll.winmm.timeBeginPeriod(1)
        print("[sys] process priority HIGH, timer resolution 1 ms", flush=True)
    except Exception as e:
        print(f"[sys] could not raise priority: {e}", flush=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from . import __version__, __author__, __company__
    print(f"CensorCam v{__version__} - {__company__} (developer: {__author__})", flush=True)
    _windows_realtime_tweaks()
    if args.list_devices:
        for i, name in list_devices():
            print(f"  {i}: {name}")
        return 0

    state = OperatorState(Mode.parse(args.start_mode))
    metrics = Metrics()

    # ---- model (loaded once, stays resident) ------------------------------
    seg, factory = None, None
    if not args.no_model:
        from .segmenter import YoloSegmenter

        def factory():
            return YoloSegmenter(args.weights, device=args.model_device, imgsz=args.imgsz,
                                 conf=args.conf, half=not args.no_half,
                                 warmup_shape=(args.height, args.width, 3))

        t0 = time.perf_counter()
        seg = factory()
        print(f"[model] {seg.name} loaded + warmed in {time.perf_counter() - t0:.1f}s; "
              f"warm inference {seg.warmup((args.height, args.width, 3), 5):.1f} ms", flush=True)

    # ---- input --------------------------------------------------------------
    if args.video:
        source = FileSource(args.video, realtime=not args.no_realtime, loop=args.loop,
                            max_frames=args.max_frames)
    else:
        source = DeviceSource(args.device or "0", args.width, args.height, args.fps,
                              args.fourcc, args.backend)
    print(f"[input] {args.video or 'device ' + str(args.device or 0)}: "
          f"{source.width}x{source.height} @ {source.fps:.2f} fps"
          f"{' (decimating x%d)' % source.decimate if getattr(source, 'decimate', 1) > 1 else ''}",
          flush=True)
    if (source.width, source.height) != (args.width, args.height) and not args.video:
        print("[input] WARNING: device did not accept the requested resolution", flush=True)

    cfg = PipelineConfig(deadline_ms=args.deadline_ms, people_block=args.people_block,
                         people_style=args.people_style, all_block=args.all_block,
                         all_style=args.all_style, latch_failures=args.latch_failures,
                         latch_window_s=args.latch_window_s, latch_hold_s=args.latch_hold_s,
                         watchdog_ms=args.watchdog_ms, keepwarm_s=args.keepwarm_s)
    out_fps = min(args.fps, source.fps) if not args.video else source.fps
    output = OutputWorker(make_sinks(args.output, args), metrics, state, fps=out_fps)
    output.gate_block, output.gate_style = cfg.all_block, cfg.all_style
    builder = MaskBuilder(args.shape, args.dilate, args.hold)
    proc = Processor(source, seg, builder, state, output, metrics, cfg, factory)
    dog = Watchdog(source, output, state, metrics, cfg)

    def status():
        failsafe = state.mode == Mode.PEOPLE and (
            proc.latched or output.last_tag in ("failsafe", "watchdog", "gate"))
        return {
            "mode": state.mode.value,
            "mode_label": state.mode.label,
            "failsafe": failsafe,
            "status_text": "FAILSAFE" if failsafe else state.mode.label,
            "latency_probe": state.latency_probe,
            "model": getattr(proc.segmenter, "name", None),
            "last_error": proc.last_error,
            "output_error": output.error,
        }

    def all_metrics():
        snap = metrics.snapshot(status())
        snap["capture"] = {"frames_in": source.frames_in, "dropped": source.dropped,
                           "fps_nominal": source.fps}
        p = snap["pipeline_latency_ms"]
        d = snap["downstream_latency_ms"]
        if p.get("n") and d.get("n"):
            snap["censorcam_to_display_ms_p50"] = round(p["p50"] + d["p50"], 1)
        if p.get("n"):
            snap["suggested_obs_audio_offset_ms"] = round(p["p50"] + 1000.0 / out_fps / 2)
        return snap

    server = ControlServer(state, status, all_metrics, metrics.downstream_ms.add,
                           port=args.port, api_key=args.api_key).start()
    print(f"[control] dashboard: http://127.0.0.1:{server.port}/  "
          f"(Stream Deck: http://127.0.0.1:{server.port}/mode/people etc.)", flush=True)

    bridge = None
    if args.obs_scenes:
        try:
            bridge = ObsSceneBridge(state, {args.scene_normal: Mode.NORMAL,
                                            args.scene_people: Mode.PEOPLE,
                                            args.scene_all: Mode.ALL},
                                    args.obs_host, args.obs_port, args.obs_password)
            print("[obs] following program scene changes", flush=True)
        except Exception as e:
            print(f"[obs] WARNING: could not connect to obs-websocket: {e}", flush=True)

    output.start()
    source.start()
    dog.start()
    proc.start()
    print(f"[run] mode = {state.mode.label}. Ctrl+C to quit.", flush=True)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *a: stop.update(flag=True))
    last_print = time.monotonic()
    try:
        while not stop["flag"]:
            time.sleep(0.1)
            if not proc.alive:
                break
            if args.status_every and time.monotonic() - last_print >= args.status_every:
                last_print = time.monotonic()
                s = all_metrics()
                inf = s["inference_ms"]
                print(f"[stat] {s['status_text']:<11} out {s['output_fps']:5.1f} fps | "
                      f"infer p50 {inf.get('p50', '-')} p95 {inf.get('p95', '-')} ms | "
                      f"pipeline p95 {s['pipeline_latency_ms'].get('p95', '-')} ms | "
                      f"drops cap {s['capture']['dropped']} | failsafe {s['counters']['failsafe_frames']} "
                      f"| watchdog {s['counters']['watchdog_frames']}", flush=True)
    finally:
        # let the output thread flush the last frame
        time.sleep(0.2)
        proc.stop()
        dog.stop()
        source.stop()
        output.stop()
        if bridge:
            bridge.stop()
        final = all_metrics()
        server.stop()
        if args.metrics_json:
            with open(args.metrics_json, "w") as f:
                json.dump(final, f, indent=2)
        print("[exit] final metrics:\n" + json.dumps(final, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
