# CensorCam - Copyright (C) 2026 Aeternum Logic. Developed by Brian Rawls.
# Licensed under the GNU Affero General Public License v3.0 - see LICENSE.
"""Output sinks and the single output thread.

All sinks are opened and driven from ONE thread (OutputWorker). Spout uses an
OpenGL context that is bound to the thread that created it, so nothing else
may call into a sink.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from . import probe
from .censor import obscure
from .metrics import Metrics
from .state import Mode, OperatorState


class Sink:
    name = "sink"

    def open(self, w: int, h: int, fps: float) -> None: ...
    def send(self, frame: np.ndarray) -> None: ...
    def close(self) -> None: ...


class NullSink(Sink):
    name = "null"

    def send(self, frame):
        pass


class SpoutSink(Sink):
    """GPU texture share into OBS via the Off World Live 'Spout2 Capture' source."""

    name = "spout"

    def __init__(self, sender_name: str = "CensorCam", flip: bool = False, fmt: str = "rgba"):
        self.sender_name = sender_name
        self.flip = flip
        self.fmt = fmt
        self.sender = None

    def open(self, w, h, fps):
        import SpoutGL
        from SpoutGL import enums
        # RGBA is the format every Spout receiver handles; some builds ignore
        # GL_BGR_EXT and treat the data as RGB, which swaps red/blue (blue tint).
        self._fmt = {"rgba": enums.GL_RGBA, "bgra": enums.GL_BGRA_EXT,
                     "bgr": enums.GL_BGR_EXT, "rgb": enums.GL_RGB}[self.fmt]
        self._conv = {"rgba": cv2.COLOR_BGR2RGBA, "bgra": cv2.COLOR_BGR2BGRA,
                      "bgr": None, "rgb": cv2.COLOR_BGR2RGB}[self.fmt]
        self.sender = SpoutGL.SpoutSender()
        self.sender.setSenderName(self.sender_name)

    def send(self, frame):
        h, w = frame.shape[:2]
        if self._conv is not None:
            frame = cv2.cvtColor(frame, self._conv)
        buf = np.ascontiguousarray(frame)
        try:
            ok = self.sender.sendImage(buf, w, h, self._fmt, self.flip, 0)
        except TypeError:  # older bindings may only accept bytes
            ok = self.sender.sendImage(buf.tobytes(), w, h, self._fmt, self.flip, 0)
        if ok is False:
            raise RuntimeError("Spout sendImage failed")
        self.sender.setFrameSync(self.sender_name)

    def close(self):
        if self.sender is not None:
            self.sender.releaseSender()
            self.sender = None


class VirtualCamSink(Sink):
    """Fallback transport: a *second* virtual camera (Unity Capture) that OBS
    ingests as a Video Capture Device. (OBS's own Virtual Camera stays free for
    TikTok LIVE Studio.) Higher CPU cost than Spout."""

    name = "vcam"

    def __init__(self, backend: str = "unitycapture", device: Optional[str] = None):
        self.backend = backend
        self.device = device
        self.cam = None

    def open(self, w, h, fps):
        import pyvirtualcam
        self.cam = pyvirtualcam.Camera(w, h, fps, fmt=pyvirtualcam.PixelFormat.BGR,
                                       backend=self.backend, device=self.device)
        print(f"[vcam] sending to {self.cam.device}", flush=True)

    def send(self, frame):
        self.cam.send(frame)

    def close(self):
        if self.cam:
            self.cam.close()


class PreviewSink(Sink):
    name = "preview"

    def __init__(self, title: str = "CensorCam output (preview)", scale: float = 0.5):
        self.title, self.scale = title, scale

    def send(self, frame):
        f = cv2.resize(frame, None, fx=self.scale, fy=self.scale) if self.scale != 1 else frame
        cv2.imshow(self.title, f)
        cv2.waitKey(1)

    def close(self):
        try:
            cv2.destroyWindow(self.title)
        except Exception:
            pass


class FileSink(Sink):
    name = "file"

    def __init__(self, path: str):
        self.path = path
        self.w = None

    def open(self, w, h, fps):
        self.w = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def send(self, frame):
        self.w.write(frame)

    def close(self):
        if self.w:
            self.w.release()


class ListSink(Sink):
    """Collects (meta, frame) for tests."""

    name = "list"

    def __init__(self):
        self.items: list = []

    def send(self, frame):
        self.items.append(frame)


@dataclass
class OutFrame:
    seq: int
    t_arrival: float
    frame: np.ndarray
    mode: Mode            # the protection level actually applied to this frame
    tag: str              # normal | people | all | failsafe | watchdog
    force: bool = False


class OutputWorker:
    def __init__(self, sinks: List[Sink], metrics: Metrics, state: OperatorState,
                 fps: float = 30.0):
        self.sinks = sinks
        self.metrics = metrics
        self.state = state
        self.fps = fps
        self._cond = threading.Condition()
        self._pending: Optional[OutFrame] = None
        self._stop = threading.Event()
        self.last_sent_seq = 0
        self.last_sent_time = time.perf_counter()
        self.last_sent_mode: Mode = Mode.ALL
        self.last_tag = ""
        self.opened = False
        self.error: Optional[str] = None
        self.on_sent = None  # test hook: fn(OutFrame)
        self.gate_block, self.gate_style = 48, "blur"
        self._thread = threading.Thread(target=self._run, name="OutputWorker", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2)

    def post(self, item: OutFrame) -> None:
        with self._cond:
            if self._pending is not None:
                # keep the stricter of two frames for the same/older content
                if (self._pending.force and not item.force
                        and item.mode.strictness < self._pending.mode.strictness):
                    return
                self.metrics.inc("output_overwritten")
            self._pending = item
            self._cond.notify()

    def _run(self):
        while not self._stop.is_set():
            with self._cond:
                while self._pending is None and not self._stop.is_set():
                    self._cond.wait(0.1)
                item, self._pending = self._pending, None
            if item is None:
                continue
            if not item.force and item.seq < self.last_sent_seq:
                continue
            frame = item.frame
            # FINAL GATE: nothing weaker than the operator's current mode leaves.
            if item.mode.strictness < self.state.mode.strictness:
                frame = obscure(frame, self.gate_block, self.gate_style)
                item = OutFrame(item.seq, item.t_arrival, frame, Mode.ALL, "gate", item.force)
                self.metrics.inc("gate_frames")
            if not self.opened:
                h, w = frame.shape[:2]
                for s in self.sinks:
                    s.open(w, h, self.fps)
                self.opened = True
            if self.state.latency_probe:
                frame = probe.stamp(frame.copy())
            t0 = time.perf_counter()
            for s in self.sinks:
                try:
                    s.send(frame)
                except Exception as e:  # keep streaming to the remaining sinks
                    self.error = f"{s.name}: {e}"
            t1 = time.perf_counter()
            self.metrics.send_ms.add((t1 - t0) * 1000)
            self.metrics.pipeline_ms.add((t0 - item.t_arrival) * 1000)
            self.metrics.mark_output()
            self.last_sent_seq = max(self.last_sent_seq, item.seq)
            self.last_sent_time = t1
            self.last_sent_mode = item.mode
            self.last_tag = item.tag
            if self.on_sent:
                self.on_sent(item)
        for s in self.sinks:
            try:
                s.close()
            except Exception:
                pass
