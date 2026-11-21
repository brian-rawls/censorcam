# CensorCam - Copyright (C) 2026 Aeternum Logic. Developed by Brian Rawls.
# Licensed under the GNU Affero General Public License v3.0 - see LICENSE.
"""Frame sources with "latest frame wins" semantics.

A dedicated thread reads the device continuously so the driver buffer never
fills (that is where hidden latency usually comes from). Consumers always get
the newest frame; any frame overwritten before it was consumed is counted as a
capture drop.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class FrameSource:
    def __init__(self):
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._t = 0.0
        self._consumed_seq = 0
        self.frames_in = 0
        self.dropped = 0
        self.ended = False
        self.error: Optional[str] = None
        self.width = 0
        self.height = 0
        self.fps = 30.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- producer side -------------------------------------------------
    def _publish(self, frame: np.ndarray) -> None:
        with self._cond:
            if self._frame is not None and self._seq > self._consumed_seq:
                self.dropped += 1
            self._frame = frame
            self._seq += 1
            self._t = time.perf_counter()
            self.frames_in += 1
            self._cond.notify_all()

    def _end(self, error: Optional[str] = None) -> None:
        with self._cond:
            self.ended = True
            self.error = error
            self._cond.notify_all()

    # -- consumer side -------------------------------------------------
    def get_latest(self, after_seq: int, timeout: float = 0.5
                   ) -> Optional[Tuple[int, float, np.ndarray]]:
        """Block until a frame newer than `after_seq` exists; return (seq, t_arrival, frame)."""
        deadline = time.perf_counter() + timeout
        with self._cond:
            while self._seq <= after_seq and not self.ended:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            if self._seq <= after_seq:
                return None
            self._consumed_seq = self._seq
            return self._seq, self._t, self._frame

    def peek(self) -> Optional[Tuple[int, float, np.ndarray]]:
        with self._cond:
            if self._frame is None:
                return None
            return self._seq, self._t, self._frame

    def start(self) -> "FrameSource":
        self._thread = threading.Thread(target=self._run, name=type(self).__name__, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


def resolve_device_index(device: str) -> int:
    """Accept an integer index or a DirectShow device name (needs pygrabber)."""
    if str(device).isdigit():
        return int(device)
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError as e:
        raise SystemExit("Device names need `pip install pygrabber`; or pass a numeric index "
                         "(run with --list-devices).") from e
    names = FilterGraph().get_input_devices()
    for i, n in enumerate(names):
        if device.lower() in n.lower():
            return i
    raise SystemExit(f"No video device matching {device!r}. Found: {names}")


def list_devices() -> list:
    try:
        from pygrabber.dshow_graph import FilterGraph
        return list(enumerate(FilterGraph().get_input_devices()))
    except Exception:
        found = []
        for i in range(8):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY)
            if cap.isOpened():
                found.append((i, f"device {i} ({int(cap.get(3))}x{int(cap.get(4))})"))
            cap.release()
        return found


class DeviceSource(FrameSource):
    """Elgato HD60 X (or any UVC/DirectShow device)."""

    def __init__(self, device: str = "0", width: int = 1920, height: int = 1080,
                 fps: float = 30.0, fourcc: str = "", backend: str = "dshow"):
        super().__init__()
        self.index = resolve_device_index(device)
        self.req = (width, height, fps)
        self.fourcc = fourcc
        api = {"dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
               "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
               "any": cv2.CAP_ANY}[backend]
        self.cap = cv2.VideoCapture(self.index, api)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open capture device {device}. Close Elgato Studio / "
                             "4K Capture Utility and remove the HD60 X *video* source from OBS - "
                             "only one app can own it.")
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        # If the device delivers 60 fps but we want 30, decimate in software.
        self.decimate = max(1, int(round(self.fps / fps))) if self.fps > fps * 1.5 else 1

    def _run(self) -> None:
        n = 0
        fails = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 60:
                    self._end("capture device stopped delivering frames")
                    break
                time.sleep(0.005)
                continue
            fails = 0
            n += 1
            if n % self.decimate:
                continue
            self._publish(frame)
        self.cap.release()


class FileSource(FrameSource):
    """Recorded gameplay. realtime=True paces playback at the file's fps so
    the pipeline sees the same timing as a live capture card."""

    def __init__(self, path: str, realtime: bool = True, loop: bool = False,
                 max_frames: int = 0):
        super().__init__()
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open video file {path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.realtime = realtime
        self.loop = loop
        self.max_frames = max_frames
        self.decimate = 1

    def _run(self) -> None:
        period = 1.0 / self.fps
        nxt = time.perf_counter()
        count = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            if self.realtime:
                nxt += period
                delay = nxt - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    nxt = time.perf_counter()
            else:
                # benchmark mode: wait until the previous frame was consumed
                with self._cond:
                    while self._seq > self._consumed_seq and not self._stop.is_set():
                        self._cond.wait(0.05)
            self._publish(frame)
            count += 1
            if self.max_frames and count >= self.max_frames:
                break
        self.cap.release()
        self._end()


class ArraySource(FrameSource):
    """Push frames manually (tests)."""

    def _run(self) -> None:
        self._stop.wait()

    def push(self, frame: np.ndarray) -> None:
        self._publish(frame)
