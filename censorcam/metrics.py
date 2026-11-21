# CensorCam - Copyright (C) 2026 Aeternum Logic. Developed by Brian Rawls.
# Licensed under the GNU Affero General Public License v3.0 - see LICENSE.
"""Rolling performance metrics."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict


class Rolling:
    def __init__(self, n: int = 300):
        self.v: Deque[float] = deque(maxlen=n)

    def add(self, x: float) -> None:
        self.v.append(x)

    def summary(self) -> Dict[str, float]:
        if not self.v:
            return {"n": 0}
        s = sorted(self.v)
        n = len(s)
        return {
            "n": n,
            "mean": round(sum(s) / n, 2),
            "p50": round(s[n // 2], 2),
            "p95": round(s[min(n - 1, int(n * 0.95))], 2),
            "max": round(s[-1], 2),
        }


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.inference_ms = Rolling()
        self.process_ms = Rolling()        # frame arrival -> processed frame ready
        self.pipeline_ms = Rolling()       # frame arrival -> handed to output (Spout etc.)
        self.send_ms = Rolling()           # time spent inside the output sink
        self.downstream_ms = Rolling(120)  # reported by tools/measure_latency.py
        self.counters: Dict[str, int] = {
            "frames_out": 0,
            "output_overwritten": 0,   # processed but replaced before sink took it
            "failsafe_frames": 0,      # BLUR PEOPLE frames sent fully obscured
            "inference_errors": 0,
            "deadline_misses": 0,
            "watchdog_frames": 0,
            "failsafe_latches": 0,
            "model_reloads": 0,
        }
        self._out_times: Deque[float] = deque(maxlen=120)

    def inc(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.counters[key] = self.counters.get(key, 0) + n

    def mark_output(self) -> None:
        self._out_times.append(time.perf_counter())
        self.inc("frames_out")

    def out_fps(self) -> float:
        t = list(self._out_times)
        if len(t) < 2 or t[-1] == t[0]:
            return 0.0
        return round((len(t) - 1) / (t[-1] - t[0]), 2)

    def snapshot(self, extra: Dict | None = None) -> Dict:
        d = {
            "uptime_s": round(time.monotonic() - self.started, 1),
            "output_fps": self.out_fps(),
            "inference_ms": self.inference_ms.summary(),
            "process_ms": self.process_ms.summary(),
            "pipeline_latency_ms": self.pipeline_ms.summary(),
            "sink_send_ms": self.send_ms.summary(),
            "downstream_latency_ms": self.downstream_ms.summary(),
            "counters": dict(self.counters),
        }
        if extra:
            d.update(extra)
        return d
