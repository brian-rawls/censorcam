# CensorCam - Copyright (C) 2026 Aeternum Logic. Developed by Brian Rawls.
# Licensed under the GNU Affero General Public License v3.0 - see LICENSE.
"""Frame processor, fail-safe logic and watchdog.

Safety model while BLUR PEOPLE is active (fail-closed):
  * The censor mask for a frame is computed from THAT frame. No stale masks.
  * Inference error / no model loaded       -> frame is sent fully obscured.
  * Inference slower than the deadline      -> frame is sent fully obscured.
  * >= N failures within a window           -> "latched": every frame is fully
                                                obscured for hold_s, extended
                                                while failures continue.
  * Processing thread stalls (hang)         -> watchdog sends fully obscured
                                                copies of the newest raw frame.
  * Operator switches mode mid-frame        -> stricter mode wins for that frame.
  * Final gate in the output thread         -> any frame weaker than the
                                                current mode is obscured.
The raw capture is never connected to OBS, so a crash freezes OBS on the
last (already censored) frame instead of revealing raw video.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .capture import FrameSource
from .censor import MaskBuilder, censor_people, obscure
from .metrics import Metrics
from .outputs import OutFrame, OutputWorker
from .state import Mode, OperatorState


@dataclass
class PipelineConfig:
    deadline_ms: float = 25.0       # max inference time before a frame fails closed
    people_block: int = 32          # pixel size of the per-character pixelation
    people_style: str = "pixelate"  # pixelate | blur
    all_block: int = 48
    all_style: str = "blur"
    latch_failures: int = 3         # failures within latch_window_s that latch fail-safe
    latch_window_s: float = 1.0
    latch_hold_s: float = 2.0
    watchdog_ms: float = 120.0      # output silence in a censored mode before watchdog acts
    keepwarm_s: float = 1.0         # run a throw-away inference this often in other modes (0=off)
    reload_after_errors: int = 5    # consecutive exceptions before reloading the model
    latch_probe_every: int = 5      # while latched, only test the model on every Nth frame
                                    # so a slow/broken model can't also drag down output fps


class Processor:
    def __init__(self, source: FrameSource, segmenter, builder: MaskBuilder,
                 state: OperatorState, output: OutputWorker, metrics: Metrics,
                 cfg: Optional[PipelineConfig] = None,
                 segmenter_factory: Optional[Callable[[], object]] = None):
        self.source = source
        self.segmenter = segmenter
        self.builder = builder
        self.state = state
        self.output = output
        self.metrics = metrics
        self.cfg = cfg or PipelineConfig()
        self.factory = segmenter_factory
        self._fail_times: deque = deque()
        self._latch_until = 0.0
        self._consecutive_errors = 0
        self._last_warm = 0.0
        self._prev_mode: Optional[Mode] = None
        self._reloading = False
        self._latched_frames = 0
        self._warming = False
        self.last_error: Optional[str] = None
        self.last_heartbeat = time.perf_counter()
        self._seg_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="Processor", daemon=True)

    # ---- fail-safe bookkeeping -------------------------------------------
    @property
    def latched(self) -> bool:
        return time.monotonic() < self._latch_until

    def _fail(self, reason: str) -> None:
        now = time.monotonic()
        self.last_error = reason
        self._fail_times.append(now)
        while self._fail_times and now - self._fail_times[0] > self.cfg.latch_window_s:
            self._fail_times.popleft()
        if self.latched:
            self._latch_until = max(self._latch_until, now + self.cfg.latch_hold_s)
        elif len(self._fail_times) >= self.cfg.latch_failures:
            self._latch_until = now + self.cfg.latch_hold_s
            self.metrics.inc("failsafe_latches")
            print(f"[failsafe] LATCHED full censorship ({reason})", flush=True)

    def _full(self, frame: np.ndarray) -> np.ndarray:
        return obscure(frame, self.cfg.all_block, self.cfg.all_style)

    def _reload_async(self) -> None:
        if self._reloading or self.factory is None:
            return
        self._reloading = True
        self.metrics.inc("model_reloads")

        def work():
            try:
                print("[model] reloading after repeated errors ...", flush=True)
                self.segmenter = None
                self.segmenter = self.factory()
                self._consecutive_errors = 0
                print("[model] reloaded", flush=True)
            except Exception as e:
                self.last_error = f"reload failed: {e}"
            finally:
                self._reloading = False

        threading.Thread(target=work, daemon=True).start()

    # ---- per-frame logic ---------------------------------------------------
    def _people(self, frame: np.ndarray):
        seg = self.segmenter
        if self.latched:
            self._latched_frames += 1
            if self._latched_frames % max(1, self.cfg.latch_probe_every):
                return self._full(frame), "failsafe"
        else:
            self._latched_frames = 0
        if seg is None:
            self._fail("model not loaded")
            return self._full(frame), "failsafe"
        t0 = time.perf_counter()
        try:
            with self._seg_lock:
                dets = seg.infer(frame)
        except Exception as e:
            self.metrics.inc("inference_errors")
            self._consecutive_errors += 1
            self._fail(f"inference error: {e!r}")
            if self._consecutive_errors >= self.cfg.reload_after_errors:
                self._reload_async()
            return self._full(frame), "failsafe"
        dt = (time.perf_counter() - t0) * 1000
        self.metrics.inference_ms.add(dt)
        self._consecutive_errors = 0
        if dt > self.cfg.deadline_ms:
            self.metrics.inc("deadline_misses")
            self._fail(f"inference {dt:.1f} ms > deadline {self.cfg.deadline_ms} ms")
            return self._full(frame), "failsafe"
        if self.latched:
            return self._full(frame), "failsafe"
        out, _ = censor_people(frame, dets, self.builder, self.cfg.people_block,
                               self.cfg.people_style)
        return out, "people"

    def process(self, seq: int, t_arrival: float, frame: np.ndarray) -> OutFrame:
        mode0 = self.state.mode
        if mode0 == Mode.PEOPLE and self._prev_mode != Mode.PEOPLE:
            self.builder.reset()
        self._prev_mode = mode0

        if mode0 == Mode.NORMAL:
            out, applied, tag = frame, Mode.NORMAL, "normal"
        elif mode0 == Mode.ALL:
            out, applied, tag = self._full(frame), Mode.ALL, "all"
        else:
            out, tag = self._people(frame)
            applied = Mode.PEOPLE if tag == "people" else Mode.ALL
            if tag == "failsafe":
                self.metrics.inc("failsafe_frames")

        # Operator may have pressed a button while we were working: stricter wins.
        mode1 = self.state.mode
        if mode1.strictness > applied.strictness:
            out, applied, tag = self._full(frame), Mode.ALL, "switch"

        self.metrics.process_ms.add((time.perf_counter() - t_arrival) * 1000)
        return OutFrame(seq, t_arrival, out, applied, tag)

    def _maybe_keepwarm(self, frame: np.ndarray) -> None:
        """Keeps CUDA kernels/clocks warm so the first BLUR PEOPLE frame is fast.
        Runs on a separate background thread so it never delays a video frame."""
        if (self.cfg.keepwarm_s <= 0 or self.state.mode == Mode.PEOPLE
                or self.segmenter is None or self._warming):
            return
        now = time.monotonic()
        if now - self._last_warm < self.cfg.keepwarm_s:
            return
        self._last_warm = now
        self._warming = True
        seg = self.segmenter

        def work():
            try:
                with self._seg_lock:
                    if self.state.mode != Mode.PEOPLE:
                        seg.infer(frame)
            except Exception:
                pass
            finally:
                self._warming = False

        threading.Thread(target=work, name="KeepWarm", daemon=True).start()

    # ---- thread -------------------------------------------------------------
    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def join(self, timeout=None):
        self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self):
        last_seq = 0
        while not self._stop.is_set():
            self.last_heartbeat = time.perf_counter()
            got = self.source.get_latest(last_seq, timeout=0.25)
            if got is None:
                if self.source.ended:
                    break
                continue
            seq, t, frame = got
            last_seq = seq
            item = self.process(seq, t, frame)
            self.output.post(item)
            self._maybe_keepwarm(frame)


class Watchdog:
    """Independent thread that guarantees a censored picture whenever the
    operator has selected a censored mode, even if the processor is stuck."""

    def __init__(self, source: FrameSource, output: OutputWorker, state: OperatorState,
                 metrics: Metrics, cfg: PipelineConfig):
        self.source, self.output, self.state = source, output, state
        self.metrics, self.cfg = metrics, cfg
        self._kick = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="Watchdog", daemon=True)
        state.add_listener(lambda old, new: self._kick.set())

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._kick.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            self._kick.wait(0.01)
            self._kick.clear()
            mode = self.state.mode
            if mode == Mode.NORMAL:
                continue
            stale = (time.perf_counter() - self.output.last_sent_time) * 1000
            weaker_on_screen = self.output.last_sent_mode.strictness < mode.strictness
            if not (weaker_on_screen or stale > self.cfg.watchdog_ms):
                continue
            p = self.source.peek()
            if p is None:
                continue
            seq, t, frame = p
            full = obscure(frame, self.cfg.all_block, self.cfg.all_style)
            self.output.post(OutFrame(seq, t, full, Mode.ALL, "watchdog", force=True))
            self.metrics.inc("watchdog_frames")
            # avoid re-posting before the output thread had a chance to send
            time.sleep(0.005)
