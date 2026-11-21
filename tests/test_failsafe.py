"""Fail-closed guarantees for BLUR PEOPLE."""

import time

import numpy as np

from censorcam.capture import ArraySource
from censorcam.censor import Detection, MaskBuilder, obscure
from censorcam.metrics import Metrics
from censorcam.outputs import ListSink, OutputWorker
from censorcam.pipeline import PipelineConfig, Processor, Watchdog
from censorcam.segmenter import FakeSegmenter
from censorcam.state import Mode, OperatorState

from conftest import frame_with_id

DET = Detection([800, 300, 1000, 900], 0.9)
REGION = (slice(320, 880), slice(820, 980))


def make(seg, mode=Mode.PEOPLE, **cfg):
    state = OperatorState(mode)
    metrics = Metrics()
    sink = ListSink()
    out = OutputWorker([sink], metrics, state)
    c = PipelineConfig(keepwarm_s=0, **cfg)
    src = ArraySource()
    proc = Processor(src, seg, MaskBuilder("box", 8, 0), state, out, metrics, c)
    return state, metrics, sink, out, src, proc


def is_full(out, frame, cfg=PipelineConfig()):
    return np.array_equal(out, obscure(frame, cfg.all_block, cfg.all_style))


def test_success_blurs_people_only(frame):
    _, m, _, _, _, proc = make(FakeSegmenter([DET]))
    item = proc.process(1, time.perf_counter(), frame)
    assert item.tag == "people" and item.mode == Mode.PEOPLE
    assert not np.array_equal(item.frame[REGION], frame[REGION])
    assert np.array_equal(item.frame[:200, :200], frame[:200, :200])


def test_exception_fails_closed(frame):
    seg = FakeSegmenter([DET])
    seg.raise_exc = RuntimeError("CUDA error: illegal memory access")
    _, m, _, _, _, proc = make(seg)
    item = proc.process(1, time.perf_counter(), frame)
    assert item.tag == "failsafe" and is_full(item.frame, frame)
    assert m.counters["inference_errors"] == 1


def test_no_model_fails_closed(frame):
    _, _, _, _, _, proc = make(None)
    item = proc.process(1, time.perf_counter(), frame)
    assert item.tag == "failsafe" and is_full(item.frame, frame)


def test_deadline_miss_fails_closed(frame):
    _, m, _, _, _, proc = make(FakeSegmenter([DET], delay_s=0.05), deadline_ms=20)
    item = proc.process(1, time.perf_counter(), frame)
    assert item.tag == "failsafe" and is_full(item.frame, frame)
    assert m.counters["deadline_misses"] == 1


def test_latch_holds_after_repeated_failures(frame):
    seg = FakeSegmenter([DET])
    seg.raise_exc = RuntimeError("boom")
    _, m, _, _, _, proc = make(seg, latch_failures=3, latch_hold_s=0.3)
    for i in range(3):
        proc.process(i, time.perf_counter(), frame)
    assert proc.latched and m.counters["failsafe_latches"] == 1
    seg.raise_exc = None  # model "recovers"...
    item = proc.process(9, time.perf_counter(), frame)
    assert item.tag == "failsafe" and is_full(item.frame, frame)  # ...but latch still holds
    time.sleep(0.35)
    assert proc.process(10, time.perf_counter(), frame).tag == "people"


def test_switch_to_all_during_inference_wins(frame):
    seg = FakeSegmenter([DET])
    state, _, _, _, _, proc = make(seg)
    seg.on_infer = lambda s: state.set_mode(Mode.ALL, "test")
    item = proc.process(1, time.perf_counter(), frame)
    assert item.tag == "switch" and is_full(item.frame, frame)


def test_final_gate_obscures_weaker_frame(frame):
    """A NORMAL frame that reaches the output thread after the operator chose
    BLUR PEOPLE must not go out raw."""
    from censorcam.outputs import OutFrame
    state, m, sink, out, _, _ = make(FakeSegmenter(), mode=Mode.NORMAL)
    out.start()
    state.set_mode(Mode.PEOPLE, "test")
    out.post(OutFrame(1, time.perf_counter(), frame, Mode.NORMAL, "normal"))
    time.sleep(0.2)
    out.stop()
    assert len(sink.items) == 1 and is_full(sink.items[0], frame)
    assert m.counters["gate_frames"] == 1


def _run_threads(seg, n_frames, mode=Mode.PEOPLE, fps=60, **cfg):
    state, m, sink, out, src, proc = make(seg, mode, **cfg)
    sent = []
    out.on_sent = lambda item: sent.append(item)
    dog = Watchdog(src, out, state, m, proc.cfg)
    out.start(); src.start(); dog.start(); proc.start()
    raws = []
    for i in range(n_frames):
        f = frame_with_id(i)
        raws.append(f)
        src.push(f)
        time.sleep(1 / fps)
    time.sleep(0.3)
    proc.stop(); dog.stop(); src.stop(); out.stop()
    return state, m, sink, sent, raws


def test_no_raw_region_ever_leaks_under_random_failures():
    """Integration: errors, slow frames and good frames interleaved. In every
    frame that left the app, the character region must be obscured."""
    seg = FakeSegmenter([DET])
    import random
    rnd = random.Random(1)

    def chaos(s):
        r = rnd.random()
        s.raise_exc = RuntimeError("x") if r < 0.2 else None
        s.delay_s = 0.04 if 0.2 <= r < 0.35 else 0.0

    seg.on_infer = chaos
    _, m, sink, sent, raws = _run_threads(seg, 60, deadline_ms=25, latch_hold_s=0.2)
    assert len(sink.items) > 20
    for f in sink.items:
        for raw in raws:
            assert not np.array_equal(f[REGION], raw[REGION])
    assert m.counters["failsafe_frames"] > 0


def test_watchdog_covers_hung_inference():
    seg = FakeSegmenter([DET])
    hung = {"n": 0}

    def hang(s):
        hung["n"] += 1
        if hung["n"] == 5:
            time.sleep(1.0)  # simulated driver hang

    seg.on_infer = hang
    _, m, sink, sent, raws = _run_threads(seg, 60, fps=30, watchdog_ms=120)
    assert m.counters["watchdog_frames"] >= 3
    # the watchdog frames are fully obscured copies of real raw frames
    wd = [it for it in sent if it.tag == "watchdog"]
    assert wd and all(any(is_full(it.frame, r) for r in raws) for it in wd[:3])


def test_mode_change_censors_instantly_even_without_new_frames():
    """Press BLUR ALL while the capture is frozen: the picture on screen must
    become obscured right away, not at the next captured frame."""
    state, m, sink, out, src, proc = make(FakeSegmenter(), mode=Mode.NORMAL)
    dog = Watchdog(src, out, state, m, proc.cfg)
    out.start(); src.start(); dog.start(); proc.start()
    raw = frame_with_id(42)
    src.push(raw)
    time.sleep(0.15)
    assert np.array_equal(sink.items[-1], raw)  # NORMAL passes through
    t0 = time.perf_counter()
    state.set_mode(Mode.ALL, "test")
    while not is_full(sink.items[-1], raw):
        time.sleep(0.001)
        assert time.perf_counter() - t0 < 0.1, "BLUR ALL took longer than 100 ms"
    proc.stop(); dog.stop(); src.stop(); out.stop()


def test_blur_all_skips_inference(frame):
    seg = FakeSegmenter([DET], delay_s=0.5)
    _, _, _, _, _, proc = make(seg, mode=Mode.ALL)
    t0 = time.perf_counter()
    item = proc.process(1, t0, frame)
    assert seg.calls == 0 and time.perf_counter() - t0 < 0.1
    assert item.tag == "all" and is_full(item.frame, frame)
