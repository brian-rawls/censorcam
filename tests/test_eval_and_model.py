import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import eval_segmentation as ev  # noqa: E402


def test_score_frame_counts_coverage():
    h, w = 200, 300
    gt1 = np.zeros((h, w), np.uint8); gt1[10:50, 10:50] = 255      # fully covered
    gt2 = np.zeros((h, w), np.uint8); gt2[100:140, 100:140] = 255  # half covered
    gt3 = np.zeros((h, w), np.uint8); gt3[150:190, 250:290] = 255  # missed
    cm = np.zeros((h, w), np.uint8); cm[0:60, 0:60] = 255; cm[100:140, 100:120] = 255
    inst, inter, gt, over = ev.score_frame([gt1, gt2, gt3], cm, 0.95)
    assert [i["status"] for i in inst] == ["covered", "partial", "missed"]
    assert inter == 1600 + 800 and gt == 3 * 1600
    assert over == pytest.approx((3600 - 1600) / (h * w))


def test_labelme_roundtrip(tmp_path):
    from censorcam.censor import Detection
    d = Detection([10, 10, 60, 90], 0.8,
                  [np.array([[10, 10], [60, 10], [60, 90], [10, 90]], np.float32)])
    p = tmp_path / "frame_000000.json"
    ev.write_labelme(p, "frame_000000.png", (100, 100, 3), [d])
    masks = ev.load_labelme(p, (100, 100, 3))
    assert len(masks) == 1 and masks[0][50, 30] == 255 and masks[0][95, 95] == 0


def _have_model():
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_model(), reason="ultralytics not installed")
def test_real_model_censors_people_in_photo():
    from ultralytics.utils import ASSETS
    from censorcam.censor import MaskBuilder, censor_people
    from censorcam.segmenter import YoloSegmenter
    img = cv2.imread(str(ASSETS / "bus.jpg"))
    seg = YoloSegmenter(os.environ.get("CENSORCAM_TEST_WEIGHTS", "yolo11n-seg.pt"),
                        device="cpu", warmup_shape=None)
    dets = seg.infer(img)
    assert len(dets) >= 3
    assert all(d.polygons for d in dets)
    out, mask = censor_people(img, dets, MaskBuilder("mask", 16, 0))
    for d in dets:  # centre of every detected person is obscured
        x1, y1, x2, y2 = map(int, d.box)
        assert mask[(y1 + y2) // 2, (x1 + x2) // 2] > 0


@pytest.mark.skipif(not _have_model(), reason="ultralytics not installed")
def test_app_end_to_end_on_clip(tmp_path):
    """Runs the real app (model on CPU) over a clip, BLUR PEOPLE mode, writes output video."""
    clip = tmp_path / "clip.mp4"
    subprocess.run([sys.executable, str(ROOT / "tools/make_test_clip.py"), "--out", str(clip),
                    "--seconds", "2"], check=True, cwd=tmp_path)
    out = tmp_path / "out.mp4"
    mj = tmp_path / "metrics.json"
    r = subprocess.run([sys.executable, "-m", "censorcam", "--video", str(clip), "--no-realtime",
                        "--output", f"file:{out}", "--start-mode", "people", "--model-device", "cpu",
                        "--weights", os.environ.get("CENSORCAM_TEST_WEIGHTS", "yolo11n-seg.pt"),
                        "--deadline-ms", "5000", "--port", "0", "--status-every", "0",
                        "--metrics-json", str(mj)],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    m = json.loads(mj.read_text())
    assert m["counters"]["frames_out"] >= 55
    assert m["counters"]["failsafe_frames"] == 0
    assert m["inference_ms"]["n"] > 0
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 55
