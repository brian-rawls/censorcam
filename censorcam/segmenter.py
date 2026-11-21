"""Person segmentation backends.

The default backend is an Ultralytics YOLO segmentation model restricted to
COCO class 0 ("person"). It is loaded once at startup and kept resident in GPU
memory for the life of the process, so switching into BLUR PEOPLE costs nothing.

IMPORTANT: COCO person models were trained on real photos. They often work on
photoreal game characters, but small/distant, dark, partially occluded, seated
in vehicles, or stylised characters can be missed. Measure this on your own
footage with tools/eval_segmentation.py before relying on it.
"""

from __future__ import annotations

import time
from typing import List, Optional, Protocol, Sequence

import numpy as np

from .censor import Detection


class Segmenter(Protocol):
    name: str

    def infer(self, frame: np.ndarray) -> List[Detection]: ...


class YoloSegmenter:
    def __init__(self, weights: str = "yolo11s-seg.pt", device: Optional[str] = None,
                 imgsz: int = 640, conf: float = 0.20, iou: float = 0.6,
                 classes: Sequence[int] = (0,), half: Optional[bool] = None,
                 warmup_shape=(1080, 1920, 3)):
        import torch
        from ultralytics import YOLO

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.is_cuda = str(device).startswith("cuda") or str(device).isdigit()
        self.half = self.is_cuda if half is None else (half and self.is_cuda)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.classes = list(classes)
        self.weights = weights
        self.name = f"{weights}@{device}{' fp16' if self.half else ''}"
        self.model = YOLO(weights, task="segment")
        # Ultralytics >= 8.4.2xx replaced `half=` with `quantize=`; support both.
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            new_api = "quantize" in DEFAULT_CFG_DICT
        except Exception:
            new_api = False
        if new_api:
            self._prec = {"quantize": 16} if self.half else {}
        else:
            self._prec = {"half": self.half}
        if warmup_shape:
            self.warmup(warmup_shape)

    def warmup(self, shape=(1080, 1920, 3), n: int = 3) -> float:
        dummy = np.zeros(shape, np.uint8)
        t = 0.0
        for _ in range(n):
            t0 = time.perf_counter()
            self.infer(dummy)
            t = (time.perf_counter() - t0) * 1000
        return t

    def infer(self, frame: np.ndarray) -> List[Detection]:
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                               classes=self.classes, device=self.device, **self._prec,
                               verbose=False, retina_masks=False)[0]
        out: List[Detection] = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        polys = list(r.masks.xy) if r.masks is not None else [None] * len(boxes)
        for b, c, p in zip(boxes, confs, polys):
            plist = [np.asarray(p, np.float32)] if p is not None and len(p) >= 3 else []
            out.append(Detection(box=b.tolist(), conf=float(c), polygons=plist))
        return out


class FakeSegmenter:
    """Deterministic stand-in for tests and for exercising the fail-safe.

    behaviour hooks: delay_s, raise_exc, hang_s, on_infer callback.
    """

    name = "fake"

    def __init__(self, dets: Optional[List[Detection]] = None, delay_s: float = 0.0):
        self.dets = dets or []
        self.delay_s = delay_s
        self.raise_exc: Optional[BaseException] = None
        self.on_infer = None
        self.calls = 0

    def infer(self, frame: np.ndarray) -> List[Detection]:
        self.calls += 1
        if self.on_infer:
            self.on_infer(self)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.raise_exc is not None:
            raise self.raise_exc
        return list(self.dets)
