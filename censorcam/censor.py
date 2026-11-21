# CensorCam - Copyright (C) 2026 Aeternum Logic. Developed by Brian Rawls.
# Licensed under the GNU Affero General Public License v3.0 - see LICENSE.
"""Image operations: irreversible obscuring + mask construction/compositing.

Obscuring is done by heavy downsampling (information is destroyed, not just
smoothed), then either nearest-neighbour upscale ("pixelate") or bilinear
upscale of a further-blurred thumbnail ("blur"). Both cost ~1-2 ms at 1080p.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np


@dataclass
class Detection:
    box: Sequence[float]                      # x1, y1, x2, y2 in frame pixels
    conf: float
    polygons: List[np.ndarray] = field(default_factory=list)  # each (N, 2) float, frame pixels


def obscure(frame: np.ndarray, block: int = 48, style: str = "blur") -> np.ndarray:
    """Return a fully obscured copy of `frame`.

    block: size in pixels of the information cell. 48 at 1080p leaves a 40x22
    grid of colours -- no faces, text or silhouettes survive.
    """
    h, w = frame.shape[:2]
    sw, sh = max(1, w // block), max(1, h // block)
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    if style == "pixelate":
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    small = cv2.GaussianBlur(small, (0, 0), 1.0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def composite(frame: np.ndarray, obscured: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy `obscured` over `frame` wherever mask != 0. Returns a new array."""
    out = frame.copy()
    cv2.copyTo(obscured, mask, out)
    return out


class MaskBuilder:
    """Turns detections into a binary censor mask (uint8, 0/255, frame size).

    Safety-oriented: masks are dilated (segmentation edges are never exact) and
    optionally OR-ed with the previous `hold` frames' masks, which can only
    *add* coverage (reduces flicker when a detection drops out for a frame).
    Work is done at 1/`scale` resolution for speed.
    """

    def __init__(self, shape: str = "mask", dilate_px: int = 16, hold: int = 1,
                 box_pad: float = 0.08, scale: int = 4):
        if shape not in ("mask", "box", "both"):
            raise ValueError("shape must be mask | box | both")
        self.shape = shape
        self.dilate_px = dilate_px
        self.hold = max(0, hold)
        self.box_pad = box_pad
        self.scale = max(1, scale)
        self._history: List[np.ndarray] = []

    def reset(self) -> None:
        self._history.clear()

    def build(self, dets: Sequence[Detection], frame_shape) -> np.ndarray:
        h, w = frame_shape[:2]
        s = self.scale
        sh, sw = (h + s - 1) // s, (w + s - 1) // s
        small = np.zeros((sh, sw), np.uint8)
        for d in dets:
            drew = False
            if self.shape in ("mask", "both") and d.polygons:
                pts = [np.round(np.asarray(p, np.float32) / s).astype(np.int32)
                       for p in d.polygons if p is not None and len(p) >= 3]
                if pts:
                    cv2.fillPoly(small, pts, 255)
                    drew = True
            if self.shape in ("box", "both") or not drew:
                # box mode, or fallback when the model gave no usable polygon
                x1, y1, x2, y2 = [float(v) for v in d.box]
                px, py = (x2 - x1) * self.box_pad, (y2 - y1) * self.box_pad
                cv2.rectangle(small,
                              (int((x1 - px) / s), int((y1 - py) / s)),
                              (int((x2 + px) / s) + 1, int((y2 + py) / s) + 1), 255, -1)
        r = int(round(self.dilate_px / s))
        if r > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            small = cv2.dilate(small, k)
        combined = small
        for prev in self._history:
            if prev.shape == small.shape:
                combined = cv2.bitwise_or(combined, prev)
        if self.hold:
            self._history.append(small)
            self._history = self._history[-self.hold:]
        if s == 1:
            return combined
        return cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)


def censor_people(frame: np.ndarray, dets: Sequence[Detection], builder: MaskBuilder,
                  block: int = 32, style: str = "pixelate") -> tuple[np.ndarray, np.ndarray]:
    mask = builder.build(dets, frame.shape)
    if not mask.any():
        return frame, mask
    return composite(frame, obscure(frame, block, style), mask), mask


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.45) -> np.ndarray:
    """Debug visualisation: tint censored area and outline it."""
    out = frame.copy()
    tint = np.zeros_like(frame)
    tint[:] = color
    m = mask > 0
    out[m] = (out[m] * (1 - alpha) + tint[m] * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out
