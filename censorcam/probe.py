# CensorCam - Copyright (C) 2026 Aeternum Logic. Developed by Brian Rawls.
# Licensed under the GNU Affero General Public License v3.0 - see LICENSE.
"""Latency probe: a small barcode strip stamped onto outgoing frames.

The strip encodes the wall-clock time (ms, mod 2^24) at which the frame left
CensorCam. tools/measure_latency.py screen-grabs the strip wherever it is
displayed (OBS preview, TikTok LIVE Studio preview, a projector) and compares
to the current time -> downstream latency of OBS + Virtual Camera + TikTok.

Layout (in cells): a 1-cell black border around [1,0 sync][24 data bits][parity],
i.e. 29 x 3 cells. White = 1. Only enable while testing - it is visible on stream.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

BITS = 24
CELLS = 2 + BITS + 1          # sync + data + parity
GRID_W, GRID_H = CELLS + 2, 3  # with border
MOD = 1 << BITS


def now_code() -> int:
    return int(time.time() * 1000) % MOD


def _bits(value: int):
    data = [(value >> (BITS - 1 - i)) & 1 for i in range(BITS)]
    return [1, 0] + data + [sum(data) % 2]


def stamp(frame: np.ndarray, value: Optional[int] = None, cell: Optional[int] = None,
          position: str = "center") -> np.ndarray:
    """Draw the strip in-place and return frame."""
    h, w = frame.shape[:2]
    cell = cell or max(4, h // 60)  # 18 px at 1080p -> 522 px wide, fits a 9:16 centre crop
    value = now_code() if value is None else value
    sw, sh = GRID_W * cell, GRID_H * cell
    if position == "center":
        x0, y0 = (w - sw) // 2, (h - sh) // 2
    else:  # bottom-left
        x0, y0 = cell, h - sh - cell
    frame[y0:y0 + sh, x0:x0 + sw] = 0
    for i, b in enumerate(_bits(value)):
        if b:
            x = x0 + (i + 1) * cell
            frame[y0 + cell:y0 + 2 * cell, x:x + cell] = 255
    return frame


def decode(strip: np.ndarray) -> Optional[int]:
    """Decode an image cropped (roughly tightly) around the strip incl. border."""
    if strip.ndim == 3:
        strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    h, w = strip.shape[:2]
    if w < GRID_W * 2 or h < 3:
        return None
    y = int(h * 1.5 / GRID_H)
    cw = w / GRID_W
    row = strip[max(0, y - 1):y + 2]
    bits = []
    for i in range(CELLS):
        cx = int((i + 1.5) * cw)
        x0, x1 = max(0, cx - max(1, int(cw / 4))), cx + max(1, int(cw / 4)) + 1
        bits.append(1 if row[:, x0:x1].mean() > 127 else 0)
    if bits[0] != 1 or bits[1] != 0:
        return None
    data = bits[2:2 + BITS]
    if sum(data) % 2 != bits[-1]:
        return None
    v = 0
    for b in data:
        v = (v << 1) | b
    return v


def age_ms(code: int, now: Optional[int] = None) -> int:
    now = now_code() if now is None else now
    return (now - code) % MOD
