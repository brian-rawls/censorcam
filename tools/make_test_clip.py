"""Make a short 1080p30 smoke-test clip containing real people (Ultralytics'
bundled sample photos, slowly panning). Only for checking that the pipeline
runs end to end - it says NOTHING about accuracy on GTA characters. Use your
own recorded gameplay for that (tools/eval_segmentation.py)."""

import argparse
import sys

import cv2
import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="smoke_test.mp4")
    p.add_argument("--seconds", type=float, default=4)
    a = p.parse_args(argv)
    from ultralytics.utils import ASSETS
    imgs = [cv2.imread(str(ASSETS / n)) for n in ("bus.jpg", "zidane.jpg")]
    canvas = np.full((1080, 2400, 3), 40, np.uint8)
    a_img = cv2.resize(imgs[0], (810 * imgs[0].shape[1] // imgs[0].shape[0], 810))
    b_img = cv2.resize(imgs[1], (1200, 1200 * imgs[1].shape[0] // imgs[1].shape[1]))
    canvas[100:100 + a_img.shape[0], 50:50 + a_img.shape[1]] = a_img
    y = 1080 - b_img.shape[0] - 60
    canvas[y:y + b_img.shape[0], 1150:1150 + b_img.shape[1]] = b_img
    n = int(a.seconds * 30)
    w = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    for i in range(n):
        x = int((2400 - 1920) * i / max(1, n - 1))
        w.write(canvas[:, x:x + 1920].copy())
    w.release()
    print(f"wrote {a.out} ({n} frames)")


if __name__ == "__main__":
    sys.exit(main())
