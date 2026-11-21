# CensorCam - Copyright (C) 2026 Aeternum Logic. Developed by Brian Rawls.
# Licensed under the GNU Affero General Public License v3.0 - see LICENSE.
"""Optional: export the model to a TensorRT FP16 engine for lower, steadier inference time.

    python tools/export_tensorrt.py --weights yolo11s-seg.pt
    -> yolo11s-seg.engine   (then run CensorCam with --weights yolo11s-seg.engine)

Takes a few minutes and is tied to your GPU + TensorRT version: re-export after
driver/TensorRT upgrades. Ultralytics will try to pip-install TensorRT itself;
if that fails on your machine, skip this - PyTorch FP16 on a 4070 Ti Super is
already well inside the 33 ms frame budget. Compare both with:
    python -m censorcam --video clip.mp4 --no-realtime --output null --start-mode people --weights X
"""

import argparse

from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--weights", default="yolo11s-seg.pt")
p.add_argument("--imgsz", type=int, default=640)
a = p.parse_args()
path = YOLO(a.weights).export(format="engine", half=True, imgsz=a.imgsz, device=0,
                              dynamic=False, batch=1)
print(f"exported: {path}")
