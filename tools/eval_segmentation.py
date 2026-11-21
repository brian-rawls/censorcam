# CensorCam - Copyright (C) 2026 Aeternum Logic. Developed by Brian Rawls.
# Licensed under the GNU Affero General Public License v3.0 - see LICENSE.
"""Measure how well the person model censors characters in YOUR gameplay footage.

Do not assume a COCO "person" model sees game characters. This tool tells you
how often it does, on footage you recorded, using the exact same mask logic
(dilation, box fallback, hold) that the live app uses.

Workflow
--------
1) Extract frames to label (every 15th frame), pre-labelled by the model so
   you only have to FIX mistakes instead of drawing everything:

     python tools/eval_segmentation.py --video gta6_city.mp4 --extract eval/frames --every 15 --prelabel

2) Open eval/frames in labelme (pip install labelme; `labelme eval/frames`).
   For each frame: delete wrong polygons, ADD every human character the model
   missed (label "person"), adjust bad outlines. Save. Frames with nobody in
   them still need their (empty) JSON saved, so they count as negatives.
   Aim for 150-300 frames covering: daylight/night, crowds, distant pedestrians,
   people in cars, cutscenes, first-person, water/beach, interiors, rain.

3) Score one or more models against your labels:

     python tools/eval_segmentation.py --video gta6_city.mp4 --labels eval/frames --every 15 \
         --weights yolo11s-seg.pt,yolo26s-seg.pt,yolo11m-seg.pt --out eval/report

   -> eval/report/<model>/report.md, per_frame.csv, failures/*.jpg (+ summary.md comparing models)

4) Without labels you still get a review video to eyeball:

     python tools/eval_segmentation.py --video gta6_city.mp4 --review eval/review.mp4

Key metrics (higher = safer)
  pixel_recall      share of all labelled character pixels that ended up obscured
  instances_covered share of characters >= --cover-thresh obscured (default 95 %)
  frames_clean      share of labelled frames where EVERY character was covered
  over_blur         share of frame area obscured that was NOT a character (cost)
Broken down by character size: small (<0.1 % of frame, e.g. distant pedestrians),
medium (0.1-1 %), large (>1 %).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from censorcam.censor import MaskBuilder, censor_people, overlay_mask  # noqa: E402

HUMAN_LABELS = {"person", "character", "npc", "human", "people", "player"}


def frame_name(idx: int) -> str:
    return f"frame_{idx:06d}"


def load_labelme(path: Path, shape, labels=HUMAN_LABELS):
    """Return list of instance masks (uint8) from a labelme JSON file."""
    d = json.loads(path.read_text())
    h, w = shape[:2]
    out = []
    for s in d.get("shapes", []):
        if s.get("label", "").lower() not in labels:
            continue
        pts = np.array(s["points"], np.float32)
        m = np.zeros((h, w), np.uint8)
        st = s.get("shape_type", "polygon")
        if st == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            cv2.rectangle(m, (int(min(x1, x2)), int(min(y1, y2))),
                          (int(max(x1, x2)), int(max(y1, y2))), 255, -1)
        elif len(pts) >= 3:
            cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255)
        else:
            continue
        out.append(m)
    return out


def write_labelme(path: Path, image_name: str, shape, dets):
    shapes = []
    for d in dets:
        polys = d.polygons or [np.array([[d.box[0], d.box[1]], [d.box[2], d.box[1]],
                                         [d.box[2], d.box[3]], [d.box[0], d.box[3]]], np.float32)]
        for p in polys:
            p = p[:: max(1, len(p) // 40)]  # thin out points so editing is pleasant
            shapes.append({"label": "person", "points": p.round(1).tolist(),
                           "group_id": None, "shape_type": "polygon", "flags": {},
                           "description": f"model conf {d.conf:.2f}"})
    path.write_text(json.dumps({"version": "5.4.1", "flags": {}, "shapes": shapes,
                                "imagePath": image_name, "imageData": None,
                                "imageHeight": shape[0], "imageWidth": shape[1]}, indent=1))


def size_bucket(area_frac: float) -> str:
    return "small" if area_frac < 0.001 else ("medium" if area_frac < 0.01 else "large")


def score_frame(gt_masks, censor_mask, cover_thresh):
    h, w = censor_mask.shape
    cm = censor_mask > 0
    union = np.zeros((h, w), bool)
    inst = []
    inter_total = gt_total = 0
    for g in gt_masks:
        gb = g > 0
        area = int(gb.sum())
        if area == 0:
            continue
        inter = int((gb & cm).sum())
        inter_total += inter
        gt_total += area
        union |= gb
        cov = inter / area
        inst.append({"area_frac": area / (h * w), "coverage": cov,
                     "bucket": size_bucket(area / (h * w)),
                     "status": "covered" if cov >= cover_thresh else ("partial" if cov >= 0.5 else "missed")})
    over = float((cm & ~union).sum()) / (h * w)
    return inst, inter_total, gt_total, over


def iter_frames(video, every, max_frames):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    idx = -1
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        idx += 1
        if idx % every:
            continue
        yield idx, f
        n += 1
        if max_frames and n >= max_frames:
            break
    cap.release()


def make_segmenter(weights, args):
    if weights == "fake":
        from censorcam.segmenter import FakeSegmenter
        return FakeSegmenter()
    from censorcam.segmenter import YoloSegmenter
    return YoloSegmenter(weights, device=args.device, imgsz=args.imgsz, conf=args.conf,
                         warmup_shape=None)


def evaluate(weights, args, outdir: Path):
    seg = make_segmenter(weights, args)
    builder = MaskBuilder(args.shape, args.dilate, args.hold if args.every == 1 else 0)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "failures").mkdir(exist_ok=True)
    rows, inst_all = [], []
    inter_sum = gt_sum = 0
    over_list, inf_ms = [], []
    labelled = clean = 0
    review = None
    for idx, frame in iter_frames(args.video, args.every, args.max_frames):
        t0 = time.perf_counter()
        dets = seg.infer(frame)
        dt = (time.perf_counter() - t0) * 1000
        inf_ms.append(dt)
        censored, mask = censor_people(frame, dets, builder)
        row = {"frame": idx, "inference_ms": round(dt, 2), "detections": len(dets),
               "censored_area_pct": round(100 * float((mask > 0).mean()), 3)}
        if args.review:
            if review is None:
                h, w = frame.shape[:2]
                path = args.review if len(args.weights_list) == 1 else str(outdir / "review.mp4")
                review = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         max(1.0, 30.0 / args.every), (w, h // 2))
            vis = np.hstack([overlay_mask(frame, mask), censored])
            vis = cv2.resize(vis, (frame.shape[1], frame.shape[0] // 2))
            cv2.putText(vis, f"#{idx} dets={len(dets)} {dt:.1f}ms", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            review.write(vis)
        lbl = Path(args.labels) / (frame_name(idx) + ".json") if args.labels else None
        if lbl is not None and lbl.exists():
            labelled += 1
            gts = load_labelme(lbl, frame.shape)
            inst, inter, gt, over = score_frame(gts, mask, args.cover_thresh)
            inter_sum += inter
            gt_sum += gt
            over_list.append(over)
            for i in inst:
                i["frame"] = idx
            inst_all += inst
            bad = [i for i in inst if i["status"] != "covered"]
            if not bad:
                clean += 1
            else:
                vis = overlay_mask(frame, mask)
                for g, i in zip(gts, inst):
                    if i["status"] != "covered":
                        cs, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(vis, cs, -1, (0, 0, 255), 3)
                cv2.imwrite(str(outdir / "failures" / f"{frame_name(idx)}.jpg"), vis,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
            row.update({"gt_instances": len(inst), "missed_or_partial": len(bad),
                        "pixel_recall": round(inter / gt, 4) if gt else None,
                        "over_blur": round(over, 4)})
        rows.append(row)
    if review is not None:
        review.release()

    with open(outdir / "per_frame.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "frame", k))
        wr = csv.DictWriter(f, keys)
        wr.writeheader()
        wr.writerows(rows)

    s = sorted(inf_ms) or [0]
    rep = {
        "model": getattr(seg, "name", weights),
        "frames_evaluated": len(rows),
        "inference_ms_p50": round(s[len(s) // 2], 2),
        "inference_ms_p95": round(s[min(len(s) - 1, int(len(s) * .95))], 2),
        "frames_with_detections_pct": round(100 * np.mean([r["detections"] > 0 for r in rows]), 1) if rows else 0,
        "settings": {"conf": args.conf, "imgsz": args.imgsz, "shape": args.shape,
                     "dilate": args.dilate, "cover_thresh": args.cover_thresh},
    }
    if labelled:
        n = len(inst_all)
        rep.update({
            "labelled_frames": labelled,
            "gt_instances": n,
            "pixel_recall": round(inter_sum / gt_sum, 4) if gt_sum else None,
            "instances_covered": round(sum(i["status"] == "covered" for i in inst_all) / n, 4) if n else None,
            "instances_missed": round(sum(i["status"] == "missed" for i in inst_all) / n, 4) if n else None,
            "frames_clean": round(clean / labelled, 4),
            "over_blur_mean": round(float(np.mean(over_list)), 4),
            "by_size": {},
        })
        for b in ("small", "medium", "large"):
            sub = [i for i in inst_all if i["bucket"] == b]
            if sub:
                rep["by_size"][b] = {"n": len(sub),
                                     "covered": round(sum(i["status"] == "covered" for i in sub) / len(sub), 4),
                                     "mean_coverage": round(float(np.mean([i["coverage"] for i in sub])), 4)}
    (outdir / "report.json").write_text(json.dumps(rep, indent=2))
    md = [f"# Segmentation eval - {rep['model']}", "", f"Video: `{args.video}`", ""]
    md += [f"- **{k}**: {v}" for k, v in rep.items() if k not in ("by_size", "settings", "model")]
    if rep.get("by_size"):
        md += ["", "| size | n | covered | mean coverage |", "|---|---|---|---|"]
        md += [f"| {b} | {v['n']} | {v['covered']:.1%} | {v['mean_coverage']:.1%} |"
               for b, v in rep["by_size"].items()]
    md += ["", f"Settings: `{json.dumps(rep['settings'])}`",
           "", "Frames with a missed/partial character are in `failures/` (red outline = missed character, green = what was obscured)."]
    (outdir / "report.md").write_text("\n".join(md))
    return rep


def extract(args):
    out = Path(args.extract)
    out.mkdir(parents=True, exist_ok=True)
    seg = make_segmenter(args.weights_list[0], args) if args.prelabel else None
    n = 0
    for idx, frame in iter_frames(args.video, args.every, args.max_frames):
        name = frame_name(idx)
        cv2.imwrite(str(out / f"{name}.png"), frame)
        if seg is not None:
            write_labelme(out / f"{name}.json", f"{name}.png", frame.shape, seg.infer(frame))
        n += 1
    print(f"wrote {n} frames to {out}" + (" with model pre-labels" if seg else ""))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--weights", default="yolo11s-seg.pt", help="comma list to compare models")
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--shape", default="mask", choices=["mask", "box", "both"])
    p.add_argument("--dilate", type=int, default=16)
    p.add_argument("--hold", type=int, default=1)
    p.add_argument("--every", type=int, default=15, help="use every Nth frame")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--extract", default=None, help="write frames (png) here for labelling")
    p.add_argument("--prelabel", action="store_true", help="with --extract: write model labels as labelme JSON")
    p.add_argument("--labels", default=None, help="folder of labelme JSON named frame_000123.json")
    p.add_argument("--cover-thresh", type=float, default=0.95)
    p.add_argument("--review", default=None, help="write side-by-side review video")
    p.add_argument("--out", default="eval/report")
    args = p.parse_args(argv)
    args.weights_list = [w.strip() for w in args.weights.split(",") if w.strip()]

    if args.extract:
        return extract(args)
    reports = []
    for w in args.weights_list:
        od = Path(args.out) / Path(w).stem if len(args.weights_list) > 1 else Path(args.out)
        rep = evaluate(w, args, od)
        reports.append(rep)
        print(json.dumps(rep, indent=2))
    if len(reports) > 1:
        lines = ["| model | p95 ms | pixel recall | chars covered | small covered | frames clean | over-blur |",
                 "|---|---|---|---|---|---|---|"]
        for r in reports:
            sm = r.get("by_size", {}).get("small", {}).get("covered")
            lines.append(f"| {r['model']} | {r['inference_ms_p95']} | {r.get('pixel_recall')} | "
                         f"{r.get('instances_covered')} | {sm} | {r.get('frames_clean')} | {r.get('over_blur_mean')} |")
        Path(args.out, "summary.md").write_text("\n".join(lines))
        print("\n".join(lines))


if __name__ == "__main__":
    main()
