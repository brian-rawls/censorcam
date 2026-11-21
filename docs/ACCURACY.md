# Measuring detection accuracy on your own footage

**Don't assume a person-detection model recognises video game characters.** The bundled models were trained on real photographs (COCO). They often work on photoreal games, but small or distant, dark, partially hidden, in-vehicle and stylised characters are common misses. `tools/eval_segmentation.py` measures this on footage *you* recorded, using exactly the same mask logic as the live app (dilation, box fallback, hold).

## 1. Record representative clips

Record from the capture card at 1080p, a few minutes per situation:

- busy daytime streets and crowds
- night, neon and rain
- distant pedestrians
- characters in or next to vehicles
- cutscenes and close-ups
- first-person view (if the game has one)
- interiors, beaches and water

## 2. Extract frames with model pre-labels

```
python tools/eval_segmentation.py --video gameplay.mp4 --extract eval/frames --every 15 --prelabel
```

This writes every 15th frame as `frame_000123.png`, plus a [labelme](https://github.com/wkentaro/labelme) JSON file containing the model's own guesses, so you only correct them instead of drawing every outline from scratch.

## 3. Correct the labels

```
pip install labelme
labelme eval/frames
```

For each frame:
- **add every character the model missed.** Use the label `person`; `character`, `npc` and `human` are also accepted. Polygons or rectangles both work;
- delete wrong labels and fix sloppy outlines;
- save, **even when a frame has nobody in it**, so it counts as a negative.

Aim for 150–300 frames across all the situations above.

> The scores only mean something **after you've added the misses**. Pre-labels left uncorrected would score the model against itself.

## 4. Score one or more models

```
python tools/eval_segmentation.py --video gameplay.mp4 --labels eval/frames --every 15 ^
    --weights yolo11s-seg.pt,yolo26s-seg.pt,yolo11m-seg.pt --out eval/report
```

Outputs:
- `eval/report/summary.md`: comparison table across models;
- `eval/report/<model>/report.md`: per-model report with a breakdown by size;
- `eval/report/<model>/per_frame.csv`: per-frame detections, timings and scores;
- `eval/report/<model>/failures/*.jpg`: frames where a character wasn't fully covered. **Red outline** = labelled character that was missed or only partly covered; **green** = what was actually blurred.

### Metrics

| Metric | Meaning | Want |
|---|---|---|
| `pixel_recall` | Share of all labelled character pixels that ended up blurred | high |
| `instances_covered` | Share of characters with at least `--cover-thresh` (95 %) of their pixels blurred | high |
| `instances_missed` | Share of characters with under 50 % blurred | low |
| `frames_clean` | Share of labelled frames where **every** character was covered | high |
| `over_blur_mean` | Share of the frame that was blurred but wasn't a character | low (cosmetic cost) |
| `by_size` | The same, split into small (< 0.1 % of the frame), medium (0.1–1 %) and large (> 1 %) | — |

Without labels, `--review review.mp4` writes a side-by-side video (mask overlay | censored output) you can watch.

## 5. Improve coverage

Try these one at a time, re-scoring after each:

1. `--conf 0.10`: accept less confident detections.
2. `--dilate 24`: bigger safety margin around each character.
3. `--shape both`: silhouette plus bounding box.
4. `--imgsz 960`: helps distant characters, costs inference time.
5. A larger model: `yolo11m-seg.pt`. Re-export it with `tools/export_tensorrt.py --weights yolo11m-seg.pt` and check its time stays under your deadline.
6. **Fine-tune on your labelled frames.** This is usually the biggest gain:
   1. convert to YOLO format (e.g. `pip install labelme2yolo`);
   2. train with `yolo train model=yolo11s-seg.pt data=your.yaml epochs=50 imgsz=640`;
   3. score the result here;
   4. export it with `tools/export_tensorrt.py --weights runs/segment/train/weights/best.pt`.

Use the same settings live as you used in the evaluation: `--conf`, `--dilate`, `--shape` and `--imgsz` all change coverage.

---
*CensorCam by Aeternum Logic · developed by Brian Rawls*
