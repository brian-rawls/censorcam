"""Measure downstream latency: CensorCam output -> OBS -> Virtual Camera -> TikTok LIVE Studio preview.

1. Start CensorCam, open http://127.0.0.1:8765/ and tick "latency probe"
   (a small black/white barcode appears in the centre of the picture -
   it IS visible on stream, so do this before going live or in a private test).
2. Take a screenshot to find the strip on screen:
       python tools/measure_latency.py --screenshot shot.png
   Open shot.png in Paint; hover the strip's top-left and bottom-right corners
   (pixel coordinates show in Paint's status bar). Include the black border.
3. Measure (repeat for the OBS preview and for the TikTok LIVE Studio preview):
       python tools/measure_latency.py --region 1200,640,300,40 --seconds 20 --report

Result = time from CensorCam handing the frame to Spout until it is visible in
that window (includes display refresh, so +/- 1 refresh interval).
Total glass-to-glass = HD60 X capture latency (not measurable in software; see
README for the phone slow-motion method) + CensorCam pipeline latency
(/metrics -> pipeline_latency_ms) + this number.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from censorcam import probe  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", help="x,y,w,h of the probe strip on screen (incl. black border)")
    p.add_argument("--seconds", type=float, default=15)
    p.add_argument("--screenshot", help="save a full screenshot (all monitors) and exit")
    p.add_argument("--report", action="store_true", help="send results to CensorCam's /metrics")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--key", default=None)
    args = p.parse_args(argv)

    import mss  # pip install mss
    import cv2

    with mss.mss() as sct:
        if args.screenshot:
            img = np.array(sct.grab(sct.monitors[0]))[:, :, :3]
            cv2.imwrite(args.screenshot, img)
            print(f"saved {args.screenshot}. Note: coordinates are relative to the virtual desktop "
                  f"origin {sct.monitors[0]['left']},{sct.monitors[0]['top']}")
            return
        if not args.region:
            p.error("--region is required (use --screenshot first)")
        x, y, w, h = [int(v) for v in args.region.split(",")]
        mon = {"left": x + sct.monitors[0]["left"], "top": y + sct.monitors[0]["top"],
               "width": w, "height": h}
        ages, last_code, bad = [], None, 0
        end = time.time() + args.seconds
        while time.time() < end:
            img = np.array(sct.grab(mon))[:, :, :3]
            now = probe.now_code()
            code = probe.decode(img)
            if code is None:
                bad += 1
            elif code != last_code:       # only the first sighting of each new frame counts
                last_code = code
                age = probe.age_ms(code, now)
                if age < 5000:
                    ages.append(age)
                    if args.report:
                        q = f"&key={args.key}" if args.key else ""
                        try:
                            urllib.request.urlopen(
                                f"http://127.0.0.1:{args.port}/report/downstream?ms={age}{q}", timeout=0.5)
                        except Exception:
                            pass
            time.sleep(0.002)
    if not ages:
        print(f"No probe decoded ({bad} unreadable grabs). Is the probe on and the region right?")
        return 1
    ages.sort()
    print(f"frames measured: {len(ages)}  (unreadable grabs: {bad})")
    print(f"downstream latency ms: median {statistics.median(ages):.0f}  "
          f"p95 {ages[int(len(ages) * .95) - 1 if len(ages) > 1 else 0]:.0f}  "
          f"min {ages[0]:.0f}  max {ages[-1]:.0f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
