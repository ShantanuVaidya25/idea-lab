"""Build a test set from the live classroom camera (the dashboard must be running).

For every frame it saves:
  images/frame_XXXX.jpg        the RAW camera picture (no boxes drawn)
  predictions/frame_XXXX.json  what the system detected at that moment (people + fan/light ON/OFF)
  labels/frame_XXXX.txt        the same detections as editable YOLO labels (pre-labels to CORRECT, not the truth)

Then correct the labels with  python tools/label_tool.py  and evaluate with  python tools/evaluate.py

Usage:  python tools/capture_dataset.py --count 60 --interval 3
Vary the scene while it runs: people walking in and out, fans and lights switched on and off.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.evaluation import CLASSES, write_yolo_labels  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def to_label_boxes(pred: dict) -> list[tuple[int, list[float]]]:
    """System predictions -> (class id, [x1, y1, x2, y2]) for the pre-labels."""
    out = [(CLASSES.index("person"), p[:4]) for p in pred["persons"]]
    for a in pred["appliances"]:
        if a["state"] in ("ON", "OFF"):
            out.append((CLASSES.index(f"{a['cls']}_{a['state'].lower()}"), a["bbox"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "testset"))
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--interval", type=float, default=3.0, help="seconds between frames")
    ap.add_argument("--url", default="http://localhost:8502/capture")
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("images", "labels", "predictions"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    (out / "dataset.yaml").write_text(f"path: {out.as_posix()}\ntest: images\nnames:\n"
                                      + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)), encoding="utf-8")
    start = len(list((out / "images").glob("*.jpg")))

    for k in range(args.count):
        try:
            with urllib.request.urlopen(args.url, timeout=10) as r:
                pred = json.loads(r.read())
        except Exception as e:
            sys.exit(f"Could not reach the dashboard camera ({e}). Start the dashboard first (run_demo.bat).")
        name = f"frame_{start + k:04d}"
        (out / "images" / f"{name}.jpg").write_bytes(base64.b64decode(pred.pop("image_jpg_b64")))
        (out / "predictions" / f"{name}.json").write_text(json.dumps(pred, indent=1), encoding="utf-8")
        write_yolo_labels(out / "labels" / f"{name}.txt", to_label_boxes(pred), pred["width"], pred["height"])
        n_app = sum(1 for a in pred["appliances"] if a["state"] in ("ON", "OFF"))
        print(f"[{k + 1}/{args.count}] {name}: {len(pred['persons'])} people, {n_app} fans/lights")
        if k + 1 < args.count:
            time.sleep(args.interval)
    print(f"\nSaved to {out}. Next: correct the labels with  python tools/label_tool.py")


if __name__ == "__main__":
    main()
