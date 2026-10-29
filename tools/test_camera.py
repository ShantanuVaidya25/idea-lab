"""Test webcam + YOLO without the dashboard. Press Q to quit.

Usage:  python tools/test_camera.py            (camera from config.yaml)
        python tools/test_camera.py 1          (camera index 1)
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import load_config  # noqa: E402

cfg = load_config()
src = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else cfg["vision"]["source"]
cap = cv2.VideoCapture(src)
if not cap.isOpened():
    sys.exit(f"Cannot open camera {src}. Try another index: python tools/test_camera.py 1")

from ultralytics import YOLO  # noqa: E402
model = YOLO(cfg["vision"]["model_path"])
print("Model classes:", model.names)
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed")
            break
        res = model.predict(frame, conf=cfg["vision"]["confidence"], verbose=False)[0]
        cv2.imshow("YOLO test (Q to quit)", res.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
