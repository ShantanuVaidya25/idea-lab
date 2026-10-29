"""Find YOLO model files and print what the model can detect.

Usage:  python tools/check_model.py                 (searches project folder + uses config path)
        python tools/check_model.py path/to/best.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import ROOT, load_config  # noqa: E402
from core.vision import resolve_classes  # noqa: E402

print("== Model files found in project ==")
found = [p for ext in ("*.pt", "*.onnx", "*.engine", "*.tflite") for p in ROOT.rglob(ext) if ".venv" not in p.parts]
for p in found:
    print("  ", p.relative_to(ROOT), f"({p.stat().st_size / 1e6:.1f} MB)")
if not found:
    print("   none (ultralytics will download yolov8n.pt = COCO, person only)")

cfg = load_config()
path = sys.argv[1] if len(sys.argv) > 1 else cfg["vision"]["model_path"]
print(f"\n== Inspecting {path} ==")
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402

print("ultralytics version:", ultralytics.__version__)
try:
    import torch
    print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
except ImportError:
    print("torch not installed")

m = YOLO(path)
print("task:", m.task)
print("classes:", dict(m.names))
imgsz = getattr(m, "overrides", {}).get("imgsz", 640)
print("trained input size:", imgsz)
cmap = resolve_classes(dict(m.names), cfg["vision"]["class_aliases"])
for k, ids in cmap.items():
    print(f"  {k:7s}: {'YES ' + str([m.names[i] for i in ids]) if ids else 'NOT SUPPORTED by this model'}")
