"""Evaluate the detection system on the labelled classroom test set.

    python tools/evaluate.py                 (only frames you marked "reviewed" in the labelling tool)
    python tools/evaluate.py --all           (every frame, even unreviewed ones - not a fair test)

People      : the person model (config.yaml vision.model_path) is run on every frame at a very low confidence,
              so the whole precision-recall curve can be measured; the live system's ignore zones are applied.
Fans/lights : what the full system (detector + glow / blade logic) reported at the moment the frame was captured.

Writes to data/testset/eval/: metrics.json, REPORT.md, confusion matrices, PR / F1 curves, per-class bars and
detection-density heatmaps.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import evaluation as ev  # noqa: E402
from core.config import load_config  # noqa: E402
from core.vision import center_in_zones, resolve_classes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def person_predictions(paths: list[Path], cfg: dict) -> dict[str, list]:
    from ultralytics import YOLO
    v = cfg["vision"]
    model = YOLO(str(ROOT / v["model_path"]) if (ROOT / v["model_path"]).exists() else v["model_path"])
    ids = resolve_classes(dict(model.names), v.get("class_aliases", {})).get("person", [])
    out = {}
    for p in paths:
        r = model.predict(str(p), conf=0.001, imgsz=v.get("imgsz", 640), classes=ids or None, verbose=False)[0]
        preds = []
        for b in r.boxes:
            box = [float(x) for x in b.xyxy[0].tolist()]
            if not center_in_zones(box, v.get("ignore_person_zones")):
                preds.append((ev.CLASSES.index("person"), float(b.conf[0]), box))
        out[p.stem] = preds
    return out


def appliance_predictions(pred_file: Path) -> list:
    if not pred_file.exists():
        return []
    pred = json.loads(pred_file.read_text(encoding="utf-8"))
    return [(ev.CLASSES.index(f"{a['cls']}_{a['state'].lower()}"), float(a.get("conf", 1.0)), a["bbox"])
            for a in pred.get("appliances", []) if a.get("state") in ("ON", "OFF")]


def fmt(v, pct=False) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    return f"{100 * v:.1f} %" if pct else f"{v:.3f}"


def report_md(r5: dict, r3: dict, state: dict, n: int, person_conf: float) -> str:
    s = r5["summary"]
    lines = [
        "# Detection evaluation - classroom test set", "",
        f"Frames evaluated: **{n}** (hand-checked labels).  IoU threshold for a hit: 0.5.  "
        f"Person confidence threshold in operation: {person_conf}.", "",
        "## Overall (5 classes: person, fan_on, fan_off, light_on, light_off)", "",
        "| Metric | Micro average (all boxes pooled) | Macro average (mean of classes) |", "|---|---|---|",
        f"| Precision | {fmt(s['micro_precision'], True)} | {fmt(s['precision'], True)} |",
        f"| Recall | {fmt(s['micro_recall'], True)} | {fmt(s['recall'], True)} |",
        f"| F1 | {fmt(s['micro_f1'], True)} | {fmt(s['f1'], True)} |",
        f"| mAP@0.5 | – | {fmt(s['map50'])} |", f"| mAP@0.5:0.95 | – | {fmt(s['map50_95'])} |", "",
        f"Boxes: {s['tp']} correct (TP), {s['fp']} false alarms (FP), {s['fn']} missed (FN). Micro averages pool every "
        "box; macro averages give each class the same weight, so a class with very few objects moves them a lot.", "",
        "## Per class", "",
        "| Class | True boxes | Predicted | TP | FP | FN | Precision | Recall | F1 | AP@0.5 | AP@0.5:0.95 |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in r5["per_class"].items():
        lines.append(f"| {name} | {c['gt']} | {c['pred']} | {c['tp']} | {c['fp']} | {c['fn']} | {fmt(c['precision'], True)} | "
                     f"{fmt(c['recall'], True)} | {fmt(c['f1'], True)} | {fmt(c['ap50'])} | {fmt(c['ap50_95'])} |")
    s3 = r3["summary"]
    lines += ["", "## Detection only (ON / OFF ignored: person, fan, light)", "",
              "| Class | Precision | Recall | F1 | AP@0.5 |", "|---|---|---|---|---|"]
    for name, c in r3["per_class"].items():
        lines.append(f"| {name} | {fmt(c['precision'], True)} | {fmt(c['recall'], True)} | {fmt(c['f1'], True)} | {fmt(c['ap50'])} |")
    lines += [f"| **mean** | {fmt(s3['precision'], True)} | {fmt(s3['recall'], True)} | {fmt(s3['f1'], True)} | {fmt(s3['map50'])} |",
              "", f"ON / OFF state correct for found fans and lights: **{fmt(state['accuracy'], True)}** "
              f"({state['correct']} of {state['matched']}).", "",
              "Plots: confusion_matrix.png, confusion_matrix_normalized.png, confusion_matrix_detection.png, "
              "PR_curve.png, F1_curve.png, metrics_per_class.png, density_heatmap_labels.png, "
              "density_heatmap_predictions.png. See EVALUATION.md for how to read them."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "testset"))
    ap.add_argument("--all", action="store_true", help="include frames not marked reviewed")
    ap.add_argument("--out", default=None, help="output folder (default: <data>/eval)")
    ap.add_argument("--frames", default=None, help="only these frame numbers, e.g. 0-29 or 0-29,40-45")
    args = ap.parse_args()
    data = Path(args.data)
    cfg = load_config()
    reviewed = json.loads((data / "reviewed.json").read_text(encoding="utf-8")) if (data / "reviewed.json").exists() else {}
    paths = [p for p in sorted((data / "images").glob("*.jpg")) if args.all or reviewed.get(p.stem)]
    if args.frames:
        keep = set()
        for part in args.frames.split(","):
            a, _, b = part.partition("-")
            keep.update(range(int(a), int(b or a) + 1))
        paths = [p for p in paths if int(p.stem.split("_")[-1]) in keep]
    if not paths:
        sys.exit("No reviewed frames yet. Label them first:  python tools/label_tool.py   (or use --all)")
    print(f"Evaluating {len(paths)} frames ...")
    persons = person_predictions(paths, cfg)
    images, frames = [], []
    for p in paths:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        frames.append(img)
        images.append({"name": p.stem, "gts": ev.read_yolo_labels(data / "labels" / f"{p.stem}.txt", w, h),
                       "preds": persons[p.stem] + appliance_predictions(data / "predictions" / f"{p.stem}.json")})
    person_conf = float(cfg["vision"].get("confidence", 0.45))
    thresholds = {"person": person_conf}
    r5 = ev.evaluate(images, ev.CLASSES, thresholds)
    merged, names, state = ev.merge_states(images)
    r3 = ev.evaluate(merged, names, thresholds)

    out = Path(args.out) if args.out else data / "eval"
    out.mkdir(exist_ok=True)
    ev.plot_confusion(r5["confusion"], out / "confusion_matrix.png")
    ev.plot_confusion(r5["confusion"], out / "confusion_matrix_normalized.png", normalize=True)
    ev.plot_confusion(r3["confusion"], out / "confusion_matrix_detection.png")
    ev.plot_curves(r5, out)
    bg = np.median(np.stack(frames[:: max(1, len(frames) // 15)]), axis=0).astype(np.uint8)
    ev.plot_density(images, ev.CLASSES, bg, out / "density_heatmap_labels.png", "gts",
                    "Detection-density heatmap - hand-checked labels")
    op_images = [{**im, "preds": [q for q in im["preds"] if q[1] >= thresholds.get(ev.CLASSES[q[0]], 0.0)]} for im in images]
    ev.plot_density(op_images, ev.CLASSES, bg, out / "density_heatmap_predictions.png", "preds",
                    "Detection-density heatmap - system predictions")
    ev.save_json({"five_class": {k: v for k, v in r5.items()}, "detection_only": r3, "state_accuracy": state,
                  "frames": len(images), "person_conf": person_conf}, out / "metrics.json")
    (out / "REPORT.md").write_text(report_md(r5, r3, state, len(images), person_conf), encoding="utf-8")

    s = r5["summary"]
    print(f"\nMicro: P {fmt(s['micro_precision'], True)}  R {fmt(s['micro_recall'], True)}  F1 {fmt(s['micro_f1'], True)}  "
          f"(TP {s['tp']}, FP {s['fp']}, FN {s['fn']})")
    print(f"Macro: P {fmt(s['precision'], True)}  R {fmt(s['recall'], True)}  F1 {fmt(s['f1'], True)}  "
          f"mAP@0.5 {fmt(s['map50'])}  mAP@0.5:0.95 {fmt(s['map50_95'])}")
    for name, c in r5["per_class"].items():
        print(f"  {name:10s} gt={c['gt']:3d}  P={fmt(c['precision'], True):>7s}  R={fmt(c['recall'], True):>7s}  "
              f"AP50={fmt(c['ap50'])}")
    print(f"ON/OFF state accuracy: {fmt(state['accuracy'], True)} ({state['correct']}/{state['matched']})")
    print(f"Saved plots and REPORT.md to {out}")


if __name__ == "__main__":
    main()
