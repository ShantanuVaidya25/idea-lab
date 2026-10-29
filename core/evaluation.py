"""Detection metrics for the classroom test set: precision, recall, F1, AP / mAP, confusion matrix, heatmaps.

Definitions follow the standard object-detection practice (as in COCO / Ultralytics):
  IoU        = area(overlap) / area(union) of a predicted box and a true box
  TP / FP    = a prediction is a true positive if it overlaps an unmatched true box of the SAME class with
               IoU >= threshold (each true box can be matched once, highest-confidence predictions first);
               otherwise it is a false positive. True boxes left unmatched are false negatives (FN).
  Precision  = TP / (TP + FP)     "of what the system reported, how much was right"
  Recall     = TP / (TP + FN)     "of what was really there, how much the system found"
  F1         = 2 P R / (P + R)
  AP         = area under the precision-recall curve (COCO 101-point interpolation)
  mAP@0.5    = mean AP over the classes, IoU threshold 0.5
  mAP@0.5:0.95 = mean AP averaged over IoU thresholds 0.50, 0.55, ..., 0.95 (stricter: boxes must fit tightly)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CLASSES = ["person", "fan_on", "fan_off", "light_on", "light_off"]
MERGED = {"person": "person", "fan_on": "fan", "fan_off": "fan", "light_on": "light", "light_off": "light"}
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


# ---------------------------------------------------------------- YOLO label files
def write_yolo_labels(path: Path, boxes: list[tuple[int, list[float]]], w: int, h: int) -> None:
    """boxes: (class id, [x1, y1, x2, y2] in pixels) -> 'cls cx cy bw bh' normalised to 0..1."""
    lines = []
    for cls, (x1, y1, x2, y2) in boxes:
        x1, x2 = sorted((max(0.0, min(w, x1)), max(0.0, min(w, x2))))
        y1, y2 = sorted((max(0.0, min(h, y1)), max(0.0, min(h, y2))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_yolo_labels(path: Path, w: int, h: int) -> list[tuple[int, list[float]]]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:])
        out.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


# ---------------------------------------------------------------- matching
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and every box in b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match_image(preds: list[tuple[int, float, list]], gts: list[tuple[int, list]], iou_thr: float,
                cls: int) -> tuple[list[tuple[float, bool]], int]:
    """For one image and one class: [(confidence, is_TP)] per prediction, and the number of true boxes."""
    p = sorted([q for q in preds if q[0] == cls], key=lambda q: -q[1])
    g = [b for c, b in gts if c == cls]
    if not p:
        return [], len(g)
    ious = iou_matrix(np.array([q[2] for q in p]), np.array(g)) if g else np.zeros((len(p), 0))
    used, out = set(), []
    for i, q in enumerate(p):
        best, best_j = iou_thr, -1
        for j in range(len(g)):
            if j not in used and ious[i, j] >= best:
                best, best_j = ious[i, j], j
        if best_j >= 0:
            used.add(best_j)
        out.append((q[1], best_j >= 0))
    return out, len(g)


def average_precision(scored: list[tuple[float, bool]], n_gt: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """COCO 101-point AP. Returns (AP, precision curve, recall curve, confidences)."""
    if n_gt == 0:
        return float("nan"), np.array([]), np.array([]), np.array([])
    if not scored:
        return 0.0, np.array([]), np.array([]), np.array([])
    scored = sorted(scored, key=lambda s: -s[0])
    conf = np.array([s[0] for s in scored]); tp = np.array([s[1] for s in scored], float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]          # precision can only go down as recall rises
    ap = float(np.mean([envelope[recall >= r].max() if (recall >= r).any() else 0.0 for r in np.linspace(0, 1, 101)]))
    return ap, precision, recall, conf


# ---------------------------------------------------------------- full evaluation
def evaluate(images: list[dict], classes: list[str], conf_thresholds: dict[str, float] | None = None) -> dict:
    """images: [{"name", "gts": [(cls, box)], "preds": [(cls, conf, box)]}] with class ids into `classes`.

    conf_thresholds: per-class confidence the SYSTEM uses in operation, for precision / recall / F1 and the
    confusion matrix (AP always uses every prediction).
    """
    conf_thresholds = conf_thresholds or {}
    per_class = {}
    for ci, name in enumerate(classes):
        aps, curves = [], None
        n_gt_total = 0
        for thr in IOU_THRESHOLDS:
            scored, n_gt = [], 0
            for im in images:
                s, g = match_image(im["preds"], im["gts"], float(thr), ci)
                scored += s
                n_gt += g
            ap, prec, rec, conf = average_precision(scored, n_gt)
            aps.append(ap)
            if thr == IOU_THRESHOLDS[0]:
                curves, n_gt_total, scored50 = (prec, rec, conf), n_gt, scored
        op = conf_thresholds.get(name, 0.0)
        tp = sum(1 for c, t in scored50 if t and c >= op)
        fp = sum(1 for c, t in scored50 if not t and c >= op)
        fn = n_gt_total - tp
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / n_gt_total if n_gt_total else float("nan")
        f1 = 2 * p * r / (p + r) if p == p and r == r and p + r > 0 else float("nan")
        per_class[name] = {"gt": n_gt_total, "pred": sum(1 for c, _ in scored50 if c >= op), "tp": tp, "fp": fp,
                           "fn": fn, "precision": p, "recall": r, "f1": f1, "ap50": aps[0],
                           "ap50_95": float(np.nanmean(aps)) if n_gt_total else float("nan"),
                           "conf_threshold": op,
                           "curve": {"precision": curves[0].tolist(), "recall": curves[1].tolist(),
                                     "conf": curves[2].tolist()}}
    with_gt = [c for c in classes if per_class[c]["gt"] > 0]
    summary = {
        "images": len(images),
        "classes_with_labels": with_gt,
        "map50": float(np.mean([per_class[c]["ap50"] for c in with_gt])) if with_gt else float("nan"),
        "map50_95": float(np.mean([per_class[c]["ap50_95"] for c in with_gt])) if with_gt else float("nan"),
        "precision": float(np.nanmean([per_class[c]["precision"] for c in with_gt])) if with_gt else float("nan"),
        "recall": float(np.nanmean([per_class[c]["recall"] for c in with_gt])) if with_gt else float("nan"),
    }
    summary["f1"] = (2 * summary["precision"] * summary["recall"] / (summary["precision"] + summary["recall"])
                     if summary["precision"] + summary["recall"] > 0 else float("nan"))
    # micro average: pool every box of every class (a class with 1 object does not weigh as much as one with 100)
    tp = sum(per_class[c]["tp"] for c in classes); fp = sum(per_class[c]["fp"] for c in classes)
    fn = sum(per_class[c]["fn"] for c in classes)
    mp = tp / (tp + fp) if tp + fp else float("nan"); mr = tp / (tp + fn) if tp + fn else float("nan")
    summary.update({"micro_precision": mp, "micro_recall": mr,
                    "micro_f1": 2 * mp * mr / (mp + mr) if mp == mp and mr == mr and mp + mr > 0 else float("nan"),
                    "tp": tp, "fp": fp, "fn": fn})
    return {"summary": summary, "per_class": per_class,
            "confusion": confusion_matrix(images, classes, conf_thresholds)}


def confusion_matrix(images: list[dict], classes: list[str], conf_thresholds: dict[str, float] | None = None,
                     iou_thr: float = 0.5) -> dict:
    """Rows = PREDICTED class (+ background), columns = TRUE class (+ background), counts.

    Each prediction is paired with the true box it overlaps most (any class, IoU >= iou_thr):
      pair             -> matrix[predicted, true]   (diagonal = right class, off-diagonal = wrong class)
      true box unpaired -> matrix[background, true]  (missed)
      prediction unpaired -> matrix[predicted, background]  (false alarm)
    """
    conf_thresholds = conf_thresholds or {}
    n = len(classes)
    m = np.zeros((n + 1, n + 1), int)
    for im in images:
        preds = [p for p in im["preds"] if p[1] >= conf_thresholds.get(classes[p[0]], 0.0)]
        gts = im["gts"]
        ious = iou_matrix(np.array([p[2] for p in preds]), np.array([g[1] for g in gts])) \
            if preds and gts else np.zeros((len(preds), len(gts)))
        pairs = sorted(((ious[i, j], i, j) for i in range(len(preds)) for j in range(len(gts)) if ious[i, j] >= iou_thr),
                       reverse=True)
        used_p, used_g = set(), set()
        for _, i, j in pairs:
            if i in used_p or j in used_g:
                continue
            used_p.add(i); used_g.add(j)
            m[preds[i][0], gts[j][0]] += 1
        for j, g in enumerate(gts):
            if j not in used_g:
                m[n, g[0]] += 1
        for i, p in enumerate(preds):
            if i not in used_p:
                m[p[0], n] += 1
    return {"labels": classes + ["background"], "matrix": m.tolist()}


def merge_states(images: list[dict]) -> tuple[list[dict], list[str], dict]:
    """fan_on / fan_off -> fan, light_on / light_off -> light: detection quality regardless of ON/OFF, plus
    how often the ON/OFF state is right for the fans / lights that were found."""
    merged = ["person", "fan", "light"]
    idx = {c: merged.index(MERGED[c]) for c in CLASSES}
    out, right, total = [], 0, 0
    for im in images:
        gts = [(idx[CLASSES[c]], b) for c, b in im["gts"]]
        preds = [(idx[CLASSES[c]], s, b) for c, s, b in im["preds"]]
        out.append({**im, "gts": gts, "preds": preds})
        # state accuracy: fan / light predictions matched to a true fan / light of the same kind
        for kind in ("fan", "light"):
            g = [(c, b) for c, b in im["gts"] if MERGED[CLASSES[c]] == kind]
            p = [(c, s, b) for c, s, b in im["preds"] if MERGED[CLASSES[c]] == kind]
            if not g or not p:
                continue
            ious = iou_matrix(np.array([q[2] for q in p]), np.array([q[1] for q in g]))
            used = set()
            for i in np.argsort([-q[1] for q in p]):
                j = int(np.argmax(ious[i])) if ious.shape[1] else -1
                if j >= 0 and ious[i, j] >= 0.5 and j not in used:
                    used.add(j)
                    total += 1
                    right += int(p[i][0] == g[j][0])
    return out, merged, {"matched": total, "correct": right, "accuracy": right / total if total else float("nan")}


# ---------------------------------------------------------------- plots
def _style(plt) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#D0D5DD",
                         "axes.labelcolor": "#344054", "xtick.color": "#475467", "ytick.color": "#475467",
                         "axes.titleweight": "bold", "axes.titlesize": 12, "figure.facecolor": "white"})


def plot_confusion(conf: dict, path: Path, normalize: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    m = np.array(conf["matrix"], float)
    labels = conf["labels"]
    shown = m / np.maximum(m.sum(axis=0, keepdims=True), 1e-9) if normalize else m
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else max(1, shown.max()))
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("True class"); ax.set_ylabel("Predicted class")
    ax.set_title("Confusion matrix" + (" (normalised per true class)" if normalize else " (counts)"))
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            v = shown[i, j]
            if (normalize and v > 0.004) or (not normalize and v > 0):
                ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}", ha="center", va="center",
                        color="white" if v > (0.55 if normalize else shown.max() * 0.55) else "#101828", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_curves(result: dict, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    colors = ["#1D4ED8", "#12B76A", "#98A2B3", "#F79009", "#7A5AF8", "#F04438"]
    # precision-recall
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for k, (name, r) in enumerate(result["per_class"].items()):
        if r["gt"] == 0 or not r["curve"]["recall"]:
            continue
        ax.plot(r["curve"]["recall"], r["curve"]["precision"], color=colors[k % len(colors)], lw=2,
                label=f"{name}  AP@0.5 = {r['ap50']:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall curve  (mAP@0.5 = {result['summary']['map50']:.3f})")
    ax.grid(color="#F2F4F7"); ax.legend(loc="lower left", frameon=False)
    fig.tight_layout(); fig.savefig(folder / "PR_curve.png", dpi=150); plt.close(fig)
    # F1 vs confidence
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for k, (name, r) in enumerate(result["per_class"].items()):
        c = np.array(r["curve"]["conf"])
        if r["gt"] == 0 or len(c) == 0:
            continue
        p, rc = np.array(r["curve"]["precision"]), np.array(r["curve"]["recall"])
        f1 = 2 * p * rc / np.maximum(p + rc, 1e-9)
        ax.plot(c, f1, color=colors[k % len(colors)], lw=2, label=name)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_xlabel("Confidence threshold"); ax.set_ylabel("F1")
    ax.set_title("F1 score vs confidence"); ax.grid(color="#F2F4F7"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(folder / "F1_curve.png", dpi=150); plt.close(fig)
    # per-class bars
    names = [n for n, r in result["per_class"].items() if r["gt"] > 0]
    if names:
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        keys = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("ap50", "AP@0.5")]
        w = 0.8 / len(keys)
        for k, (key, lab) in enumerate(keys):
            vals = [np.nan_to_num(result["per_class"][n][key]) for n in names]
            ax.bar(np.arange(len(names)) + (k - 1.5) * w, vals, w, label=lab, color=colors[k])
        ax.set_xticks(range(len(names)), names); ax.set_ylim(0, 1.05); ax.set_title("Metrics per class")
        ax.grid(axis="y", color="#F2F4F7"); ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        fig.tight_layout(); fig.savefig(folder / "metrics_per_class.png", dpi=150); plt.close(fig)


def plot_density(images: list[dict], classes: list[str], background: np.ndarray, path: Path,
                 source: str = "gts", title: str = "Where things are detected") -> None:
    """Detection-density heatmap: how often each part of the picture holds a person / fan / light."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    h, w = background.shape[:2]
    groups = {"person": ["person"], "fan": ["fan_on", "fan_off", "fan"], "light": ["light_on", "light_off", "light"]}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    for ax, (g, members) in zip(axes, groups.items()):
        acc = np.zeros((h, w), np.float32)
        n = 0
        for im in images:
            for item in im[source]:
                c, box = (item[0], item[-1])
                if classes[c] in members:
                    x1, y1, x2, y2 = (int(max(0, v)) for v in box)
                    acc[y1:min(h, y2), x1:min(w, x2)] += 1
                    n += 1
        acc = cv2.GaussianBlur(acc, (0, 0), 6)
        ax.imshow(gray, cmap="gray", alpha=0.55)
        if acc.max() > 0:
            ax.imshow(np.ma.masked_less(acc / acc.max(), 0.02), cmap="inferno", alpha=0.65, vmin=0, vmax=1)
        ax.set_title(f"{g}  ({n} boxes in {len(images)} frames)")
        ax.axis("off")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(to_jsonable(obj), indent=1), encoding="utf-8")
