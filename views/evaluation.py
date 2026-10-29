"""Evaluation: detection accuracy of the system on the hand-labelled classroom test set."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import ROOT
from core.ui import kpi, kpi_row, page_head, section

EVAL = ROOT / "data" / "testset" / "eval"
page_head("Evaluation", "Precision, recall, mAP and confusion matrix on the hand-labelled classroom test set")

metrics_file = EVAL / "metrics.json"
if not metrics_file.exists():
    st.info("No evaluation yet. Build and label the test set, then run the evaluation:")
    st.code("python tools/capture_dataset.py --count 60 --interval 3\n"
            "python tools/label_tool.py        # correct the boxes at http://localhost:8600\n"
            "python tools/evaluate.py", language="bash")
    st.caption("EVALUATION.md explains every metric and plot.")
    st.stop()

m = json.loads(metrics_file.read_text(encoding="utf-8"))
five, det, state = m["five_class"], m["detection_only"], m["state_accuracy"]
s = five["summary"]
pct = lambda v: "–" if v is None else f"{100 * v:.1f} %"
num = lambda v: "–" if v is None else f"{v:.3f}"
st.caption(f"{m['frames']} hand-checked frames  ·  IoU threshold 0.5  ·  person confidence {m['person_conf']}  ·  "
           f"evaluated {datetime.fromtimestamp(metrics_file.stat().st_mtime):%d %b %Y %H:%M}")

kpi_row([kpi("Precision", pct(s.get("micro_precision")), f"all boxes  ·  macro {pct(s['precision'])}", "blue"),
         kpi("Recall", pct(s.get("micro_recall")), f"all boxes  ·  macro {pct(s['recall'])}", "blue"),
         kpi("F1 score", pct(s.get("micro_f1")), f"all boxes  ·  macro {pct(s['f1'])}", "green"),
         kpi("mAP@0.5", num(s["map50"]), "mean average precision", "green"),
         kpi("mAP@0.5:0.95", num(s["map50_95"]), "strict box fit", "gray"),
         kpi("ON / OFF state", pct(state["accuracy"]), f"{state['correct']} of {state['matched']} fans / lights", "amber")])

section("Per class")
rows = [{"Class": n, "True boxes": c["gt"], "Predicted": c["pred"], "TP": c["tp"], "FP": c["fp"], "FN": c["fn"],
         "Precision": c["precision"], "Recall": c["recall"], "F1": c["f1"], "AP@0.5": c["ap50"],
         "AP@0.5:0.95": c["ap50_95"]} for n, c in five["per_class"].items()]
cfg = {k: st.column_config.NumberColumn(k, format="%.3f") for k in ("Precision", "Recall", "F1", "AP@0.5", "AP@0.5:0.95")}
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config=cfg)
d = det["summary"]
st.caption(f"Detection only (ON / OFF ignored, classes person / fan / light): precision {pct(d['precision'])}, "
           f"recall {pct(d['recall'])}, mAP@0.5 {num(d['map50'])}.")


def show(name: str, caption: str) -> None:
    p = EVAL / name
    if p.exists():
        st.image(str(p), use_container_width=True)
        st.caption(caption)


t1, t2, t3, t4 = st.tabs(["Confusion matrix", "Curves", "Detection heatmaps", "Report"])
with t1:
    c1, c2 = st.columns(2)
    with c1:
        show("confusion_matrix.png", "Counts. Rows: what the system said. Columns: what was really there. "
             "Diagonal = correct. 'background' row = missed objects, 'background' column = false alarms.")
    with c2:
        show("confusion_matrix_normalized.png", "Each column divided by its total: the diagonal is the recall of "
             "that class.")
    show("confusion_matrix_detection.png", "Detection only: fan_on / fan_off merged into fan, light_on / light_off into light.")
with t2:
    c1, c2 = st.columns(2)
    with c1:
        show("PR_curve.png", "Precision against recall as the confidence threshold is lowered. AP = area under it.")
    with c2:
        show("F1_curve.png", "F1 at every confidence threshold. The peak is the best operating threshold.")
    show("metrics_per_class.png", "Precision, recall, F1 and AP@0.5 for each class.")
with t3:
    show("density_heatmap_labels.png", "Where people, fans and lights really are in this camera view (hand-checked labels).")
    show("density_heatmap_predictions.png", "Where the system reports them. Hot spots that appear here but not above "
         "are false alarms; the reverse are misses.")
with t4:
    rp = EVAL / "REPORT.md"
    if rp.exists():
        st.markdown(rp.read_text(encoding="utf-8"))
