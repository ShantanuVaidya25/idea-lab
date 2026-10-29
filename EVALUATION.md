# Model evaluation: how it is measured and how to explain it

This document explains how the detection accuracy of the Smart Classroom system is measured, what every number
and plot means, and how to present it. The numbers themselves are in `data/testset/eval/REPORT.md` and on the
**Evaluation** page of the dashboard.

---

## 1. What is being evaluated

The system detects three kinds of object in the classroom camera picture and decides the state of two of them:

| Class | Meaning | Produced by |
|---|---|---|
| `person` | a person in view | YOLOv8 person detector (`yolov8n.pt`) |
| `fan_on` / `fan_off` | a ceiling fan, spinning / stopped | YOLO-World open-vocabulary detector + blade-visibility and motion logic |
| `light_on` / `light_off` | a light fitting, lit / not lit | glow detector (brightness relative to surroundings) + YOLO-World |

So the evaluation answers two questions at once:

1. **Detection**: did the system find the person / fan / light, and is the box in the right place?
2. **State**: for fans and lights, did it also get ON / OFF right?

That is why there are 5 classes. Confusing `fan_on` with `fan_off` counts as an error, exactly like confusing a
fan with a light.

---

## 2. How the test set is built (ground truth)

Metrics are only meaningful against **ground truth**: labels a human has checked. The steps:

1. **Capture**: `python tools/capture_dataset.py --count 60 --interval 3` saves real frames from the classroom
   camera while the dashboard runs. For every frame it stores:
   - `images/frame_XXXX.jpg`: the raw picture (no boxes drawn on it)
   - `predictions/frame_XXXX.json`: what the system detected at that exact moment (the thing being tested)
   - `labels/frame_XXXX.txt`: a copy of those detections as editable **pre-labels**
2. **Label**: `python tools/label_tool.py`, open http://localhost:8600. A person goes through every frame and
   **corrects** the pre-labels: adds boxes the system missed, deletes boxes on things that are not there, fixes
   wrong classes (for example a spinning fan marked `fan_off`) and tightens loose boxes. Each frame is marked
   *reviewed* when done. The corrected labels are the ground truth.
3. **Evaluate**: `python tools/evaluate.py` compares the system's predictions with the ground truth and
   produces all metrics and plots in `data/testset/eval/`.

Only frames marked *reviewed* are used, so unchecked pre-labels never count as truth.

> Because the labels start from the system's own output, the labeller must check every frame carefully: a box
> the labeller leaves unchanged is a statement that the system was right there.

**Label format** (YOLO): one text file per image, one line per object:
`class_id  x_center  y_center  width  height`, all as a fraction (0 to 1) of the image width / height.
Class ids: 0 person, 1 fan_on, 2 fan_off, 3 light_on, 4 light_off.

---

## 3. The metrics, in plain words

### IoU: how well two boxes overlap
```
IoU = area of overlap / area of union
```
1.0 = identical boxes, 0 = no overlap. A predicted box counts as finding an object when **IoU >= 0.5** with a
labelled box of the **same class**.

### TP, FP, FN
- **TP (true positive)**: a predicted box that matches a real object of the same class (IoU >= 0.5).
- **FP (false positive / false alarm)**: a predicted box with no matching real object, or the wrong class.
- **FN (false negative / miss)**: a real object no prediction matched.

Each real object can be matched only once; the most confident prediction gets it first.

### Precision: "when the system says something is there, how often is it right?"
```
Precision = TP / (TP + FP)
```
Low precision means many false alarms (for example white walls reported as lights).

### Recall: "of everything really there, how much did the system find?"
```
Recall = TP / (TP + FN)
```
Low recall means many misses (for example people at the back of the room not detected).

### F1 score: one number balancing the two
```
F1 = 2 x Precision x Recall / (Precision + Recall)
```
F1 is high only when precision AND recall are both high.

### AP (average precision) and the precision-recall curve
Every prediction has a **confidence**. Lowering the confidence threshold finds more objects (recall rises) but
also admits more mistakes (precision falls). Plotting precision against recall for every threshold gives the
**PR curve**. **AP is the area under that curve** (0 to 1), a summary of the detector at all thresholds, not
just one. Computed with the COCO 101-point method: precision is read at recall 0, 0.01, ..., 1.00 (using the
best precision at that recall or higher) and averaged.

### mAP@0.5 and mAP@0.5:0.95
- **mAP@0.5** = the mean of the per-class AP, with a hit meaning IoU >= 0.5. The headline number in most papers.
- **mAP@0.5:0.95** = the same averaged over ten IoU thresholds 0.50, 0.55, ..., 0.95. Much stricter: boxes must
  fit tightly. It is always lower than mAP@0.5.

Only classes that appear in the labels are averaged (a class with no labelled objects cannot be scored).

### ON / OFF state accuracy
Among fans and lights that were found (box matched, IoU >= 0.5, ignoring ON/OFF), the share whose ON/OFF
state was right. This isolates the state logic from the box detection.

### Operating threshold
Precision, recall, F1 and the confusion matrix use the confidence the system **uses in operation** (person:
`vision.confidence` in `config.yaml`, 0.45). AP / mAP use every prediction, because they describe all
thresholds. For people, the person model is re-run on the saved frames at a very low confidence so the whole PR
curve can be measured. Fans and lights are scored on exactly what the running system reported.

---

## 4. The plots

All plots are saved in `data/testset/eval/` and shown on the dashboard's **Evaluation** page.

### Confusion matrix (heatmap): `confusion_matrix.png`
A grid, coloured by count:
- **rows** = what the system predicted, **columns** = what was really there
- the **diagonal** = correct (predicted class = true class)
- off the diagonal = mix-ups, for example row `fan_off`, column `fan_on` = "a spinning fan reported as stopped"
- the **background row** = real objects the system **missed** (nothing was predicted on them)
- the **background column** = **false alarms** (something predicted where nothing was)

`confusion_matrix_normalized.png` divides each column by its total, so the diagonal shows the **recall of each
class** and each column shows where that class's objects ended up.

`confusion_matrix_detection.png` merges the states (fan_on + fan_off = fan, light_on + light_off = light) to show
pure detection.

### Precision-recall curve: `PR_curve.png`
One curve per class; the legend shows its AP@0.5. A curve that stays high (near the top right) is a good detector.

### F1-confidence curve: `F1_curve.png`
F1 at every confidence threshold. The peak shows the threshold that balances misses and false alarms best,
which is useful for choosing `vision.confidence`.

### Metrics per class: `metrics_per_class.png`
Bars of precision, recall, F1 and AP@0.5 for each class, the quickest way to see which class is weakest.

### Detection-density heatmaps: `density_heatmap_labels.png`, `density_heatmap_predictions.png`
For each kind of object (person / fan / light) the boxes from every frame are added up and drawn as a heatmap over
the camera picture: bright = something is there often.
- **labels** version: where objects really are in this camera view (from the hand-checked labels)
- **predictions** version: where the system reports them

Comparing the two shows the system's habits: a hot spot only in the predictions is a place it keeps reporting
something that is not there (a false-alarm area); a hot spot only in the labels is a place it keeps missing.

---

## 5. How to present it (viva notes)

1. **What was tested.** "We captured N frames from our classroom camera, corrected every label by hand and
   compared the system's output with those labels using the standard COCO detection metrics."
2. **Headline numbers.** Quote precision, recall, F1, mAP@0.5 and mAP@0.5:0.95 from `REPORT.md`, plus the ON/OFF
   state accuracy for fans and lights.
3. **Use the confusion matrix to explain errors.** For example: "most errors are in the background row for
   `person`, which means misses: people partly hidden behind others".
4. **Use the heatmaps to explain where.** For example: "false alarms concentrate at the window, where sunlight
   looks like a lit light".
5. **Be clear about the limits.**
   - The test set is small (tens of frames, one camera, one room, one day). The numbers have a wide margin of
     error and should not be read as general accuracy.
   - Frames a few seconds apart look similar, so they are not fully independent samples.
   - Fan / light states are labelled from the picture by a human; a fan turning very slowly can be ambiguous.
   - The person detector is the standard pre-trained YOLOv8n (COCO dataset), not fine-tuned on classroom images.
     Fans and lights use an open-vocabulary detector plus rules, not a model trained on this room.
6. **How to improve.** Label more frames across different days, lighting and camera angles; fine-tune a YOLOv8
   model on them (`yolo train data=data/testset/dataset.yaml model=yolov8n.pt`), then re-run `tools/evaluate.py`
   on frames that were **not** used for training.

### Likely questions

- *Why mAP and not just accuracy?* Detection has no single "right answer" per image: there can be any number of
  objects. Accuracy cannot count misses and false alarms separately; precision, recall and mAP can.
- *Why is mAP@0.5:0.95 so much lower?* It requires boxes that fit almost exactly (IoU up to 0.95). Hand-drawn
  boxes themselves vary by a few pixels, so this metric is strict by design.
- *What is the "background" class in the confusion matrix?* Not a real class: the row is objects nobody detected,
  the column is detections of objects that do not exist.
- *Why 5 classes and not 3?* Because the project's goal is to know whether a fan or light is ON, not only that it
  exists. Splitting by state makes ON/OFF mistakes visible in the confusion matrix.

---

## 6. Commands

```bash
python tools/capture_dataset.py --count 60 --interval 3   # dashboard must be running
python tools/label_tool.py                                # http://localhost:8600
python tools/evaluate.py                                  # writes data/testset/eval/
```
