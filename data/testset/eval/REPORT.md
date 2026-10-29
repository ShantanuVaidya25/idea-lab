# Detection evaluation - classroom test set

Frames evaluated: **30** (hand-checked labels).  IoU threshold for a hit: 0.5.  Person confidence threshold in operation: 0.45.

## Overall (5 classes: person, fan_on, fan_off, light_on, light_off)

| Metric | Micro average (all boxes pooled) | Macro average (mean of classes) |
|---|---|---|
| Precision | 72.0 % | 59.5 % |
| Recall | 72.6 % | 83.1 % |
| F1 | 72.3 % | 69.4 % |
| mAP@0.5 | – | 0.530 |
| mAP@0.5:0.95 | – | 0.490 |

Boxes: 170 correct (TP), 66 false alarms (FP), 64 missed (FN). Micro averages pool every box; macro averages give each class the same weight, so a class with very few objects moves them a lot.

## Per class

| Class | True boxes | Predicted | TP | FP | FN | Precision | Recall | F1 | AP@0.5 | AP@0.5:0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| person | 116 | 112 | 95 | 17 | 21 | 84.8 % | 81.9 % | 83.3 % | 0.823 | 0.660 |
| fan_on | 24 | 28 | 23 | 5 | 1 | 82.1 % | 95.8 % | 88.5 % | 0.817 | 0.817 |
| fan_off | 0 | 1 | 0 | 1 | 0 | 0.0 % | – | – | – | – |
| light_on | 93 | 87 | 51 | 36 | 42 | 58.6 % | 54.8 % | 56.7 % | 0.357 | 0.357 |
| light_off | 1 | 8 | 1 | 7 | 0 | 12.5 % | 100.0 % | 22.2 % | 0.125 | 0.125 |

## Detection only (ON / OFF ignored: person, fan, light)

| Class | Precision | Recall | F1 | AP@0.5 |
|---|---|---|---|---|
| person | 84.8 % | 81.9 % | 83.3 % | 0.823 |
| fan | 82.8 % | 100.0 % | 90.6 % | 0.858 |
| light | 54.7 % | 55.3 % | 55.0 % | 0.330 |
| **mean** | 74.1 % | 79.1 % | 76.5 % | 0.670 |

ON / OFF state correct for found fans and lights: **98.7 %** (75 of 76).

Plots: confusion_matrix.png, confusion_matrix_normalized.png, confusion_matrix_detection.png, PR_curve.png, F1_curve.png, metrics_per_class.png, density_heatmap_labels.png, density_heatmap_predictions.png. See EVALUATION.md for how to read them.
