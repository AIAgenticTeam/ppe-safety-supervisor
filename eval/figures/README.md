# Evaluation figures

Report-ready artifacts from the final detector run (`ppe-presence-s960-3`, YOLO26s @ 960px).
The full set of six training runs stays local under `eval/training-runs/` — gitignored,
38 MB, re-downloadable from the pod archive.

| file | what it shows |
| --- | --- |
| `train_curves.png` | loss and mAP over 116 epochs — early stop at 66, patience 50 |
| `train_results.csv` | the same numbers, per epoch |
| `train_args.yaml` | exact training config, for reproducibility |
| `val_confusion_matrix.png` | val split, normalised |
| `test_confusion_matrix.png` | **test split** — the honest one |
| `test_pr_curve.png` | precision–recall per class |
| `test_f1_curve.png` | F1 vs confidence — use this to pick the deployment threshold |
| `test_batch0_ground_truth.jpg` | labelled boxes |
| `test_batch0_predictions.jpg` | what the model produced on the same images |

That last pair is the evidence for the claim that mAP50-95 (~0.435) is capped by annotation
quality rather than model capacity. Compare them side by side before writing that section.
