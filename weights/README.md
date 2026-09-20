# Model weights

Weights are **not** stored in git — a 20 MB binary in history is carried by every
clone forever, and retraining would add another copy each time. They live as a
**GitHub Release asset** instead.

## Get them

```bash
python scripts/get_weights.py
```

Downloads `best.pt` into this folder. Everything that needs a model defaults to
`weights/best.pt`.

## Publish a new model

```bash
gh release create v1-detector weights/best.pt --notes "YOLO26s @960, 6 presence classes"
# or: GitHub -> Releases -> Draft a new release -> attach best.pt
```

## Current model

| | |
| --- | --- |
| run | `ppe-presence-s960-3` |
| arch | YOLO26s, imgsz 960 |
| classes | helmet, gloves, vest, boots, goggles, Person |
| test mAP50 | 0.817 |
| test mAP50-95 | 0.433 |
| trained | early stop at epoch 66 of 200, patience 50 |

Per-class recall is recorded in `perception/events.py::DETECTOR_RECALL` and ships inside every
event. **If you retrain, update that table** — the agent uses it to weigh how much
an absent detection is worth.
