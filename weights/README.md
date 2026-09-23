# Model weights

Weights are **not** stored in git — a 20 MB binary in history is carried by every
clone forever, and retraining would add another copy each time.

## Get them

**Today:** no release is published. The weights are shared with the team directly —
put the file you were given at `weights/best.pt`. Everything that needs a model
defaults to that path.

**Once a release exists:**

```bash
python scripts/get_weights.py
```

downloads `best.pt` into this folder, and says what to do instead if it cannot.

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
