# ViolationEvent — the contract

`SCHEMA_VERSION = "1.0"` · defined in [`events.py`](../events.py)

This is the interface between perception (lane A) and everything downstream. Lane A produces
these; lanes C and D consume them. **Nothing downstream ever sees a bounding box tensor, a
confidence map, or a frame.**

If you need a new field: bump `SCHEMA_VERSION`, update this document, and tell the team in
the group chat. Do not add one quietly — the fixtures and tests will catch it, but only after
someone has wasted an afternoon.

A machine-readable JSON Schema is exported as `events.JSON_SCHEMA`. Use it in tests:

```python
from events import JSON_SCHEMA
import jsonschema, json
jsonschema.validate(json.load(open("fixtures/events/....json")), JSON_SCHEMA)
```

---

## Example

```json
{
  "event_id": "evt_20260907T072600_cam_3_t17",
  "schema_version": "1.0",
  "captured_at": "2026-09-07T07:26:00+03:00",
  "camera_id": "cam_3",
  "zone": {
    "name": "grinding_station",
    "label": "Bay 3 - grinding station",
    "required_ppe": ["helmet", "gloves", "goggles"],
    "severity_multiplier": 2.0
  },
  "subject": {
    "track_id": 17,
    "bbox": [520, 285, 700, 705],
    "detection_confidence": 0.94,
    "height_px": 420,
    "worker_ref": "W-0412"
  },
  "ppe": {
    "present": { "helmet": 0.9, "goggles": 0.71 },
    "missing": ["gloves"],
    "indeterminate": []
  },
  "status": "violation",
  "confirmation": {
    "frames_observed": 10,
    "frames_missing": 10,
    "window_seconds": 2.0,
    "rule": "8_of_10",
    "confirmed": true
  },
  "evidence": { "crop_path": "...", "frame_path": "..." },
  "detector": {
    "weights": "ppe-presence-s960-3/weights/best.pt",
    "imgsz": 960,
    "recall": { "helmet": 0.796, "gloves": 0.733, "...": 0 }
  },
  "reasons": []
}
```

---

## Fields that carry uncertainty

Three fields exist so the agent can reason honestly about what the vision system does **not**
know. Using them correctly is the difference between a credible system and one that
confidently accuses people.

### `ppe.missing` vs `ppe.indeterminate`

| field | meaning | may support a violation? |
| --- | --- | --- |
| `missing` | required, the zone was assessable, the item was not detected | yes |
| `indeterminate` | required, but could not be assessed at all | **never** |

`indeterminate` is populated when the relevant body zone is outside the frame, the subject is
too small for reliable small-PPE detection, or the person detection confidence is too low.
"I looked and it was not there" and "I could not look" are different claims.

### `detector.recall`

The measured per-class recall on the held-out test split, shipped inside every event.

```
helmet 0.796 · vest 0.816 · boots 0.742 · goggles 0.723 · gloves 0.733 · Person 0.867
```

Read it as: at gloves 0.733, **roughly one worn glove in four is not detected**. An absence is
therefore weaker evidence than a presence, and weaker for gloves than for vests. The agent
should let this shape its confidence language and its severity score — never assert certainty
the detector does not have.

`event.weakest_evidence()` returns the lowest recall among the missing items.

### `confirmation`

Temporal evidence from the tracker. A single-frame absence is not a violation; sustained
absence across a tracked window is.

`confirmed` is true only when the N-of-M rule was satisfied. `event.is_actionable` requires
**both** `status == "violation"` and `confirmation.confirmed` — use that property rather than
checking `status` alone.

---

## `status`

| value | meaning | who handles it |
| --- | --- | --- |
| `compliant` | all zone-required PPE present | logged, no action |
| `violation` | required PPE assessably absent | Compliance Agent |
| `review` | something could not be assessed, or gates not met | human |

---

## `subject.worker_ref` — read this before building history features

**This field is not produced by the vision system.** ByteTrack IDs do not survive a session, a
camera hand-off, or a worker leaving and re-entering frame. Person re-identification is out of
scope.

So `worker_ref` is bound by a supervisor in the console, or by a badge/roster integration.
When it is `null`, history lookups fall back to `track_id` within the current session only.

Any feature that claims "third time this week" must read `worker_ref`, and must degrade
gracefully when it is absent. The fallback is per-zone aggregation — "the grinding station
generated five violations this week" — which needs no identity at all and is arguably the more
useful insight.

---

## Zones

Defined in [`zones.json`](../zones.json), resolved by [`zones.py`](../zones.py).

A person is located by their **foot point** — the bottom-centre of their box — not the box
centre. A worker leaning over a machine has a box overlapping three zones; their feet are in
one. Where zones overlap, the smallest wins, so a machine station drawn inside a general work
area takes precedence.

Validate and visualise before trusting any event:

```bash
python zones.py zones.json
python zones.py zones.json --overlay frame.jpg cam_3 zones_overlay.jpg
```

A polygon that looks right in JSON is routinely ten metres off on the actual floor. Have
someone who knows the site check the overlay.
