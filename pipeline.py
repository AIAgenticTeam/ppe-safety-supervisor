"""
frame -> detections -> zone-aware assessment -> ViolationEvent

The live path. Produces exactly the same JSON shape as `make_fixtures.py`, so
anything built against `fixtures/` works unchanged against real footage.

    python pipeline.py frame.jpg --camera cam_3
    python pipeline.py footage/ --camera cam_3 --out events/

NOT YET TEMPORAL. Every event comes from a single frame, so `confirmation.rule`
is "single_frame" and `confirmation.confirmed` is always False -- which means
`event.is_actionable` is always False too. That is deliberate and correct: at
helmet recall 0.796, a single frame is not enough evidence to accuse anyone.
Day 2 adds ByteTrack and N-of-M confirmation, and that is what will flip
`confirmed` to True.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from events import RIYADH, Confirmation, Evidence, ViolationEvent, build_event
from ppe_compliance import Policy, assess_detections, save_evidence
from zones import ZoneMap

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_image(model, image_path: Path, zmap: ZoneMap, camera_id: str,
                  policy: Policy, out_dir: Path, imgsz: int | None = None,
                  save_crops: bool = True) -> list[ViolationEvent]:
    """Run one frame end to end and return its events (one per person)."""
    kwargs = {"conf": min(policy.person_conf_min, policy.ppe_conf_min), "verbose": False}
    if imgsz:
        kwargs["imgsz"] = imgsz
    result = model.predict(str(image_path), **kwargs)[0]

    img_h, img_w = result.orig_shape
    people, unassigned = assess_detections(
        result.boxes, model.names, img_w, img_h, policy, zmap, camera_id)

    when = datetime.fromtimestamp(image_path.stat().st_mtime, RIYADH)
    events: list[ViolationEvent] = []

    for person in people:
        evidence = Evidence()
        if save_crops:
            crop_dir = out_dir / "crops"
            evidence.crop_path = save_evidence(image_path, person, crop_dir)
            evidence.frame_path = str(image_path)

        event = build_event(
            assessment=person,
            zone=person.zone or zmap.zone_or_default(camera_id, person.bbox),
            camera_id=camera_id,
            # Without a tracker, ids are per-frame only. Day 2 replaces this with
            # a ByteTrack id that is stable across the confirmation window.
            track_id=person.person_id,
            confirmation=Confirmation(
                frames_observed=1, frames_missing=1 if person.missing else 0,
                window_seconds=0.0, rule="single_frame", confirmed=False),
            when=when,
            evidence=evidence,
        )
        events.append(event)

    if unassigned:
        # PPE detected but not attributable to anyone -- a helmet on a bench, or
        # a person the detector missed. Worth surfacing; never an accusation.
        (out_dir / "unassigned").mkdir(parents=True, exist_ok=True)
        (out_dir / "unassigned" / f"{image_path.stem}.json").write_text(
            json.dumps(unassigned, indent=2), encoding="utf-8")

    return events


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="image file or a directory of frames")
    ap.add_argument("--camera", required=True, help="camera id in the zone config")
    ap.add_argument("--zones", default="zones.json")
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--out", default="events")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="omit to inherit the checkpoint's training resolution")
    ap.add_argument("--no-crops", action="store_true")
    args = ap.parse_args()

    zmap = ZoneMap.load(args.zones)
    problems = zmap.validate()
    if problems:
        raise SystemExit("zone config invalid:\n  " + "\n  ".join(problems))
    if args.camera not in zmap.cameras:
        raise SystemExit(f"unknown camera {args.camera!r}; "
                         f"known: {sorted(zmap.cameras)}")

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"no checkpoint at {weights} -- pass --weights")

    from ultralytics import YOLO
    model = YOLO(str(weights))

    src = Path(args.source)
    frames = ([src] if src.is_file()
              else sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT))
    if not frames:
        raise SystemExit(f"no images found at {src}")

    out = Path(args.out)
    (out / "events").mkdir(parents=True, exist_ok=True)

    all_events: list[ViolationEvent] = []
    for frame in frames:
        for e in process_image(model, frame, zmap, args.camera, Policy(), out,
                               args.imgsz, not args.no_crops):
            (out / "events" / f"{e.event_id}.json").write_text(e.to_json(),
                                                               encoding="utf-8")
            all_events.append(e)

    from collections import Counter
    by_status = Counter(e.status for e in all_events)
    by_zone = Counter(e.zone.name for e in all_events)

    print(f"\n{len(frames)} frames -> {len(all_events)} events -> {out}/events/\n")
    for k, v in by_status.most_common():
        print(f"  {k:<11} {v:>4}")
    print("\n  by zone:")
    for k, v in by_zone.most_common():
        print(f"    {k:<20} {v:>4}")

    actionable = sum(1 for e in all_events if e.is_actionable)
    print(f"\n  actionable: {actionable}   "
          "(always 0 until the day-2 tracker confirms across frames)")


if __name__ == "__main__":
    main()
