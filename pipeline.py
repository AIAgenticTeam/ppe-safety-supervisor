"""
frame -> detections -> zone-aware assessment -> ViolationEvent

The live path. Produces exactly the same JSON shape as `make_fixtures.py`, so
anything built against `fixtures/` works unchanged against real footage.

    python pipeline.py clip.mp4    --camera cam_3 --out events/   # video, tracked
    python pipeline.py frame.jpg   --camera cam_3                 # one still
    python pipeline.py footage/    --camera cam_3                 # a folder of stills

Two modes, and the difference is not cosmetic:

  VIDEO  runs ByteTrack and N-of-M temporal confirmation (see tracking.py). Only
         sustained absences are emitted, `confirmed` is True, and the resulting
         events are actionable. Evidence is captured at the confirming frame.

  STILLS have no temporal dimension, so `confirmation.rule` stays "single_frame"
         and `confirmed` is always False -- meaning `is_actionable` is False too.
         That is correct, not a limitation: at helmet recall 0.796 a single frame
         is not evidence enough to accuse anyone. Stills are for inspecting the
         assessment layer, never for issuing findings.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from events import (RIYADH, Confirmation, Evidence, ViolationEvent, build_event,
                    make_event_id)
from evidence import EvidenceStore
from ppe_compliance import Policy, assess_detections
from zones import ZoneMap

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


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

    store = EvidenceStore(out_dir / "evidence")

    for person in people:
        # The id is minted before anything is written, because the evidence filenames
        # are built from it. Deriving it twice and trusting the two to agree kept the
        # event pointing at files only by coincidence.
        event_id = make_event_id(camera_id, person.person_id, when)

        evidence = Evidence()
        if save_crops:
            # Same store as the video path. The stills path used to write a bare crop
            # with no frame and no annotation, so a reviewer opening it saw a
            # photograph of a worker with nothing marking what was alleged.
            label = f"missing {'+'.join(person.missing)}" if person.missing else ""
            stored = store.save_from_image_file(event_id, image_path, person.bbox,
                                                label=label)
            evidence = stored.as_event_evidence()

        event = build_event(
            assessment=person,
            zone=person.zone or zmap.zone_or_default(camera_id, person.bbox),
            camera_id=camera_id,
            event_id=event_id,
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


def process_video(video_path: Path, weights: Path, zmap: ZoneMap, camera_id: str,
                  policy: Policy, out_dir: Path, n: int, m: int, stride: int,
                  imgsz: int | None = None) -> list[ViolationEvent]:
    """Track people through a clip and emit only temporally confirmed findings.

    Evidence is captured at the confirming frame, because video frames are transient --
    once the loop moves on, that image is gone.
    """
    from datetime import timedelta

    from tracking import track_video

    store = EvidenceStore(out_dir / "evidence")
    events: list[ViolationEvent] = []
    started = datetime.now(RIYADH)

    print(f"tracking {video_path.name}  rule {n}_of_{m}"
          f"{f'  stride {stride}' if stride > 1 else ''}\n")

    for confirmation, person, frame_index, frame in track_video(
            video_path, weights, zmap, camera_id, policy,
            n=n, m=m, stride=stride, imgsz=imgsz):

        # Timestamp the finding at its position in the clip, not at wall-clock now.
        when = started + timedelta(seconds=confirmation.window_seconds)
        event_id = make_event_id(camera_id, confirmation.track_id, when)

        stored = store.save(
            event_id, frame, person.bbox,
            label=f"missing {'+'.join(confirmation.items)}",
            present_boxes=None,
        )

        event = build_event(
            assessment=person,
            zone=person.zone or zmap.zone_or_default(camera_id, person.bbox),
            camera_id=camera_id,
            track_id=confirmation.track_id,
            confirmation=Confirmation(
                frames_observed=confirmation.frames_observed,
                frames_missing=confirmation.frames_missing,
                window_seconds=confirmation.window_seconds,
                rule=confirmation.rule,
                confirmed=True,
            ),
            when=when,
            evidence=stored.as_event_evidence(),
            event_id=event_id,
        )
        events.append(event)

    return events


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="a video, an image, or a directory of frames")
    ap.add_argument("--camera", required=True, help="camera id in the zone config")
    ap.add_argument("--zones", default="zones.json")
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--out", default="events")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="omit to inherit the checkpoint's training resolution")
    ap.add_argument("--no-crops", action="store_true",
                    help="stills mode only; video always captures evidence")
    ap.add_argument("-n", type=int, default=8,
                    help="video: frames an item must be missing (default 8)")
    ap.add_argument("-m", type=int, default=10,
                    help="video: window size in observed frames (default 10)")
    ap.add_argument("--stride", type=int, default=1,
                    help="video: process every Nth frame (default 1)")
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

    src = Path(args.source)
    out = Path(args.out)
    (out / "events").mkdir(parents=True, exist_ok=True)
    is_video = src.is_file() and src.suffix.lower() in VIDEO_EXT

    if is_video:
        all_events = process_video(src, weights, zmap, args.camera, Policy(), out,
                                   args.n, args.m, args.stride, args.imgsz)
        for e in all_events:
            (out / "events" / f"{e.event_id}.json").write_text(e.to_json(),
                                                               encoding="utf-8")
        frames = [src]
    else:
        from ultralytics import YOLO
        model = YOLO(str(weights))

        frames = ([src] if src.is_file()
                  else sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT))
        if not frames:
            raise SystemExit(f"no images found at {src}")

        all_events = []
        for frame in frames:
            for e in process_image(model, frame, zmap, args.camera, Policy(), out,
                                   args.imgsz, not args.no_crops):
                (out / "events" / f"{e.event_id}.json").write_text(e.to_json(),
                                                                   encoding="utf-8")
                all_events.append(e)

    from collections import Counter
    by_status = Counter(e.status for e in all_events)
    by_zone = Counter(e.zone.name for e in all_events)

    source = src.name if is_video else f"{len(frames)} frames"
    print(f"\n{source} -> {len(all_events)} events -> {out}/events/\n")
    for k, v in by_status.most_common():
        print(f"  {k:<11} {v:>4}")
    if by_zone:
        print("\n  by zone:")
        for k, v in by_zone.most_common():
            print(f"    {k:<20} {v:>4}")

    actionable = sum(1 for e in all_events if e.is_actionable)
    print(f"\n  actionable: {actionable}")
    if is_video:
        store = EvidenceStore(out / "evidence")
        u = store.usage()
        print(f"  evidence : {u['events']} pairs, {u['megabytes']} MB "
              f"-> {out}/evidence/")
    else:
        print("  stills have no temporal dimension, so nothing is actionable by design")


if __name__ == "__main__":
    main()
