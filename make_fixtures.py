"""
Generate a realistic set of ViolationEvents without a GPU, a model, or a camera.

This is what unblocks Lanes C and D on day one. The agent graph, the Streamlit
console, the SQLite schema and the zone report can all be built and tested
against these files today, while Lane A is still wiring the tracker.

    python make_fixtures.py --out fixtures

Produces fixtures/events/*.json and fixtures/index.json. The set deliberately
covers every branch the agent has to handle, including the three that are easy
to forget: an indeterminate assessment, a zone where the "missing" item is not
actually required, and a repeat offender across several days.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from events import (RIYADH, Confirmation, DetectorInfo, Evidence, PPEState, Subject,
                    ViolationEvent, ZoneRef, make_event_id)
from zones import ZoneMap

# Monday of the demo week, 07:00 local.
WEEK_START = datetime(2026, 9, 7, 7, 0, tzinfo=RIYADH)


def _zone_ref(zmap: ZoneMap, camera_id: str, zone_name: str) -> ZoneRef:
    cam = zmap.cameras[camera_id]
    z = next(z for z in cam.zones if z.name == zone_name)
    return ZoneRef(name=z.name, label=z.label,
                   required_ppe=list(z.required_ppe),
                   severity_multiplier=z.severity_multiplier)


def _event(zmap, *, day, hour, minute, camera_id, zone_name, track_id, bbox,
           conf, present, missing, indeterminate, status, reasons=(),
           frames_observed=10, frames_missing=0, worker_ref=None) -> ViolationEvent:
    when = WEEK_START + timedelta(days=day, hours=hour - 7, minutes=minute)
    x1, y1, x2, y2 = bbox
    confirmed = status == "violation" and frames_missing >= 8
    return ViolationEvent(
        event_id=make_event_id(camera_id, track_id, when),
        captured_at=when.isoformat(timespec="seconds"),
        camera_id=camera_id,
        zone=_zone_ref(zmap, camera_id, zone_name),
        subject=Subject(track_id=track_id, bbox=bbox, detection_confidence=conf,
                        height_px=y2 - y1, worker_ref=worker_ref),
        ppe=PPEState(present=present, missing=list(missing),
                     indeterminate=list(indeterminate)),
        status=status,
        confirmation=Confirmation(
            frames_observed=frames_observed,
            frames_missing=frames_missing,
            window_seconds=round(frames_observed / 5.0, 1),
            rule="8_of_10",
            confirmed=confirmed,
        ),
        evidence=Evidence(
            crop_path=f"fixtures/crops/{make_event_id(camera_id, track_id, when)}.jpg",
            frame_path=f"fixtures/frames/{camera_id}_{when.strftime('%Y%m%dT%H%M%S')}.jpg",
        ),
        detector=DetectorInfo(),
        reasons=list(reasons),
    )


def build(zmap: ZoneMap) -> list[ViolationEvent]:
    E = []

    # --- DEMO BEAT 1 -------------------------------------------------------
    # Same worker, no gloves, in the walkway. The walkway requires only a helmet,
    # so this is COMPLIANT. Context decides, not the detection.
    E.append(_event(zmap, day=0, hour=7, minute=12, camera_id="cam_3",
                    zone_name="walkway", track_id=17, bbox=(120, 300, 290, 700),
                    conf=0.93, present={"helmet": 0.91}, missing=[], indeterminate=[],
                    status="compliant", worker_ref="W-0412"))

    # --- DEMO BEAT 2 -------------------------------------------------------
    # Same worker, minutes later, at the grinder. Gloves now required. VIOLATION.
    E.append(_event(zmap, day=0, hour=7, minute=26, camera_id="cam_3",
                    zone_name="grinding_station", track_id=17,
                    bbox=(520, 285, 700, 705), conf=0.94,
                    present={"helmet": 0.90, "goggles": 0.71}, missing=["gloves"],
                    indeterminate=[], status="violation",
                    frames_missing=10, worker_ref="W-0412"))

    # --- DEMO BEAT 3 -------------------------------------------------------
    # Distant worker: too small to assess small PPE. INDETERMINATE, not a violation.
    E.append(_event(zmap, day=0, hour=9, minute=3, camera_id="cam_3",
                    zone_name="grinding_station", track_id=22,
                    bbox=(610, 300, 668, 395), conf=0.88,
                    present={}, missing=[], indeterminate=["gloves", "goggles"],
                    status="review",
                    reasons=["person is 95px tall, below the 120px threshold for "
                             "reliable PPE detection"],
                    frames_observed=6, frames_missing=0))

    # Occluded by machinery: head zone not visible.
    E.append(_event(zmap, day=0, hour=11, minute=41, camera_id="cam_3",
                    zone_name="welding_bay", track_id=23, bbox=(900, 2, 1080, 540),
                    conf=0.90, present={"vest": 0.84, "gloves": 0.66},
                    missing=[], indeterminate=["helmet", "goggles"], status="review",
                    reasons=["helmet: body zone not fully inside the frame",
                             "goggles: body zone not fully inside the frame"]))

    # Low-confidence person: missing item, but not asserted as a violation.
    E.append(_event(zmap, day=1, hour=8, minute=15, camera_id="cam_3",
                    zone_name="loading_dock", track_id=31, bbox=(900, 590, 1040, 715),
                    conf=0.56, present={"helmet": 0.77}, missing=["vest"],
                    indeterminate=[], status="review",
                    reasons=["person detected at 0.56, below the 0.70 gate for "
                             "asserting a violation"],
                    frames_missing=9))

    # --- DEMO BEAT 4: repeat offender, same worker across the week ----------
    for day, hour, minute in ((2, 10, 8), (4, 14, 32)):
        E.append(_event(zmap, day=day, hour=hour, minute=minute, camera_id="cam_3",
                        zone_name="grinding_station", track_id=17,
                        bbox=(505, 290, 690, 700), conf=0.92,
                        present={"helmet": 0.89}, missing=["gloves", "goggles"],
                        indeterminate=[], status="violation",
                        frames_missing=10, worker_ref="W-0412"))

    # --- Routine compliant traffic (the majority of real events) ------------
    compliant = [
        (0, 8, 2, "cam_3", "assembly", 41, {"helmet": 0.93, "gloves": 0.81}),
        (1, 9, 45, "cam_1", "assembly_line", 52, {"helmet": 0.88, "gloves": 0.79}),
        (2, 7, 30, "cam_1", "assembly_line", 53, {"helmet": 0.91, "gloves": 0.74}),
        (3, 13, 11, "cam_1", "assembly_line", 55, {"helmet": 0.86, "gloves": 0.83}),
        (4, 8, 55, "cam_1", "assembly_line", 58, {"helmet": 0.94, "gloves": 0.77}),
    ]
    for day, hour, minute, cam, zone, tid, present in compliant:
        if cam == "cam_3":
            cam, zone = "cam_1", "assembly_line"
        E.append(_event(zmap, day=day, hour=hour, minute=minute, camera_id=cam,
                        zone_name=zone, track_id=tid, bbox=(300, 250, 470, 690),
                        conf=0.92, present=present, missing=[], indeterminate=[],
                        status="compliant"))

    # Welding bay: full PPE, correct.
    E.append(_event(zmap, day=1, hour=10, minute=20, camera_id="cam_3",
                    zone_name="welding_bay", track_id=61,
                    bbox=(880, 270, 1070, 555), conf=0.95,
                    present={"helmet": 0.92, "gloves": 0.80, "goggles": 0.76, "vest": 0.88},
                    missing=[], indeterminate=[], status="compliant"))

    # --- Other genuine violations, spread across days so the zone report and
    # --- the repeat-offender query have something to aggregate ---------------
    others = [
        (1, 11, 5,  "welding_bay",     72, {"helmet": 0.90, "gloves": 0.74, "goggles": 0.70}, ["vest"]),
        (2, 15, 40, "loading_dock",    77, {"helmet": 0.88, "boots": 0.69}, ["vest"]),
        (3, 7, 50,  "grinding_station", 81, {"helmet": 0.91, "gloves": 0.78}, ["goggles"]),
        (3, 16, 22, "walkway",          84, {}, ["helmet"]),
        (4, 12, 9,  "grinding_station", 88, {"helmet": 0.87, "goggles": 0.72}, ["gloves"]),
        (5, 9, 37,  "loading_dock",     91, {"helmet": 0.90, "vest": 0.85}, ["boots"]),
        (5, 14, 3,  "welding_bay",      93, {"helmet": 0.89, "vest": 0.86, "gloves": 0.71}, ["goggles"]),
    ]
    for day, hour, minute, zone, tid, present, missing in others:
        E.append(_event(zmap, day=day, hour=hour, minute=minute, camera_id="cam_3",
                        zone_name=zone, track_id=tid, bbox=(540, 280, 720, 700),
                        conf=0.91, present=present, missing=missing,
                        indeterminate=[], status="violation", frames_missing=9))

    E.sort(key=lambda e: e.captured_at)
    return E


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zones", default="zones.json")
    ap.add_argument("--out", default="fixtures")
    args = ap.parse_args()

    zmap = ZoneMap.load(args.zones)
    problems = zmap.validate()
    if problems:
        raise SystemExit("zone config invalid:\n  " + "\n  ".join(problems))

    out = Path(args.out)
    (out / "events").mkdir(parents=True, exist_ok=True)

    events = build(zmap)
    for e in events:
        (out / "events" / f"{e.event_id}.json").write_text(e.to_json(), encoding="utf-8")

    index = [{"event_id": e.event_id, "captured_at": e.captured_at,
              "camera_id": e.camera_id, "zone": e.zone.name, "status": e.status,
              "missing": e.ppe.missing, "indeterminate": e.ppe.indeterminate,
              "track_id": e.subject.track_id, "worker_ref": e.subject.worker_ref,
              "actionable": e.is_actionable}
             for e in events]
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    from collections import Counter
    by_status = Counter(e.status for e in events)
    by_zone = Counter(e.zone.name for e in events if e.status == "violation")

    print(f"wrote {len(events)} events -> {out}/events/")
    print(f"index -> {out}/index.json\n")
    print("by status:")
    for k, v in by_status.most_common():
        print(f"  {k:<11} {v:>3}")
    print(f"  {'actionable':<11} {sum(1 for e in events if e.is_actionable):>3}"
          "   (confirmed violations only)")
    print("\nviolations by zone:")
    for k, v in by_zone.most_common():
        print(f"  {k:<20} {v:>3}")
    repeats = Counter(e.subject.worker_ref for e in events
                      if e.status == "violation" and e.subject.worker_ref)
    if repeats:
        print("\nrepeat offenders:")
        for k, v in repeats.most_common():
            print(f"  {k:<10} {v:>3} violations this week")


if __name__ == "__main__":
    main()
