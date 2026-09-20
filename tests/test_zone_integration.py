"""
Integration tests for the live path: detections -> zone-aware assessment -> event.

These use stub detection boxes rather than a model, so they run anywhere without
a GPU, weights, or torch. What they prove is the join between `zones.py` and
`ppe_compliance.py` -- the part that makes the walkway/grinder distinction real
rather than something the fixtures merely assert.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from perception.events import Confirmation, build_event  # noqa: E402
from perception.ppe_compliance import Policy, assess_detections  # noqa: E402
from perception.zones import ZoneMap  # noqa: E402

NAMES = {0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles", 5: "Person"}
W, H = 1280, 720


class StubBox:
    """Minimal stand-in for an ultralytics Boxes row."""

    def __init__(self, cls: int, conf: float, xyxy):
        self.cls, self.conf, self.xyxy = cls, conf, [xyxy]


def worker_without_gloves(person_box):
    """A worker wearing a helmet and goggles, but no gloves."""
    x1, y1, x2, y2 = person_box
    cx, h = (x1 + x2) / 2, y2 - y1
    return [
        StubBox(5, 0.94, list(person_box)),
        StubBox(0, 0.90, [cx - 40, y1 + 0.02 * h, cx + 40, y1 + 0.14 * h]),   # helmet
        StubBox(4, 0.71, [cx - 28, y1 + 0.10 * h, cx + 28, y1 + 0.18 * h]),   # goggles
    ]


@pytest.fixture(scope="module")
def zmap():
    return ZoneMap.load(ROOT / "config" / "zones.json")


# ------------------------------------------------------------------ the thesis

def test_same_detections_different_zones_different_outcomes(zmap):
    """No gloves is compliant in the walkway and a violation at the grinder.

    This is the claim slide 5 makes. Here it is on the live code path.
    """
    walkway_box = (120, 300, 290, 700)      # feet at (205, 700) -> walkway
    grinder_box = (520, 285, 700, 705)      # feet at (610, 705) -> grinding_station

    in_walkway, _ = assess_detections(worker_without_gloves(walkway_box), NAMES,
                                      W, H, Policy(), zmap, "cam_3")
    at_grinder, _ = assess_detections(worker_without_gloves(grinder_box), NAMES,
                                      W, H, Policy(), zmap, "cam_3")

    a, b = in_walkway[0], at_grinder[0]

    assert a.zone.name == "walkway"
    assert a.required_ppe == ("helmet",)
    assert a.missing == []
    assert a.status == "compliant"

    assert b.zone.name == "grinding_station"
    assert "gloves" in b.required_ppe
    assert b.missing == ["gloves"]
    assert b.status == "violation"


def test_zone_severity_multiplier_reaches_the_event(zmap):
    """The agent needs the multiplier; check it survives the whole path."""
    boxes = worker_without_gloves((520, 285, 700, 705))
    people, _ = assess_detections(boxes, NAMES, W, H, Policy(), zmap, "cam_3")

    event = build_event(assessment=people[0], zone=people[0].zone,
                        camera_id="cam_3", track_id=1,
                        confirmation=Confirmation(rule="single_frame"))

    assert event.zone.name == "grinding_station"
    assert event.zone.severity_multiplier == 2.0
    assert event.zone.required_ppe == ["helmet", "gloves", "goggles"]
    assert event.ppe.missing == ["gloves"]


def test_welding_bay_requires_more_than_grinder(zmap):
    """Same worker, same kit, stricter zone -> more missing items."""
    boxes = worker_without_gloves((880, 270, 1070, 555))   # feet -> welding_bay
    people, _ = assess_detections(boxes, NAMES, W, H, Policy(), zmap, "cam_3")
    p = people[0]

    assert p.zone.name == "welding_bay"
    assert set(p.missing) == {"gloves", "vest"}
    assert p.status == "violation"


# ------------------------------------------------------- fallback and guardrails

def test_without_a_zone_map_policy_required_applies(zmap):
    """Backward compatible: no zone map means one rule for everyone in frame."""
    boxes = worker_without_gloves((520, 285, 700, 705))
    people, _ = assess_detections(boxes, NAMES, W, H, Policy(required=("helmet", "vest")))
    p = people[0]

    assert p.zone is None
    assert p.required_ppe == ("helmet", "vest")
    assert p.missing == ["vest"]          # gloves not required by the fallback policy


def test_unzoned_position_uses_camera_default(zmap):
    """Standing outside every drawn polygon still gets the camera's baseline."""
    boxes = worker_without_gloves((600, 40, 760, 240))     # feet at (680, 240), above all zones
    people, _ = assess_detections(boxes, NAMES, W, H, Policy(), zmap, "cam_3")
    p = people[0]

    assert p.zone.name == "unzoned"
    assert p.required_ppe == ("helmet",)
    assert p.status == "compliant"        # helmet is present


def test_single_frame_event_is_never_actionable(zmap):
    """Until the day-2 tracker confirms across frames, nothing is actionable."""
    boxes = worker_without_gloves((520, 285, 700, 705))
    people, _ = assess_detections(boxes, NAMES, W, H, Policy(), zmap, "cam_3")

    event = build_event(assessment=people[0], zone=people[0].zone,
                        camera_id="cam_3", track_id=1,
                        confirmation=Confirmation(rule="single_frame", confirmed=False))

    assert event.status == "violation"
    assert not event.is_actionable, "a single frame must never accuse anyone"


def test_two_people_in_different_zones_judged_separately(zmap):
    """One frame, two workers, two different zone requirements."""
    boxes = (worker_without_gloves((120, 300, 290, 700))      # walkway
             + worker_without_gloves((520, 285, 700, 705)))   # grinder
    people, _ = assess_detections(boxes, NAMES, W, H, Policy(), zmap, "cam_3")

    assert len(people) == 2
    by_zone = {p.zone.name: p for p in people}
    assert by_zone["walkway"].status == "compliant"
    assert by_zone["grinding_station"].status == "violation"


def test_ppe_on_the_ground_is_not_credited_to_anyone(zmap):
    """A helmet lying on a bench must not make an unhelmeted worker compliant."""
    boxes = [
        StubBox(5, 0.94, [520, 285, 700, 705]),      # worker, bare head
        StubBox(0, 0.88, [1000, 640, 1060, 690]),    # helmet on the floor, far away
    ]
    people, unassigned = assess_detections(boxes, NAMES, W, H, Policy(), zmap, "cam_3")

    assert "helmet" in people[0].missing
    assert len(unassigned) == 1 and unassigned[0]["class"] == "helmet"
