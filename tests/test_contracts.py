"""
Contract tests. These protect the interface every lane depends on.

If a change here fails, it means the ViolationEvent shape or the zone geometry moved --
which breaks lanes C and D silently. Fix the change, or bump SCHEMA_VERSION deliberately
and update docs/EVENT_SCHEMA.md and the fixtures together.

    python -m pytest tests -q
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from events import SCHEMA_VERSION, ViolationEvent  # noqa: E402
from zones import ZoneMap  # noqa: E402

FIXTURES = sorted((ROOT / "fixtures" / "events").glob("*.json"))


@pytest.fixture(scope="module")
def zmap():
    return ZoneMap.load(ROOT / "zones.json")


# --------------------------------------------------------------------- zones

def test_zone_config_is_valid(zmap):
    assert zmap.validate() == [], "zones.json has problems; run `python zones.py zones.json`"


def test_person_located_by_feet_not_box_centre(zmap):
    """A worker leaning across zones is judged by where they stand."""
    leaning = (700, 280, 1000, 700)          # box overlaps grinder and welding bay
    zone = zmap.zone_or_default("cam_3", leaning)
    assert zone.name == "loading_dock"       # feet at (850, 700)


@pytest.mark.parametrize("bbox,expected", [
    ((120, 300, 290, 700), "walkway"),
    ((520, 285, 700, 705), "grinding_station"),
    ((880, 270, 1070, 555), "welding_bay"),
    ((900, 590, 1040, 715), "loading_dock"),
])
def test_zone_lookup(zmap, bbox, expected):
    assert zmap.zone_or_default("cam_3", bbox).name == expected


def test_unzoned_falls_back_to_camera_default(zmap):
    zone = zmap.zone_or_default("cam_3", (600, 40, 700, 150))
    assert zone.name == "unzoned"
    assert zone.required_ppe == ("helmet",)


def test_walkway_requires_less_than_grinder(zmap):
    """The whole thesis of the project, as a test."""
    walkway = zmap.zone_or_default("cam_3", (120, 300, 290, 700))
    grinder = zmap.zone_or_default("cam_3", (520, 285, 700, 705))
    assert "gloves" not in walkway.required_ppe
    assert "gloves" in grinder.required_ppe
    assert grinder.severity_multiplier > walkway.severity_multiplier


def test_unknown_camera_raises(zmap):
    with pytest.raises(KeyError):
        zmap.locate("cam_99", (0, 0, 10, 10))


# -------------------------------------------------------------------- events

def test_fixtures_exist():
    assert len(FIXTURES) >= 20, "run `python make_fixtures.py --out fixtures`"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_round_trips(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    event = ViolationEvent.from_dict(raw)
    assert json.loads(event.to_json()) == raw


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_is_internally_consistent(path):
    e = ViolationEvent.from_dict(json.loads(path.read_text(encoding="utf-8")))

    assert e.schema_version == SCHEMA_VERSION
    assert e.status in ("compliant", "violation", "review")

    # an item cannot be both absent-and-assessed and unassessable
    assert not (set(e.ppe.missing) & set(e.ppe.indeterminate))

    # anything reported missing or present must actually be required in this zone
    for item in e.ppe.missing + e.ppe.indeterminate:
        assert item in e.zone.required_ppe, f"{item} not required in {e.zone.name}"

    # a compliant subject has nothing outstanding
    if e.status == "compliant":
        assert not e.ppe.missing and not e.ppe.indeterminate


def test_indeterminate_never_actionable():
    """Guardrail: 'I could not look' must never become an accusation."""
    for path in FIXTURES:
        e = ViolationEvent.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if e.ppe.indeterminate:
            assert e.status != "violation", f"{e.event_id} asserts a violation it cannot see"
            assert not e.is_actionable


def test_actionable_requires_temporal_confirmation():
    for path in FIXTURES:
        e = ViolationEvent.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if e.is_actionable:
            assert e.status == "violation"
            assert e.confirmation.confirmed
            assert e.confirmation.frames_missing >= 8


def test_detector_recall_travels_with_every_event():
    """The agent needs this to size its own confidence."""
    for path in FIXTURES:
        e = ViolationEvent.from_dict(json.loads(path.read_text(encoding="utf-8")))
        assert e.detector.recall["helmet"] == pytest.approx(0.796)
        if e.ppe.missing:
            assert 0 < e.weakest_evidence() <= 1


def test_the_demo_thesis_holds():
    """Same worker, no gloves, two zones, two different outcomes."""
    by_id = {}
    for path in FIXTURES:
        e = ViolationEvent.from_dict(json.loads(path.read_text(encoding="utf-8")))
        by_id.setdefault(e.subject.track_id, []).append(e)

    worker = sorted(by_id[17], key=lambda e: e.captured_at)
    walkway, grinder = worker[0], worker[1]

    assert walkway.zone.name == "walkway"
    assert walkway.status == "compliant"          # no gloves, but none required here
    assert grinder.zone.name == "grinding_station"
    assert grinder.status == "violation"
    assert grinder.ppe.missing == ["gloves"]


def test_repeat_offender_is_traceable():
    refs = [ViolationEvent.from_dict(json.loads(p.read_text(encoding="utf-8")))
            for p in FIXTURES]
    w0412 = [e for e in refs if e.subject.worker_ref == "W-0412" and e.status == "violation"]
    assert len(w0412) >= 3, "the escalation demo beat needs a repeat offender in the fixtures"
