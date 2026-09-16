"""
The event store's query surface.

These are the methods the Lane D console is built on -- the identification queue, the
approval queue, the dashboard counts. They had no tests, which meant the console would
have been the first thing to discover whether they worked.

The queries are small. What they get wrong is not arithmetic, it is *which rows they
are allowed to see*: an unattributed event must not count toward anybody's history, an
unconfirmed one must not count as a violation, and a decision already signed must not
appear in the queue of things awaiting a signature.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import EventStore, Worker  # noqa: E402


def event(event_id, *, worker=None, zone="grinding_station", camera="cam_3",
          missing=("helmet",), confirmed=True, status="violation", days_ago=0):
    captured = (datetime.now() - timedelta(days=days_ago)).isoformat(timespec="seconds")
    return {
        "event_id": event_id, "captured_at": captured, "camera_id": camera,
        "schema_version": "1.0",
        "zone": {"name": zone, "label": zone, "required_ppe": ["helmet"]},
        "subject": {"track_id": 1, "bbox": [0, 0, 10, 10],
                    "detection_confidence": 0.9, "height_px": 400,
                    "worker_ref": worker},
        "ppe": {"present": {}, "missing": list(missing), "indeterminate": []},
        "status": status,
        "confirmation": {"frames_observed": 10, "frames_missing": 10,
                         "confirmed": confirmed, "rule": "8_of_10"},
        "detector": {"recall": {"helmet": 0.796}}, "evidence": {}, "reasons": [],
    }


@pytest.fixture
def db(tmp_path):
    store = EventStore(tmp_path / "t.db")
    store.add_worker(Worker("W-1", "Alpha"))
    store.add_worker(Worker("W-2", "Beta"))
    return store


def bind(db, event_id, worker="W-1"):
    assert db.attach_identity(event_id, worker, "supervisor:test")


# ------------------------------------------------------- the roster

def test_roster_is_what_the_dropdown_is_built_from(db):
    assert [w.worker_id for w in db.roster()] == ["W-1", "W-2"]   # ordered by name


def test_an_inactive_worker_leaves_the_dropdown(db):
    import sqlite3
    with sqlite3.connect(db.path) as conn:
        conn.execute("UPDATE workers SET active = 0 WHERE worker_id = 'W-2'")
    assert [w.worker_id for w in db.roster()] == ["W-1"]


# --------------------------------------- the supervisor's identification queue

def test_unattributed_lists_only_what_nobody_has_named(db):
    db.record_event(event("e1"))
    db.record_event(event("e2"))
    bind(db, "e2")
    assert [row["event_id"] for row in db.unattributed()] == ["e1"]


def test_unattributed_puts_actionable_findings_first(db):
    """A supervisor working down the queue should meet the real violations first."""
    db.record_event(event("quiet", confirmed=False, days_ago=0))
    db.record_event(event("real", confirmed=True, days_ago=3))
    assert [row["event_id"] for row in db.unattributed()] == ["real", "quiet"]


def test_unattributed_decodes_the_missing_list(db):
    """It is stored as JSON text; a console that got a string would render it as one."""
    db.record_event(event("e1", missing=["helmet", "gloves"]))
    assert db.unattributed()[0]["missing"] == ["helmet", "gloves"]


def test_unattributed_respects_its_limit(db):
    for i in range(5):
        db.record_event(event(f"e{i}"))
    assert len(db.unattributed(limit=2)) == 2


# ---------------------------------------------------------- approval

def _decide(db, event_id, *, action="escalation", severity=7.0, approval=True):
    db.record_decision(event_id=event_id, action=action, severity=severity,
                       band="escalation", citations=["1926.100"], rationale="r",
                       draft_body="b", confidence=0.8, requires_approval=approval)


def test_nothing_is_sent_before_a_human_signs(db):
    db.record_event(event("e1"))
    _decide(db, "e1")
    assert [r["event_id"] for r in db.pending_approval()] == ["e1"]

    assert db.approve("e1", "supervisor:khalid")
    assert db.pending_approval() == []


def test_a_decision_needing_no_approval_never_enters_the_queue(db):
    db.record_event(event("e1"))
    _decide(db, "e1", action="warning", severity=3.0, approval=False)
    assert db.pending_approval() == []


def test_approving_twice_does_not_overwrite_the_first_signature(db):
    """Whose name is on it matters. The second call must not quietly replace the first."""
    db.record_event(event("e1"))
    _decide(db, "e1")
    assert db.approve("e1", "supervisor:first")
    assert not db.approve("e1", "supervisor:second")
    assert db.get_decision("e1")["approved_by"] == "supervisor:first"


def test_approving_something_that_does_not_exist_is_false_not_an_error(db):
    assert not db.approve("no_such_event", "supervisor:khalid")


def test_the_approval_queue_is_worst_first(db):
    for eid, sev in (("low", 5.0), ("worst", 9.0), ("mid", 7.0)):
        db.record_event(event(eid))
        _decide(db, eid, severity=sev)
    assert [r["event_id"] for r in db.pending_approval()] == ["worst", "mid", "low"]


# -------------------------------------------------------- the dashboard

def test_by_zone_counts_findings_and_confirmed_ones_separately(db):
    db.record_event(event("e1", zone="walkway", confirmed=True))
    db.record_event(event("e2", zone="walkway", confirmed=False))
    row = next(r for r in db.by_zone() if r["zone"] == "walkway")
    assert row["n"] == 2 and row["confirmed"] == 1


def test_by_zone_separates_the_same_zone_name_on_different_cameras(db):
    """cam_3/walkway and d_view02/walkway are different places. Merging them would
    average away whichever one is actually dangerous."""
    db.record_event(event("e1", zone="walkway", camera="cam_3"))
    db.record_event(event("e2", zone="walkway", camera="d_view02"))
    cameras = {r["camera_id"] for r in db.by_zone() if r["zone"] == "walkway"}
    assert cameras == {"cam_3", "d_view02"}


def test_by_zone_ignores_what_fell_outside_the_window(db):
    db.record_event(event("old", zone="walkway", days_ago=30))
    assert not any(r["zone"] == "walkway" for r in db.by_zone(days=7))


def test_repeat_offenders_needs_attributed_confirmed_violations(db):
    for i in range(3):
        db.record_event(event(f"e{i}"))
        bind(db, f"e{i}")
    assert db.repeat_offenders(minimum=2)[0]["violations"] == 3


def test_repeat_offenders_cannot_see_unattributed_events(db):
    """The safety property, from the reporting side. Three unnamed findings must not
    become a repeat offender."""
    for i in range(3):
        db.record_event(event(f"e{i}", worker="W-1"))     # a ref, but nobody bound it
    assert db.repeat_offenders(minimum=2) == []


def test_repeat_offenders_ignores_unconfirmed_findings(db):
    for i in range(3):
        db.record_event(event(f"e{i}", confirmed=False))
        bind(db, f"e{i}")
    assert db.repeat_offenders(minimum=2) == []


def test_repeat_offenders_honours_the_minimum(db):
    db.record_event(event("e0"))
    bind(db, "e0")
    assert db.repeat_offenders(minimum=2) == []


def test_repeat_offenders_carries_the_name_for_the_console(db):
    for i in range(2):
        db.record_event(event(f"e{i}"))
        bind(db, f"e{i}")
    assert db.repeat_offenders(minimum=2)[0]["name"] == "Alpha"


def test_stats_counts_what_the_header_shows(db):
    db.record_event(event("e1"))
    db.record_event(event("e2", confirmed=False))
    bind(db, "e1")
    _decide(db, "e1")
    s = db.stats()
    assert s == {"events": 2, "actionable": 1, "unattributed": 1,
                 "decisions": 1, "workers": 2}


def test_stats_on_an_empty_store_returns_zeros_not_nulls(db):
    """SUM() over no rows is NULL in SQLite, and a dashboard that renders None as a
    count looks broken on the one screen everyone sees first."""
    assert db.stats() == {"events": 0, "actionable": 0, "unattributed": 0,
                          "decisions": 0, "workers": 2}
