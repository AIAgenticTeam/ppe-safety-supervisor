"""
Drift monitoring.

Two things are worth testing about a drift report, and neither is the statistics --
Evidently's tests are Evidently's problem.

The first is that it *detects* something when something has genuinely changed. A
monitor that always says "fine" is worse than no monitor, because it is reassuring.

The second is that it refuses when it cannot tell. The original version of the parser
read a key that does not exist and reported "0 columns checked, 0 drifted", which
renders identically to a clean bill of health. That is the shape of failure this file
exists to catch.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("evidently", reason="needs evidently; install into .venv")
pytest.importorskip("pandas")

from app.db import EventStore  # noqa: E402
from app.monitoring import (MIN_ROWS, ColumnDrift, DriftResult, _column_drift,  # noqa: E402
                            drift_report, event_frame, run_drift, split_by_time)

NOW = datetime.now(timezone.utc)


def event(i, when, *, confidence=0.9, zone="grinding_station", missing=("helmet",),
          camera="cam_3", status="violation", recall=0.796):
    return {
        "event_id": f"e{i}", "captured_at": when.isoformat(), "camera_id": camera,
        "schema_version": "1.0",
        "zone": {"name": zone, "label": zone, "required_ppe": ["helmet"]},
        "subject": {"track_id": 1, "bbox": [0, 0, 10, 10],
                    "detection_confidence": confidence, "height_px": 400,
                    "worker_ref": None},
        "ppe": {"present": {}, "missing": list(missing), "indeterminate": []},
        "status": status,
        "confirmation": {"frames_observed": 10, "frames_missing": 9,
                         "confirmed": True, "rule": "8_of_10"},
        "detector": {"recall": {"helmet": recall, "gloves": 0.733}},
        "evidence": {}, "reasons": [],
    }


def store(tmp_path, events):
    db = EventStore(tmp_path / "m.db")
    for e in events:
        db.record_event(e)
    return db


def steady(n=30, start_id=0, days_ago=14, **kw):
    return [event(start_id + i, NOW - timedelta(days=days_ago) + timedelta(minutes=7 * i),
                  **kw) for i in range(n)]


# ------------------------------------------------------------- the frame

def test_the_frame_carries_the_features_that_move_when_perception_degrades(tmp_path):
    db = store(tmp_path, steady(3))
    frame = event_frame(db)
    for column in ("person_confidence", "weakest_recall", "missing_ratio",
                   "missing_count", "zone", "camera_id", "status", "attributed"):
        assert column in frame.columns


def test_weakest_recall_is_the_worst_of_the_missing_items(tmp_path):
    """The item we are least able to see is what limits the finding."""
    db = store(tmp_path, [event(0, NOW, missing=["helmet", "gloves"])])
    assert event_frame(db).iloc[0]["weakest_recall"] == 0.733


def test_a_compliant_finding_has_no_recall_to_speak_of(tmp_path):
    db = store(tmp_path, [event(0, NOW, missing=[], status="compliant")])
    assert event_frame(db).iloc[0]["weakest_recall"] is None or \
        str(event_frame(db).iloc[0]["weakest_recall"]) == "nan"


def test_attribution_is_tracked_because_losing_it_is_silent(tmp_path):
    """If identity binding stops, the memory loop stops, every third offence reads as
    a first, and nothing in the output looks wrong. This column is where that shows."""
    from app.db import Worker

    db = store(tmp_path, [event(0, NOW)])
    assert event_frame(db).iloc[0]["attributed"] == "no"

    db.add_worker(Worker("W-1", "Someone"))
    db.attach_identity("e0", "W-1", "supervisor:test")
    assert event_frame(db).iloc[0]["attributed"] == "yes"


def test_the_frame_is_empty_not_broken_when_nothing_is_recorded(tmp_path):
    assert event_frame(EventStore(tmp_path / "m.db")).empty


# --------------------------------------------------- it refuses when it cannot tell

def test_too_little_data_is_a_refusal_with_a_reason(tmp_path):
    db = store(tmp_path, steady(5, days_ago=14) + steady(5, start_id=5, days_ago=1))
    result = run_drift(db, days=7)
    assert not result.ran
    assert "not enough data" in result.reason
    assert str(MIN_ROWS) in result.reason


def test_an_empty_store_says_so_rather_than_passing(tmp_path):
    result = run_drift(EventStore(tmp_path / "m.db"))
    assert not result.ran and "no findings" in result.reason


def test_a_refusal_is_not_a_clean_bill_of_health(tmp_path):
    """The distinction the console has to render: "cannot tell" and "nothing wrong"
    must not produce the same summary."""
    refused = run_drift(EventStore(tmp_path / "m.db")).summary()
    assert refused["ran"] is False
    assert refused["reason"]


# ------------------------------------------------------ it detects a real change

@pytest.fixture
def drifted(tmp_path):
    """A camera that was moved: confidence collapses and the zone changes with it."""
    import random
    random.seed(7)
    before = [event(i, NOW - timedelta(days=14) + timedelta(minutes=7 * i),
                    confidence=round(random.gauss(0.91, 0.03), 3),
                    zone="grinding_station") for i in range(30)]
    after = [event(30 + i, NOW - timedelta(days=2) + timedelta(minutes=7 * i),
                   confidence=round(random.gauss(0.58, 0.06), 3),
                   zone="walkway") for i in range(30)]
    return store(tmp_path, before + after)


def test_collapsed_detection_confidence_is_caught(drifted):
    """The whole point. A monitor that never fires is worse than none, because it
    reassures."""
    result = run_drift(drifted, days=7)
    assert result.ran, result.reason
    assert "person_confidence" in result.drifted_columns


def test_it_does_not_flag_what_did_not_change(drifted):
    """A monitor that flags everything is also useless."""
    result = run_drift(drifted, days=7)
    for column in ("missing_count", "status", "camera_id", "attributed"):
        assert column not in result.drifted_columns


def test_every_feature_that_could_move_was_checked(drifted):
    """The original bug: a parser reading a key that does not exist reported zero
    checked and zero drifted, which renders as "all clear".

    The invariant is not a fixed count — it is that the number of features checked
    matches the number that had any way of changing. A column holding one value across
    both windows is excluded deliberately, so counting it would be wrong too.
    """
    from app.monitoring import usable_columns

    frame = event_frame(drifted)
    reference, current = split_by_time(frame, days=7)
    expected = usable_columns(reference, current)

    result = run_drift(drifted, days=7)
    assert result.checked_columns > 0, "nothing was parsed out of the report"
    assert result.checked_columns == len(expected), (
        f"checked {result.checked_columns} of {len(expected)} movable features: "
        f"{sorted(expected)}")


def test_constant_features_are_excluded_rather_than_counted_as_unchanged(drifted):
    """Counting a column that could not move as a column that did not move understates
    the drift share — the denominator has to be features that had a choice."""
    result = run_drift(drifted, days=7)
    assert "camera_id" not in [c.column for c in result.columns]
    assert result.share == 1.0, "both movable features drifted, so the share is 1.0"


def test_steady_operation_reports_no_drift(tmp_path):
    import random
    random.seed(3)
    before = [event(i, NOW - timedelta(days=14) + timedelta(minutes=7 * i),
                    confidence=round(random.gauss(0.90, 0.03), 3)) for i in range(30)]
    after = [event(30 + i, NOW - timedelta(days=2) + timedelta(minutes=7 * i),
                   confidence=round(random.gauss(0.90, 0.03), 3)) for i in range(30)]
    result = run_drift(store(tmp_path, before + after), days=7)
    assert result.ran
    assert result.drifted_columns == [], result.drifted_columns


def test_the_summary_survives_json(drifted):
    """It is served over HTTP. numpy floats do not serialise, and finding that out in
    the browser is finding it out late."""
    json.dumps(run_drift(drifted, days=7).summary())


def test_the_full_report_can_be_written(drifted, tmp_path):
    out = tmp_path / "reports" / "drift.html"
    result = run_drift(drifted, days=7, html_path=out)
    assert out.exists() and out.stat().st_size > 1000
    assert result.html == str(out)


# ------------------------------------------------------------- the parser

def test_the_parser_reads_config_not_the_display_name():
    raw = {"metrics": [
        {"metric_name": "DriftedColumnsCount(drift_share=0.5)",
         "value": {"count": 2.0, "share": 0.22}},
        {"metric_name": "ValueDrift(column=person_confidence,method=K-S p_value)",
         "config": {"column": "person_confidence", "threshold": 0.05,
                    "method": "K-S p_value"},
         "value": 1.7e-17},
        {"metric_name": "ValueDrift(column=status,method=Z-test p_value)",
         "config": {"column": "status", "threshold": 0.05, "method": "Z-test p_value"},
         "value": 1.0},
    ]}
    columns = _column_drift(raw)
    assert [c.column for c in columns] == ["person_confidence", "status"]
    assert columns[0].drifted and not columns[1].drifted


def test_an_unparseable_metric_is_omitted_rather_than_counted_clean():
    """If the shape changes in a future Evidently, this must look like "nothing
    checked", never like "nothing wrong"."""
    raw = {"metrics": [
        {"metric_name": "ValueDrift(column=x)", "config": {}, "value": None},
        {"metric_name": "ValueDrift(column=y)", "value": 0.01},
    ]}
    assert _column_drift(raw) == []


def test_drift_share_is_zero_when_nothing_was_checked():
    assert DriftResult(ran=True).share == 0.0
    assert DriftResult(ran=True).checked_columns == 0


def test_share_counts_only_the_drifted():
    result = DriftResult(ran=True, columns=[
        ColumnDrift("a", 0.001, 0.05, True), ColumnDrift("b", 0.9, 0.05, False),
        ColumnDrift("c", 0.9, 0.05, False), ColumnDrift("d", 0.9, 0.05, False)])
    assert result.share == 0.25 and result.drifted_columns == ["a"]


# ------------------------------------------------------------- the split

def test_the_split_puts_older_findings_in_the_reference(tmp_path):
    db = store(tmp_path, steady(10, days_ago=30) + steady(10, start_id=10, days_ago=1))
    reference, current = split_by_time(event_frame(db), days=7)
    assert len(reference) == 10 and len(current) == 10


def test_splitting_an_empty_frame_is_not_an_error(tmp_path):
    reference, current = split_by_time(event_frame(EventStore(tmp_path / "m.db")))
    assert reference.empty and current.empty
