"""
Tests for temporal confirmation.

These are what justify the claim that the system does not accuse people on one frame.
They run without a model or a video -- the confirmation logic is pure by design.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from tracking import TemporalConfirmer, TrackHistory, Observation  # noqa: E402


def feed(confirmer, track_id, frames, start=0):
    """Feed a list of (status, missing, indeterminate) tuples; return all confirmations."""
    out = []
    for i, (status, missing, indet) in enumerate(frames, start=start):
        out.extend(confirmer.ingest(i, track_id, status, tuple(missing), tuple(indet)))
    return out


MISSING_HELMET = ("violation", ["helmet"], [])
COMPLIANT = ("compliant", [], [])
UNSEEN_HEAD = ("review", [], ["helmet"])


# --------------------------------------------------------------- the core rule

def test_sustained_absence_confirms():
    c = TemporalConfirmer(n=8, m=10, fps=10)
    got = feed(c, 1, [MISSING_HELMET] * 10)
    assert len(got) == 1
    assert got[0].items == ("helmet",)
    assert got[0].frames_missing == 10
    assert got[0].confirmed


def test_single_frame_never_confirms():
    """The whole point of this module."""
    c = TemporalConfirmer(n=8, m=10)
    assert feed(c, 1, [MISSING_HELMET]) == []


def test_below_threshold_does_not_confirm():
    """7 of 10 missing, rule needs 8."""
    c = TemporalConfirmer(n=8, m=10)
    frames = [MISSING_HELMET] * 7 + [COMPLIANT] * 3
    assert feed(c, 1, frames) == []


def test_exactly_at_threshold_confirms():
    c = TemporalConfirmer(n=8, m=10)
    frames = [MISSING_HELMET] * 8 + [COMPLIANT] * 2
    got = feed(c, 1, frames)
    assert len(got) == 1 and got[0].frames_missing == 8


def test_short_track_cannot_confirm():
    """Someone glimpsed crossing a corner of frame is never judged."""
    c = TemporalConfirmer(n=3, m=10)
    assert feed(c, 1, [MISSING_HELMET] * 9) == []      # 9 < m, no verdict
    # the tenth observation completes the window and only then can it fire
    assert len(list(c.ingest(9, 1, "violation", ("helmet",)))) == 1


def test_intermittent_detection_does_not_confirm():
    """A helmet detected every other frame is flicker, not a bare head."""
    c = TemporalConfirmer(n=8, m=10)
    frames = [MISSING_HELMET, COMPLIANT] * 5          # 5 of 10 missing
    assert feed(c, 1, frames) == []


# ------------------------------------------------- indeterminate is not missing

def test_indeterminate_never_counts_as_missing():
    """'I could not see their head' is not evidence of a bare head."""
    c = TemporalConfirmer(n=8, m=10)
    assert feed(c, 1, [UNSEEN_HEAD] * 10) == []


def test_indeterminate_frames_dilute_the_window():
    """An occluded worker never accumulates enough evidence to be accused."""
    c = TemporalConfirmer(n=8, m=10)
    frames = [MISSING_HELMET] * 7 + [UNSEEN_HEAD] * 3   # only 7 assessable absences
    assert feed(c, 1, frames) == []


# ------------------------------------------------------------------- debounce

def test_confirmation_fires_once_not_every_frame():
    """A worker at a grinder for ten seconds must not generate 300 tickets."""
    c = TemporalConfirmer(n=8, m=10)
    got = feed(c, 1, [MISSING_HELMET] * 300)
    assert len(got) == 1, f"expected one ticket, got {len(got)}"


def test_a_second_item_confirms_separately():
    c = TemporalConfirmer(n=8, m=10)
    feed(c, 1, [("violation", ["helmet"], [])] * 10)
    got = feed(c, 1, [("violation", ["helmet", "gloves"], [])] * 10, start=10)
    assert len(got) == 1
    assert got[0].items == ("gloves",)          # helmet already ticketed


# ------------------------------------------------------------- multiple people

def test_tracks_are_independent():
    c = TemporalConfirmer(n=8, m=10)
    for i in range(10):
        list(c.ingest(i, 1, "violation", ("helmet",)))
        list(c.ingest(i, 2, "compliant"))
    assert sorted(c.state()[1]["confirmed"]) == ["helmet"]
    assert c.state()[2]["confirmed"] == []


def test_dropping_a_track_forgets_it():
    c = TemporalConfirmer(n=8, m=10)
    feed(c, 1, [MISSING_HELMET] * 10)
    c.drop(1)
    assert 1 not in c.state()


# ----------------------------------------------------------------- bookkeeping

def test_state_reports_progress_toward_confirmation():
    c = TemporalConfirmer(n=8, m=10)
    feed(c, 1, [MISSING_HELMET] * 5 + [COMPLIANT] * 5)
    assert c.state()[1]["pending"]["helmet"] == "5/8"


def test_window_slides():
    """Old frames fall out; a worker who puts a helmet on stops being confirmed-eligible."""
    c = TemporalConfirmer(n=8, m=10)
    feed(c, 1, [MISSING_HELMET] * 7)
    feed(c, 1, [COMPLIANT] * 10, start=7)       # window now entirely compliant
    h = c.tracks[1]
    assert h.missing_count("helmet") == 0
    assert h.observed == 10


def test_window_seconds_uses_fps():
    c = TemporalConfirmer(n=8, m=10, fps=30)
    got = feed(c, 1, [MISSING_HELMET] * 10)
    assert got[0].window_seconds == pytest.approx(10 / 30, abs=0.01)


def test_rule_string_is_reported():
    assert TemporalConfirmer(n=8, m=10).rule == "8_of_10"
    assert TemporalConfirmer(n=3, m=5).rule == "3_of_5"


@pytest.mark.parametrize("n,m", [(11, 10), (0, 10)])
def test_invalid_rules_rejected(n, m):
    with pytest.raises(ValueError):
        TemporalConfirmer(n=n, m=m)


def test_history_window_is_bounded():
    h = TrackHistory(track_id=1, window_size=10)
    for i in range(50):
        h.add(Observation(i, "compliant"))
    assert h.observed == 10
