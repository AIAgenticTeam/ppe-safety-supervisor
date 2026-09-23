"""
Agent tool tests.

These protect the boundary that makes the agent layer defensible: the model contributes
wording and proportionality, never facts. A tool that guesses, or that lets an unknown
be read as a zero, undoes that.

No API key, no model, no network.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from agents.tools import (assess_confidence, event_facts, final_severity,  # noqa: E402
                          get_worker_history, lookup_clause, score_baseline)

FIXTURES = sorted((ROOT / "fixtures" / "events").glob("*.json"))


def load(name_fragment: str) -> dict:
    for f in FIXTURES:
        if name_fragment in f.name:
            return json.loads(f.read_text(encoding="utf-8"))
    raise FileNotFoundError(name_fragment)


# ------------------------------------------------------------- lookup_clause

def test_clause_lookup_is_deterministic():
    assert lookup_clause("helmet", "walkway") == lookup_clause("helmet", "walkway")


def test_clause_lookup_reports_its_basis():
    """A notice must never present site policy as a regulatory requirement."""
    helmet = lookup_clause("helmet", "walkway")
    gloves = lookup_clause("gloves", "walkway")
    assert helmet.is_regulatory and helmet.basis == "regulation"
    assert not gloves.is_regulatory and gloves.basis == "site_policy"
    assert "site policy" in gloves.notice_phrase.lower()


def test_zone_can_raise_an_item_to_a_specific_clause():
    """Hi-vis is site policy generally, but 1926.201 squarely covers flaggers."""
    general = lookup_clause("vest", "walkway")
    dock = lookup_clause("vest", "loading_dock")
    assert general.clause_id == "1926.95" and not general.is_regulatory
    assert dock.clause_id == "1926.201" and dock.is_regulatory
    assert dock.override_reason


def test_boots_carry_their_specification_clause():
    """1926.96 says what footwear must MEET; the duty to wear is elsewhere."""
    boots = lookup_clause("boots")
    assert boots.clause_id == "1926.95"
    assert boots.specification_clause == "1926.96"
    assert not boots.is_regulatory


def test_unknown_item_raises_rather_than_guessing():
    with pytest.raises(KeyError):
        lookup_clause("respirator")


# --------------------------------------------------------------- severity

def test_baseline_needs_no_history():
    """It runs before anyone is identified, to set the prompt's urgency."""
    s = score_baseline("walkway", ["helmet"], camera="d_view02")
    assert s.baseline == 5 and s.priors == 0


def test_priors_only_enter_at_the_final_score():
    base = score_baseline("walkway", ["helmet"], camera="d_view02")
    final = final_severity("walkway", ["helmet"], priors=2, camera="d_view02")
    assert final.final == base.baseline + 4
    assert final.band == "stop_work"


# --------------------------------------------------------------- history

def test_unknown_identity_is_not_zero_priors():
    """The distinction the whole escalation path rests on: 'cannot say' is not 'none'."""
    h = get_worker_history(None)
    assert h.resolved is False
    assert h.prior_violations == 0          # but resolved=False says do not trust it
    assert "must not be assumed empty" in h.note


def test_known_identity_without_a_store_is_also_unresolved():
    h = get_worker_history("W-0412", db=None)
    assert h.resolved is False


def test_history_reads_the_store_when_one_exists():
    class FakeDB:                       # mirrors EventStore.violations_for exactly
        def violations_for(self, ref, since=None, exclude_event=None):
            return [{"event_id": "a"}, {"event_id": "b"}]

    h = get_worker_history("W-0412", db=FakeDB())
    assert h.resolved is True and h.prior_violations == 2


# ------------------------------------------------------------ confidence

def test_confidence_is_separate_from_severity():
    """Two events, same missing item and zone, different evidence strength."""
    strong = {"ppe": {"missing": ["helmet"]},
              "detector": {"recall": {"helmet": 0.796}},
              "confirmation": {"frames_observed": 10, "frames_missing": 10},
              "subject": {"detection_confidence": 0.95}}
    weak = {"ppe": {"missing": ["helmet"]},
            "detector": {"recall": {"helmet": 0.796}},
            "confirmation": {"frames_observed": 10, "frames_missing": 8},
            "subject": {"detection_confidence": 0.52}}
    assert assess_confidence(strong).score > assess_confidence(weak).score


def test_weakest_missing_item_sets_the_recall():
    """A finding is only as strong as its least reliable class."""
    ev = {"ppe": {"missing": ["helmet", "goggles"]},
          "detector": {"recall": {"helmet": 0.796, "goggles": 0.723}},
          "confirmation": {"frames_observed": 10, "frames_missing": 10},
          "subject": {"detection_confidence": 0.9}}
    assert assess_confidence(ev).weakest_recall == pytest.approx(0.723)


def test_low_confidence_is_flagged_for_a_human():
    ev = {"ppe": {"missing": ["gloves"]},
          "detector": {"recall": {"gloves": 0.733}},
          "confirmation": {"frames_observed": 10, "frames_missing": 4},
          "subject": {"detection_confidence": 0.55}}
    c = assess_confidence(ev)
    assert c.is_low
    assert "human review" in c.note


def test_compliant_event_has_no_weak_evidence():
    ev = {"ppe": {"missing": []}, "detector": {"recall": {}},
          "confirmation": {"frames_observed": 10, "frames_missing": 0},
          "subject": {"detection_confidence": 0.9}}
    assert assess_confidence(ev).weakest_recall == 1.0


# ------------------------------------------------------------ event_facts

def test_agent_never_sees_the_evidence_path_or_boxes():
    """An agent handed a crop path starts describing a photograph it cannot see."""
    facts = event_facts(load("t17"))
    for leaked in ("evidence", "crop_path", "bbox", "subject"):
        assert leaked not in facts


def test_event_facts_keeps_what_the_agent_must_reason_from():
    facts = event_facts(load("t17"))
    for needed in ("zone", "missing", "indeterminate", "confirmed", "worker_ref"):
        assert needed in facts
