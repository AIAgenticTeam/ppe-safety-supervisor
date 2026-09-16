"""
Clause map and severity tests.

These protect the two things Lane C depends on: that a citation is correct and that a
severity score is reproducible. No model, no index, no network.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from kb.clause_map import ClauseMap  # noqa: E402


@pytest.fixture(scope="module")
def cm():
    return ClauseMap.load()


# ------------------------------------------------------------------ integrity

def test_clause_map_is_self_consistent(cm):
    assert cm.validate() == [], "run: python kb/clause_map.py"


def test_every_detected_class_has_a_rule(cm):
    """If the detector can report it missing, the map must be able to cite it."""
    detected = {"helmet", "gloves", "vest", "boots", "goggles"}
    assert detected <= set(cm.known_items), (
        f"no clause rule for {detected - set(cm.known_items)}")


# ---------------------------------------------------------------- citations

@pytest.mark.parametrize("item,clause", [
    ("helmet", "1926.100"),
    ("goggles", "1926.102"),
    ("boots", "1926.96"),
])
def test_specific_clauses(cm, item, clause):
    rule = cm.for_item(item)
    assert rule.clause.clause_id == clause
    assert rule.is_regulatory


def test_scope_is_construction_not_general_industry(cm):
    """1910 is general industry. Citing it on a construction site is the right topic
    from the wrong body of law."""
    for cid in cm.clauses:
        assert cid.startswith("1926."), f"{cid} is not part 1926"


def test_gloves_do_not_cite_the_general_industry_clause(cm):
    """Construction has no equivalent of 1910.138. Citing it would be wrong law."""
    rule = cm.for_item("gloves")
    assert rule.clause.clause_id != "1910.138"
    assert rule.clause.clause_id == "1926.95"
    assert not rule.is_regulatory, "gloves are site policy in construction, not regulation"


def test_site_policy_items_say_so_in_the_notice(cm):
    """A notice must never present site policy as a regulatory requirement."""
    for item in cm.known_items:
        rule = cm.for_item(item)
        if not rule.is_regulatory:
            assert "site policy" in rule.notice_phrase.lower(), (
                f"{item} is site policy but its notice phrase reads as regulation")


def test_vest_is_site_policy_generally(cm):
    """1926.201 covers flaggers only; it must not be cited for general site hi-vis."""
    rule = cm.for_item("vest", "walkway")
    assert rule.clause.clause_id == "1926.95"
    assert not rule.is_regulatory


def test_vest_becomes_regulatory_for_flaggers(cm):
    """The loading dock override is the one place hi-vis is squarely regulated."""
    rule = cm.for_item("vest", "loading_dock")
    assert rule.clause.clause_id == "1926.201"
    assert rule.is_regulatory
    assert rule.override_reason, "an override without a reason is unexplainable"


def test_unknown_item_raises_rather_than_guessing(cm):
    with pytest.raises(KeyError):
        cm.for_item("respirator")


# ----------------------------------------------------------------- severity

def test_score_sums_missing_not_remainder(cm):
    """The bug this design exists to avoid.

    Zone-total-minus-missing makes an absolute threshold mean something different in
    every zone: the walkway totals 7, below a threshold of 9, so a fully compliant
    worker would be reported severe.
    """
    compliant = cm.score([], "walkway")
    assert compliant.baseline == 0
    assert compliant.band == "compliant"


def test_missing_helmet_outweighs_missing_vest(cm):
    assert cm.score(["helmet"], "walkway").baseline > cm.score(["vest"], "walkway").baseline


def test_zone_weights_differ(cm):
    """Per-zone weights are what carry context, now that severity_multiplier is gone."""
    assert (cm.score(["gloves"], "grinding_station").baseline
            > cm.score(["gloves"], "work_area").baseline)


def test_priors_escalate(cm):
    """Demo beat 4: third time this week reaches stop-work by arithmetic."""
    first = cm.score(["helmet"], "grinding_station", priors=0)
    third = cm.score(["helmet"], "grinding_station", priors=2)
    assert first.band == "escalation"
    assert third.band == "stop_work"
    assert third.final == first.final + 4


@pytest.mark.parametrize("score,band", [
    (0, "compliant"), (1, "log_only"), (2, "log_only"),
    (3, "warning"), (4, "warning"),
    (5, "escalation"), (7, "escalation"),
    (8, "stop_work"), (99, "stop_work"),
])
def test_bands_tile_the_range(cm, score, band):
    assert cm.band_for(score) == band


def test_score_is_reproducible(cm):
    """Deterministic scoring is what makes Macro-F1 against human labels computable."""
    a = cm.score(["helmet", "vest"], "walkway", priors=1)
    b = cm.score(["helmet", "vest"], "walkway", priors=1)
    assert a == b


def test_explain_is_human_readable(cm):
    text = cm.score(["helmet", "vest"], "walkway", priors=2).explain()
    assert "helmet 5" in text and "vest 2" in text and "->" in text
