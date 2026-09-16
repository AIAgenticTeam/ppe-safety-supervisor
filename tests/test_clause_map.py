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
    """Per-zone weights are what carry context, now that severity_multiplier is gone.

    Weights are keyed by camera/zone, so the camera must be passed -- a bare zone name
    falls back to the defaults and the distinction disappears.
    """
    assert (cm.score(["gloves"], "grinding_station", camera="cam_3").baseline
            > cm.score(["gloves"], "work_area", camera="d_view02").baseline)


def test_bare_zone_name_falls_back_to_default(cm):
    """Documented behaviour, not an accident: without a camera there is no way to know
    which site's 'walkway' is meant, so the defaults apply. Production always has a
    camera_id on the event, and check_against_zones() guarantees every real zone has a
    scoped entry."""
    assert cm.weights_for("grinding_station") == cm.weights_for("nonexistent_zone")


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


# --------------------------------------------------- cross-file agreement (item 1 & 2)

def test_zones_and_clauses_agree(cm):
    """zones.json and clauses.yaml each hold half of what a zone requires.

    The dangerous drift is a required item with no weight: it scores zero, the violation
    is still reported, and it never escalates -- with nothing in the output looking
    wrong. This is the test that keeps the two files honest.
    """
    problems = cm.check_against_zones()
    assert problems == [], "\n  " + "\n  ".join(problems)


def test_every_zone_has_camera_scoped_weights(cm):
    """A bare zone name is shared between cameras. Two sites can each have a 'walkway'
    with different requirements, and a bare key would silently give them one table."""
    import json
    from pathlib import Path

    raw = json.loads((ROOT / "zones.json").read_text(encoding="utf-8"))
    for cam_id, cam in raw["cameras"].items():
        for zone in cam["zones"]:
            key = f"{cam_id}/{zone['name']}"
            assert cm.weights_for(zone["name"], cam_id) == cm.weights_for(key.split("/")[1], cam_id)
            assert set(cm.weights_for(zone["name"], cam_id)) == set(zone["required_ppe"]), (
                f"{key} weights do not match its required_ppe")


def test_same_zone_name_on_different_cameras_can_differ(cm):
    """cam_3/walkway requires helmet only; d_view02/walkway also requires a vest.
    Before camera scoping they shared one weight table."""
    cam3 = cm.weights_for("walkway", "cam_3")
    dview = cm.weights_for("walkway", "d_view02")
    assert cam3 != dview, "camera scoping is not taking effect"
    assert "vest" in dview and "vest" not in cam3


def test_scoring_an_unweighted_required_item_raises(cm):
    """Fail loudly rather than score zero."""
    with pytest.raises(ValueError, match="no severity weight"):
        cm.score(["boots"], "walkway", camera="cam_3", required=["helmet", "boots"])


def test_strict_can_be_disabled_for_exploration(cm):
    """The guard is on by default; ad-hoc scoring may opt out."""
    result = cm.score(["boots"], "walkway", camera="cam_3", strict=False)
    assert result.baseline == 0            # still zero, but now it was a deliberate choice
