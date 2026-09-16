"""
The seam between Lane A and Lane C.

Every other graph test builds its event as a dict inside the test file, which proves the
agents work on an event of the shape the *test* believes in. Nothing proved they work on
the shape Lane A actually writes.

So these run the real artefacts: the event `pipeline.py` produced from real CCTV in
`events/run1/`, and the hand-built fixtures in `fixtures/events/`. If Lane A's output
drifts -- a renamed field, a moved nesting level, a bumped schema -- this is what fails,
rather than a supervisor noticing that nothing has escalated for a week.

They use a stub model, so the seam is checked without an API key.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).absolute().parent))   # for the stub LLM below

from agents.graph import run_case  # noqa: E402
from agents.guardrails import SUPPORTED_SCHEMA_MAJOR, gate_schema_understood  # noqa: E402
from agents.state import Blocker, CaseState  # noqa: E402
from agents.tools import assess_confidence, event_facts  # noqa: E402
from app.db import EventStore  # noqa: E402
from test_agents import StubCall, StubClient, _adjudicator_script  # noqa: E402

PIPELINE_EVENTS = sorted((ROOT / "events").glob("run*/events/*.json"))
FIXTURE_EVENTS = sorted((ROOT / "fixtures" / "events").glob("*.json"))

pytestmark = pytest.mark.skipif(not PIPELINE_EVENTS,
                                reason="no pipeline output in events/")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(params=PIPELINE_EVENTS, ids=lambda p: p.stem[:28])
def real_event(request):
    return load(request.param)


# ------------------------------------------------- the shape Lane A writes

def test_pipeline_output_declares_a_schema_the_agents_understand(real_event):
    assert real_event.get("schema_version", "").split(".")[0] == SUPPORTED_SCHEMA_MAJOR
    assert gate_schema_understood(CaseState(event=real_event))


def test_every_fixture_also_declares_a_supported_schema():
    """Fixtures are what most tests run on. If they drift from the pipeline's real
    output, the suite keeps passing while the system stops working."""
    for path in FIXTURE_EVENTS:
        assert gate_schema_understood(CaseState(event=load(path))), path.name


def test_the_facts_an_agent_reasons_from_are_actually_populated(real_event):
    """A missing key does not raise -- event_facts uses .get() throughout -- so a
    renamed field would silently hand the agent a blank where the zone should be."""
    facts = event_facts(real_event)
    for key in ("event_id", "captured_at", "camera_id", "zone", "zone_name",
                "required_ppe", "status"):
        assert facts[key], f"{key} came back empty from real pipeline output"


def test_confidence_can_be_computed_from_real_detector_metadata(real_event):
    """assess_confidence reads detector recall and the frame counts. If Lane A stops
    carrying them, every finding silently scores as maximally confident."""
    result = assess_confidence(real_event)
    assert 0.0 < result.score <= 1.0
    assert result.band in ("high", "moderate", "low")
    assert 0.0 < result.weakest_recall <= 1.0
    assert result.frames_observed > 0


def test_missing_items_are_covered_by_the_clause_map(real_event):
    """Lane A names PPE classes; Lane B maps them to clauses. A class renamed on one
    side and not the other produces a finding nobody can cite."""
    from agents.tools import lookup_clause
    zone = real_event.get("zone", {}).get("name") or "default"
    for item in real_event.get("ppe", {}).get("missing", []):
        rule = lookup_clause(item, zone)
        assert rule.clause_id, f"no clause covers {item!r} in {zone!r}"


def test_the_zone_on_a_real_event_has_a_severity_weight(real_event):
    """The REVIEW #1 failure, checked against real output rather than config: a zone
    that requires an item nobody weighted scores 0 and never escalates."""
    from agents.tools import score_baseline
    zone = real_event.get("zone", {})
    missing = real_event.get("ppe", {}).get("missing", [])
    if not missing:
        pytest.skip("compliant event")
    result = score_baseline(zone.get("name"), missing, real_event.get("camera_id"),
                            required=zone.get("required_ppe"))
    assert result.baseline > 0


# --------------------------------------------- the whole way through

def test_a_real_pipeline_event_runs_end_to_end(tmp_path, real_event):
    """The claim this file exists to defend: Lane A's output goes in, a recorded and
    verified decision comes out."""
    if not (real_event.get("status") == "violation"
            and real_event.get("confirmation", {}).get("confirmed")):
        pytest.skip("not an actionable finding; the entry gate correctly stops it")

    db = EventStore(tmp_path / "t.db")
    script = [[StubCall("submit_draft", {
        "summary": "A worker was without required protection; some items rest on "
                   "site policy.",
        "clause_ids": ["1926.102", "1926.100", "1926.95"]})]] \
        + _adjudicator_script("warning")

    out = run_case(real_event, db, client=StubClient(script))

    # Deliberately strict. Accepting "review" here would let this test pass in exactly
    # the situation it exists to catch: a broken seam sending every event to the review
    # queue while the pipeline looks healthy.
    assert out["outcome"] == "done", out.get("reason")
    assert out["case"].record.verified
    assert out["case"].decision is not None
    assert out["case"].draft.citations, "a real event produced no citation"
    assert db.get_event(real_event["event_id"]) is not None
    assert db.get_decision(real_event["event_id"]) is not None


def test_a_bumped_major_schema_raises_an_alert_rather_than_going_quiet(tmp_path,
                                                                      real_event):
    """The silent failure this guards. Without the gate, unfamiliar events look
    unconfirmed, route to review, and the pipeline appears healthy while escalating
    nothing."""
    db = EventStore(tmp_path / "t.db")
    future = {**real_event, "schema_version": "2.0"}
    out = run_case(future, db, client=StubClient([]))
    assert out["outcome"] == "alert"
    assert out["case"].blocked_on == Blocker.SCHEMA_UNSUPPORTED


def test_an_event_with_no_schema_version_is_refused(tmp_path, real_event):
    db = EventStore(tmp_path / "t.db")
    stripped = {k: v for k, v in real_event.items() if k != "schema_version"}
    out = run_case(stripped, db, client=StubClient([]))
    assert out["outcome"] == "alert"


# ------------------------------------------------- config agrees with itself

def test_the_clause_map_agrees_with_zones_on_load():
    """Runs the check that `clause_map()` now performs at startup, so a config edit
    that breaks the agreement fails here rather than mid-case during a demo."""
    from agents.tools import clause_map
    assert clause_map().validate() == []


# ------------------------------- drift that nobody declared

@pytest.mark.parametrize("label,mutate", [
    ("renamed confirmation field",
     lambda e: {**e, "confirmation": {"is_confirmed": True}}),
    ("re-nested zone", lambda e: {**e, "zone": {"label": "walkway"}}),
    ("detector metadata dropped",
     lambda e: {k: v for k, v in e.items() if k != "detector"}),
    ("ppe block flattened", lambda e: {**e, "ppe": {"missing": ["helmet"]}}),
])
def test_undeclared_drift_is_loud_not_quiet(tmp_path, real_event, label, mutate):
    """The failure mode this seam is built against.

    Every reader below the gate uses .get() with a default, so a renamed field does not
    raise -- it makes findings look unconfirmed and routes them to review. The pipeline
    then looks healthy while escalating nothing, which is the one failure nobody goes
    looking for. These must reach `alert`, never `review`.
    """
    db = EventStore(tmp_path / "t.db")
    out = run_case(mutate(real_event), db, client=StubClient([]))
    assert out["outcome"] == "alert", f"{label} degraded quietly to {out['outcome']}"
    assert out["case"].blocked_on == Blocker.SCHEMA_UNSUPPORTED


def test_an_empty_ppe_block_is_still_well_formed(real_event):
    """Presence, not truth. A worker wearing everything has an empty `missing` list and
    that is an ordinary event, not a broken one."""
    from agents.guardrails import gate_event_is_well_formed
    quiet = {**real_event,
             "ppe": {"present": {}, "missing": [], "indeterminate": []}}
    assert gate_event_is_well_formed(CaseState(event=quiet))


def test_every_required_path_is_present_in_real_pipeline_output(real_event):
    """If Lane A stops writing one of these, this names which one."""
    from agents.guardrails import gate_event_is_well_formed
    verdict = gate_event_is_well_formed(CaseState(event=real_event))
    assert verdict, verdict.reason


def test_every_fixture_is_well_formed_too():
    from agents.guardrails import gate_event_is_well_formed
    for path in FIXTURE_EVENTS:
        verdict = gate_event_is_well_formed(CaseState(event=load(path)))
        assert verdict, f"{path.name}: {verdict.reason}"
