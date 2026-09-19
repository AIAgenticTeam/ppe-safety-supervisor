"""
Agent and graph tests.

They run against a stub LLM, so the whole suite stays free, offline and deterministic.
What is being tested is not the model's prose -- it is that the guardrails hold whatever
the model does, including when it misbehaves.

The stub lets each test script exactly the tool calls an agent "chooses", which is how a
test can ask: what happens when the assessor skips the clause lookup? When the
adjudicator escalates on unknown identity? Those are the cases that matter, and a real
model would rarely produce them on demand.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from agents.adjudicator import adjudicate  # noqa: E402
from agents.assessor import assess  # noqa: E402
from agents.guardrails import (ENTRY_GATES, POST_ASSESSOR, gate_citation_present,  # noqa: E402
                               gate_citations_are_construction, gate_human_approval,
                               gate_low_confidence_goes_to_a_human, gate_write_verified,
                               run_gates)
from agents.recorder import record  # noqa: E402
from agents.state import (Action, Blocker, CaseState, Citation, Decision,  # noqa: E402
                          Draft, SeverityScore)
from agents.tools import assess_confidence  # noqa: E402
from app.db import EventStore, Worker  # noqa: E402

FIXTURES = sorted((ROOT / "fixtures" / "events").glob("*.json"))


def load(fragment: str) -> dict:
    for f in FIXTURES:
        if fragment in f.name:
            return json.loads(f.read_text(encoding="utf-8"))
    raise FileNotFoundError(fragment)


def confirmed_event(missing=("helmet",), zone="grinding_station", camera="cam_3",
                    worker=None, frames_missing=10, person_conf=0.9):
    """A minimal event that passes the entry gate."""
    return {
        "event_id": "evt_test_1", "captured_at": "2026-09-16T10:00:00+03:00",
        "camera_id": camera, "schema_version": "1.0",
        "zone": {"name": zone, "label": zone,
                 "required_ppe": ["helmet", "goggles", "gloves"]},
        "subject": {"track_id": 1, "bbox": [0, 0, 10, 10],
                    "detection_confidence": person_conf, "height_px": 400,
                    "worker_ref": worker},
        "ppe": {"present": {}, "missing": list(missing), "indeterminate": []},
        "status": "violation",
        "confirmation": {"frames_observed": 10, "frames_missing": frames_missing,
                         "confirmed": True, "rule": "8_of_10"},
        "detector": {"recall": {"helmet": 0.796, "goggles": 0.723, "gloves": 0.733}},
        "evidence": {}, "reasons": [],
    }


# ---------------------------------------------------------------- the stub

class StubMessage:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls or []
        self.content = content
        self.role = "assistant"


class StubCall:
    def __init__(self, name, args, call_id="c1"):
        self.id = call_id
        self.function = type("F", (), {"name": name,
                                       "arguments": json.dumps(args)})()


class StubClient:
    """Replays a scripted sequence of tool-call turns."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kwargs):
        turn = self.turns.pop(0) if self.turns else []
        message = StubMessage(tool_calls=turn)
        return type("R", (), {"choices": [type("Ch", (), {"message": message})()]})()


# ------------------------------------------------------------- entry gates

def test_unconfirmed_violation_never_reaches_an_agent():
    """A single frame cannot accuse. This is the gate that enforces it."""
    event = confirmed_event()
    event["confirmation"]["confirmed"] = False
    verdict = run_gates(CaseState(event=event), ENTRY_GATES)
    assert not verdict and "confirmed" in verdict.reason


def test_indeterminate_cannot_also_be_missing():
    event = confirmed_event()
    event["ppe"]["indeterminate"] = ["helmet"]
    assert not run_gates(CaseState(event=event), ENTRY_GATES)


def test_compliant_event_is_rejected():
    event = confirmed_event()
    event["status"] = "compliant"
    assert not run_gates(CaseState(event=event), ENTRY_GATES)


# ----------------------------------------------------------------- assessor

def test_assessor_citation_is_re_derived_not_trusted():
    """The model names clause ids; the map decides what they mean. It cannot smuggle
    a clause in that does not govern a missing item."""
    client = StubClient([[StubCall("submit_draft", {
        "summary": "A worker was without head protection at the grinding station.",
        "clause_ids": ["1926.100", "1910.138"],   # the second is general industry
    })]])
    state = assess(CaseState(event=confirmed_event(["helmet"])), client=client)
    ids = [c.clause_id for c in state.draft.citations]
    assert ids == ["1926.100"], f"1910.138 should never survive, got {ids}"


def test_assessor_that_submits_nothing_is_refused_not_invented():
    """An agent that stops without a draft produces no draft. The gate then refuses the
    case, rather than the system fabricating one."""
    client = StubClient([[], [], [], [], [], []])       # never calls a tool
    state = assess(CaseState(event=confirmed_event()), client=client)
    assert state.draft is None
    assert not gate_citation_present(state)


def test_assessor_skipping_lookup_produces_no_citation():
    """Autonomy is safe because the OUTPUT is checked, not the process."""
    client = StubClient([[StubCall("submit_draft",
                                   {"summary": "Something happened.", "clause_ids": []})]])
    state = assess(CaseState(event=confirmed_event()), client=client)
    verdict = run_gates(state, POST_ASSESSOR)
    assert not verdict and verdict.blocker == Blocker.NO_CITATION


def test_general_industry_citation_is_blocked():
    state = CaseState(event=confirmed_event())
    state.draft = Draft(summary="x", citations=[Citation("1910.138", "Hand protection")])
    assert not gate_citations_are_construction(state)


# -------------------------------------------------------------- adjudicator

def _adjudicator_script(action, priors=0, worker=None):
    return [
        [StubCall("get_worker_history", {"worker_ref": worker})],
        [StubCall("final_severity", {"zone": "grinding_station",
                                     "missing": ["helmet"], "priors": priors})],
        [StubCall("assess_confidence", {})],
        [StubCall("submit_decision", {"action": action, "rationale": "because",
                                      "draft_body": "note to supervisor"})],
    ]


def test_unknown_identity_blocks_the_case():
    """resolved=False means 'cannot say', never 'no priors'."""
    state = CaseState(event=confirmed_event(worker=None))
    state.draft = Draft(summary="x", citations=[Citation("1926.100", "Head protection")])
    state = adjudicate(state, db=None, client=StubClient(_adjudicator_script("warning")))
    assert state.blocked_on == Blocker.WORKER_IDENTITY
    assert state.decision.prior_violations == 0     # but the block says do not trust it


def test_escalation_always_requires_approval():
    state = CaseState(event=confirmed_event())
    state.draft = Draft(summary="x", citations=[Citation("1926.100", "Head protection")])
    state = adjudicate(state, db=None,
                       client=StubClient(_adjudicator_script("escalation")))
    assert state.decision.action == Action.ESCALATION
    assert state.decision.requires_approval
    assert gate_human_approval(state)


def test_warning_does_not_require_approval():
    state = CaseState(event=confirmed_event())
    state.draft = Draft(summary="x", citations=[Citation("1926.100", "Head protection")])
    state = adjudicate(state, db=None, client=StubClient(_adjudicator_script("warning")))
    assert not state.decision.requires_approval


def test_priors_reach_the_decision_when_identity_is_known(tmp_path):
    db = EventStore(tmp_path / "t.db")
    db.add_worker(Worker("W-1", "Someone"))
    for i in range(2):
        prior = confirmed_event(worker="W-1")
        prior["event_id"] = f"prior_{i}"
        db.record_event(prior)
        db.attach_identity(f"prior_{i}", "W-1", "supervisor:test")

    state = CaseState(event=confirmed_event(worker="W-1"))
    state.draft = Draft(summary="x", citations=[Citation("1926.100", "Head protection")])
    state = adjudicate(state, db=db,
                       client=StubClient(_adjudicator_script("escalation", priors=2,
                                                             worker="W-1")))
    assert state.decision.prior_violations == 2
    assert state.blocked_on is None


# ------------------------------------------------------------- confidence

def test_high_severity_low_confidence_goes_to_a_human():
    """Serious but uncertain is urgent review, not a quiet downgrade."""
    event = confirmed_event(["gloves"], frames_missing=4, person_conf=0.55)
    state = CaseState(event=event)
    state.decision = Decision(action=Action.ESCALATION,
                              severity=SeverityScore(base=5), requires_approval=True)
    verdict = gate_low_confidence_goes_to_a_human(state, assess_confidence(event))
    assert not verdict and verdict.blocker == Blocker.HUMAN_APPROVAL


# ---------------------------------------------------------------- recorder

def test_record_then_verify(tmp_path):
    db = EventStore(tmp_path / "t.db")
    state = CaseState(event=confirmed_event())
    state.decision = Decision(action=Action.WARNING, severity=SeverityScore(base=3))
    state = record(state, db=db)
    assert state.record.verified and state.is_complete
    assert gate_write_verified(state)


def test_unverified_write_blocks_the_case(tmp_path):
    """The memory loop's single point of failure, made loud."""
    class BrokenDB(EventStore):
        def verify(self, event_id):     # the write silently did not land
            return False

    state = CaseState(event=confirmed_event())
    state = record(state, db=BrokenDB(tmp_path / "t.db"))
    assert not state.record.verified
    assert state.blocked_on == Blocker.WRITE_UNCONFIRMED
    assert not state.is_complete


def test_identity_off_the_roster_is_refused(tmp_path):
    """Free-text identity is how one person's history splits across spellings."""
    db = EventStore(tmp_path / "t.db")
    state = CaseState(event=confirmed_event(worker="W-NOT-ON-ROSTER"))
    state = record(state, db=db, bound_by="supervisor:test")
    assert state.blocked_on == Blocker.WORKER_IDENTITY
    assert state.record.resolved_by == "unresolved"


def test_recorded_event_is_readable_by_the_next_case(tmp_path):
    """The loop that makes the system remember."""
    db = EventStore(tmp_path / "t.db")
    db.add_worker(Worker("W-1", "Someone"))
    first = confirmed_event(worker="W-1")
    record(CaseState(event=first), db=db, bound_by="supervisor:test")
    assert len(db.violations_for("W-1")) == 1


# ------------------------------------------------- what the store refuses to do

def test_an_off_roster_reference_does_not_destroy_the_finding(tmp_path):
    """The worst failure available here: a confirmed violation lost because its label
    was wrong. The event is kept; only the attribution is withheld."""
    db = EventStore(tmp_path / "t.db")
    db.record_event(confirmed_event(worker="W-GHOST"))
    stored = db.get_event("evt_test_1")
    assert stored is not None
    assert stored["subject"]["worker_ref"] == "W-GHOST"      # preserved for a human
    assert db.violations_for("W-GHOST") == []                # but attributed to nobody


def test_a_worker_ref_alone_never_attributes(tmp_path):
    """Even an id that IS on the roster must not bind itself. An attribution with no
    bound_by is an accusation with no author."""
    db = EventStore(tmp_path / "t.db")
    db.add_worker(Worker("W-1", "Someone"))
    db.record_event(confirmed_event(worker="W-1"))
    assert db.violations_for("W-1") == []

    db.attach_identity("evt_test_1", "W-1", "supervisor:test")
    assert len(db.violations_for("W-1")) == 1


def test_re_recording_does_not_erase_a_supervisors_binding(tmp_path):
    """Re-running a video is routine. Losing a supervisor's work to it would not be."""
    db = EventStore(tmp_path / "t.db")
    db.add_worker(Worker("W-1", "Someone"))
    event = confirmed_event(worker="W-1")
    db.record_event(event)
    db.attach_identity("evt_test_1", "W-1", "supervisor:test")

    db.record_event(event)                                   # the pipeline runs again
    assert len(db.violations_for("W-1")) == 1


# ---------------------------------------------------------------- the graph
#
# Routing is the whole safety story: what happens to a case the agents could not
# handle. A case must never quietly disappear, and it must never quietly proceed.

from agents.graph import run_case  # noqa: E402


def _full_script(action="warning", clause_ids=("1926.100",), priors=0, worker=None):
    """One assessor turn, then the adjudicator's four."""
    return [[StubCall("submit_draft", {"summary": "A worker was without head protection.",
                                       "clause_ids": list(clause_ids)})]] \
        + _adjudicator_script(action, priors, worker)


def test_happy_path_runs_to_done(tmp_path):
    db = EventStore(tmp_path / "t.db")
    out = run_case(confirmed_event(), db, client=StubClient(_full_script()))
    assert out["outcome"] == "done"
    assert out["case"].decision.action == Action.WARNING
    assert out["case"].record.verified


def test_unconfirmed_event_is_routed_to_review_without_calling_a_model(tmp_path):
    """No LLM turns are scripted at all -- if the graph reached an agent, this fails."""
    db = EventStore(tmp_path / "t.db")
    event = confirmed_event()
    event["confirmation"]["confirmed"] = False
    out = run_case(event, db, client=StubClient([]))
    assert out["outcome"] == "review"
    assert out["case"].draft is None


def test_uncitable_draft_is_parked_not_sent(tmp_path):
    db = EventStore(tmp_path / "t.db")
    out = run_case(confirmed_event(), db,
                   client=StubClient(_full_script(clause_ids=[])))
    assert out["outcome"] == "parked"
    assert out["case"].blocked_on == Blocker.NO_CITATION
    assert out["case"].decision is None          # it never reached the adjudicator


def test_a_write_that_did_not_land_raises_an_alert(tmp_path):
    """The memory loop failing silently is the one failure nobody would notice."""
    class BrokenDB(EventStore):
        def verify(self, event_id):
            return False

    out = run_case(confirmed_event(), BrokenDB(tmp_path / "t.db"),
                   client=StubClient(_full_script()))
    assert out["outcome"] == "alert"
    assert not out["case"].is_complete


def test_escalation_reaching_done_still_awaits_a_human(tmp_path):
    """'done' means the case is recorded, not that anything was sent."""
    db = EventStore(tmp_path / "t.db")
    out = run_case(confirmed_event(), db,
                   client=StubClient(_full_script("escalation")))
    assert out["outcome"] == "done"
    assert out["case"].decision.requires_approval
    assert db.get_decision("evt_test_1")["requires_approval"] == 1


def test_a_model_that_forgets_approval_has_it_forced_on(tmp_path):
    """A drafting slip must not become a sent escalation. The graph corrects it rather
    than losing the case."""
    db = EventStore(tmp_path / "t.db")
    out = run_case(confirmed_event(), db,
                   client=StubClient(_full_script("stop_work")))
    assert out["case"].decision.requires_approval
    assert out["outcome"] == "done"


def test_the_trace_records_every_tool_each_agent_chose(tmp_path):
    """The demo's evidence that the agents are choosing, not following a script."""
    db = EventStore(tmp_path / "t.db")
    out = run_case(confirmed_event(), db, client=StubClient(_full_script()))
    actors = {entry.agent for entry in out["case"].trace}
    assert {"graph", "assessor", "adjudicator", "recorder"} <= actors


# ------------------------------------------- the score is not the model's to supply

def test_a_model_supplied_prior_count_is_ignored(tmp_path):
    """The bug this was written for: the adjudicator read "prior_violations: 2" and
    then asked for a score with priors=0, producing a warning-band number underneath
    an escalation. The tool now reads the confirmed count itself."""
    db = EventStore(tmp_path / "t.db")
    db.add_worker(Worker("W-1", "Someone"))
    for i in range(2):
        prior = confirmed_event(worker="W-1")
        prior["event_id"] = f"prior_{i}"
        db.record_event(prior)
        db.attach_identity(f"prior_{i}", "W-1", "supervisor:test")

    state = CaseState(event=confirmed_event(worker="W-1"))
    state.draft = Draft(summary="x", citations=[Citation("1926.100", "Head protection")])
    # The script asks for priors=0 -- exactly what the real model did.
    state = adjudicate(state, db=db,
                       client=StubClient(_adjudicator_script("warning", priors=0,
                                                             worker="W-1")))
    assert state.decision.severity.total == 9, "the repeat penalty must be in the score"
    assert state.decision.prior_violations == 2


def test_scoring_before_looking_anyone_up_still_counts_priors(tmp_path):
    """Tool order is the agent's to choose, so the score must not depend on it."""
    db = EventStore(tmp_path / "t.db")
    db.add_worker(Worker("W-1", "Someone"))
    prior = confirmed_event(worker="W-1")
    prior["event_id"] = "prior_0"
    db.record_event(prior)
    db.attach_identity("prior_0", "W-1", "supervisor:test")

    state = CaseState(event=confirmed_event(worker="W-1"))
    state.draft = Draft(summary="x", citations=[Citation("1926.100", "Head protection")])
    state = adjudicate(state, db=db, client=StubClient([
        [StubCall("final_severity", {})],          # scored first, no history yet
        [StubCall("submit_decision", {"action": "escalation", "rationale": "r",
                                      "draft_body": "b"})],
    ]))
    assert state.decision.severity.total == 7      # 5 baseline + one prior


def test_log_only_is_a_real_action(tmp_path):
    """It is in the tool schema and it is the default, so an Action that cannot
    represent it crashes the graph on an ordinary decision."""
    db = EventStore(tmp_path / "t.db")
    out = run_case(confirmed_event(), db,
                   client=StubClient(_full_script("log_only")))
    assert out["outcome"] == "done"
    assert out["case"].decision.action == Action.LOG_ONLY
    assert not out["case"].decision.requires_approval


def test_an_action_harsher_than_the_score_needs_a_human(tmp_path):
    """Arguing a case down is the agent's to do. Arguing one up is a human's."""
    db = EventStore(tmp_path / "t.db")
    event = confirmed_event(["gloves"], zone="walkway", camera="d_view02_test")
    out = run_case(event, db, client=StubClient(
        [[StubCall("submit_draft", {
            "summary": "Gloves were absent; hand protection rests on site policy.",
            "clause_ids": ["1926.95"]})]]
        + _adjudicator_script("stop_work")))
    assert out["case"].decision.requires_approval
    assert any(s.tool == "gate_action_matches_the_score" for s in out["case"].trace)


def test_a_gentler_action_than_the_score_passes_freely(tmp_path):
    """The departure the agent is allowed to make on its own."""
    db = EventStore(tmp_path / "t.db")
    out = run_case(confirmed_event(), db,
                   client=StubClient(_full_script("log_only")))
    assert not any(s.tool == "gate_action_matches_the_score" for s in out["case"].trace)


# ------------------------------------------------ when the model is not there
#
# An unguarded chat.completions.create raises through LangGraph and takes the run down.
# On a demo night that is a traceback instead of a system. These check it degrades.

from agents import llm  # noqa: E402
from agents.guardrails import gate_no_unsupported_pattern_claim  # noqa: E402
from agents.llm import ModelUnavailable, complete  # noqa: E402


class RateLimit(Exception):
    pass


class AuthenticationError(Exception):
    pass


class FlakyClient:
    """Fails `failures` times, then behaves."""

    def __init__(self, failures, exc=RateLimit, turns=None):
        self.failures, self.exc, self.calls = failures, exc, 0
        self.turns = list(turns or [])
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc("upstream said no")
        turn = self.turns.pop(0) if self.turns else []
        message = StubMessage(tool_calls=turn)
        return type("R", (), {"choices": [type("Ch", (), {"message": message})()]})()


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    monkeypatch.setattr(llm, "BASE_DELAY", 0)


def test_a_transient_failure_is_retried():
    client = FlakyClient(failures=2)
    complete(client, model="m", messages=[])
    assert client.calls == 3


def test_retries_are_not_infinite():
    client = FlakyClient(failures=99)
    with pytest.raises(ModelUnavailable):
        complete(client, model="m", messages=[])
    assert client.calls == llm.ATTEMPTS


def test_a_bad_key_is_not_retried():
    """It will still be wrong in two seconds, and each attempt costs a call."""
    client = FlakyClient(failures=99, exc=AuthenticationError)
    with pytest.raises(ModelUnavailable):
        complete(client, model="m", messages=[])
    assert client.calls == 1


def test_an_unreachable_model_parks_the_case_and_keeps_the_finding(tmp_path):
    """Degraded, not lost: no decision, but the event is on file and a human sees it."""
    db = EventStore(tmp_path / "t.db")
    out = run_case(confirmed_event(), db, client=FlakyClient(failures=99))
    assert out["outcome"] == "parked"
    assert out["case"].blocked_on == Blocker.MODEL_UNAVAILABLE
    assert out["case"].decision is None
    assert db.get_event("evt_test_1") is not None, "the finding must survive an outage"


def test_the_adjudicator_going_dark_still_keeps_the_finding(tmp_path):
    """The assessor succeeds, then the model dies before a decision."""
    db = EventStore(tmp_path / "t.db")
    client = FlakyClient(failures=0, turns=[
        [StubCall("submit_draft", {"summary": "s", "clause_ids": ["1926.100"]})]])
    client.failures, client.calls = 99, -1      # first call works, the rest do not
    out = run_case(confirmed_event(), db, client=client)
    assert out["outcome"] == "parked"
    assert db.get_event("evt_test_1") is not None


# ------------------------------------- a pattern the record does not support

def _decision_saying(text, priors=0):
    state = CaseState(event=confirmed_event())
    state.decision = Decision(action=Action.WARNING, severity=SeverityScore(base=3),
                              prior_violations=priors, draft_body=text, rationale="")
    return state


def test_alleging_a_third_violation_with_nothing_on_record_is_blocked():
    """The one harm the model can do with prose alone. A supervisor reading this
    would act on a pattern that does not exist."""
    verdict = gate_no_unsupported_pattern_claim(
        _decision_saying("This is the worker's third violation this month."))
    assert not verdict and verdict.blocker == Blocker.WORKER_IDENTITY


@pytest.mark.parametrize("claim", [
    "The worker has prior violations on file.",
    "This is a repeated failure to wear head protection.",
    "The worker was again without a helmet.",
    "This shows a pattern of non-compliance.",
    "There is a history of similar findings.",
    "The worker continues to disregard PPE requirements.",
])
def test_pattern_language_is_caught_in_its_usual_forms(claim):
    assert not gate_no_unsupported_pattern_claim(_decision_saying(claim))


def test_the_same_claim_passes_when_the_record_supports_it(tmp_path):
    verdict = gate_no_unsupported_pattern_claim(
        _decision_saying("This is the worker's third violation.", priors=2))
    assert verdict


def test_a_plain_single_incident_notice_passes():
    assert gate_no_unsupported_pattern_claim(
        _decision_saying("A worker was without head protection at the grinding "
                         "station. Please address this."))


def test_priority_does_not_trip_the_prior_check():
    """Word boundaries matter: 'priority' is not 'prior violation'."""
    assert gate_no_unsupported_pattern_claim(
        _decision_saying("This is a high priority finding requiring attention."))


def test_an_unsupported_claim_reaching_the_graph_demands_a_human(tmp_path):
    db = EventStore(tmp_path / "t.db")
    script = _full_script()
    script[-1] = [StubCall("submit_decision", {
        "action": "warning", "rationale": "r",
        "draft_body": "The worker has prior violations and continues to offend."})]
    out = run_case(confirmed_event(), db, client=StubClient(script))
    assert out["case"].decision.requires_approval
    assert out["case"].blocked_on == Blocker.WORKER_IDENTITY


# --------------------------------------- site policy must not read as regulation

from agents.guardrails import gate_site_policy_is_labelled  # noqa: E402


def test_gloves_presented_as_a_regulation_are_refused():
    """Construction has no hand-protection clause. Gloves rest on the employer's
    general duty, and saying otherwise overstates what the law requires."""
    state = CaseState(event=confirmed_event(["gloves"]))
    state.draft = Draft(summary="The worker was not wearing gloves, required under "
                                "29 CFR 1926.95.",
                        citations=[Citation("1926.95", "General requirements",
                                            is_site_policy=True)])
    assert not gate_site_policy_is_labelled(state)


def test_the_same_finding_passes_when_it_says_site_policy():
    state = CaseState(event=confirmed_event(["gloves"]))
    state.draft = Draft(summary="The worker was not wearing gloves; hand protection "
                                "here rests on site policy under 29 CFR 1926.95.",
                        citations=[Citation("1926.95", "General requirements",
                                            is_site_policy=True)])
    assert gate_site_policy_is_labelled(state)


def test_a_genuine_regulation_needs_no_such_label():
    state = CaseState(event=confirmed_event(["helmet"]))
    state.draft = Draft(summary="The worker was without head protection.",
                        citations=[Citation("1926.100", "Head protection",
                                            is_site_policy=False)])
    assert gate_site_policy_is_labelled(state)


def test_the_zombie_multiplier_never_reaches_an_agent():
    """`severity_multiplier` still rides along in the event schema (REVIEW #6) even
    though the design dropped it as a double count. An agent handed it would
    reasonably apply it on top of the zone weights that already encode the same thing.

    event_facts is what keeps that contained, so this asserts the containment rather
    than trusting it."""
    from agents.tools import event_facts
    event = confirmed_event()
    event["zone"]["severity_multiplier"] = 2.5
    facts = event_facts(event)
    assert "severity_multiplier" not in json.dumps(facts)


def test_a_missing_api_key_parks_the_case_instead_of_raising(monkeypatch, tmp_path):
    """OpenAI() raises in its CONSTRUCTOR when there is no key, before complete() is
    ever reached — so the ModelUnavailable handling around the call did not cover it.

    It surfaced as an HTTP 500 from POST /events that said nothing useful. A missing
    credential is an unreachable model like any other: keep the finding, park the case,
    tell a human why.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "")

    state = assess(CaseState(event=confirmed_event()), client=None)
    assert state.draft is None
    assert state.blocked_on == Blocker.MODEL_UNAVAILABLE

    db = EventStore(tmp_path / "t.db")
    out = run_case(confirmed_event(), db, client=None)
    assert out["outcome"] == "parked"
    assert "model" in out["reason"].lower()
    assert db.get_event("evt_test_1") is not None, "the finding must survive"


def test_an_unconfirmed_finding_is_kept_not_dropped(tmp_path):
    """Intake rejection was the only terminal path that returned without writing.

    Found by replaying a whole night's concrete pour: every event came back `review`,
    the API answered 201 Created each time, and the database stayed empty. Refusing to
    let a finding become an accusation is not the same as refusing to remember it — the
    review queue is precisely where a human decides what the system could not, and it
    cannot decide about a record that was never written.
    """
    db = EventStore(tmp_path / "t.db")
    event = confirmed_event()
    event["confirmation"]["confirmed"] = False

    out = run_case(event, db, client=StubClient([]))

    assert out["outcome"] == "review"
    assert db.get_event("evt_test_1") is not None, "the finding must survive the refusal"
    assert db.get_decision("evt_test_1") is None, "but it must not acquire a decision"
    assert [e["event_id"] for e in db.unattributed()] == ["evt_test_1"]


def test_a_compliant_finding_is_kept_too(tmp_path):
    """Compliant observations are the denominator every rate is measured against, and
    the drift monitor reads them. Dropping them silently skews both."""
    db = EventStore(tmp_path / "t.db")
    event = confirmed_event()
    event["status"] = "compliant"
    event["ppe"]["missing"] = []

    out = run_case(event, db, client=StubClient([]))
    assert out["outcome"] == "review"
    assert db.get_event("evt_test_1") is not None


def test_every_terminal_outcome_leaves_the_finding_on_file(tmp_path):
    """The invariant behind all of it: whatever happens, the event is recorded."""
    cases = {
        "review":  lambda e: e["confirmation"].update(confirmed=False) or e,
        "alert":   lambda e: {**e, "schema_version": "2.0"},
        "parked":  lambda e: e,          # stub model says nothing -> no draft
    }
    for name, mutate in cases.items():
        db = EventStore(tmp_path / f"{name}.db")
        out = run_case(mutate(confirmed_event()), db, client=StubClient([]))
        assert db.get_event("evt_test_1") is not None, (
            f"a {out['outcome']!r} outcome dropped the finding")
