"""
The LangGraph state machine.

    ViolationEvent
         |
      [gate: actionable?] --no--> review queue
         |
     ASSESSOR            agent chooses its own tools
         |
      [gate: citation? construction? site policy labelled?] --no--> parked
         |
     ADJUDICATOR         agent chooses its own tools
         |
      [gate: approval required? confidence adequate?]
         |
     RECORDER            deterministic write, verified by read-back
         |
      [gate: verified?] --no--> alert
         |
       done

Guardrails sit on the EDGES, not inside the agents. That is what lets each agent call
whatever tools it judges useful: nothing it does can produce a finding that fails a gate,
so it does not need to be marched through a fixed sequence.

    from agents.graph import build_graph
    graph = build_graph(db)
    final = graph.invoke({"case": CaseState(event=event)})
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, TypedDict

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from agents.adjudicator import adjudicate  # noqa: E402
from agents.assessor import assess  # noqa: E402
from agents.guardrails import (ENTRY_GATES, POST_ASSESSOR, POST_RECORDER,  # noqa: E402
                               gate_action_matches_the_score, gate_human_approval,
                               gate_event_is_well_formed,
                               gate_schema_understood,
                               gate_no_unsupported_pattern_claim,
                               gate_low_confidence_goes_to_a_human, run_gates)
from agents.recorder import record  # noqa: E402
from agents.state import Blocker, CaseState  # noqa: E402
from agents.tools import assess_confidence, final_severity  # noqa: E402


def final_severity_for(case):
    """Re-score the case from the event alone, to compare against what was decided."""
    try:
        event = case.event
        return final_severity(
            event.get("zone", {}).get("name") or "default",
            event.get("ppe", {}).get("missing", []),
            case.decision.prior_violations if case.decision else 0,
            event.get("camera_id"),
            required=event.get("zone", {}).get("required_ppe"))
    except Exception:
        return None


class GraphState(TypedDict, total=False):
    case: CaseState
    outcome: str            # "done" | "review" | "parked" | "alert"
    reason: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _park(gs: GraphState, db, reason: str, outcome: str = "parked") -> GraphState:
    """End a case the agents could not finish -- but keep the finding.

    A parked case never reaches the Recorder, so without this the event exists only in
    whatever JSON Lane A wrote. A model outage would then mean confirmed violations
    leave no trace in the store at all, which is the difference between a degraded
    system and a lossy one. The write is best-effort: if the database is the thing
    that is broken, parking is still the right outcome.
    """
    case = gs["case"]
    try:
        db.record_event(case.event)
        case.log("graph", "park", {}, "event kept, no decision")
        # A parked case is the one somebody will actually want to explain, so its trace
        # matters more than a clean run's, not less.
        db.record_trace(case.event["event_id"], case.trace)
    except Exception as exc:                    # noqa: BLE001
        case.log("graph", "park", {}, f"event NOT kept: {exc}")
    return {**gs, "case": case, "outcome": outcome, "reason": reason}


def make_node_intake(db):
    def node_intake(gs: GraphState) -> GraphState:
        """Only a temporally confirmed violation enters the agent layer."""
        case = gs["case"]

        # Before anything reads a field: do we understand this event at all? An
        # unrecognised schema is a system fault, not a safety judgement, so it raises
        # an alert rather than joining the review queue where a human would read it
        # as an ordinary uncertain finding.
        for check in (gate_schema_understood, gate_event_is_well_formed):
            understood = check(case)
            if not understood:
                case.block(Blocker.SCHEMA_UNSUPPORTED)
                case.log("graph", "intake", {}, understood.reason)
                return _park({**gs, "case": case}, db, understood.reason,
                             outcome="alert")

        # A decision a human has signed is final. Re-judging it -- a replay, a second
        # run of the demo, a request from another website -- used to overwrite the
        # decision and erase the signature. It is now refused before any model is
        # called, and the event is left exactly as it was when it was approved.
        if db is not None:
            signed = db.get_decision(case.event.get("event_id"))
            if signed and signed.get("approved_by"):
                reason = (f"already decided and approved by {signed['approved_by']}; "
                          f"a signed decision is not re-judged")
                case.log("graph", "intake", {}, reason)
                return {**gs, "case": case, "outcome": "done", "reason": reason}

        verdict = run_gates(case, ENTRY_GATES)
        if not verdict:
            case.log("graph", "intake", {}, f"rejected: {verdict.reason}")
            # Keep it. This is the only terminal path that used to return without
            # writing, and it is the most common one on hard footage -- a whole night
            # concrete pour produced nothing but `review`, every event was dropped, and
            # the API still answered 201 Created. Refusing to let a finding become an
            # accusation is not the same as refusing to remember it: the review queue
            # is exactly where a human decides what the system could not.
            return _park({**gs, "case": case}, db, verdict.reason, outcome="review")
        case.log("graph", "intake", {}, "accepted")
        return {**gs, "outcome": ""}
    return node_intake


def _agent_failed(gs: GraphState, db, agent: str, exc: Exception) -> GraphState:
    """Last line of defence: an agent raised something nobody anticipated.

    Every other terminal path keeps the finding, and this one has to as well. A model
    returning truncated JSON once crashed the whole case, the API answered 500, and the
    event was never written -- a confused model lost data a dead model would not have.
    The agents now handle malformed output themselves; this catches whatever they don't.
    """
    case = gs["case"]
    case.log("graph", agent, {}, f"failed: {type(exc).__name__}: {exc}")
    case.block(Blocker.AGENT_ERROR)
    return _park({**gs, "case": case}, db,
                 f"the {agent} failed unexpectedly ({type(exc).__name__}); "
                 f"the finding is kept for a human")


def make_node_assess(db, client, model):
    def node_assess(gs: GraphState) -> GraphState:
        try:
            case = assess(gs["case"], client=client, model=model)
        except Exception as exc:                    # noqa: BLE001
            return _agent_failed(gs, db, "assessor", exc)
        if case.blocked_on is Blocker.MODEL_UNAVAILABLE:
            return _park({**gs, "case": case}, db, "the model could not be reached")
        verdict = run_gates(case, POST_ASSESSOR)
        if not verdict:
            if verdict.blocker:
                case.block(verdict.blocker)
            return _park({**gs, "case": case}, db, verdict.reason)
        return {**gs, "case": case, "outcome": ""}
    return node_assess


def make_node_adjudicate(db, client, model):
    def node_adjudicate(gs: GraphState) -> GraphState:
        try:
            case = adjudicate(gs["case"], db=db, client=client, model=model)
        except Exception as exc:                    # noqa: BLE001
            return _agent_failed(gs, db, "adjudicator", exc)
        if case.decision is None:
            reason = ("the model could not be reached"
                      if case.blocked_on is Blocker.MODEL_UNAVAILABLE
                      else "the adjudicator reached no decision")
            return _park({**gs, "case": case}, db, reason)

        approval = gate_human_approval(case)
        if not approval:
            # Force it rather than refuse: an escalation that forgot to ask for
            # approval is a drafting slip, not grounds to lose the case.
            case.decision.requires_approval = True
            case.log("graph", "gate_human_approval", {}, "forced approval on")

        confidence = assess_confidence(case.event)
        weak = gate_low_confidence_goes_to_a_human(case, confidence)
        if not weak:
            case.block(Blocker.HUMAN_APPROVAL)
            case.decision.requires_approval = True
            case.log("graph", "gate_low_confidence", {"score": confidence.score},
                     weak.reason)

        # A notice alleging a pattern the record does not show is the one harm the
        # model can do with prose alone. Nothing downstream checks the draft text.
        supported = gate_no_unsupported_pattern_claim(case)
        if not supported:
            case.block(Blocker.WORKER_IDENTITY)
            case.decision.requires_approval = True
            case.log("graph", "gate_no_unsupported_pattern_claim", {}, supported.reason)

        # An action harsher than the rubric supports is not refused -- it may well be
        # the right call -- but it never goes out without someone signing for it.
        severity = final_severity_for(case)
        if severity is not None:
            fits = gate_action_matches_the_score(case, severity)
            if not fits:
                case.block(Blocker.HUMAN_APPROVAL)
                case.decision.requires_approval = True
                case.log("graph", "gate_action_matches_the_score",
                         {"band": severity.band}, fits.reason)

        # Only now, after every gate has read the model's own words: state the record
        # in words the model did not write. The pattern gate is a word list, and a word
        # list can be paraphrased past. A supervisor who reads "habitually ignores PPE"
        # next to "Record: identity not confirmed -- prior violations unknown" is not
        # misled, whatever the model managed to phrase.
        line = record_line(case.decision)
        case.decision.draft_body = "\n\n".join(p for p in (case.decision.draft_body, line) if p)
        case.log("graph", "record_line", {}, line)
        return {**gs, "case": case, "outcome": ""}
    return node_adjudicate


def record_line(decision) -> str:
    """What the history lookup established, from the lookup rather than the prose."""
    if decision.history_resolved is None:
        return "Record: the worker's history was not checked for this notice."
    if not decision.history_resolved:
        return ("Record: identity not confirmed. Prior violations are unknown, "
                "not zero.")
    n, days = decision.prior_violations, decision.history_window_days
    if n == 0:
        return f"Record: no confirmed prior violations in the last {days} days."
    return (f"Record: {n} confirmed prior violation{'s' if n != 1 else ''} "
            f"in the last {days} days.")


def make_node_record(db, bound_by):
    def node_record(gs: GraphState) -> GraphState:
        try:
            case = record(gs["case"], db=db, bound_by=bound_by)
        except Exception as exc:                    # noqa: BLE001
            # The write itself failed, so there is nowhere left to keep the finding.
            # That is an alert -- a human must look at the system, not the case.
            case = gs["case"]
            case.log("graph", "recorder", {}, f"failed: {type(exc).__name__}: {exc}")
            case.block(Blocker.WRITE_UNCONFIRMED)
            return {**gs, "case": case, "outcome": "alert",
                    "reason": f"the write failed ({type(exc).__name__}): {exc}"}
        verdict = run_gates(case, POST_RECORDER)
        if not verdict:
            return {**gs, "case": case, "outcome": "alert", "reason": verdict.reason}
        return {**gs, "case": case, "outcome": "done"}
    return node_record


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

def _route(gs: GraphState) -> str:
    """A node that set an outcome has ended the run; otherwise continue."""
    return "stop" if gs.get("outcome") else "continue"


def build_graph(db, client=None, model: str = "gpt-4o-mini",
                bound_by: str | None = None):
    """Compile the three-agent graph. `db` is an app.db.EventStore."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(GraphState)
    g.add_node("intake", make_node_intake(db))
    g.add_node("assess", make_node_assess(db, client, model))
    g.add_node("adjudicate", make_node_adjudicate(db, client, model))
    g.add_node("record", make_node_record(db, bound_by))

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", _route, {"continue": "assess", "stop": END})
    g.add_conditional_edges("assess", _route, {"continue": "adjudicate", "stop": END})
    g.add_conditional_edges("adjudicate", _route, {"continue": "record", "stop": END})
    g.add_edge("record", END)
    return g.compile()


def run_case(event: dict, db, client=None, model: str = "gpt-4o-mini",
             bound_by: str | None = None) -> GraphState:
    """Convenience wrapper: one event in, the finished GraphState out."""
    graph = build_graph(db, client=client, model=model, bound_by=bound_by)
    return graph.invoke({"case": CaseState(event=event), "outcome": "", "reason": ""})
