"""
Guardrails: constraints on OUTCOMES, not on process.

This is what lets the agents be genuinely autonomous. They may call whatever tools they
judge useful, in whatever order, and loop as long as they need -- because nothing they do
can produce a finding that fails these checks. The agent is free; the output is not.

Forcing a fixed sequence of tool calls would be the alternative, and it buys less than it
looks: an agent made to call `lookup_clause` can still write a notice that ignores what
came back. Checking the finished draft is both stricter and less constraining.

Each check returns a Verdict rather than raising, so the graph can route a blocked case
to a human instead of losing it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from agents.state import BAND_ACTION, Action, Blocker, CaseState  # noqa: E402


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str = ""
    blocker: Blocker | None = None

    def __bool__(self) -> bool:
        return self.passed


OK = Verdict(True)


# ---------------------------------------------------------------------------
# Entry gate
# ---------------------------------------------------------------------------

def gate_actionable(state: CaseState) -> Verdict:
    """Only a temporally confirmed violation may enter the agent layer.

    `status == "violation"` alone is not enough: it ignores whether the finding was
    sustained across frames, and at helmet recall 0.796 a single frame mislabels a
    compliant worker about one time in five.
    """
    event = state.event
    if event.get("status") != "violation":
        return Verdict(False, f"status is {event.get('status')!r}, not a violation")
    if not event.get("confirmation", {}).get("confirmed", False):
        return Verdict(False, "not confirmed across frames -- a single frame cannot accuse")
    return OK


def gate_no_indeterminate_accusation(state: CaseState) -> Verdict:
    """'I could not look' must never become an accusation.

    An item the vision system could not assess may not appear in a finding at all.
    """
    ppe = state.event.get("ppe", {})
    overlap = set(ppe.get("missing", [])) & set(ppe.get("indeterminate", []))
    if overlap:
        return Verdict(False, f"{sorted(overlap)} is both missing and unassessable")
    return OK


# ---------------------------------------------------------------------------
# After the Assessor
# ---------------------------------------------------------------------------

def gate_citation_present(state: CaseState) -> Verdict:
    """No citation, no decision. Slide 9's guarantee.

    The agent chose freely how to get here; it does not get to arrive without one.
    """
    if state.draft is None:
        return Verdict(False, "no draft", Blocker.NO_CITATION)
    if not state.draft.has_citation:
        return Verdict(False, "the draft cites no regulation", Blocker.NO_CITATION)
    return OK


def gate_citations_are_construction(state: CaseState) -> Verdict:
    """1910 is general industry. On a construction site it is the right topic from the
    wrong body of law, and a real-looking wrong citation is worse than none."""
    for c in state.draft.citations if state.draft else []:
        if not c.clause_id.startswith("1926."):
            return Verdict(False, f"{c.clause_id} is not part 1926", Blocker.NO_CITATION)
    return OK


def gate_site_policy_is_labelled(state: CaseState) -> Verdict:
    """A notice must never present site policy as a regulatory requirement.

    Construction has no hand-protection clause and no general hi-vis clause, so gloves
    and vests rest on the employer's general duty. Saying "required under 1926.95" would
    overstate it.
    """
    if state.draft is None:
        return Verdict(False, "no draft")
    body = (state.draft.summary or "").lower()
    for c in state.draft.citations:
        if c.is_site_policy and body and "site policy" not in body:
            return Verdict(
                False,
                f"{c.clause_id} is cited as site policy but the summary does not say so")
    return OK


# ---------------------------------------------------------------------------
# After the Adjudicator
# ---------------------------------------------------------------------------

HUMAN_REQUIRED = {Action.ESCALATION, Action.STOP_WORK}


def gate_human_approval(state: CaseState) -> Verdict:
    """Escalation and stop-work pause for a person. Always.

    The system drafts; a supervisor sends. Nothing leaves the building on a model's say-so.
    """
    d = state.decision
    if d is None:
        return Verdict(False, "no decision")
    if d.action in HUMAN_REQUIRED and not d.requires_approval:
        return Verdict(False, f"{d.action.value} must require approval",
                       Blocker.HUMAN_APPROVAL)
    return OK


def gate_low_confidence_goes_to_a_human(state: CaseState, confidence) -> Verdict:
    """High severity with weak evidence is urgent REVIEW, not a quiet downgrade.

    Severity says how bad this is if true; confidence says how sure we are. They are
    separate axes, and the wrong response to 'serious but uncertain' is to act on it.
    """
    d = state.decision
    if d is None:
        return Verdict(False, "no decision")
    if confidence.is_low and d.action in HUMAN_REQUIRED:
        return Verdict(False,
                       f"confidence {confidence.score} is low for a {d.action.value}",
                       Blocker.HUMAN_APPROVAL)
    return OK


def gate_escalation_needs_priors(state: CaseState, history) -> Verdict:
    """A repeat-offence escalation must rest on history that was actually read.

    `resolved=False` means identity was never bound, so priors are unknown -- and unknown
    must not be read as zero, nor may an escalation claim a pattern nobody confirmed.
    """
    d = state.decision
    if d is None or d.prior_violations == 0:
        return OK
    if not history.resolved:
        return Verdict(False,
                       "claims prior violations but identity was never bound",
                       Blocker.WORKER_IDENTITY)
    return OK


def gate_action_matches_the_score(state: CaseState, severity) -> Verdict:
    """An action may be gentler than the score, never harsher, without a human.

    The Adjudicator is allowed to argue a case down -- weak evidence, a zone that reads
    worse on paper than in life. Arguing one UP is different: it is the model deciding
    someone deserves more than the rubric says, and the rubric is the part that is
    auditable.

    This is the gate that catches a scoring input going astray. An agent that reads
    "2 prior violations" and then scores the case as a first offence produces exactly
    this shape -- a warning-band number under an escalation -- and the mismatch is
    visible here even when the action itself happens to be right.
    """
    d = state.decision
    if d is None:
        return Verdict(False, "no decision")
    expected = BAND_ACTION.get(getattr(severity, "band", ""), None)
    if expected is None:
        return OK
    if d.action.rank > expected.rank:
        return Verdict(False,
                       f"{d.action.value} is harsher than the score supports "
                       f"({severity.final} = {severity.band})",
                       Blocker.HUMAN_APPROVAL)
    return OK


# ---------------------------------------------------------------------------
# After the Recorder
# ---------------------------------------------------------------------------

def gate_write_verified(state: CaseState) -> Verdict:
    """A case closes only on a read-back.

    The Recorder writes the history the Adjudicator reads on the next event. An
    unconfirmed write silently turns a third offence into a first, and nothing in the
    output looks wrong.
    """
    if state.record is None:
        return Verdict(False, "nothing was recorded", Blocker.WRITE_UNCONFIRMED)
    if not state.record.verified:
        return Verdict(False, "the write was not confirmed by read-back",
                       Blocker.WRITE_UNCONFIRMED)
    return OK


# ---------------------------------------------------------------------------

ENTRY_GATES = (gate_actionable, gate_no_indeterminate_accusation)
POST_ASSESSOR = (gate_citation_present, gate_citations_are_construction,
                 gate_site_policy_is_labelled)
POST_RECORDER = (gate_write_verified,)


def run_gates(state: CaseState, gates) -> Verdict:
    """First failure wins, so the reason a case was blocked is the first real problem."""
    for gate in gates:
        verdict = gate(state)
        if not verdict:
            return verdict
    return OK
