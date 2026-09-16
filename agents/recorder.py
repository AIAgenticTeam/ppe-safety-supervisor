"""
Agent 3 · Recorder -- what goes on file.

The least agentic of the three, deliberately. Committing a row is not a judgement call,
and an LLM that might forget to call `commit_record` is a system that silently loses the
history the next escalation depends on.

So the write itself is unconditional and verified by read-back. The agent's genuine
contribution is the part that IS a judgement: **resolving identity**. Deciding whether a
supervisor-supplied id matches this finding, or whether the case should be parked as
unattributed, is exactly the call a model should not make alone and a human should not
have to make blind.

The loop this closes is the system's memory. The Recorder writes what the Adjudicator
reads on the NEXT event. An unwritten record turns a third offence into a first, and
nothing in the output looks wrong -- which is why a case does not close until the write
has been read back.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from agents.state import Blocker, CaseState, CommittedRecord  # noqa: E402


def record(state: CaseState, db, bound_by: str | None = None) -> CaseState:
    """Commit the event and its decision, then verify both by reading them back.

    Deterministic on purpose. `bound_by` is present for the case where a supervisor has
    already identified the worker in the console; without it the event is stored
    unattributed, which is correct rather than incomplete.
    """
    event = state.event
    event_id = event["event_id"]

    db.record_event(event)

    worker_ref = event.get("subject", {}).get("worker_ref")
    resolved_by = "unresolved"
    if worker_ref and bound_by:
        if db.attach_identity(event_id, worker_ref, bound_by):
            resolved_by = "supervisor"
        else:
            # A worker id that is not on the roster is refused rather than stored.
            # Free-text identity is how one person's history splits across spellings.
            state.log("recorder", "attach_identity", {"worker_ref": worker_ref},
                      "refused: not on the roster")
            state.block(Blocker.WORKER_IDENTITY)
    elif worker_ref:
        # The event carries a reference, but nobody has put their name to it. It stays
        # unattributed and counts toward nobody's history. A tracker's guess is not an
        # identification.
        state.log("recorder", "attach_identity", {"worker_ref": worker_ref},
                  "held: no supervisor has confirmed this identity")

    if state.decision is not None:
        d = state.decision
        db.record_decision(
            event_id=event_id,
            action=d.action.value,
            severity=float(d.severity.total),
            band=getattr(d.severity, "band", "") or "",
            citations=[c.clause_id for c in (state.draft.citations if state.draft else [])],
            rationale=d.rationale,
            draft_body=d.draft_body,
            confidence=d.severity.evidence_strength,
            requires_approval=d.requires_approval,
        )
        state.log("recorder", "commit_record", {"event_id": event_id},
                  f"decision {d.action.value}")

    # ---- read it back ---------------------------------------------------
    verified = db.verify(event_id)
    if state.decision is not None:
        verified = verified and db.get_decision(event_id) is not None
    state.log("recorder", "verify_record", {"event_id": event_id},
              "verified" if verified else "NOT CONFIRMED")

    if not verified:
        state.block(Blocker.WRITE_UNCONFIRMED)

    state.record = CommittedRecord(
        record_id=event_id,
        worker_ref=worker_ref,
        resolved_by=resolved_by,
        written_at=datetime.now().isoformat(timespec="seconds"),
        verified=verified,
    )
    return state
