"""
The object that flows through the agent graph.

One CaseState per ViolationEvent. Each agent appends to it; nothing overwrites `event`,
because the finding is evidence and evidence does not change after the fact.

Lane C builds the graph around this. Lane D builds the SQLite schema from it. Both code
against these types rather than against a description, so a field rename breaks a test
instead of a demo.

See docs/AGENT_WORKFLOW.md for the graph and the reasoning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1.0"


class Action(str, Enum):
    """What the Adjudicator recommends. Nothing here is ever executed automatically."""

    NO_ACTION = "no_action"          # nothing issued, nothing recorded as a finding
    LOG_ONLY = "log_only"            # recorded, no notice raised
    WARNING = "warning"              # draft note for the supervisor to have a word
    ESCALATION = "escalation"        # formal; the supervisor must respond
    STOP_WORK = "stop_work"          # immediate halt; always needs human approval

    @property
    def rank(self) -> int:
        """How severe, as an ordering. Lets a guardrail ask whether a chosen action
        outruns the score that was supposed to justify it."""
        return ORDER.index(self)


class Blocker(str, Enum):
    HUMAN_APPROVAL = "human_approval"
    WORKER_IDENTITY = "worker_identity"
    NO_CITATION = "no_citation"
    WRITE_UNCONFIRMED = "write_unconfirmed"


ORDER = (Action.NO_ACTION, Action.LOG_ONLY, Action.WARNING,
         Action.ESCALATION, Action.STOP_WORK)

BAND_ACTION = {"compliant": Action.NO_ACTION, "log_only": Action.LOG_ONLY,
               "warning": Action.WARNING, "escalation": Action.ESCALATION,
               "stop_work": Action.STOP_WORK}


@dataclass
class Step:
    """One tool call, for the observability work in week 6."""

    agent: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    latency_ms: int = 0
    at: str = ""


@dataclass
class Citation:
    """A regulation reference. `clause_id` is chosen deterministically from the clause
    map; `text` is what RAG retrieved for it. Never the other way round."""

    clause_id: str                   # e.g. "1926.100"
    title: str                       # e.g. "Head protection"
    text: str = ""                   # retrieved body, for the notice
    source_url: str = ""
    is_site_policy: bool = False     # true when no clause squarely covers the item


@dataclass
class Draft:
    """Agent 1 · Assessor. What happened, and under what rule."""

    summary: str                                  # one sentence, facts only
    citations: list[Citation] = field(default_factory=list)
    items_missing: list[str] = field(default_factory=list)
    confidence_note: str = ""                     # how detector recall shapes this
    agent: str = "assessor"

    @property
    def has_citation(self) -> bool:
        return bool(self.citations)


@dataclass
class SeverityScore:
    """Filled-in rubric rather than free-form judgement, so Macro-F1 against human
    labels is computable in week 6."""

    base: int                        # 1-5, from the item and zone
    zone_multiplier: float = 1.0
    repeat_count: int = 0
    evidence_strength: float = 1.0   # lowest detector recall among missing items
    total: float = 0.0
    rationale: str = ""


@dataclass
class Decision:
    """Agent 2 · Adjudicator. Given history, what response is proportionate."""

    action: Action
    severity: SeverityScore
    prior_violations: int = 0
    draft_body: str = ""             # the warning email or escalation letter
    recipient_role: str = ""         # "worker" | "supervisor"
    requires_approval: bool = False
    rationale: str = ""
    agent: str = "adjudicator"


@dataclass
class CommittedRecord:
    """Agent 3 · Recorder. Who this was, and what is now on file.

    `verified` is set only after a read-back. Agent 2's history on the NEXT event is
    exactly this table, so an unverified write silently turns a third offence into a
    first -- and nothing in the output looks wrong.
    """

    record_id: str = ""
    worker_ref: str | None = None
    resolved_by: str = ""            # "supervisor" | "roster" | "unresolved"
    written_at: str = ""
    verified: bool = False
    agent: str = "recorder"


@dataclass
class CaseState:
    """Everything known about one finding as it moves through the graph."""

    event: dict                                   # the ViolationEvent, immutable
    draft: Draft | None = None
    decision: Decision | None = None
    record: CommittedRecord | None = None
    trace: list[Step] = field(default_factory=list)
    blocked_on: Blocker | None = None
    schema_version: str = SCHEMA_VERSION

    # ---- gates the graph reads --------------------------------------

    @property
    def event_id(self) -> str:
        return self.event["event_id"]

    @property
    def is_actionable(self) -> bool:
        """Lane A's gate. Status alone is not enough -- it ignores whether the finding
        was confirmed across frames, which is what stops single-frame accusations."""
        return (self.event.get("status") == "violation"
                and self.event.get("confirmation", {}).get("confirmed", False))

    @property
    def may_proceed_to_adjudication(self) -> bool:
        """No citation, no decision. The guardrail from slide 9."""
        return self.draft is not None and self.draft.has_citation

    @property
    def needs_human(self) -> bool:
        return self.blocked_on is not None or (
            self.decision is not None and self.decision.requires_approval)

    @property
    def is_complete(self) -> bool:
        """Only a verified write closes a case."""
        return self.record is not None and self.record.verified

    # ---- bookkeeping -------------------------------------------------

    def log(self, agent: str, tool: str, arguments: dict | None = None,
            result_summary: str = "", latency_ms: int = 0) -> None:
        self.trace.append(Step(agent=agent, tool=tool, arguments=arguments or {},
                               result_summary=result_summary, latency_ms=latency_ms,
                               at=datetime.now().isoformat(timespec="seconds")))

    def block(self, reason: Blocker) -> None:
        self.blocked_on = reason

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.blocked_on:
            d["blocked_on"] = self.blocked_on.value
        if self.decision:
            d["decision"]["action"] = self.decision.action.value
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
