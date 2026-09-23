"""
The tools the agents call.

Almost everything here is deterministic on purpose. The clause is looked up, the severity
is arithmetic, the history is a query. What the agents contribute is judgement about
*wording and proportionality* -- not facts.

That split is the whole architecture: a model that cannot invent a citation, cannot
invent a score, and cannot invent a prior offence has a much smaller surface on which to
be confidently wrong.

Every tool returns a typed result and raises rather than guessing. A tool that silently
returns a default is a tool that lets an agent proceed on a fiction.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from kb.clause_map import ClauseMap, ItemRule, SeverityResult  # noqa: E402

_CLAUSE_MAP: ClauseMap | None = None
_RETRIEVER = None


class ClauseMapInvalid(RuntimeError):
    """clauses.yaml and zones.json disagree. Nothing should run on that."""


def clause_map(validate: bool = True) -> ClauseMap:
    """The clause map, checked against zones.json the first time it is loaded.

    The check has to happen here rather than in a test, because the thing it catches is
    a config edit: someone adds a zone to zones.json, forgets the matching weight in
    clauses.yaml, and `score(strict=True)` raises on the first event that touches that
    zone. That is a crash in the middle of a case, during a demo, rather than a refusal
    to start -- which is the same information delivered at the worst possible time.

    Validation runs once, on first load, and costs a YAML parse.
    """
    global _CLAUSE_MAP
    if _CLAUSE_MAP is None:
        candidate = ClauseMap.load()
        if validate:
            problems = candidate.validate()
            if problems:
                joined = "\n  - ".join(problems)
                raise ClauseMapInvalid(
                    f"clauses.yaml does not agree with zones.json:\n  - {joined}")
        _CLAUSE_MAP = candidate
    return _CLAUSE_MAP


def retriever():
    """Loaded lazily -- importing sentence-transformers costs seconds, and the clause
    lookup tools do not need it at all."""
    global _RETRIEVER
    if _RETRIEVER is None:
        from kb.retrieve import Retriever
        _RETRIEVER = Retriever.load()
    return _RETRIEVER


# ---------------------------------------------------------------------------
# Agent 1 · Assessor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClauseResult:
    item: str
    clause_id: str
    citation: str
    title: str
    basis: str                       # "regulation" | "site_policy"
    notice_phrase: str
    is_regulatory: bool
    supporting_clause: str = ""
    specification_clause: str = ""
    override_reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def lookup_clause(item: str, zone: str = "default") -> ClauseResult:
    """Which regulation covers this missing item here, and on what basis.

    DETERMINISTIC. Retrieval plays no part. Cosine similarity choosing which law a worker
    allegedly breached yields a confident, real-looking, wrong citation -- worse than no
    citation, and the failure the citation guardrail exists to catch.

    `basis` matters as much as the clause: "regulation" means the clause squarely requires
    the item, "site_policy" means the site requires it and the regulation only supports
    that through a general duty. A notice must never present the second as the first.
    """
    rule: ItemRule = clause_map().for_item(item, zone)
    base = clause_map()._items[item]                    # for the optional extra clauses
    return ClauseResult(
        item=item,
        clause_id=rule.clause.clause_id,
        citation=rule.clause.citation,
        title=rule.clause.title,
        basis=rule.basis,
        notice_phrase=" ".join(rule.notice_phrase.split()),
        is_regulatory=rule.is_regulatory,
        supporting_clause=base.get("supporting_clause", "") or "",
        specification_clause=base.get("specification_clause", "") or "",
        override_reason=rule.override_reason,
    )


def retrieve_clause_text(clause_id: str, max_chars: int = 900) -> str:
    """The words of a clause, fetched BY ID. No similarity search involved.

    This is where RAG actually sits in the citation path: the clause is already chosen,
    and retrieval supplies the text a notice quotes.
    """
    text = retriever().text_for(clause_id, max_chars=max_chars)
    if not text:
        raise ValueError(
            f"{clause_id} is citable but has no text in the index. Rebuild with "
            f"`python kb/ingest.py && python kb/build_index.py`.")
    return text


def score_baseline(zone: str, missing: list[str], camera: str | None = None,
                   required: list[str] | None = None) -> SeverityResult:
    """Severity from the zone's item weights, before history is known.

    Enough to set the urgency of the supervisor's identification prompt, which is why it
    runs before anyone is identified. Sums the weights of MISSING items -- never
    zone-total-minus-missing, which reports a compliant worker as severe in any zone
    whose total sits below the threshold.
    """
    return clause_map().score(missing, zone, priors=0, camera=camera, required=required)


# ---------------------------------------------------------------------------
# Agent 2 · Adjudicator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HistoryResult:
    worker_ref: str | None
    window_days: int
    prior_violations: int
    priors: list[dict]
    resolved: bool                   # False when identity is unknown
    note: str = ""


def get_worker_history(worker_ref: str | None, db=None, window_days: int = 7,
                       now: datetime | None = None,
                       exclude_event: str | None = None) -> HistoryResult:
    """Prior confirmed violations for this worker inside the window.

    Returns `resolved=False` when identity is unknown, and callers MUST treat that as
    "cannot say", not as "no priors". The pipeline never sets worker_ref -- ByteTrack ids
    do not survive a session -- so identity arrives only when a supervisor binds it. An
    unattributed event counts toward nobody, which is what stops the system escalating
    against someone on the basis of incidents nobody confirmed were them.
    """
    if not worker_ref:
        return HistoryResult(
            worker_ref=None, window_days=window_days, prior_violations=0, priors=[],
            resolved=False,
            note="identity not bound; history cannot be read and must not be assumed empty")

    if db is None:
        return HistoryResult(
            worker_ref=worker_ref, window_days=window_days, prior_violations=0, priors=[],
            resolved=False, note="no event store connected")

    since = (now or datetime.now()) - timedelta(days=window_days)
    # The finding being judged is never its own prior. Without this, judging an
    # incident a second time -- a replay, a re-run of the demo -- counted it against
    # the worker it was already attributed to: priors 2 -> 3, severity 9 -> 11, from
    # one event.
    rows = db.violations_for(worker_ref, since=since, exclude_event=exclude_event)
    return HistoryResult(
        worker_ref=worker_ref, window_days=window_days,
        prior_violations=len(rows), priors=list(rows), resolved=True)


def final_severity(zone: str, missing: list[str], priors: int,
                   camera: str | None = None,
                   required: list[str] | None = None) -> SeverityResult:
    """Baseline plus the repeat penalty. Deterministic, so Macro-F1 against human labels
    is computable in week 6."""
    return clause_map().score(missing, zone, priors=priors, camera=camera,
                              required=required)


@dataclass(frozen=True)
class ConfidenceResult:
    score: float                     # 0..1
    weakest_recall: float
    frames_missing: int
    frames_observed: int
    person_confidence: float
    band: str                        # "high" | "moderate" | "low"
    note: str

    @property
    def is_low(self) -> bool:
        return self.band == "low"


def assess_confidence(event: dict) -> ConfidenceResult:
    """How much the finding can be trusted -- a SEPARATE axis from how serious it is.

    Detector recall is deliberately kept out of severity. A missing helmet is equally
    dangerous whether the detector is 79% or 99% reliable; folding recall into severity
    would mean "we are less sure, therefore it is less serious", which is bad reasoning.

    High severity with low confidence is urgent human review, not a downgraded
    escalation.
    """
    ppe = event.get("ppe", {})
    missing = ppe.get("missing", [])
    recalls = event.get("detector", {}).get("recall", {})
    conf = event.get("confirmation", {})

    weakest = min((recalls.get(i, 0.7) for i in missing), default=1.0)
    observed = max(conf.get("frames_observed", 1), 1)
    frames_missing = conf.get("frames_missing", 0)
    person_conf = event.get("subject", {}).get("detection_confidence", 0.0)

    sustained = frames_missing / observed
    score = round(weakest * sustained * min(person_conf / 0.7, 1.0), 3)

    if score >= 0.65:
        band, note = "high", "sustained across frames on a confidently detected person"
    elif score >= 0.4:
        band, note = "moderate", "supported, but not strongly"
    else:
        band, note = "low", ("weak evidence -- route to human review rather than acting "
                             "on it automatically")

    return ConfidenceResult(
        score=score, weakest_recall=weakest, frames_missing=frames_missing,
        frames_observed=observed, person_confidence=person_conf, band=band, note=note)


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def event_facts(event: dict) -> dict:
    """The subset of a ViolationEvent an agent is allowed to reason from.

    Deliberately narrow. An agent given the whole event will reach for the evidence path
    or the bounding box and start describing a photograph it cannot see.
    """
    return {
        "event_id": event.get("event_id"),
        "captured_at": event.get("captured_at"),
        "camera_id": event.get("camera_id"),
        "zone": event.get("zone", {}).get("label") or event.get("zone", {}).get("name"),
        "zone_name": event.get("zone", {}).get("name"),
        "required_ppe": event.get("zone", {}).get("required_ppe", []),
        "present": list(event.get("ppe", {}).get("present", {})),
        "missing": event.get("ppe", {}).get("missing", []),
        "indeterminate": event.get("ppe", {}).get("indeterminate", []),
        "status": event.get("status"),
        "confirmed": event.get("confirmation", {}).get("confirmed", False),
        "worker_ref": event.get("subject", {}).get("worker_ref"),
    }
