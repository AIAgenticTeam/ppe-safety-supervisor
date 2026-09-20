"""
The service layer: FastAPI between the pipeline and the console.

Lane A posts findings, the agent graph judges them, the supervisor console reads queues
and sends back the two things only a human may decide -- who this was, and whether an
escalation goes out. The console is a set of server-rendered pages (app/web/) served by
this same app, so one process and one port -- open http://127.0.0.1:8000/.

    uvicorn app.api:app --reload
    python -m app.api                      # same thing, with the banner

The pages read the event store directly, but never decide anything themselves: identity,
approval, roster and replay are all written through the JSON endpoints below, so the
human gate has exactly one implementation and the tests cover it.

Endpoints, grouped by who calls them:

    pipeline      POST /events
                  GET  /replay/{source}   POST /replay/{source}?name=
    console       GET  /queue/identification   GET /queue/approvals
                  POST /events/{id}/identity   POST /decisions/{id}/approve
                  GET  /events  /events/{id}  /events/{id}/trace
                  GET  /evidence/{id}/frame   /evidence/{id}/crop
    dashboard     GET  /report  /stats  /roster
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import Body, FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agents.graph import run_case  # noqa: E402
from agents.guardrails import gate_event_is_well_formed, gate_schema_understood  # noqa: E402
from agents.state import CaseState  # noqa: E402
from app.db import DEFAULT_DB_PATH, EventStore, Worker  # noqa: E402
from app.web import replay  # noqa: E402
from app.web.routes import install as install_pages  # noqa: E402

DEFAULT_DB = os.getenv("SAFETY_DB", DEFAULT_DB_PATH)

# The key lives in .env, which is gitignored. Loaded here because this process is the
# one that calls the model -- scripts/run_agents.py loaded it for its own process and
# nothing did it for the service, so judging over HTTP failed with a 500 that said
# nothing useful. Never loaded inside `agents/`, so the test suite still runs with no
# credentials at all.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# Evidence is served by event id, never by a path from the request. The stored path is
# still checked against these roots before anything is opened, because the event body
# arrived over the network and a path inside it is not trustworthy either.
EVIDENCE_ROOTS = [ROOT / "events", ROOT / "fixtures", Path.cwd()]


# ---------------------------------------------------------------- payloads

class IdentityBinding(BaseModel):
    """Who a supervisor says this was. The one judgement the vision system never makes."""

    worker_id: str = Field(min_length=1, description="must be on the roster")
    bound_by: str = Field(min_length=1, description="the supervisor putting their name to it")


class Approval(BaseModel):
    approved_by: str = Field(min_length=1)


class WorkerIn(BaseModel):
    worker_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    email: str = ""
    role: str = ""


# ---------------------------------------------------------------- helpers

def _case_summary(state: CaseState, outcome: str, reason: str) -> dict[str, Any]:
    """What the console needs to render a case, and nothing it does not."""
    decision = state.decision
    return {
        "event_id": state.event.get("event_id"),
        "outcome": outcome,
        "reason": reason,
        "blocked_on": state.blocked_on.value if state.blocked_on else None,
        "summary": state.draft.summary if state.draft else "",
        "citations": [
            {"clause_id": c.clause_id, "title": c.title, "site_policy": c.is_site_policy}
            for c in (state.draft.citations if state.draft else [])
        ],
        "action": decision.action.value if decision else None,
        "severity": decision.severity.total if decision else None,
        "prior_violations": decision.prior_violations if decision else 0,
        "requires_approval": decision.requires_approval if decision else False,
        "draft_body": decision.draft_body if decision else "",
        "rationale": decision.rationale if decision else "",
        "recorded": state.record.verified if state.record else False,
        "steps": len(state.trace),
    }


def _safe_evidence_path(raw: str) -> Path:
    """Resolve a stored evidence path and refuse anything outside the known roots.

    The path comes from an event body that arrived over HTTP. Serving it unchecked
    would turn this endpoint into a file reader for the whole disk.
    """
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (ROOT / candidate)
    resolved = candidate.resolve()
    for root in EVIDENCE_ROOTS:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
        break
    raise HTTPException(404, "evidence not available")


# ---------------------------------------------------------------- the app

def create_app(db_path: str | Path = DEFAULT_DB, client=None,
               model: str = "gpt-4o-mini") -> FastAPI:
    """`client` is injectable so the tests can drive the whole API with a stub model."""
    app = FastAPI(
        title="Autonomous Industrial Safety Supervisor",
        description="PPE findings in, judged and cited, with a human on the escalations.",
        version="1.0",
    )
    app.state.db = EventStore(db_path)
    app.state.client = client
    app.state.model = model

    def db() -> EventStore:
        return app.state.db

    # ---- pipeline -------------------------------------------------------

    @app.post("/events", status_code=201, tags=["pipeline"])
    def post_event(event: dict = Body(...),
                   judge: bool = Query(True, description="run the agents now")):
        """Accept one ViolationEvent from the pipeline.

        Validation reuses the graph's own intake gates rather than restating the schema
        as a Pydantic model -- two descriptions of the same shape is how they drift.
        A malformed event is refused here with the reason, instead of being stored and
        quietly routed to review later.
        """
        case = CaseState(event=event)
        for check in (gate_schema_understood, gate_event_is_well_formed):
            verdict = check(case)
            if not verdict:
                raise HTTPException(422, verdict.reason)

        if not judge:
            return {"event_id": db().record_event(event), "outcome": "stored",
                    "reason": "judging skipped"}

        out = run_case(event, db(), client=app.state.client, model=app.state.model)
        return _case_summary(out["case"], out.get("outcome", ""), out.get("reason", ""))

    # ---- replay ---------------------------------------------------------

    @app.get("/replay/{source}", tags=["pipeline"])
    def replay_list(source: str):
        """Event files this source offers. A fixed allowlist, not a path from the client."""
        return {"source": source, "files": replay.list_files(source)}

    @app.post("/replay/{source}", tags=["pipeline"])
    def replay_one(source: str, name: str = Query(...),
                   judge: bool = Query(True, description="run the agents now")):
        """Post one listed event file through the real intake, one at a time so the
        page can show progress -- judging is a model call and 20 of them are not quick."""
        event = replay.load(source, name)
        if event is None:
            raise HTTPException(404, "not a replayable file")
        return post_event(event, judge)

    # ---- the console's two queues ---------------------------------------

    @app.get("/queue/identification", tags=["console"])
    def identification_queue(limit: int = Query(50, ge=1, le=500)):
        """Findings nobody has put a name to. Actionable ones first."""
        return {"events": db().unattributed(limit=limit)}

    @app.get("/queue/approvals", tags=["console"])
    def approval_queue():
        """Escalations waiting on a signature. Nothing here has been sent."""
        return {"decisions": db().pending_approval()}

    @app.post("/events/{event_id}/identity", tags=["console"])
    def bind_identity(event_id: str, binding: IdentityBinding):
        """A supervisor identifies the worker. Refused unless they are on the roster --
        free-text identity is how one person's history splits across spellings."""
        if db().get_event(event_id) is None:
            raise HTTPException(404, "no such event")
        if not db().attach_identity(event_id, binding.worker_id, binding.bound_by):
            raise HTTPException(422, f"{binding.worker_id} is not on the roster")
        return {"event_id": event_id, "worker_id": binding.worker_id,
                "bound_by": binding.bound_by}

    @app.post("/decisions/{event_id}/approve", tags=["console"])
    def approve(event_id: str, approval: Approval):
        """The human-in-the-loop gate. Approving twice is refused rather than silently
        replacing whoever signed first."""
        decision = db().get_decision(event_id)
        if decision is None:
            raise HTTPException(404, "no decision on that event")
        if not db().approve(event_id, approval.approved_by):
            raise HTTPException(409, "already approved by "
                                     f"{decision.get('approved_by') or 'someone else'}")
        return {"event_id": event_id, "approved_by": approval.approved_by}

    # ---- reading --------------------------------------------------------

    @app.get("/events", tags=["console"])
    def list_events(limit: int = Query(50, ge=1, le=500),
                    unattributed_only: bool = False):
        if unattributed_only:
            return {"events": db().unattributed(limit=limit)}
        return {"events": db().recent(limit=limit)}

    @app.get("/events/{event_id}", tags=["console"])
    def get_event(event_id: str):
        event = db().get_event(event_id)
        if event is None:
            raise HTTPException(404, "no such event")
        return {"event": event, "decision": db().get_decision(event_id)}

    @app.get("/events/{event_id}/trace", tags=["console"])
    def get_trace(event_id: str):
        """Every tool each agent chose, in order. The observability requirement, and the
        thing that shows a panel the agents are deciding rather than following a script."""
        trace = db().get_trace(event_id)
        if trace is None:
            raise HTTPException(404, "no trace for that event")
        return {"event_id": event_id, "trace": trace}

    @app.get("/evidence/{event_id}/{kind}", tags=["console"])
    def get_evidence(event_id: str, kind: Literal["frame", "crop"]):
        """The annotated frame or the crop. A finding a supervisor cannot check for
        themselves is not auditable."""
        event = db().get_event(event_id)
        if event is None:
            raise HTTPException(404, "no such event")
        raw = (event.get("evidence") or {}).get(f"{kind}_path")
        if not raw:
            raise HTTPException(404, f"no {kind} stored for that event")
        return FileResponse(_safe_evidence_path(raw), media_type="image/jpeg")

    # ---- dashboard ------------------------------------------------------

    @app.get("/report", tags=["dashboard"])
    def report(days: int = Query(7, ge=1, le=365)):
        """Zone hotspots and repeat offenders.

        Zone counts need no identity at all, and are arguably the more actionable
        number: you fix the zone, not the person.
        """
        return {
            "days": days,
            "by_zone": db().by_zone(days=days),
            "repeat_offenders": db().repeat_offenders(days=days),
            "stats": db().stats(),
        }

    @app.get("/drift", tags=["dashboard"])
    def drift(days: int = Query(7, ge=1, le=365),
              save_html: bool = Query(False, description="also write the full report")):
        """Has the detector walked off the distribution it was trained on?

        The failure this exists for is quiet: recall does not announce itself when it
        drops, findings simply stop appearing, and an empty queue looks exactly like a
        safe site. A moved camera or a new site shows up in the confidence distribution
        well before anybody notices the queue is thin.

        Returns `ran: false` with a reason when there is too little data, rather than a
        verdict computed from a handful of rows.
        """
        from app.monitoring import run_drift

        out = ROOT / "reports" / f"drift_{days}d.html" if save_html else None
        result = run_drift(db(), days=days, html_path=out)
        body = result.summary()
        if result.html:
            body["report_html"] = result.html
        return body

    @app.get("/stats", tags=["dashboard"])
    def stats():
        return db().stats()

    @app.get("/roster", tags=["dashboard"])
    def roster():
        return {"workers": [w.__dict__ for w in db().roster()]}

    @app.post("/roster", status_code=201, tags=["dashboard"])
    def add_worker(worker: WorkerIn):
        db().add_worker(Worker(worker.worker_id, worker.name, worker.email, worker.role))
        return {"worker_id": worker.worker_id}

    @app.get("/health", tags=["dashboard"])
    def health():
        return {"status": "ok", "db": db().path, "events": db().stats()["events"]}

    install_pages(app)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    print(f"  db      {DEFAULT_DB}")
    print("  docs    http://127.0.0.1:8000/docs")
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
