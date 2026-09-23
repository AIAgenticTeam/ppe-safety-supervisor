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
                  POST /roster/import (CSV; a preview unless dry_run=false)
                  POST /roster/remove
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
from urllib.parse import urlsplit

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import Body, FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agents.graph import run_case  # noqa: E402
from agents.guardrails import gate_event_is_well_formed, gate_schema_understood  # noqa: E402
from agents.state import CaseState  # noqa: E402
from app import roster_import  # noqa: E402
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
# still checked before anything is opened, because the event body arrived over the
# network and a path inside it is not trustworthy either.
#
# This list used to include Path.cwd(). The server runs from the repo root, so that one
# entry made the whole repository servable: an event claiming its evidence was `.env`
# got the OpenAI key back byte-for-byte, labelled as a JPEG. The old test only tried
# paths OUTSIDE the repo, which is why it passed. Roots are now only where evidence is
# actually written, plus any extra directories named in SAFETY_EVIDENCE_ROOTS.
EVIDENCE_ROOTS = [ROOT / "events", ROOT / "fixtures"] + [
    Path(p) for p in os.getenv("SAFETY_EVIDENCE_ROOTS", "").split(os.pathsep) if p]

# Belt and braces: even inside an allowed root, only an actual image is served.
# Extension alone can be a lie, so the first bytes have to agree with it.
EVIDENCE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
_MAGIC = {"image/jpeg": b"\xff\xd8", "image/png": b"\x89PNG"}


# ---------------------------------------------------------------- payloads

class IdentityBinding(BaseModel):
    """Who a supervisor says this was. The one judgement the vision system never makes."""

    worker_id: str = Field(min_length=1, description="must be on the roster")
    bound_by: str = Field(min_length=1, description="the supervisor putting their name to it")


class Approval(BaseModel):
    approved_by: str = Field(min_length=1)


class RosterRemoval(BaseModel):
    worker_ids: list[str] = Field(min_length=1, max_length=5000)


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
    media = EVIDENCE_TYPES.get(resolved.suffix.lower())
    if media is None:
        raise HTTPException(404, "evidence not available")
    for root in EVIDENCE_ROOTS:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        if resolved.is_file():
            with open(resolved, "rb") as fh:
                if fh.read(4).startswith(_MAGIC[media]):
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

    @app.middleware("http")
    async def refuse_cross_site_writes(request: Request, call_next):
        """Refuse any write a browser sends on behalf of a different site.

        While the console is open, any other page the supervisor visits can fire
        requests at 127.0.0.1:8000. Most endpoints were already safe -- they require a
        JSON body, and a cross-site JSON request needs a CORS preflight that this app
        never grants. Two were not: /roster/import took a text/plain body and /replay
        took no body at all, so another website could plant people on the roster, or
        replay a real event -- which re-judged it and, before that was fixed, erased
        its approval.

        Browsers label cross-site requests themselves (Sec-Fetch-Site) and send the
        page's Origin with every POST. The pipeline, the scripts and the tests are not
        browsers and send neither, so they are unaffected. This covers every write
        endpoint, including ones not written yet.
        """
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            site = request.headers.get("sec-fetch-site")
            origin = request.headers.get("origin")
            host = request.headers.get("host", "")
            foreign = site in ("cross-site", "same-site") or (
                origin is not None
                and (origin == "null" or urlsplit(origin).netloc != host))
            if foreign:
                return JSONResponse({"detail": "cross-site request refused"},
                                    status_code=403)
        return await call_next(request)

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
        path = _safe_evidence_path(raw)
        return FileResponse(path, media_type=EVIDENCE_TYPES[path.suffix.lower()])

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

    @app.post("/roster/remove", tags=["dashboard"])
    def remove_workers(removal: RosterRemoval):
        """Take one or more people off the roster.

        Someone with no findings is deleted. Someone with findings is hidden instead
        ("deactivated"): their history stays attributed to them and their name stays in the
        reports, but they leave the roster and can no longer be picked. Importing their id
        again restores them."""
        outcome = db().remove_workers(removal.worker_ids)
        if not outcome:
            raise HTTPException(404, "none of those people are on the roster")
        return {
            "removed": len(outcome),
            "deleted": [w for w, o in outcome.items() if o == "deleted"],
            "deactivated": [w for w, o in outcome.items() if o == "deactivated"],
            "not_on_roster": [w for w in dict.fromkeys(removal.worker_ids) if w not in outcome],
        }

    @app.get("/roster/template.csv", tags=["dashboard"])
    def roster_template():
        """A header and one example row, for whoever is about to build the file."""
        return Response(roster_import.TEMPLATE, media_type="text/csv", headers={
            "Content-Disposition": 'attachment; filename="roster-template.csv"'})

    @app.post("/roster/import", tags=["dashboard"])
    async def import_roster(
            request: Request,
            dry_run: bool = Query(True, description="preview only; nothing is written"),
            skip_invalid: bool = Query(False, description="import the valid rows even if "
                                                          "some rows have problems")):
        """Add or update many people from a CSV file (the request body is the file).

        A preview by default: the same parse and the same plan, with nothing written, so
        the person sees what would change before it does. Applying is all-or-nothing --
        one bad row blocks the import -- unless `skip_invalid` says otherwise, because a
        half-applied roster is the outcome nobody notices until a name is missing."""
        declared = int(request.headers.get("content-length") or 0)
        if declared > roster_import.MAX_BYTES:
            raise HTTPException(413, "The file is too large for a roster import.")
        data = await request.body()
        if len(data) > roster_import.MAX_BYTES:
            raise HTTPException(413, "The file is too large for a roster import.")
        try:
            parsed = roster_import.parse(data)
        except roster_import.RosterFileError as exc:
            raise HTTPException(422, str(exc)) from exc

        plan = roster_import.plan(parsed, db().roster())
        status = {w.worker_id: "new" for w in plan.new}
        status.update({w.worker_id: "updated" for w in plan.updated})
        status.update({w.worker_id: "unchanged" for w in plan.unchanged})
        issues = [{"row": i.row, "problem": i.problem} for i in parsed.issues]

        applied = 0
        if not dry_run:
            if issues and not skip_invalid:
                raise HTTPException(422, f"{len(issues)} row(s) have problems, so nothing was "
                                         "imported. Fix them, or import the valid rows only.")
            if not parsed.rows:
                raise HTTPException(422, "There are no valid rows to import.")
            applied = db().add_workers(plan.to_write)

        return {
            "dry_run": dry_run, "applied": applied,
            "encoding": parsed.encoding,
            "delimiter": {",": "comma", ";": "semicolon", "	": "tab", "|": "pipe"}[parsed.delimiter],
            "columns": parsed.columns, "ignored_columns": parsed.ignored_columns,
            "rows": len(parsed.rows) + len(issues), "valid": len(parsed.rows),
            "new": len(plan.new), "updated": len(plan.updated), "unchanged": len(plan.unchanged),
            "issue_count": len(issues), "issues": issues[:200],
            "preview": [{"worker_id": w.worker_id, "name": w.name, "email": w.email,
                         "role": w.role, "status": status[w.worker_id]}
                        for _, w in parsed.rows[:8]],
        }

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
