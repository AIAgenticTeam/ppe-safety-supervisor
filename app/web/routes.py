"""
The supervisor console, as pages.

Server-rendered HTML in the same process as the API, replacing the Streamlit console.
Reads go straight to the event store; every write (identity, approval, roster, replay)
is made by the page's script against the same JSON endpoints the tests already cover,
so the human gate lives in one place and this file has nothing to approve with.

The HTML paths avoid the JSON ones on purpose (`/findings`, not `/events`), because the
pipeline and the scripts POST to `/events` and that contract does not move.

    /                  overview                     /weekly       zones and repeat offenders
    /identify          findings with no name        /monitoring   detector drift
    /approvals         escalations awaiting a sign  /team         the roster
    /findings          everything recorded          /about /faq   how it works, and why
    /findings/{id}     one finding, with its trace
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.web import replay

HERE = Path(__file__).absolute().parent

# ANSI Z535 safety colours, unchanged from the Streamlit console. They are not part of
# the erepair palette and are deliberately kept out of it: a site already reads red as
# stop and orange as escalate, so a severity band inherits a meaning the viewer has
# before they read the label. Recolouring them to the theme would make them decoration.
BAND_COLOUR = {"stop_work": "#C8102E", "escalation": "#E35205",
               "warning": "#B08400", "log_only": "#6B7887", "no_action": "#00843D"}

templates = Jinja2Templates(directory=str(HERE / "templates"))


def _stamp(value: str | None) -> str:
    """2026-09-07T09:03:00+03:00 -> 2026-09-07 09:03"""
    return (value or "")[:16].replace("T", " ")


def _severity(value) -> str:
    # A bare "None" in a column reads as a fault. An undecided finding is not one.
    return f"{value:g}" if value is not None else "—"


templates.env.filters["stamp"] = _stamp
templates.env.filters["severity"] = _severity
templates.env.filters["action_label"] = lambda a: (a or "undecided").replace("_", " ")
templates.env.globals["band_colour"] = lambda a: BAND_COLOUR.get(a, "#6B7887")


def install(app: FastAPI) -> None:
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    router = APIRouter(include_in_schema=False)

    def store():
        return app.state.db

    def render(request: Request, name: str, *, active: str, title: str,
               crumb: str | None = None, status_code: int = 200, **ctx):
        """One place that adds what the header and footer show on every page."""
        db = store()
        stats = db.stats()
        ctx.update(
            active=active, title=title, crumb=crumb or title, stats=stats,
            n_identify=stats["unattributed"], n_approve=len(db.pending_approval()),
            db_name=Path(db.path).name)
        return templates.TemplateResponse(request, name, ctx, status_code=status_code)

    # ---- overview -------------------------------------------------------

    @router.get("/")
    def home(request: Request):
        db = store()
        return render(request, "home.html", active="home", title="Home",
                      recent=db.recent(limit=5), pending=db.pending_approval()[:3])

    # ---- the two queues -------------------------------------------------

    @router.get("/identify")
    def identify(request: Request):
        db = store()
        return render(request, "identify.html", active="identify",
                      title="Identify findings",
                      queue=db.unattributed(limit=50),
                      roster=[w.__dict__ for w in db.roster()])

    @router.get("/approvals")
    def approvals(request: Request):
        db = store()
        rows = []
        for row in db.pending_approval():
            rows.append({**row,
                         "decision": db.get_decision(row["event_id"]) or {},
                         "trace": db.get_trace(row["event_id"]) or []})
        return render(request, "approvals.html", active="approvals",
                      title="Approvals", queue=rows)

    # ---- the audit view -------------------------------------------------

    @router.get("/findings")
    def findings(request: Request):
        return render(request, "findings.html", active="findings",
                      title="Findings", rows=store().recent(limit=100),
                      sources=replay.summary())

    @router.get("/findings/{event_id}")
    def finding(request: Request, event_id: str):
        db = store()
        event = db.get_event(event_id)
        if event is None:
            return render(request, "404.html", active="", title="404 Error",
                          status_code=404)
        return render(request, "finding.html", active="findings",
                      title="Finding", crumb=event_id, event=event,
                      decision=db.get_decision(event_id),
                      trace=db.get_trace(event_id) or [])

    # ---- reports --------------------------------------------------------

    @router.get("/weekly")
    def weekly(request: Request, days: int = Query(7, ge=1, le=365)):
        db = store()
        by_zone = db.by_zone(days=days)
        # Label by zone, and add the camera only where a zone name is ambiguous.
        # "cam_3/walkway" and "d_view02/walkway" are different places, so the camera
        # cannot always be dropped -- but prefixing every row pushes the part that
        # matters off the end of the axis.
        seen = Counter(r["zone"] for r in by_zone)
        peak = max((r["confirmed"] or 0 for r in by_zone), default=0)
        zones = [{"label": (r["zone"] if seen[r["zone"]] == 1
                            else f"{r['zone']} · {r['camera_id']}"),
                  "confirmed": r["confirmed"] or 0, "all": r["n"],
                  "pct": round(100 * (r["confirmed"] or 0) / peak) if peak else 0}
                 for r in by_zone]
        return render(request, "weekly.html", active="weekly", title="The week",
                      days=days, zones=zones, repeats=db.repeat_offenders(days=days))

    @router.get("/monitoring")
    def monitoring(request: Request, days: int = Query(7, ge=1, le=60)):
        # The drift run is slow (Evidently), so the page draws immediately and its
        # script fetches /drift -- a click on any other page never waits on it.
        return render(request, "monitoring.html", active="monitoring",
                      title="Monitoring", days=days)

    # ---- people and explanation ----------------------------------------

    @router.get("/team")
    def team(request: Request):
        db = store()
        return render(request, "team.html", active="team", title="Team",
                      roster=[w.__dict__ for w in db.roster()],
                      findings=db.finding_counts())

    @router.get("/about")
    def about(request: Request):
        return render(request, "about.html", active="about", title="How it works")

    @router.get("/faq")
    def faq(request: Request):
        return render(request, "faq.html", active="faq", title="FAQ")

    app.include_router(router)

    # A browser that follows a dead link gets the themed 404. API clients send no
    # `text/html` in Accept, so the JSON routes keep returning their JSON errors.
    @app.exception_handler(StarletteHTTPException)
    async def not_found(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404 and "text/html" in request.headers.get("accept", ""):
            return render(request, "404.html", active="", title="404 Error",
                          status_code=404)
        return await http_exception_handler(request, exc)
