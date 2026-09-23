"""
The FastAPI service.

Driven with a stub model, so the whole API is exercised without a key or a network.

The endpoints worth being careful about are the two that accept input from outside:
`POST /events` takes a body the pipeline built, and the evidence routes serve files
whose paths came from inside that body. Everything else is a read over SQLite.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).absolute().parent))

pytest.importorskip("fastapi", reason="needs fastapi; install into .venv")
from fastapi.testclient import TestClient  # noqa: E402

from app.api import create_app  # noqa: E402
from test_agents import (StubCall, StubClient, _adjudicator_script,  # noqa: E402
                         confirmed_event)


def script(action="warning", clause_ids=("1926.100",)):
    return [[StubCall("submit_draft", {
        "summary": "A worker was without head protection at the grinding station.",
        "clause_ids": list(clause_ids)})]] + _adjudicator_script(action)


@pytest.fixture
def client(tmp_path):
    def _make(action="warning"):
        app = create_app(tmp_path / "api.db", client=StubClient(script(action)))
        return TestClient(app)
    return _make


@pytest.fixture
def api(client):
    return client()


# ------------------------------------------------------------- the pipeline

def test_posting_an_event_judges_and_records_it(api):
    r = api.post("/events", json=confirmed_event())
    assert r.status_code == 201
    body = r.json()
    assert body["outcome"] == "done"
    assert body["action"] == "warning"
    assert body["recorded"] is True
    assert body["citations"][0]["clause_id"] == "1926.100"


def test_an_event_of_an_unknown_schema_is_refused_at_the_door(api):
    """Not stored and quietly routed to review -- refused, with the reason. The caller
    is a program, and a program can be fixed if it is told."""
    r = api.post("/events", json={**confirmed_event(), "schema_version": "2.0"})
    assert r.status_code == 422
    assert "schema" in r.json()["detail"].lower()


def test_a_malformed_event_names_the_missing_fields(api):
    broken = confirmed_event()
    del broken["confirmation"]
    r = api.post("/events", json=broken)
    assert r.status_code == 422
    assert "confirmation.confirmed" in r.json()["detail"]


def test_judging_can_be_skipped_to_store_a_finding_cheaply(api):
    """Backfilling a night of footage should not spend a model call per frame."""
    r = api.post("/events?judge=false", json=confirmed_event())
    assert r.status_code == 201
    assert r.json()["outcome"] == "stored"
    assert api.get("/events/evt_test_1").json()["decision"] is None


# ---------------------------------------------------------- the two queues

def test_a_judged_event_waits_in_the_identification_queue(api):
    api.post("/events", json=confirmed_event())
    queue = api.get("/queue/identification").json()["events"]
    assert [e["event_id"] for e in queue] == ["evt_test_1"]


def test_binding_identity_clears_it_from_that_queue(api):
    api.post("/events", json=confirmed_event())
    api.post("/roster", json={"worker_id": "W-1", "name": "Someone"})

    r = api.post("/events/evt_test_1/identity",
                 json={"worker_id": "W-1", "bound_by": "supervisor:khalid"})
    assert r.status_code == 200
    assert api.get("/queue/identification").json()["events"] == []


def test_an_identity_off_the_roster_is_refused(api):
    api.post("/events", json=confirmed_event())
    r = api.post("/events/evt_test_1/identity",
                 json={"worker_id": "W-GHOST", "bound_by": "supervisor:khalid"})
    assert r.status_code == 422
    assert "roster" in r.json()["detail"]


def test_binding_identity_on_an_unknown_event_is_a_404(api):
    api.post("/roster", json={"worker_id": "W-1", "name": "Someone"})
    r = api.post("/events/nope/identity",
                 json={"worker_id": "W-1", "bound_by": "k"})
    assert r.status_code == 404


def test_an_identification_must_name_who_made_it(api):
    """An attribution with no author is an accusation with no author."""
    api.post("/events", json=confirmed_event())
    r = api.post("/events/evt_test_1/identity", json={"worker_id": "W-1"})
    assert r.status_code == 422        # pydantic refuses the payload


# --------------------------------------------------------------- approvals

def test_an_escalation_waits_for_a_signature(client):
    api = client("escalation")
    api.post("/events", json=confirmed_event())
    queue = api.get("/queue/approvals").json()["decisions"]
    assert [d["event_id"] for d in queue] == ["evt_test_1"]


def test_approving_clears_the_queue(client):
    api = client("escalation")
    api.post("/events", json=confirmed_event())
    r = api.post("/decisions/evt_test_1/approve", json={"approved_by": "supervisor:k"})
    assert r.status_code == 200
    assert api.get("/queue/approvals").json()["decisions"] == []


def test_a_second_approval_is_refused_rather_than_replacing_the_first(client):
    """Whose name is on it matters."""
    api = client("escalation")
    api.post("/events", json=confirmed_event())
    api.post("/decisions/evt_test_1/approve", json={"approved_by": "supervisor:first"})
    r = api.post("/decisions/evt_test_1/approve", json={"approved_by": "supervisor:second"})
    assert r.status_code == 409
    assert "supervisor:first" in r.json()["detail"]


def test_a_warning_never_enters_the_approval_queue(api):
    api.post("/events", json=confirmed_event())
    assert api.get("/queue/approvals").json()["decisions"] == []


def test_approving_something_undecided_is_a_404(api):
    r = api.post("/decisions/nope/approve", json={"approved_by": "k"})
    assert r.status_code == 404


# ------------------------------------------------------------ the trace

def test_the_trace_survives_the_request_that_made_it(api):
    """It used to live in CaseState and die with the process, which meant the one case
    somebody wanted explained was the one with no explanation."""
    api.post("/events", json=confirmed_event())
    trace = api.get("/events/evt_test_1/trace").json()["trace"]
    assert {step["agent"] for step in trace} >= {"graph", "assessor", "adjudicator",
                                                 "recorder"}
    assert trace == sorted(trace, key=lambda s: s["step_index"])


def test_a_parked_case_keeps_its_trace_too(tmp_path):
    """The case most worth explaining is the one that did not finish."""
    app = create_app(tmp_path / "api.db", client=StubClient([]))   # model says nothing
    api = TestClient(app)
    api.post("/events", json=confirmed_event())
    assert api.get("/events/evt_test_1/trace").json()["trace"]


def test_no_trace_for_an_unknown_event(api):
    assert api.get("/events/nope/trace").status_code == 404


# ------------------------------------------------------------- evidence

def test_evidence_is_served_when_it_exists(api, tmp_path):
    frame = ROOT / "events" / "_test_frame.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    try:
        event = confirmed_event()
        event["evidence"] = {"frame_path": str(frame), "crop_path": str(frame)}
        api.post("/events", json=event)
        r = api.get("/evidence/evt_test_1/frame")
        assert r.status_code == 200
        assert r.content.startswith(b"\xff\xd8")
    finally:
        frame.unlink(missing_ok=True)


def test_an_evidence_path_outside_the_known_roots_is_refused(api):
    """The body arrived over HTTP. Serving a path out of it unchecked would turn this
    endpoint into a file reader for the whole disk."""
    event = confirmed_event()
    event["evidence"] = {"frame_path": r"C:\Windows\win.ini",
                         "crop_path": "/etc/passwd"}
    api.post("/events", json=event)
    assert api.get("/evidence/evt_test_1/frame").status_code == 404
    assert api.get("/evidence/evt_test_1/crop").status_code == 404


def test_a_traversal_dressed_as_a_relative_path_is_refused(api):
    event = confirmed_event()
    event["evidence"] = {"frame_path": "events/../../../../Windows/win.ini"}
    api.post("/events", json=event)
    assert api.get("/evidence/evt_test_1/frame").status_code == 404


def test_an_event_with_no_evidence_says_so(api):
    api.post("/events", json=confirmed_event())
    r = api.get("/evidence/evt_test_1/frame")
    assert r.status_code == 404
    assert "no frame" in r.json()["detail"]


def test_only_frame_and_crop_are_valid_evidence_kinds(api):
    api.post("/events", json=confirmed_event())
    assert api.get("/evidence/evt_test_1/payload").status_code == 422


# ------------------------------------------------------------ dashboard

def test_the_report_counts_by_zone_and_needs_no_identity(api):
    api.post("/events", json=confirmed_event())
    report = api.get("/report").json()
    assert report["by_zone"][0]["zone"] == "grinding_station"
    assert report["by_zone"][0]["confirmed"] == 1
    assert report["repeat_offenders"] == []       # nobody has been named


def test_the_roster_round_trips(api):
    api.post("/roster", json={"worker_id": "W-1", "name": "Alpha", "role": "fitter"})
    workers = api.get("/roster").json()["workers"]
    assert workers[0]["worker_id"] == "W-1" and workers[0]["role"] == "fitter"


def test_stats_start_at_zero_not_null(api):
    assert api.get("/stats").json() == {"events": 0, "actionable": 0, "unattributed": 0,
                                        "decisions": 0, "workers": 0}


def test_health_reports_the_database_it_is_actually_using(api, tmp_path):
    body = api.get("/health").json()
    assert body["status"] == "ok"
    assert body["db"].endswith("api.db")


def test_recent_events_carry_their_decision_for_the_console_list(api):
    api.post("/events", json=confirmed_event())
    row = api.get("/events").json()["events"][0]
    # grinding_station + helmet scores 5, which is the escalation band. The stub chose
    # "warning" -- gentler than the score, which the agent is allowed to do on its own.
    assert row["action"] == "warning" and row["severity"] == 5.0
    assert row["missing"] == ["helmet"] and row["worker_id"] is None


# ----------------------------------------------- evidence: only evidence is served
#
# The roots used to include the working directory, which is the repository. Any event
# could name any file in it -- .env with the OpenAI key, the database, the source --
# and /evidence served it byte for byte. Evidence is now an image, under events/ or
# fixtures/, that starts like an image.

@pytest.mark.parametrize("path", [
    "README.md",                                   # relative to the old root, the repo
    str(ROOT / "app" / "api.py"),
    str(ROOT / "requirements.txt"),
    "events/../README.md",
])
def test_a_file_from_the_repository_is_not_evidence(api, path):
    event = confirmed_event()
    event["evidence"] = {"frame_path": path, "crop_path": path}
    api.post("/events", json=event)
    assert api.get("/evidence/evt_test_1/frame").status_code == 404
    assert api.get("/evidence/evt_test_1/crop").status_code == 404


def test_an_image_name_on_something_that_is_not_an_image_is_refused(api):
    """Inside the right folder, with the right extension, and still not a photograph."""
    fake = ROOT / "events" / "_test_not_an_image.jpg"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("OPENAI_API_KEY=placeholder-not-a-key\n", encoding="utf-8")
    try:
        event = confirmed_event()
        event["evidence"] = {"frame_path": str(fake)}
        api.post("/events", json=event)
        assert api.get("/evidence/evt_test_1/frame").status_code == 404
    finally:
        fake.unlink(missing_ok=True)


def test_a_file_in_the_evidence_folder_that_is_not_an_image_type_is_refused(api):
    notes = ROOT / "events" / "_test_notes.txt"
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_bytes(b"\xff\xd8 starts like a jpeg, named like a note")
    try:
        event = confirmed_event()
        event["evidence"] = {"frame_path": str(notes)}
        api.post("/events", json=event)
        assert api.get("/evidence/evt_test_1/frame").status_code == 404
    finally:
        notes.unlink(missing_ok=True)


# --------------------------------------------- writes from another website
#
# While the console is open, any page the supervisor visits can send requests to it.
# /roster/import accepted a text/plain body and /replay accepted no body, so neither
# needed the CORS preflight that protects the JSON endpoints.

FOREIGN = [
    {"Sec-Fetch-Site": "cross-site"},
    {"Sec-Fetch-Site": "same-site"},               # another port on this machine
    {"Origin": "https://attacker.example"},
    {"Origin": "null"},                            # a sandboxed iframe
]
PLANT = "worker_id,name\nW-PLANTED,Planted Person\n"


def _a_fixture():
    names = sorted(p.name for p in (ROOT / "fixtures" / "events").glob("*.json"))
    if not names:
        pytest.skip("no fixture events to replay")
    return names[0]


@pytest.mark.parametrize("headers", FOREIGN)
def test_another_website_cannot_import_a_roster(api, headers):
    r = api.post("/roster/import?dry_run=false", content=PLANT,
                 headers={**headers, "Content-Type": "text/plain"})
    assert r.status_code == 403
    assert api.get("/roster").json()["workers"] == []


@pytest.mark.parametrize("headers", FOREIGN)
def test_another_website_cannot_replay_or_post_an_event(api, headers):
    assert api.post(f"/replay/fixtures?name={_a_fixture()}&judge=false",
                    headers=headers).status_code == 403
    assert api.post("/events?judge=false", json=confirmed_event(),
                    headers=headers).status_code == 403
    assert api.get("/events").json()["events"] == []


def test_the_console_itself_can_still_write(api):
    same = {"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"}
    r = api.post("/roster/import?dry_run=false", content=PLANT,
                 headers={**same, "Content-Type": "text/csv"})
    assert r.status_code == 200
    assert api.post(f"/replay/fixtures?name={_a_fixture()}&judge=false",
                    headers=same).status_code == 200


def test_programs_that_are_not_browsers_are_unaffected(api):
    """The pipeline and the scripts send neither header."""
    assert api.post("/events?judge=false", json=confirmed_event()).status_code == 201


def test_reading_is_not_refused(api):
    """Only writes are checked. Following a link to the console must still work."""
    assert api.get("/events", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 200


# ------------------------------------------------ a signed decision is final

def test_a_replay_does_not_erase_an_approval(tmp_path):
    """Posting an approved event again used to re-judge it and overwrite the decision,
    approval and all. The second run is scripted with no model turns: if it reached an
    agent, the case would park instead of finishing."""
    app = create_app(tmp_path / "api.db", client=StubClient(script("escalation")))
    api = TestClient(app)
    api.post("/events", json=confirmed_event())
    api.post("/decisions/evt_test_1/approve", json={"approved_by": "supervisor:k"})

    again = api.post("/events", json=confirmed_event())
    assert again.status_code == 201
    assert again.json()["outcome"] == "done"
    assert "supervisor:k" in again.json()["reason"]
    decision = app.state.db.get_decision("evt_test_1")
    assert decision["approved_by"] == "supervisor:k"
    assert decision["action"] == "escalation"
