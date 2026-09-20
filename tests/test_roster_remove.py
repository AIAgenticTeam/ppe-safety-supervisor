"""
Removing people from the roster.

The one rule that shapes everything here: a finding belongs to the record, and the reports
read a worker's name from the workers table. So someone with findings attributed to them is
*hidden* (deactivated), never erased -- and someone with none is genuinely deleted. Either
way they leave the roster and can no longer be picked or newly attributed.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi", reason="needs fastapi; install into .venv")

from fastapi.testclient import TestClient  # noqa: E402

from app.api import create_app  # noqa: E402
from app.db import EventStore, Worker  # noqa: E402

FIXTURES = sorted((ROOT / "fixtures" / "events").glob("*.json"))


def event(i=0):
    return json.loads(FIXTURES[i].read_text(encoding="utf-8"))


@pytest.fixture
def store(tmp_path):
    db = EventStore(tmp_path / "rm.db")
    for wid, name in [("W-1", "Alpha"), ("W-2", "Beta"), ("W-3", "Gamma")]:
        db.add_worker(Worker(wid, name, f"{name.lower()}@x.com", "Welder"))
    return db


def attribute(db, worker_id, i=0):
    """Record a real fixture event and attribute it, so the worker has a finding."""
    e = event(i)
    db.record_event(e)
    assert db.attach_identity(e["event_id"], worker_id, "supervisor:test")
    return e["event_id"]


def ids(db):
    return sorted(w.worker_id for w in db.roster())


# ------------------------------------------------------------------ the store

def test_someone_with_no_findings_is_deleted(store):
    assert store.remove_workers(["W-2"]) == {"W-2": "deleted"}
    assert ids(store) == ["W-1", "W-3"]
    # genuinely gone from the table, not merely hidden
    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM workers WHERE worker_id = 'W-2'").fetchone()[0] == 0


def test_someone_with_findings_is_hidden_not_erased(store):
    event_id = attribute(store, "W-1")
    assert store.remove_workers(["W-1"]) == {"W-1": "deactivated"}
    assert "W-1" not in ids(store)
    # the record is intact: the finding is still theirs
    row = next(r for r in store.recent() if r["event_id"] == event_id)
    assert row["worker_id"] == "W-1"


def test_reports_still_know_the_name_of_someone_who_was_removed(store):
    """repeat_offenders reads the name from the workers table, so hiding must keep the row."""
    for i in range(2):
        e = event(i)
        e["confirmation"]["confirmed"] = True
        store.record_event(e)
        store.attach_identity(e["event_id"], "W-1", "supervisor:test")
    store.remove_workers(["W-1"])
    offenders = store.repeat_offenders(days=3650, minimum=1)
    assert any(o["worker_id"] == "W-1" and o["name"] == "Alpha" for o in offenders)


def test_a_removed_person_cannot_be_newly_attributed(store):
    """Removed from the roster means not pickable -- through the API as well as the page."""
    store.remove_workers(["W-2"])            # deleted
    store.remove_workers(["W-1"])            # nothing attached yet: also deleted
    e = event(3)
    store.record_event(e)
    assert store.attach_identity(e["event_id"], "W-2", "supervisor:test") is False
    # and one who was hidden rather than deleted
    attribute(store, "W-3", i=4)
    store.remove_workers(["W-3"])
    assert store.attach_identity(e["event_id"], "W-3", "supervisor:test") is False


def test_the_worker_count_is_the_roster_not_the_archive(store):
    attribute(store, "W-1")
    store.remove_workers(["W-1", "W-2"])
    assert store.stats()["workers"] == 1


def test_unknown_and_already_removed_ids_are_ignored_and_duplicates_count_once(store):
    attribute(store, "W-1")
    assert store.remove_workers(["W-1", "W-1", "nobody", "W-2", "W-2"]) == {
        "W-1": "deactivated", "W-2": "deleted"}
    assert store.remove_workers(["W-1", "W-2"]) == {}       # both already gone


def test_finding_counts_say_who_has_history(store):
    attribute(store, "W-1", 0)
    attribute(store, "W-1", 1)
    attribute(store, "W-3", 2)
    assert store.finding_counts() == {"W-1": 2, "W-3": 1}


def test_a_hidden_person_comes_back_when_their_id_is_added_again(store):
    """Importing the roster again restores them, with their findings still attached."""
    event_id = attribute(store, "W-1")
    store.remove_workers(["W-1"])
    store.add_worker(Worker("W-1", "Alpha", "alpha@x.com", "Foreman"))
    assert "W-1" in ids(store)
    assert next(r for r in store.recent() if r["event_id"] == event_id)["worker_id"] == "W-1"


# ------------------------------------------------------------------ the endpoint

@pytest.fixture
def api(tmp_path):
    app = create_app(tmp_path / "api.db")
    client = TestClient(app)
    for wid, name in [("W-1", "Alpha"), ("W-2", "Beta"), ("W-3", "Gamma")]:
        client.post("/roster", json={"worker_id": wid, "name": name})
    return client, app


def roster_ids(client):
    return sorted(w["worker_id"] for w in client.get("/roster").json()["workers"])


def test_removing_people_reports_what_happened_to_each(api):
    client, app = api
    attribute(app.state.db, "W-1")
    r = client.post("/roster/remove", json={"worker_ids": ["W-1", "W-2", "ghost"]})
    body = r.json()
    assert r.status_code == 200 and body["removed"] == 2
    assert body["deleted"] == ["W-2"] and body["deactivated"] == ["W-1"]
    assert body["not_on_roster"] == ["ghost"]
    assert roster_ids(client) == ["W-3"]


def test_removing_nobody_on_the_roster_is_a_404(api):
    client, _ = api
    r = client.post("/roster/remove", json={"worker_ids": ["ghost"]})
    assert r.status_code == 404
    assert roster_ids(client) == ["W-1", "W-2", "W-3"]


@pytest.mark.parametrize("body", [{}, {"worker_ids": []}, {"worker_ids": "W-1"},
                                  {"worker_ids": [f"W-{i}" for i in range(5001)]}])
def test_a_malformed_request_is_refused_and_removes_nothing(api, body):
    client, _ = api
    assert client.post("/roster/remove", json=body).status_code == 422
    assert roster_ids(client) == ["W-1", "W-2", "W-3"]


def test_after_removal_identity_binding_is_refused_like_any_unknown_person(api):
    client, app = api
    event_id = app.state.db.record_event(event())
    client.post("/roster/remove", json={"worker_ids": ["W-2"]})
    r = client.post(f"/events/{event_id}/identity",
                    json={"worker_id": "W-2", "bound_by": "supervisor:test"})
    assert r.status_code == 422 and "not on the roster" in r.json()["detail"]


def test_a_removed_person_keeps_their_findings_visible_in_the_audit_list(api):
    client, app = api
    event_id = attribute(app.state.db, "W-1")
    client.post("/roster/remove", json={"worker_ids": ["W-1"]})
    row = next(r for r in client.get("/events").json()["events"] if r["event_id"] == event_id)
    assert row["worker_id"] == "W-1"


def test_importing_a_removed_persons_id_restores_them(api):
    client, app = api
    attribute(app.state.db, "W-1")
    client.post("/roster/remove", json={"worker_ids": ["W-1"]})
    r = client.post("/roster/import", params={"dry_run": "false"}, headers={"content-type": "text/csv"},
                    content=b"worker_id,name\nW-1,Alpha Restored\n")
    assert r.status_code == 200 and r.json()["new"] == 1
    assert "W-1" in roster_ids(client)


# ------------------------------------------------------------------ the page

def test_the_team_page_offers_removal_and_says_who_has_findings(api):
    client, app = api
    attribute(app.state.db, "W-1")
    page = client.get("/team", headers={"accept": "text/html"}).text
    assert "data-remove-dialog" in page and "data-bulk" in page
    assert page.count("data-remove") >= 3                    # a control per person, plus the dialog
    assert 'data-id="W-1"' in page and 'data-findings="1"' in page
    assert 'data-id="W-2"' in page and 'data-findings="0"' in page
