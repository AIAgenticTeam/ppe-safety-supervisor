"""
The supervisor console.

Streamlit ships `AppTest`, which executes the real script and exposes what it rendered,
so these are not mocks of the UI -- they run the console against the real FastAPI app
with a stub model behind it. `httpx.request` is redirected to the API's own test client,
which means the console's HTTP calls hit the actual routes without a socket.

What is worth testing here is not the styling. It is the two claims the console makes to
a supervisor: that nothing has been sent without a signature, and that identity is picked
rather than typed.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).absolute().parent))

pytest.importorskip("streamlit", reason="needs streamlit; install into .venv")
pytest.importorskip("fastapi", reason="needs fastapi; install into .venv")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from app.api import create_app  # noqa: E402
from app.db import EventStore  # noqa: E402
from test_agents import (StubCall, StubClient, _adjudicator_script,  # noqa: E402
                         confirmed_event)

CONSOLE = str(ROOT / "app" / "console.py")


def script(action):
    return [[StubCall("submit_draft", {
        "summary": "A worker was without head protection at the grinding station.",
        "clause_ids": ["1926.100"]})]] + _adjudicator_script(action)


@pytest.fixture
def console(tmp_path, monkeypatch):
    """A console wired to a real API over a real client, with a stub model."""
    def _make(action="warning", seed=True, roster=True):
        app = create_app(tmp_path / "c.db", client=StubClient(script(action)))
        client = TestClient(app)

        def routed(method, url, **kwargs):
            kwargs.pop("timeout", None)
            return client.request(method, str(url).replace("http://127.0.0.1:8000", ""))\
                if method == "GET" else client.request(
                    method, str(url).replace("http://127.0.0.1:8000", ""),
                    json=kwargs.get("json"))

        monkeypatch.setattr(httpx, "request", routed)
        if roster:
            client.post("/roster", json={"worker_id": "W-1", "name": "Alpha"})
        if seed:
            client.post("/events", json=confirmed_event())

        at = AppTest.from_file(CONSOLE, default_timeout=30)
        at.run()
        return at
    return _make


def text_of(at) -> str:
    """Everything the page rendered, flattened -- markdown, captions, successes."""
    parts = []
    for attr in ("markdown", "caption", "success", "info", "warning", "error",
                 "subheader", "title", "header"):
        parts += [getattr(el, "value", "") for el in getattr(at, attr, [])]
    return "\n".join(str(p) for p in parts)


# --------------------------------------------------------------- it renders

def test_the_console_runs_without_an_exception(console):
    at = console()
    assert not at.exception


def test_the_four_tabs_are_there(console):
    at = console()
    labels = " ".join(t.label for t in at.tabs)
    for name in ("Identify", "Approve", "Events", "Report"):
        assert name in labels


def test_queue_counts_appear_on_the_tabs(console):
    """A supervisor should see there is work without opening anything."""
    at = console()
    assert any("Identify" in t.label and "1" in t.label for t in at.tabs)


# ------------------------------------------------- identity is picked, not typed

def test_identity_is_a_dropdown_off_the_roster(console):
    """Typed identity is how one person's history splits across two spellings, so the
    console must not offer a free-text box for who someone was."""
    at = console()
    options = [o for s in at.selectbox for o in s.options]
    assert any("W-1" in str(o) for o in options)


def test_with_an_empty_roster_the_console_says_so_rather_than_offering_a_box(console):
    at = console(roster=False)
    assert "roster is empty" in text_of(at).lower()


def test_confirming_an_identity_needs_a_name_behind_it(console):
    """An attribution with no author is an accusation with no author, so the button is
    disabled until the supervisor says who they are."""
    at = console()
    confirm = [b for b in at.button if "Confirm identity" in b.label]
    assert confirm and confirm[0].disabled


# ------------------------------------------------------- nothing is sent alone

def test_an_escalation_shows_its_draft_evidence_and_citation_before_the_button(console):
    """Approving something you have not seen is the habit this console must not create."""
    at = console("escalation")
    approve_tab = next(t for t in at.tabs if "Approve" in t.label)
    rendered = text_of(at)
    assert "drafted for you to send" in rendered.lower()
    assert "1926.100" in rendered
    assert approve_tab is not None


def test_the_approve_button_is_disabled_until_someone_signs(console):
    at = console("escalation")
    approve = [b for b in at.button if "Approve and send" in b.label]
    assert approve and approve[0].disabled


def test_a_warning_never_reaches_the_approval_queue(console):
    at = console("warning")
    assert "nothing awaiting approval" in text_of(at).lower()


def test_the_console_says_plainly_that_nothing_has_been_sent(console):
    at = console("escalation")
    assert "has been sent" in text_of(at).lower()


# ------------------------------------------------------------ the report

def test_the_report_is_labelled_as_deterministic_not_an_agent(console):
    """The weekly view is SQL. Letting it read as agent output would overstate the
    system in the one place nobody checks."""
    at = console()
    assert "not an agent" in text_of(at).lower()


def test_no_repeat_offenders_is_distinguished_from_nobody_identified(console):
    """Those are different facts and the empty state must not conflate them."""
    at = console()
    rendered = text_of(at).lower()
    assert "not been identified" in rendered


# ------------------------------------------------- the service being down

def test_a_dead_service_gives_an_instruction_not_a_traceback(monkeypatch):
    """A console that throws a stack trace onto the projector is a console that fails
    in front of the room."""
    def refuse(*a, **k):
        raise httpx.ConnectError("nothing listening")

    monkeypatch.setattr(httpx, "request", refuse)
    at = AppTest.from_file(CONSOLE, default_timeout=30)
    at.run()

    assert not at.exception, "a down service must not raise"
    errors = " ".join(str(e.value) for e in at.error)
    assert "uvicorn app.api:app" in errors


# ------------------------------------------------------------- monitoring

def test_the_monitoring_tab_exists(console):
    at = console()
    assert any("Monitoring" in t.label for t in at.tabs)


def test_too_little_data_reads_as_a_refusal_not_a_pass(console):
    """The distinction that matters on this tab. "Cannot tell" and "nothing wrong"
    must not look the same to a supervisor, or an unmonitored system reads as a
    healthy one."""
    at = console()                      # one event: far below the drift minimum
    rendered = text_of(at).lower()
    assert "not enough data" in rendered or "no drift check" in rendered
    assert "refusal, not a pass" in rendered


def test_the_tab_says_what_it_is_watching_for(console):
    at = console()
    assert "empty queue looks exactly like" in text_of(at).lower()
