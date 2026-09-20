"""
The supervisor console, as server-rendered pages.

These replace the Streamlit console's tests and keep its claims. What is worth testing is
not the styling but what the pages promise a supervisor: that nothing has been sent without
a signature, that identity is picked rather than typed, and that "cannot tell" never reads
as "nothing wrong". They run the real pages over the real API with a stub model behind it.

The writes themselves (identity, approval) are made by the page's script against the JSON
endpoints, which tests/test_api.py covers. What the pages must get right is the state they
present *before* that script runs -- a disabled button, a dropdown, a draft on screen.

New here, because the pages are new: model output is escaped rather than trusted, the JSON
API keeps returning JSON errors for API clients, replay can only name files it lists, and
no icon-font files are shipped (the template's fonts are separately licensed).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).absolute().parent))

pytest.importorskip("fastapi", reason="needs fastapi; install into .venv")
pytest.importorskip("jinja2", reason="needs jinja2; install into .venv")

from fastapi.testclient import TestClient  # noqa: E402

from app.api import create_app  # noqa: E402
from test_agents import (StubCall, StubClient, _adjudicator_script,  # noqa: E402
                         confirmed_event)

HTML = {"accept": "text/html"}
PAGES = ["/", "/identify", "/approvals", "/findings", "/weekly", "/monitoring",
         "/team", "/about", "/faq"]


def flat(response) -> str:
    """What a reader sees: the page's text with source line-wrapping collapsed."""
    return " ".join(response.text.split())


def script(action):
    return [[StubCall("submit_draft", {
        "summary": "A worker was without head protection at the grinding station.",
        "clause_ids": ["1926.100"]})]] + _adjudicator_script(action)


@pytest.fixture
def web(tmp_path):
    """A real app over a real client, with a stub model. Returns (client, app, event_id)."""
    def _make(action="warning", seed=True, roster=True):
        app = create_app(tmp_path / "w.db", client=StubClient(script(action)))
        client = TestClient(app)
        if roster:
            client.post("/roster", json={"worker_id": "W-1", "name": "Alpha"})
        event_id = None
        if seed:
            event_id = client.post("/events", json=confirmed_event()).json()["event_id"]
        return client, app, event_id
    return _make


# --------------------------------------------------------------- it renders

@pytest.mark.parametrize("path", PAGES)
def test_every_page_renders_with_the_shared_chrome(web, path):
    client, _, _ = web()
    r = client.get(path, headers=HTML)
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    for landmark in ('class="main-header"', 'class="site-footer', "/static/css/app.css"):
        assert landmark in r.text


def test_the_menu_reaches_every_page(web):
    client, _, _ = web()
    page = client.get("/").text
    for href in ("/identify", "/approvals", "/findings", "/weekly", "/monitoring",
                 "/team", "/about", "/faq"):
        assert f'href="{href}"' in page


def test_queue_counts_appear_in_the_header(web):
    """A supervisor should see there is work without opening anything."""
    client, _, _ = web("escalation")
    page = client.get("/").text
    assert 'menu-badge">1<' in page             # the unidentified finding
    assert "Approvals · 1" in page              # the waiting escalation


def test_the_json_api_still_answers_json_at_the_same_paths(web):
    """The pages moved out of the way of the pipeline's contract, not the other way round."""
    client, _, _ = web()
    assert "events" in client.get("/events").json()
    assert "stats" in client.get("/report").json()
    assert client.get("/health").json()["status"] == "ok"


# ------------------------------------------------- identity is picked, not typed

def test_identity_is_a_dropdown_off_the_roster(web):
    """Typed identity is how one person's history splits across two spellings, so the
    page must not offer a free-text box for who someone was. The single text input is the
    supervisor's own name."""
    client, _, _ = web()
    page = client.get("/identify").text
    assert "<select" in page and "Alpha · W-1" in page
    assert page.count('type="text"') == 1


def test_with_an_empty_roster_the_page_says_so_rather_than_offering_a_box(web):
    client, _, _ = web(roster=False)
    page = client.get("/identify").text
    assert "roster is empty" in page.lower()
    assert "<select" not in page


def test_confirming_an_identity_needs_a_name_behind_it(web):
    """An attribution with no author is an accusation with no author, so the button is
    rendered disabled and only the page's script can enable it once a name is typed."""
    client, _, _ = web()
    assert "data-submit disabled" in client.get("/identify").text


def test_an_unconfirmed_finding_says_it_can_never_be_an_accusation(web):
    client, app, _ = web(seed=False)
    event = confirmed_event()
    event["confirmation"]["confirmed"] = False
    client.post("/events?judge=false", json=event)
    assert "can never become an accusation" in client.get("/identify").text


# ------------------------------------------------------- nothing is sent alone

def test_an_escalation_shows_its_draft_and_citation_before_the_button(web):
    """Approving something you have not seen is the habit this page must not create."""
    client, _, _ = web("escalation")
    page = client.get("/approvals").text
    assert "drafted for you to send" in page.lower()
    assert "1926.100" in page
    assert page.index("drafted for you to send") < page.index("Approve and send")


def test_the_approve_button_is_disabled_until_someone_signs(web):
    client, _, _ = web("escalation")
    page = client.get("/approvals").text
    assert "data-submit disabled" in page and "Approve and send" in page


def test_a_warning_never_reaches_the_approval_queue(web):
    client, _, _ = web("warning")
    assert "nothing awaiting approval" in client.get("/approvals").text.lower()


def test_the_page_says_plainly_that_nothing_has_been_sent(web):
    client, _, _ = web("escalation")
    assert "has been sent" in client.get("/approvals").text.lower()


def test_model_output_is_escaped_not_trusted(web):
    """The drafted notice and the rationale are model output. The Streamlit console
    rendered them as HTML; these pages must show them as text."""
    client, app, event_id = web(seed=False)
    event_id = client.post("/events?judge=false", json=confirmed_event()).json()["event_id"]
    app.state.db.record_decision(
        event_id, "escalation", 6.0, "escalation", ["1926.100"],
        rationale="<img src=x onerror=alert(2)>", draft_body="<script>alert(1)</script>",
        requires_approval=True)
    page = client.get("/approvals").text
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<img src=x" not in page
    detail = client.get(f"/findings/{event_id}").text
    assert "<script>alert(1)</script>" not in detail


# ------------------------------------------------------------- the audit view

def test_a_finding_page_shows_its_evidence_decision_and_trace(web):
    client, _, event_id = web("escalation")
    page = client.get(f"/findings/{event_id}").text
    assert f"/evidence/{event_id}/crop" in page and f"/evidence/{event_id}/frame" in page
    assert "The decision" in page and "What the agents did" in page
    assert "waiting for a signature" in page


def test_the_findings_list_links_each_row_to_its_page(web):
    client, _, event_id = web()
    assert f"/findings/{event_id}" in client.get("/findings").text


def test_an_unknown_finding_is_a_404_page_for_a_browser_and_json_for_the_api(web):
    client, _, _ = web()
    browser = client.get("/findings/evt_nope", headers=HTML)
    assert browser.status_code == 404 and "can't find" in browser.text
    api = client.get("/events/evt_nope")            # no text/html in Accept
    assert api.status_code == 404 and api.json() == {"detail": "no such event"}


def test_a_dead_link_gets_the_themed_404_only_in_a_browser(web):
    client, _, _ = web()
    assert "can't find" in client.get("/nowhere", headers=HTML).text
    assert client.get("/nowhere").json() == {"detail": "Not Found"}


# ------------------------------------------------------------ the report

def test_the_report_is_labelled_as_deterministic_not_an_agent(web):
    """The weekly view is SQL. Letting it read as agent output would overstate the
    system in the one place nobody checks."""
    client, _, _ = web()
    assert "not an agent" in client.get("/weekly").text.lower()


def test_no_repeat_offenders_is_distinguished_from_nobody_identified(web):
    """Those are different facts and the empty state must not conflate them."""
    client, _, _ = web()
    assert "not been identified" in client.get("/weekly").text.lower()


def test_the_window_is_bounded(web):
    client, _, _ = web()
    assert client.get("/weekly?days=0", headers=HTML).status_code == 422
    assert client.get("/weekly?days=90", headers=HTML).status_code == 200


# ------------------------------------------------------------- monitoring

def test_the_monitoring_page_says_what_it_is_watching_for(web):
    client, _, _ = web()
    assert "empty queue looks exactly like" in flat(client.get("/monitoring")).lower()


def test_too_little_data_reads_as_a_refusal_not_a_pass(web):
    """"Cannot tell" and "nothing wrong" must not look the same to a supervisor, or an
    unmonitored system reads as a healthy one. The verdict is drawn by the page's script
    from /drift, so this checks both halves: the service refuses, and the script says so."""
    client, _, _ = web()                # one event: far below the drift minimum
    body = client.get("/drift").json()
    assert body["ran"] is False and "not enough data" in body["reason"]
    script_text = client.get("/static/js/app.js").text
    assert "refusal, not a pass" in script_text


# ------------------------------------------------------------- team

def test_the_team_page_lists_the_roster_and_offers_to_add_someone(web):
    client, _, _ = web()
    page = client.get("/team").text
    assert "Alpha" in page and "W-1" in page and "data-roster-form" in page


def test_the_team_page_offers_csv_import_with_a_template(web):
    client, _, _ = web()
    page = client.get("/team").text
    assert "data-import" in page and 'type="file"' in page
    assert 'href="/roster/template.csv"' in page
    assert client.get("/roster/template.csv").status_code == 200


def test_a_large_roster_is_searchable_and_paginated_in_the_page(web):
    """After an import the roster can be thousands long. The page carries a search box and
    a show-all control, and every person is in the page for search to find."""
    client, _, _ = web(roster=False)
    lines = ["worker_id,name"] + [f"W-{i:04d},Person {i}" for i in range(1, 251)]
    client.post("/roster/import", params={"dry_run": "false"},
                content="\n".join(lines).encode(), headers={"content-type": "text/csv"})
    page = client.get("/team").text
    assert "data-roster-filter" in page and "data-roster-more" in page
    assert page.count("data-search=") == 250
    assert "250 people" in page.lower()


# ------------------------------------------------------------- replay

def test_replay_lists_only_its_own_sources(web):
    client, _, _ = web()
    assert len(client.get("/replay/fixtures").json()["files"]) >= 20
    assert client.get("/replay/not-a-source").json()["files"] == []


def test_replay_posts_a_listed_file_through_the_real_intake(web):
    client, _, _ = web(seed=False)
    name = client.get("/replay/fixtures").json()["files"][0]
    r = client.post("/replay/fixtures", params={"name": name, "judge": "false"})
    assert r.status_code == 200 and r.json()["outcome"] == "stored"
    assert client.get("/stats").json()["events"] == 1


@pytest.mark.parametrize("name", ["../../.env", "..\\..\\.env", "/etc/passwd",
                                  "not-a-listed-file.json", "index.json"])
def test_replay_refuses_a_name_it_did_not_list(web, name):
    client, _, _ = web(seed=False)
    r = client.post("/replay/fixtures", params={"name": name, "judge": "false"})
    assert r.status_code == 404
    assert client.get("/stats").json()["events"] == 0


def test_the_findings_page_offers_replay_from_each_source(web):
    client, _, _ = web()
    page = client.get("/findings").text
    assert "data-replay" in page and "Sample events" in page and "Pipeline output" in page


# ------------------------------------------------------------- assets

def test_the_stylesheets_and_scripts_are_served(web):
    client, _, _ = web()
    for path in ("/static/css/app.css", "/static/css/style.css", "/static/js/app.js",
                 "/static/js/theme.js"):
        assert client.get(path).status_code == 200, path


def test_no_icon_font_files_are_shipped():
    """The template's icon fonts (Font Awesome Pro, icomoon) are separately licensed, so
    they are not copied here and icons are inline SVG. This is the guard on that decision."""
    static = ROOT / "app" / "web" / "static"
    fonts = [p for p in static.rglob("*") if p.suffix.lower() in
             {".woff", ".woff2", ".ttf", ".eot", ".otf"}]
    assert fonts == [], f"font files shipped: {fonts}"
    declaring = [c.name for c in static.rglob("*.css")
                 if "@font-face" in c.read_text(encoding="utf-8", errors="ignore")]
    assert declaring == [], f"stylesheets declaring their own fonts: {declaring}"
