"""
The supervisor console.

    streamlit run app/console.py

Talks to the FastAPI service over HTTP and holds no state of its own. That matters more
than it looks: Streamlit re-executes this entire file on every click, so anything
expensive or stateful living here would be rebuilt dozens of times a session. The API
owns the database, the model and the index; this file only renders.

Four tabs, in the order a supervisor actually works:

    Identify   findings nobody has put a name to      -- the queue that gates history
    Approve    escalations waiting on a signature     -- nothing here has been sent
    Events     everything recorded, with its trace    -- the audit view
    Report     zones and repeat offenders             -- deterministic, no agent

The Approve tab is the one that matters in the demo. It is where a human stays in the
loop, and it is deliberately built so that the evidence, the citation and the draft are
all on screen *before* the button -- approving something you have not seen is exactly the
habit this system should not create.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import httpx
import streamlit as st

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

API = os.getenv("SAFETY_API", "http://127.0.0.1:8000")
TIMEOUT = 60.0          # judging an event is a model call; it is not instant

# ANSI Z535 safety colours. They are not decoration: a site already reads red as stop
# and yellow as caution, so the severity bands inherit a meaning the viewer has before
# they read the label. Theme is pinned in .streamlit/config.toml.
BAND_COLOUR = {"stop_work": "#C8102E", "escalation": "#E35205",
               "warning": "#B08400", "log_only": "#6B7887", "no_action": "#00843D"}

STYLE = """
<style>
  :root {
    --ink:      #10151C;
    --ink-2:    #3D4855;
    --ink-3:    #6B7887;
    --line:     #D3DAE2;
    --surface:  #FFFFFF;
    --surface-2:#F6F8FA;
    --accent:   #005EB8;
  }

  /* Streamlit's default top padding wastes a third of the fold. */
  .block-container { padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1320px; }

  h1, h2, h3 { letter-spacing: -.015em; }
  h1 { font-size: 1.85rem !important; font-weight: 700 !important; margin-bottom: .1rem !important; }
  h2 { font-size: 1.15rem !important; font-weight: 600 !important; }
  h3 { font-size: 1.02rem !important; font-weight: 600 !important; }

  /* Tabs: a rule under the row, weight on the selected one. */
  .stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid var(--line); }
  .stTabs [data-baseweb="tab"] {
    height: 42px; padding: 0 18px;
    font-size: .94rem; font-weight: 500; color: var(--ink-3);
  }
  .stTabs [aria-selected="true"] { color: var(--accent); font-weight: 600; }

  /* One card style, used for every queue row. */
  .card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-left: 3px solid var(--accent);
    border-radius: 7px;
    padding: 15px 18px 6px;
    margin-bottom: 14px;
  }
  .card.urgent   { border-left-color: #C8102E; }
  .card.caution  { border-left-color: #E35205; }
  .card.quiet    { border-left-color: #AEB9C6; }

  .card-head {
    display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
    margin-bottom: 2px;
  }
  .card-title { font-size: 1rem; font-weight: 600; color: var(--ink); }
  .card-meta  { font-size: .8rem; color: var(--ink-3); font-variant-numeric: tabular-nums; }

  .chip {
    display: inline-block; padding: 2px 9px; border-radius: 3px;
    font-size: .68rem; font-weight: 700; letter-spacing: .06em;
    text-transform: uppercase; color: #fff; white-space: nowrap;
  }
  .chip.ghost { background: transparent; color: var(--ink-3);
                border: 1px solid var(--line); font-weight: 600; }

  /* Section labels above a block of controls. */
  .label {
    font-size: .68rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: var(--ink-3); margin: 14px 0 4px;
  }

  /* The drafted notice. It is the thing a supervisor reads before signing, so it gets
     to look like a document rather than an alert. */
  .draft {
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 13px 16px;
    font-size: .93rem; line-height: 1.6; color: var(--ink);
  }

  .stMetric { background: var(--surface-2); border: 1px solid var(--line);
              border-radius: 7px; padding: 12px 14px; }
  [data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; }
  [data-testid="stMetricLabel"] { font-size: .72rem; letter-spacing: .07em;
                                  text-transform: uppercase; color: var(--ink-3); }

  section[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
  section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

  .stButton > button { border-radius: 6px; font-weight: 600; }
  div[data-testid="stImage"] img { border-radius: 6px; border: 1px solid var(--line); }
  hr { margin: .9rem 0; border-color: var(--line); }
  #MainMenu, footer { visibility: hidden; }
</style>
"""


# ------------------------------------------------------------------ transport

def api(method: str, path: str, **kwargs):
    """One place to talk to the service, and one place for it to be down.

    A console that raises a traceback onto the screen when the API is not running is a
    console that fails in front of the room. It says what is wrong and what to do.
    """
    try:
        r = httpx.request(method, f"{API}{path}", timeout=TIMEOUT, **kwargs)
    except httpx.ConnectError:
        st.error(f"No service at {API}.\n\n"
                 "Start it with:  `python -m uvicorn app.api:app`")
        st.stop()
    except httpx.ReadTimeout:
        st.warning("The service did not answer in time. If it was judging an event, the "
                   "model call may still be running.")
        return None
    if r.status_code >= 500:
        st.error(f"{path} failed: {r.status_code}")
        return None
    return r


def get(path: str, default=None):
    r = api("GET", path)
    return r.json() if r is not None and r.status_code == 200 else default


def evidence_bytes(event_id: str, kind: str) -> bytes | None:
    r = api("GET", f"/evidence/{event_id}/{kind}")
    return r.content if r is not None and r.status_code == 200 else None


# ------------------------------------------------------------------ fragments

def chip(text: str, colour: str | None = None) -> str:
    if colour is None:
        return f"<span class='chip ghost'>{text}</span>"
    return f"<span class='chip' style='background:{colour}'>{text}</span>"


def severity_chip(action: str | None, severity=None) -> str:
    if not action:
        return chip("undecided")
    score = f" · {severity:g}" if severity is not None else ""
    return chip(f"{action.replace('_', ' ')}{score}", BAND_COLOUR.get(action, "#6B7887"))


def card_head(title: str, chips: list[str], meta: str = "") -> None:
    """One header treatment for every queue row, so the eye learns it once."""
    st.markdown(
        f"<div class='card-head'><span class='card-title'>{title}</span>"
        + "".join(chips)
        + (f"<span class='card-meta'>{meta}</span>" if meta else "")
        + "</div>",
        unsafe_allow_html=True)


def label(text: str) -> None:
    st.markdown(f"<div class='label'>{text}</div>", unsafe_allow_html=True)


def show_evidence(event_id: str, columns=2):
    """The crop and the frame side by side. A finding a supervisor cannot check for
    themselves is not auditable, so this is not decoration."""
    crop = evidence_bytes(event_id, "crop")
    frame = evidence_bytes(event_id, "frame")
    if not crop and not frame:
        st.caption("No evidence stored for this finding.")
        return
    cols = st.columns(columns)
    if crop:
        cols[0].image(crop, caption="the worker", width="stretch")
    if frame and columns > 1:
        cols[1].image(frame, caption="the scene, annotated", width="stretch")


def show_trace(event_id: str):
    trace = (get(f"/events/{event_id}/trace") or {}).get("trace")
    if not trace:
        st.caption("No trace recorded.")
        return
    st.caption("Every tool each agent chose, in the order it chose them. "
               "Nobody scripted this sequence.")
    st.dataframe(
        [{"#": s["step_index"], "agent": s["agent"], "tool": s["tool"],
          "result": (s["result"] or "")[:80]} for s in trace],
        hide_index=True, width="stretch")


# ------------------------------------------------------------------ the tabs

def tab_identify():
    st.subheader("Findings waiting to be identified")
    st.caption("The vision system never says who someone is. Until a supervisor does, "
               "a finding counts toward nobody's history — which is why this queue "
               "gates every repeat-offender escalation.")

    queue = get("/queue/identification", {}).get("events", [])
    roster = get("/roster", {}).get("workers", [])

    if not queue:
        st.success("Nothing waiting. Every recorded finding has a name against it.")
        return
    if not roster:
        st.warning("The roster is empty, so nobody can be identified yet. "
                   "Add people in the sidebar.")

    for row in queue:
        missing = ", ".join(row["missing"]) or "nothing"
        chips = [chip("confirmed", "#C8102E") if row["actionable"]
                 else chip("not confirmed")]

        with st.container(border=True):
            card_head(row["zone"], chips,
                      f"{row['camera_id']} · {row['captured_at'][:16]}")
            st.markdown(f"<span class='card-meta'>missing {missing}</span>",
                        unsafe_allow_html=True)

            left, right = st.columns([2, 3], gap="large")
            with left:
                show_evidence(row["event_id"], columns=1)
            with right:
                if not row["actionable"]:
                    st.caption("Not temporally confirmed, so this can never become an "
                               "accusation — only a record.")
                if roster:
                    names = {f"{w['name']} · {w['worker_id']}": w["worker_id"]
                             for w in roster}
                    label("identify")
                    # A dropdown, never a text box. Typed identity is how one person's
                    # history splits across two spellings.
                    choice = st.selectbox("Who was this?", list(names),
                                          key=f"who_{row['event_id']}",
                                          label_visibility="collapsed")
                    who = st.text_input("Your name", key=f"by_{row['event_id']}",
                                        placeholder="your name, e.g. supervisor:khalid",
                                        label_visibility="collapsed")
                    if st.button("Confirm identity", key=f"bind_{row['event_id']}",
                                 type="primary", disabled=not who.strip(),
                                 width="stretch"):
                        r = api("POST", f"/events/{row['event_id']}/identity",
                                json={"worker_id": names[choice],
                                      "bound_by": who.strip()})
                        if r is not None and r.status_code == 200:
                            st.success(f"Attributed to {choice}.")
                            st.rerun()
                        elif r is not None:
                            st.error(r.json().get("detail", "refused"))


def tab_approve():
    st.subheader("Waiting for a human")
    st.caption("Nothing on this page has been sent. The system drafts; a supervisor "
               "decides. Escalations and stop-work always land here.")

    queue = get("/queue/approvals", {}).get("decisions", [])
    if not queue:
        st.success("Nothing awaiting approval.")
        return

    for row in queue:
        detail = get(f"/events/{row['event_id']}")
        decision = (detail or {}).get("decision") or {}

        with st.container(border=True):
            card_head(row["zone"],
                      [severity_chip(row["action"], row["severity"])],
                      row["captured_at"][:16])

            left, right = st.columns([3, 2], gap="large")
            with right:
                show_evidence(row["event_id"], columns=1)
            with left:
                if decision.get("draft_body"):
                    label("drafted for you to send")
                    st.markdown(f"<div class='draft'>{decision['draft_body']}</div>",
                                unsafe_allow_html=True)

                if decision.get("citations"):
                    import json as _json
                    cites = decision["citations"]
                    if isinstance(cites, str):
                        cites = _json.loads(cites)
                    label("cited")
                    st.markdown(" ".join(chip(f"29 CFR {c}") for c in cites),
                                unsafe_allow_html=True)

                if decision.get("rationale"):
                    with st.expander("Why the agent proposed this"):
                        st.write(decision["rationale"])
                with st.expander("What the agents did"):
                    show_trace(row["event_id"])

                label("sign off")
                who = st.text_input("Approving as", key=f"ap_by_{row['event_id']}",
                                    placeholder="your name, e.g. supervisor:khalid",
                                    label_visibility="collapsed")
                c1, c2 = st.columns([2, 1])
                if c1.button("Approve and send", key=f"ap_{row['event_id']}",
                             type="primary", disabled=not who.strip(),
                             width="stretch"):
                    r = api("POST", f"/decisions/{row['event_id']}/approve",
                            json={"approved_by": who.strip()})
                    if r is not None and r.status_code == 200:
                        st.success("Approved.")
                        st.rerun()
                    elif r is not None:
                        st.error(r.json().get("detail", "refused"))
                c2.button("Later", key=f"skip_{row['event_id']}",
                          width="stretch")


def tab_events():
    st.subheader("Everything recorded")
    rows = get("/events", {}).get("events", [])
    if not rows:
        st.info("No findings yet. Replay some from the sidebar.")
        return

    st.dataframe(
        [{"when": r["captured_at"][:16], "camera": r["camera_id"], "zone": r["zone"],
          "missing": ", ".join(r["missing"]) or "—",
          "confirmed": bool(r["actionable"]),
          "action": r["action"] or "—",
          # A bare "None" in a column reads as a fault. An undecided finding is not one.
          "severity": f"{r['severity']:g}" if r["severity"] is not None else "—",
          "worker": r["worker_id"] or "unattributed",
          "approved by": r["approved_by"] or "—"} for r in rows],
        hide_index=True, width="stretch")

    chosen = st.selectbox("Inspect one", [r["event_id"] for r in rows])
    if chosen:
        detail = get(f"/events/{chosen}") or {}
        c1, c2 = st.columns([2, 3])
        with c1:
            show_evidence(chosen, columns=1)
        with c2:
            st.json(detail.get("decision") or {"decision": "none recorded"},
                    expanded=False)
        show_trace(chosen)


def tab_report():
    st.subheader("The week")
    st.caption("Deterministic aggregation, not an agent. Zone counts need no identity "
               "at all — and are arguably the more actionable number, because you fix "
               "the zone, not the person.")

    days = st.slider("Days", 1, 90, 7)
    report = get(f"/report?days={days}") or {}

    stats = report.get("stats", {})
    cols = st.columns(4)
    cols[0].metric("Findings", stats.get("events", 0))
    cols[1].metric("Confirmed", stats.get("actionable", 0))
    cols[2].metric("Unattributed", stats.get("unattributed", 0))
    cols[3].metric("Decisions", stats.get("decisions", 0))

    by_zone = report.get("by_zone", [])
    if not by_zone:
        st.info("Nothing in this window.")
    else:
        label("where findings happen")

        # Label by zone, and add the camera only where a zone name is ambiguous.
        # "cam_3/walkway" and "d_view02/walkway" are genuinely different places, so the
        # camera cannot always be dropped -- but prefixing every row with it pushes the
        # part that matters off the end of the axis.
        seen = Counter(r["zone"] for r in by_zone)
        rows = [{"zone": (r["zone"] if seen[r["zone"]] == 1
                          else f"{r['zone']} · {r['camera_id']}"),
                 "confirmed": r["confirmed"] or 0,
                 "all findings": r["n"]} for r in by_zone]

        # A bar chart of one category is a rectangle, and a rectangle is not a finding.
        # Horizontal bars because zone names are words, not codes -- vertical ones turn
        # "d_view02_test / walkway" into rotated, truncated nonsense.
        if len(rows) > 1:
            st.bar_chart(rows, x="zone", y="confirmed", horizontal=True,
                         height=max(150, 46 * len(rows)), color="#005EB8")
        st.dataframe(rows, hide_index=True, width="stretch")

    repeats = report.get("repeat_offenders", [])
    st.markdown("**Repeat offenders**")
    if repeats:
        st.dataframe(repeats, hide_index=True, width="stretch")
    else:
        st.caption("None — which may mean nobody repeated, or may mean findings have "
                   "not been identified yet. Those are different things, and the "
                   "Identify tab is where the second one is fixed.")


# ------------------------------------------------------------------ sidebar

def sidebar():
    st.sidebar.title("Safety Supervisor")
    health = get("/health")
    if health:
        st.sidebar.success(f"service up · {health['events']} findings")
        # The filename, not the path. A wrapped absolute path is four lines of noise
        # for something nobody reads except when it is wrong.
        st.sidebar.caption(f"`{Path(health['db']).name}`", help=health["db"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Replay footage")
    st.sidebar.caption("Posts events Lane A already produced, through the real API and "
                       "the real agents.")
    pattern = st.sidebar.text_input("Event files", "events/run1/events/*.json")
    judge = st.sidebar.checkbox("Judge with the agents", value=True,
                                help="Unticked just records the findings and spends "
                                     "nothing. This is a setting — the button below "
                                     "is what runs it.")
    # The label says what the button will do, because a tickbox called "run the agents"
    # reads as the action and gets clicked instead of the button.
    if st.sidebar.button("Replay and judge" if judge else "Replay without judging",
                         type="primary", width="stretch"):
        import glob
        import json
        paths = sorted(glob.glob(str(ROOT / pattern)))
        if not paths:
            st.sidebar.error("No files matched.")
        else:
            bar = st.sidebar.progress(0.0, "posting…")
            done = 0
            for i, path in enumerate(paths, 1):
                event = json.loads(Path(path).read_text(encoding="utf-8"))
                r = api("POST", f"/events?judge={str(judge).lower()}", json=event)
                done += r is not None and r.status_code == 201
                bar.progress(i / len(paths), f"{i}/{len(paths)}")
            st.sidebar.success(f"{done} of {len(paths)} accepted")
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Roster")
    with st.sidebar.form("add_worker", clear_on_submit=True):
        wid = st.text_input("Worker id", placeholder="W-0412")
        name = st.text_input("Name")
        if st.form_submit_button("Add") and wid.strip() and name.strip():
            api("POST", "/roster", json={"worker_id": wid.strip(), "name": name.strip()})
            st.rerun()
    for w in get("/roster", {}).get("workers", []):
        st.sidebar.caption(f"{w['name']} · `{w['worker_id']}`")


def main():
    st.set_page_config(page_title="Safety Supervisor", page_icon="🦺", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    sidebar()

    st.title("Supervisor console")
    st.markdown(
        "<div class='card-meta' style='margin:-2px 0 18px'>"
        "PPE findings, judged against 29 CFR 1926 and cited. "
        "Nothing leaves this screen without a signature."
        "</div>", unsafe_allow_html=True)

    counts_identify = len(get("/queue/identification", {}).get("events", []))
    counts_approve = len(get("/queue/approvals", {}).get("decisions", []))

    identify, approve, events, report = st.tabs([
        f"Identify{f'  ·  {counts_identify}' if counts_identify else ''}",
        f"Approve{f'  ·  {counts_approve}' if counts_approve else ''}",
        "Events", "Report"])
    with identify:
        tab_identify()
    with approve:
        tab_approve()
    with events:
        tab_events()
    with report:
        tab_report()


main()
