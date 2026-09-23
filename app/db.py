"""
The event store. SQLite, because a demo that needs a database server is a demo that
fails on the night.

Three tables and one rule: **identity is nullable and arrives late.**

    workers       the roster. Entered once, picked from a dropdown, never typed.
    events        every finding, attributed or not.
    decisions     what was decided, by whom, and whether a human approved it.

`events.worker_id` is NULL until a supervisor binds it. That is not a gap to be filled
with a guess -- it is the safety property. An unattributed event counts toward nobody's
history, so the system cannot escalate against someone on the basis of incidents nobody
confirmed were them.

The Recorder writes here and the Adjudicator reads here on the NEXT event. That loop is
the system's memory, and its single point of failure: an unwritten record silently turns
a third offence into a first, with nothing in the output looking wrong. Every write is
therefore verified by reading it back.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# Under data/ (gitignored) so a run never leaves a database in the repo root.
DEFAULT_DB_PATH = "data/safety.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS workers (
    worker_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT,
    role        TEXT,
    active      INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS events (
    event_id     TEXT PRIMARY KEY,
    captured_at  TEXT NOT NULL,
    camera_id    TEXT NOT NULL,
    zone         TEXT NOT NULL,
    status       TEXT NOT NULL,
    missing      TEXT NOT NULL DEFAULT '[]',
    actionable   INTEGER NOT NULL DEFAULT 0,
    -- NULL until a supervisor binds it. Never inferred, never guessed.
    worker_id    TEXT REFERENCES workers(worker_id),
    bound_by     TEXT,
    bound_at     TEXT,
    payload      TEXT NOT NULL,
    recorded_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    event_id     TEXT PRIMARY KEY REFERENCES events(event_id),
    action       TEXT NOT NULL,
    severity     REAL NOT NULL,
    band         TEXT NOT NULL,
    confidence   REAL,
    citations    TEXT NOT NULL DEFAULT '[]',
    rationale    TEXT,
    draft_body   TEXT,
    requires_approval INTEGER NOT NULL DEFAULT 0,
    approved_by  TEXT,
    approved_at  TEXT,
    sent_at      TEXT,
    decided_at   TEXT NOT NULL
);

-- Every tool each agent chose, in order. Week 6 is graded on observability, but the
-- reason to keep it is narrower: when a decision looks wrong, the only way to find out
-- why is to see what the agent actually asked for and what came back. The trace lived
-- in CaseState and died with the process, which meant the answer was never available
-- for the one case anybody wanted to examine.
CREATE TABLE IF NOT EXISTS trace_steps (
    event_id     TEXT NOT NULL REFERENCES events(event_id),
    step_index   INTEGER NOT NULL,
    agent        TEXT NOT NULL,
    tool         TEXT NOT NULL,
    arguments    TEXT NOT NULL DEFAULT '{}',
    result       TEXT NOT NULL DEFAULT '',
    latency_ms   INTEGER NOT NULL DEFAULT 0,
    at           TEXT NOT NULL,
    PRIMARY KEY (event_id, step_index)
);

CREATE INDEX IF NOT EXISTS idx_events_worker ON events(worker_id, captured_at);
CREATE INDEX IF NOT EXISTS idx_events_zone   ON events(camera_id, zone, captured_at);
CREATE INDEX IF NOT EXISTS idx_events_recent ON events(recorded_at DESC);
"""


@dataclass(frozen=True)
class Worker:
    worker_id: str
    name: str
    email: str = ""
    role: str = ""


class EventStore:
    """Every method is a plain query. Nothing here is a judgement call."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ---- roster ---------------------------------------------------------

    def add_worker(self, worker: Worker) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO workers (worker_id, name, email, role, active) "
                "VALUES (?, ?, ?, ?, 1)",
                (worker.worker_id, worker.name, worker.email, worker.role))
            conn.commit()

    def add_workers(self, workers: list[Worker]) -> int:
        """Write a whole roster in one transaction, so a failure part-way leaves the
        roster as it was rather than half imported. Same upsert as `add_worker`."""
        with closing(self._connect()) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO workers (worker_id, name, email, role, active) "
                "VALUES (?, ?, ?, ?, 1)",
                [(w.worker_id, w.name, w.email, w.role) for w in workers])
            conn.commit()
        return len(workers)

    def finding_counts(self) -> dict[str, int]:
        """How many findings are attributed to each worker. Tells the roster page which
        removals will delete someone and which can only hide them."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT worker_id, COUNT(*) AS n FROM events "
                "WHERE worker_id IS NOT NULL GROUP BY worker_id").fetchall()
        return {r["worker_id"]: r["n"] for r in rows}

    def remove_workers(self, worker_ids: list[str]) -> dict[str, str]:
        """Take people off the roster. Returns {worker_id: "deleted" | "deactivated"} for
        each one that was on it; ids that are not (or already removed) are left out.

        Someone with findings attributed to them is deactivated, not deleted: their history
        belongs to the record, and the reports read their name from this table. Deactivated
        people vanish from the roster and the identify dropdown, cannot be newly attributed,
        and come back if their id is imported again. Someone with no findings is deleted.

        One transaction, so a failure part-way leaves the roster as it was."""
        outcome: dict[str, str] = {}
        with closing(self._connect()) as conn:
            for wid in dict.fromkeys(worker_ids):        # de-duplicated, order kept
                row = conn.execute("SELECT active FROM workers WHERE worker_id = ?",
                                   (wid,)).fetchone()
                if row is None or not row["active"]:
                    continue
                has_findings = conn.execute(
                    "SELECT 1 FROM events WHERE worker_id = ? LIMIT 1", (wid,)).fetchone()
                if has_findings:
                    conn.execute("UPDATE workers SET active = 0 WHERE worker_id = ?", (wid,))
                    outcome[wid] = "deactivated"
                else:
                    conn.execute("DELETE FROM workers WHERE worker_id = ?", (wid,))
                    outcome[wid] = "deleted"
            conn.commit()
        return outcome

    def roster(self) -> list[Worker]:
        """What the console's dropdown is built from. Supervisors pick; they never type
        an email, because `ahmed.ali@` and `a.ali@` silently split one person in two."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT worker_id, name, email, role FROM workers "
                "WHERE active = 1 ORDER BY name").fetchall()
        return [Worker(**dict(r)) for r in rows]

    # ---- events ---------------------------------------------------------

    def record_event(self, event: dict) -> str:
        """Store a finding immediately, always unattributed.

        Two things this deliberately does not do.

        It does not copy `subject.worker_ref` into `worker_id`. That field holds whatever
        the tracker or the caller put there, and binding it here would attribute a finding
        to a person with no `bound_by` and no `bound_at` -- an accusation with no author.
        `attach_identity` is the only route to a name, because it is the only route that
        checks the roster and records who said so.

        It also does not lose the event when that reference is unusable. The raw
        `worker_ref` survives in the payload for a supervisor to confirm or reject. A
        confirmed violation must never vanish because its label was wrong.

        Re-recording an event_id refreshes the finding but preserves any binding a
        supervisor has already made; re-running a video must not erase their work.
        """
        event_id = event["event_id"]
        with closing(self._connect()) as conn:
            bound = conn.execute(
                "SELECT worker_id, bound_by, bound_at FROM events WHERE event_id = ?",
                (event_id,)).fetchone()
            conn.execute(
                "INSERT OR REPLACE INTO events "
                "(event_id, captured_at, camera_id, zone, status, missing, actionable, "
                " worker_id, bound_by, bound_at, payload, recorded_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (event_id,
                 event["captured_at"],
                 event["camera_id"],
                 event.get("zone", {}).get("name", ""),
                 event.get("status", ""),
                 json.dumps(event.get("ppe", {}).get("missing", [])),
                 int(event.get("status") == "violation"
                     and event.get("confirmation", {}).get("confirmed", False)),
                 bound["worker_id"] if bound else None,
                 bound["bound_by"] if bound else None,
                 bound["bound_at"] if bound else None,
                 json.dumps(event),
                 datetime.now().isoformat(timespec="seconds")))
            conn.commit()
        return event_id

    def get_event(self, event_id: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT payload FROM events WHERE event_id = ?",
                               (event_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def attach_identity(self, event_id: str, worker_id: str, bound_by: str) -> bool:
        """A supervisor says who this was. The one act the vision system never performs.

        Returns False if the worker is not on the roster -- a free-text id would split
        one person's history across spellings.
        """
        with closing(self._connect()) as conn:
            known = conn.execute("SELECT 1 FROM workers WHERE worker_id = ? AND active = 1",
                                 (worker_id,)).fetchone()
            if not known:
                return False
            conn.execute(
                "UPDATE events SET worker_id = ?, bound_by = ?, bound_at = ? "
                "WHERE event_id = ?",
                (worker_id, bound_by, datetime.now().isoformat(timespec="seconds"),
                 event_id))
            conn.commit()
        return True

    def violations_for(self, worker_id: str, since: datetime | None = None,
                       exclude_event: str | None = None) -> list[dict]:
        """Prior CONFIRMED violations attributed to this worker.

        Unattributed events are invisible here by construction, which is the point.
        """
        sql = ("SELECT event_id, captured_at, camera_id, zone, missing FROM events "
               "WHERE worker_id = ? AND actionable = 1")
        params: list = [worker_id]
        if since:
            sql += " AND captured_at >= ?"
            params.append(since.isoformat(timespec="seconds"))
        if exclude_event:
            sql += " AND event_id != ?"
            params.append(exclude_event)
        sql += " ORDER BY captured_at DESC"

        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [{**dict(r), "missing": json.loads(r["missing"])} for r in rows]

    def unattributed(self, limit: int = 50) -> list[dict]:
        """The supervisor's identification queue, urgent first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT event_id, captured_at, camera_id, zone, missing, actionable "
                "FROM events WHERE worker_id IS NULL "
                "ORDER BY actionable DESC, captured_at DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(r), "missing": json.loads(r["missing"])} for r in rows]

    # ---- decisions ------------------------------------------------------

    def record_decision(self, event_id: str, action: str, severity: float, band: str,
                        citations: list[str], rationale: str = "", draft_body: str = "",
                        confidence: float | None = None,
                        requires_approval: bool = False) -> bool:
        """Write the decision for an event. Returns False, writing nothing, if a human has
        already signed the decision on file.

        This was INSERT OR REPLACE, which deletes the row and writes a new one -- so
        re-judging an event quietly set approved_by back to NULL, put a signed escalation
        back in the queue, and destroyed the record of who had approved it. A signed
        decision is now final; an unsigned one can still be replaced.
        """
        with closing(self._connect()) as conn:
            cur = conn.execute(
                "INSERT INTO decisions "
                "(event_id, action, severity, band, confidence, citations, rationale, "
                " draft_body, requires_approval, decided_at) VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(event_id) DO UPDATE SET "
                "  action=excluded.action, severity=excluded.severity, "
                "  band=excluded.band, confidence=excluded.confidence, "
                "  citations=excluded.citations, rationale=excluded.rationale, "
                "  draft_body=excluded.draft_body, "
                "  requires_approval=excluded.requires_approval, "
                "  decided_at=excluded.decided_at "
                "WHERE decisions.approved_by IS NULL",
                (event_id, action, severity, band, confidence, json.dumps(citations),
                 rationale, draft_body, int(requires_approval),
                 datetime.now().isoformat(timespec="seconds")))
            conn.commit()
        return cur.rowcount > 0

    def get_decision(self, event_id: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM decisions WHERE event_id = ?",
                               (event_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["citations"] = json.loads(d["citations"])
        d["requires_approval"] = bool(d["requires_approval"])
        return d

    def approve(self, event_id: str, approved_by: str) -> bool:
        """A human signs off. Nothing may be sent before this."""
        with closing(self._connect()) as conn:
            cur = conn.execute(
                "UPDATE decisions SET approved_by = ?, approved_at = ? "
                "WHERE event_id = ? AND approved_by IS NULL",
                (approved_by, datetime.now().isoformat(timespec="seconds"), event_id))
            conn.commit()
        return cur.rowcount > 0

    def pending_approval(self) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT d.event_id, d.action, d.severity, d.band, e.zone, e.captured_at "
                "FROM decisions d JOIN events e USING (event_id) "
                "WHERE d.requires_approval = 1 AND d.approved_by IS NULL "
                "ORDER BY d.severity DESC").fetchall()
        return [dict(r) for r in rows]

    def recent(self, limit: int = 50) -> list[dict]:
        """Most recently recorded findings, for the console's main list."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT e.event_id, e.captured_at, e.camera_id, e.zone, e.status, "
                "       e.missing, e.actionable, e.worker_id, e.bound_by, "
                "       d.action, d.severity, d.band, d.requires_approval, d.approved_by "
                "FROM events e LEFT JOIN decisions d USING (event_id) "
                "ORDER BY e.recorded_at DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(r), "missing": json.loads(r["missing"])} for r in rows]

    # ---- the trace ------------------------------------------------------

    def record_trace(self, event_id: str, steps) -> int:
        """Persist what each agent chose to do.

        Replaces any earlier trace for the event rather than appending, so re-running a
        case leaves one account of it instead of two interleaved ones.
        """
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM trace_steps WHERE event_id = ?", (event_id,))
            conn.executemany(
                "INSERT INTO trace_steps "
                "(event_id, step_index, agent, tool, arguments, result, latency_ms, at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [(event_id, i, s.agent, s.tool, json.dumps(s.arguments),
                  s.result_summary, s.latency_ms, s.at)
                 for i, s in enumerate(steps)])
            conn.commit()
        return len(steps)

    def get_trace(self, event_id: str) -> list[dict] | None:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT step_index, agent, tool, arguments, result, latency_ms, at "
                "FROM trace_steps WHERE event_id = ? ORDER BY step_index",
                (event_id,)).fetchall()
        if not rows:
            return None
        return [{**dict(r), "arguments": json.loads(r["arguments"])} for r in rows]

    # ---- verification ---------------------------------------------------

    def verify(self, event_id: str) -> bool:
        """Read the write back.

        The Recorder writes the history the Adjudicator reads next time. An unconfirmed
        write silently turns a third offence into a first, so a case does not close until
        this returns True.
        """
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT 1 FROM events WHERE event_id = ?",
                               (event_id,)).fetchone()
        return row is not None

    # ---- reporting ------------------------------------------------------

    def by_zone(self, days: int = 7) -> list[dict]:
        """Zone-level counts, which need no identity at all -- and are arguably the more
        actionable number: you fix the zone, not the person."""
        since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT camera_id, zone, COUNT(*) AS n, "
                "       SUM(actionable) AS confirmed "
                "FROM events WHERE captured_at >= ? "
                "GROUP BY camera_id, zone ORDER BY confirmed DESC, n DESC",
                (since,)).fetchall()
        return [dict(r) for r in rows]

    def repeat_offenders(self, days: int = 7, minimum: int = 2) -> list[dict]:
        since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT e.worker_id, w.name, COUNT(*) AS violations "
                "FROM events e LEFT JOIN workers w USING (worker_id) "
                "WHERE e.worker_id IS NOT NULL AND e.actionable = 1 "
                "  AND e.captured_at >= ? "
                "GROUP BY e.worker_id HAVING violations >= ? "
                "ORDER BY violations DESC", (since, minimum)).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS events, "
                "       SUM(actionable) AS actionable, "
                "       SUM(worker_id IS NULL) AS unattributed FROM events").fetchone()
            decisions = conn.execute("SELECT COUNT(*) AS n FROM decisions").fetchone()["n"]
            workers = conn.execute(
                "SELECT COUNT(*) AS n FROM workers WHERE active = 1").fetchone()["n"]
        return {**{k: (v or 0) for k, v in dict(row).items()},
                "decisions": decisions, "workers": workers}
