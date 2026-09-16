"""
Run the agent graph over events Lane A produced.

    python scripts/run_agents.py events/run1/events/*.json --db safety.db

Lane A writes event JSON; this reads it and runs the three agents over each one. It is
the seam between the two halves of the system, and the thing Lane D's API will wrap.

Identity is supplied here, never inferred. A supervisor who has recognised someone
passes it explicitly:

    python scripts/run_agents.py evt.json --db safety.db --bind W-0412 --by khalid

Without that, the case runs to a decision and is recorded unattributed -- which is the
correct outcome, not a degraded one.

Exit codes: 0 all cases closed, 1 one or more raised an alert (a write that did not
land), 2 nothing could be read.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from agents.graph import run_case  # noqa: E402
from app.db import EventStore, Worker  # noqa: E402

BANNER = {"done": "closed", "review": "sent to review", "parked": "parked",
          "alert": "ALERT"}


def show(case, outcome: str, reason: str, verbose: bool) -> None:
    event = case.event
    print(f"\n{'=' * 72}")
    print(f"{event['event_id']}  |  {event['camera_id']}  |  "
          f"{event.get('zone', {}).get('name', '?')}")
    print(f"missing: {', '.join(event.get('ppe', {}).get('missing', [])) or 'nothing'}")
    print(f"{'-' * 72}")

    if verbose and case.trace:
        print("tools each agent chose:")
        for step in case.trace:
            print(f"  {step.agent:<12} {step.tool:<22} {step.result_summary[:44]}")
        print(f"{'-' * 72}")

    if case.decision:
        d = case.decision
        print(f"action    {d.action.value}")
        print(f"severity  {d.severity.total}   priors {d.prior_violations}")
        awaiting = "REQUIRED before anything is sent" if d.requires_approval else "not required"
        print(f"approval  {awaiting}")
        if case.draft and case.draft.citations:
            print("cited     " + ", ".join(
                f"29 CFR {c.clause_id}{' (site policy)' if c.is_site_policy else ''}"
                for c in case.draft.citations))
        if d.draft_body:
            print(f"\nto the supervisor:\n  {d.draft_body}")
        if d.rationale:
            print(f"\nwhy:\n  {d.rationale}")

    print(f"\n-> {BANNER.get(outcome, outcome)}"
          + (f": {reason}" if reason else ""))
    if case.blocked_on:
        print(f"   blocked on: {case.blocked_on.value}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events", nargs="+", help="event JSON files (globs accepted)")
    ap.add_argument("--db", default="safety.db")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--bind", help="worker id a supervisor has identified")
    ap.add_argument("--by", default="", help="who made that identification")
    ap.add_argument("--add-worker", action="append", default=[], metavar="ID:Name",
                    help="put someone on the roster first")
    ap.add_argument("-q", "--quiet", action="store_true", help="hide the tool trace")
    args = ap.parse_args(argv)

    paths: list[Path] = []
    for pattern in args.events:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    if not paths:
        print("no event files matched", file=sys.stderr)
        return 2

    if args.bind and not args.by:
        ap.error("--bind needs --by: an identification has to have someone behind it")

    # The key lives in .env, which is gitignored. Loaded here rather than in the
    # agents so the test suite keeps running with no credentials at all.
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("no OPENAI_API_KEY -- put it in .env at the repo root", file=sys.stderr)
        return 2

    db = EventStore(args.db)
    for spec in args.add_worker:
        worker_id, _, name = spec.partition(":")
        db.add_worker(Worker(worker_id.strip(), name.strip() or worker_id.strip()))

    alerts = 0
    for path in paths:
        event = json.loads(path.read_text(encoding="utf-8"))
        if args.bind:
            event.setdefault("subject", {})["worker_ref"] = args.bind

        out = run_case(event, db, model=args.model,
                       bound_by=args.by or None)
        show(out["case"], out.get("outcome", ""), out.get("reason", ""),
             verbose=not args.quiet)
        alerts += out.get("outcome") == "alert"

    print(f"\n{len(paths)} event(s), {alerts} alert(s)")
    return 1 if alerts else 0


if __name__ == "__main__":
    raise SystemExit(main())
