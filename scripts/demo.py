"""
Put the system into a known state for a live demonstration.

    python scripts/demo.py --reset          # build the demo database, then print the run sheet
    python scripts/demo.py                  # just print the run sheet

Everything slow or chargeable happens here, in advance. Tracking the 4K pour clip takes
minutes and judging an event is a 10-15 second model call; neither belongs in front of an
audience. What is left live is the part worth watching -- the agents reasoning, and a
human signing off.

What --reset builds:

    roster          W-0412, so identity can be bound from a dropdown
    two priors      earlier in the week, bound to W-0412 by a named supervisor,
                    so the repeat-offender path has real history to find
    pour findings   events/run2, the night concrete pour -- recorded as `review`,
                    which is the point of that clip rather than a shortcoming

Deliberately NOT seeded: the events/run1 finding. That one is judged live, because a
demo where nothing is computed in front of anyone proves nothing.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import EventStore, Worker  # noqa: E402

DB = ROOT / "demo.db"
WORKER = Worker("W-0412", "Worker 0412", role="Concrete finisher")
SUPERVISOR = "supervisor:khalid"


def reset(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()
    db = EventStore(db_path)
    db.add_worker(WORKER)

    # Two earlier findings, bound to a named worker by a named supervisor. Without
    # these the escalation beat has no history to stand on, and the Adjudicator is
    # right to refuse to claim a pattern.
    template = next(iter(sorted(glob.glob(str(ROOT / "events" / "run1" / "events" / "*.json")))), None)
    if template is None:
        print("  no events/run1 to build priors from", file=sys.stderr)
    else:
        base = json.loads(Path(template).read_text(encoding="utf-8"))
        for i, days in enumerate((5, 3)):
            prior = json.loads(json.dumps(base))
            prior["event_id"] = f"prior_{i}"
            prior["captured_at"] = (datetime.now() - timedelta(days=days)).isoformat(
                timespec="seconds")
            db.record_event(prior)
            db.attach_identity(prior["event_id"], WORKER.worker_id, SUPERVISOR)
        print(f"  priors        {len(db.violations_for(WORKER.worker_id))} bound to "
              f"{WORKER.worker_id}")

    # The pour clip. Recorded, not judged: these are `review`, so the intake gate
    # correctly refuses to hand them to an agent at all.
    pour = sorted(glob.glob(str(ROOT / "events" / "run2" / "events" / "*.json")))
    for path in pour:
        db.record_event(json.loads(Path(path).read_text(encoding="utf-8")))
    print(f"  pour findings {len(pour)} recorded from events/run2")

    stats = db.stats()
    print(f"  database      {db_path.name} - {stats['events']} events, "
          f"{stats['workers']} worker(s), {stats['unattributed']} unattributed")


RUN_SHEET = """
RUN SHEET
=========

  Before they arrive
  ------------------
  python scripts/demo.py --reset
  python scripts/run_app.py --db demo.db
      The app opens on http://127.0.0.1:8000 Leave it running.
      Check the Findings page lists 4 findings and the top bar says "Service up".

  1  What the detector actually did                              ~60s
     Show data/footage/clip_e_pour_night_stable18s.mp4 playing, then:

     Console -> Findings -> open a pour_site finding -> the annotated frame.

     "Concrete pour, night. The worker has a cap, no vest, flip-flops. The
      system tracked him for ten frames before it was willing to say anything."

  2  It declines to accuse when it cannot see well               ~45s
     Same finding, read the reason field aloud:

        person detected at 0.63, below the 0.70 gate for asserting a violation

     "It found the violation and then refused to assert it, because the night
      lighting left it unsure there was even a person there. That is the design.
      I have not lowered the gate to make this demo look better."

  3  A finding it IS sure about, judged live                    ~60s
     Terminal:

     python scripts/run_agents.py "events/run1/events/*.json" --db demo.db

     "No worker bound, so watch what it refuses to conclude."
     -> warning, cited 29 CFR 1926.102, blocked_on: worker_identity
     Point at the trace: the agent chose those tools itself.

  4  The same finding, with history                             ~60s
     Terminal:

     python scripts/run_agents.py "events/run1/events/*.json" --db demo.db \\
         --bind W-0412 --by "supervisor:khalid"

     -> escalation, severity 7.0, approval REQUIRED
     "Same event, same evidence. The only thing that changed is that a supervisor
      said who it was, and the system could then see two prior findings."

  5  A human signs, or nothing happens                          ~45s
     Console -> Approvals. Evidence, citation and draft are all on screen
     BEFORE the button. Type a name, approve.

     "Nothing on that screen had been sent. The system drafts; a person sends."

  6  The week, and whether the detector still works             ~45s
     Console -> Weekly report: findings by zone, repeat offenders
     Console -> Monitoring: "not enough data" - a refusal, not a pass.

     "It will not give a drift verdict on four events. A confident number
      computed from nothing is worse than no number."

  Rehearsing more than once
  -------------------------
  Step 4 BINDS the run1 event to W-0412. Run it twice and that event becomes a third
  prior, severity goes 7 -> 9, and the action becomes stop_work. Not wrong, but not
  what you just said would happen. Re-run `--reset` between rehearsals.

  If something breaks
  -------------------
  App not answering     restart it: python scripts/run_app.py --db demo.db
  Model call times out  the case parks, the finding is kept - say so, it is the design
  Anything else         Console -> Findings still works offline from the database
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reset", action="store_true", help="rebuild the demo database")
    ap.add_argument("--db", default=str(DB))
    args = ap.parse_args(argv)

    if args.reset:
        print("\nbuilding the demo database")
        reset(Path(args.db))

    print(RUN_SHEET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
