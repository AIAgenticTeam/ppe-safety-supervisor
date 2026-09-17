"""
Start the service and the console together.

    python scripts/run_app.py
    python scripts/run_app.py --db demo.db --seed events/run1/events/*.json

Two processes, one command. Worth having for its own sake, but mostly because a demo
that begins with "now let me open a second terminal" has already lost the room.

The API owns the database, the model and the index. The console only renders, and finds
the API through SAFETY_API. Ctrl-C stops both.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

PY = sys.executable


def wait_for(url: str, seconds: float = 30.0) -> bool:
    """Poll until the service answers. Starting the console first shows an error page
    for as long as uvicorn takes to import torch, which is not a short time."""
    import httpx

    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return True
        except Exception:               # noqa: BLE001 -- not up yet is the normal case
            time.sleep(0.4)
    return False


def seed(api_url: str, patterns: list[str], judge: bool) -> None:
    import httpx

    paths: list[Path] = []
    for pattern in patterns:
        paths += [Path(p) for p in sorted(glob.glob(pattern))]
    if not paths:
        print(f"  seed: nothing matched {patterns}")
        return

    ok = 0
    for path in paths:
        event = json.loads(path.read_text(encoding="utf-8"))
        try:
            r = httpx.post(f"{api_url}/events", params={"judge": str(judge).lower()},
                           json=event, timeout=120.0)
            ok += r.status_code == 201
            if r.status_code != 201:
                print(f"  seed: {path.name} refused — {r.status_code} {r.text[:90]}")
        except Exception as exc:        # noqa: BLE001
            print(f"  seed: {path.name} failed — {exc}")
    print(f"  seed: {ok} of {len(paths)} accepted")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="safety.db")
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--console-port", type=int, default=8501)
    ap.add_argument("--seed", nargs="*", default=[], metavar="GLOB",
                    help="event JSON to post once the service is up")
    ap.add_argument("--no-judge", action="store_true",
                    help="seed without running the agents, and spend nothing")
    ap.add_argument("--api-only", action="store_true")
    args = ap.parse_args(argv)

    api_url = f"http://127.0.0.1:{args.api_port}"
    env = {**os.environ, "SAFETY_DB": args.db, "SAFETY_API": api_url}

    print(f"  db       {args.db}")
    print(f"  api      {api_url}/docs")

    procs: list[subprocess.Popen] = []
    try:
        procs.append(subprocess.Popen(
            [PY, "-m", "uvicorn", "app.api:app",
             "--host", "127.0.0.1", "--port", str(args.api_port)],
            cwd=ROOT, env=env))

        if not wait_for(f"{api_url}/health"):
            print("  the service did not come up; see the output above", file=sys.stderr)
            return 1
        print("  api      up")

        if args.seed:
            seed(api_url, args.seed, judge=not args.no_judge)

        if not args.api_only:
            print(f"  console  http://127.0.0.1:{args.console_port}")
            procs.append(subprocess.Popen(
                [PY, "-m", "streamlit", "run", "app/console.py",
                 "--server.port", str(args.console_port)],
                cwd=ROOT, env=env))

        print("\n  Ctrl-C to stop both.\n")
        while all(p.poll() is None for p in procs):
            time.sleep(0.5)
        # One died on its own; do not leave the other orphaned.
        return next((p.returncode for p in procs if p.poll() is not None), 0)

    except KeyboardInterrupt:
        return 0
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()           # SIGTERM on posix, TerminateProcess on Windows
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()


if __name__ == "__main__":
    raise SystemExit(main())
