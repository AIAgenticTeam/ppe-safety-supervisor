"""
Start the app: the service and the supervisor console, which are one process now.

    python scripts/run_app.py
    python scripts/run_app.py --db demo.db --seed events/run1/events/*.json

One command, one port. Worth having for its own sake, but mostly because a demo that
begins with "now let me open a second terminal" has already lost the room.

The pages and the JSON API are served by the same FastAPI app (app/web/ and app/api.py),
so there is nothing to keep in step and nothing for a page to find. Ctrl-C stops it.
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

from app.db import DEFAULT_DB_PATH  # noqa: E402

PY = sys.executable


def wait_for(url: str, seconds: float = 30.0) -> bool:
    """Poll until the service answers. Importing the agent graph is not instant, and an
    error page for the first few seconds is a bad way to meet the app."""
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
    ap.add_argument("--db", default=DEFAULT_DB_PATH)
    ap.add_argument("--port", "--api-port", dest="port", type=int, default=8000)
    ap.add_argument("--seed", nargs="*", default=[], metavar="GLOB",
                    help="event JSON to post once the service is up")
    ap.add_argument("--no-judge", action="store_true",
                    help="seed without running the agents, and spend nothing")
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open the app automatically")
    args = ap.parse_args(argv)

    # Line-buffer, so the URL a person is waiting for arrives when it is printed and
    # not when the block fills. Redirected stdout buffers in 8K chunks otherwise, which
    # is exactly the case where somebody is staring at a blank terminal.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:                   # noqa: BLE001 -- not every stream supports it
        pass

    url = f"http://127.0.0.1:{args.port}"
    env = {**os.environ, "SAFETY_DB": args.db}
    print(f"\n  database   {args.db}")

    proc = subprocess.Popen(
        [PY, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", str(args.port)],
        cwd=ROOT, env=env)
    try:
        if not wait_for(f"{url}/health"):
            print("  the service did not come up; see the output above", file=sys.stderr)
            return 1

        if args.seed:
            seed(url, args.seed, judge=not args.no_judge)

        print("\n" + "=" * 58)
        print(f"  OPEN THIS  ->     {url}")
        print(f"  api docs          {url}/docs")
        print("=" * 58)

        if not args.no_browser:
            import webbrowser
            webbrowser.open(url)

        print("\n  Ctrl-C to stop.\n")
        return proc.wait()

    except KeyboardInterrupt:
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()            # SIGTERM on posix, TerminateProcess on Windows
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
