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
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open the console automatically")
    args = ap.parse_args(argv)

    # Line-buffer, so the URL a person is waiting for arrives when it is printed and
    # not when the block fills. Redirected stdout buffers in 8K chunks otherwise, which
    # is exactly the case where somebody is staring at a blank terminal.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:                   # noqa: BLE001 -- not every stream supports it
        pass

    api_url = f"http://127.0.0.1:{args.api_port}"
    env = {**os.environ, "SAFETY_DB": args.db, "SAFETY_API": api_url}

    console_url = f"http://127.0.0.1:{args.console_port}"
    print(f"\n  database   {args.db}")

    procs: list[subprocess.Popen] = []
    try:
        procs.append(subprocess.Popen(
            [PY, "-m", "uvicorn", "app.api:app",
             "--host", "127.0.0.1", "--port", str(args.api_port)],
            cwd=ROOT, env=env))

        if not wait_for(f"{api_url}/health"):
            print("  the service did not come up; see the output above", file=sys.stderr)
            return 1

        if args.seed:
            seed(api_url, args.seed, judge=not args.no_judge)

        if not args.api_only:
            procs.append(subprocess.Popen(
                [PY, "-m", "streamlit", "run", "app/console.py",
                 "--server.port", str(args.console_port)],
                cwd=ROOT, env=env))

        # The console is the thing a person opens; the API is the thing it talks to.
        # Printing them the other way round sends people to the Swagger page and leaves
        # them wondering where the application went.
        print("\n" + "=" * 58)
        if args.api_only:
            print(f"  API only          {api_url}/docs")
        else:
            print(f"  OPEN THIS  ->     {console_url}")
            print(f"  api (not the ui)  {api_url}/docs")
        print("=" * 58)

        if not args.api_only and not args.no_browser:
            # Opened here rather than by Streamlit, because Streamlit only opens a
            # browser in non-headless mode -- and non-headless is what triggers its
            # first-run "enter your email" prompt, which blocks on any machine that has
            # not run it before. A demo laptop is exactly that machine.
            import webbrowser
            if wait_for(console_url, seconds=25):
                webbrowser.open(console_url)
            else:
                print("  (console slow to start — open the URL above yourself)")

        print("\n  Ctrl-C to stop.\n")
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
