"""Download the trained detector from the GitHub release into weights/.

    python scripts/get_weights.py

No release is published at the moment: the weights are shared with the team directly,
and the file goes in weights/best.pt by hand. This script is for when a release exists.
"""

import sys
import urllib.request
from pathlib import Path

REPO = "AIAgenticTeam/ppe-safety-supervisor"
TAG = "v1-detector"
ASSET = "best.pt"

DEST = Path(__file__).absolute().parents[1] / "weights" / ASSET
URL = f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}"


def main() -> None:
    if DEST.exists():
        print(f"already have {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    try:
        urllib.request.urlretrieve(URL, DEST)
    except Exception as e:
        DEST.unlink(missing_ok=True)            # never leave a half-written model behind
        sys.exit(f"failed: {e}\n\n"
                 f"No release {TAG!r} is published on {REPO}. The weights are shared "
                 f"with the team directly: copy best.pt to {DEST}. See weights/README.md")
    print(f"saved {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
