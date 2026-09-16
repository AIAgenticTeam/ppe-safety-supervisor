"""
Guard against a real credential reaching git.

.env.example is committed and .env is not, so a key pasted into the wrong one is an
easy mistake with a permanent consequence -- once a secret is in history, removing the
commit is not enough. This test fails loudly instead.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]

SECRET_SHAPES = [
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),        # OpenAI
    re.compile(r"AIza[A-Za-z0-9_\-]{30,}"),       # Google
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),          # GitHub PAT
    re.compile(r"AKIA[A-Z0-9]{16}"),              # AWS
]


def _tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return [ROOT / line for line in out.stdout.splitlines() if line]


def test_env_example_has_no_values():
    """The template must ship with empty placeholders."""
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if "KEY=" in line or "SECRET=" in line:
            _, _, value = line.partition("=")
            assert not value.strip().strip("'\""), (
                f"{line.split('=')[0]} has a value in .env.example, which is COMMITTED. "
                f"Real keys belong in .env, which is gitignored.")


def test_env_is_gitignored():
    result = subprocess.run(["git", "check-ignore", ".env"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, ".env is not gitignored -- a key here would be committed"


@pytest.mark.parametrize("path", _tracked_files(),
                         ids=lambda p: str(p.relative_to(ROOT)).replace("\\", "/"))
def test_tracked_file_has_no_credential(path):
    """Nothing git tracks may contain something shaped like a live credential."""
    if not path.exists() or path.suffix in {".jpg", ".png", ".faiss", ".ipynb"}:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return
    for pattern in SECRET_SHAPES:
        found = pattern.search(text)
        assert not found, (
            f"{path.relative_to(ROOT)} contains something shaped like a credential "
            f"({found.group(0)[:12]}...). Move it to .env and rotate it.")
