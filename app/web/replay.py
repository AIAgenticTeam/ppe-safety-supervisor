"""
Which event files the console may replay through the pipeline.

The Streamlit console took a free-text glob from the sidebar. That was fine when the
console was a process on the operator's own machine; as a web page it would let any
client name a path and have the server read it. So the choice is now a fixed list of
sources, and a file is replayable only if it appears in that source's own listing -- a
name from the request is looked up, never opened directly.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).absolute().parents[2]

# label shown in the page -> (directory, glob relative to it)
SOURCES: dict[str, tuple[str, Path, str]] = {
    "fixtures": ("Sample events (fixtures/)", ROOT / "fixtures" / "events", "*.json"),
    "pipeline": ("Pipeline output (events/)", ROOT / "events", "*/events/*.json"),
}


def list_files(source: str) -> list[str]:
    """Replayable file names for one source, relative to its directory. Empty for an
    unknown source, so a bad name is 'nothing to replay' and not an error path."""
    if source not in SOURCES:
        return []
    _, base, pattern = SOURCES[source]
    return sorted(p.relative_to(base).as_posix() for p in base.glob(pattern) if p.is_file())


def load(source: str, name: str) -> dict | None:
    """The parsed event, or None if `name` is not one this source lists."""
    if name not in set(list_files(source)):
        return None
    return json.loads((SOURCES[source][1] / name).read_text(encoding="utf-8"))


def summary() -> list[dict]:
    """What the page needs to draw the source picker."""
    return [{"key": k, "label": label, "count": len(list_files(k))}
            for k, (label, _, _) in SOURCES.items()]
