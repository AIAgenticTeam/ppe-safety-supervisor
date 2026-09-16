"""
29 CFR 1926 -> paragraph chunks with citation metadata.

Chunks at the paragraph level, never a fixed token window. A window slices clause text
away from the citation that identifies it, and a citation is the one thing this system
must never get wrong -- so the regulation's own structure is the chunk boundary.

Each chunk carries its full citation path, so a retrieved passage can always say where
it came from:

    1926.100(b)(1)(i)  ->  "29 CFR 1926.100(b)(1)(i)"

    python kb/ingest.py                 # parse kb/corpus/*.xml -> kb/chunks.json
    python kb/ingest.py --stats         # what came out, without rewriting

Source XML is pulled from the eCFR API (see kb/corpus/README.md).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from html import unescape
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).absolute().parents[1]
CORPUS = ROOT / "kb" / "corpus"
OUT = ROOT / "kb" / "chunks.json"

# Paragraph designators as the CFR nests them: (a) (1) (i) (A)
#
# (i), (v) and (x) are ambiguous -- lowercase letters at depth 0 AND roman numerals at
# depth 2. Matching greedily gives citations like 1926.102(i)(ii)(iv), which name a
# paragraph that does not exist. Since a wrong citation is the one failure this system
# cannot tolerate, the level is resolved by asking which depth the designator would
# legally continue, given what has been seen so far.
LEVEL_PATTERNS = [
    re.compile(r"^\(([a-z])\)"),          # (a)
    re.compile(r"^\((\d+)\)"),            # (1)
    re.compile(r"^\(([ivxlc]+)\)"),       # (i)
    re.compile(r"^\(([A-Z])\)"),          # (A)
]

ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
         "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"]


def _is_continuation(depth: int, label: str, current: str | None) -> bool:
    """Would `label` legally follow `current` at this nesting depth?"""
    value = label.strip("()")
    if current is None:                       # opening a new level
        return value in ("a", "1", "i", "A")
    prev = current.strip("()")
    try:
        if depth == 0:
            return len(value) == 1 and ord(value) == ord(prev) + 1
        if depth == 1:
            return value.isdigit() and int(value) == int(prev) + 1
        if depth == 2:
            return ROMAN.index(value) == ROMAN.index(prev) + 1
        return len(value) == 1 and ord(value) == ord(prev) + 1
    except (ValueError, IndexError):
        return False

# Which PPE class a section is about, so retrieval can be filtered before it is ranked.
# Filtering first is what stops a glove query surfacing a respirator clause because both
# say "the employer shall provide".
SECTION_TOPICS = {
    "1926.95": ["helmet", "goggles", "gloves", "boots", "vest"],   # general criteria
    "1926.96": ["boots"],
    "1926.100": ["helmet"],
    "1926.101": [],          # hearing
    "1926.102": ["goggles"],
    "1926.103": [],          # respiratory
    "1926.104": [], "1926.105": [], "1926.106": [], "1926.107": [],
    "1926.28": ["helmet", "goggles", "gloves", "boots", "vest"],   # general duty
    "1926.201": ["vest"],
}


@dataclass
class Chunk:
    chunk_id: str
    section: str                 # "1926.100"
    paragraph: str               # "(b)(1)(i)" or "" for the lead paragraph
    citation: str                # "29 CFR 1926.100(b)(1)(i)"
    section_title: str
    subpart: str
    text: str
    topics: list[str]
    authority: str = ""          # specific | general | narrow, from clauses.yaml
    char_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


def _clean(element) -> str:
    """Flatten a <P> to text, dropping italics markup but keeping the words."""
    text = "".join(element.itertext())
    return unescape(re.sub(r"\s+", " ", text)).strip()


def _designator(text: str, path: list[str]) -> tuple[int, str] | None:
    """Which nesting level this paragraph opens, resolved against what came before.

    Several patterns can match the same token -- "(i)" is both a letter and a roman
    numeral. Prefer the depth where it legally continues the sequence; fall back to the
    deepest match, which is the safer guess since over-nesting keeps a paragraph inside
    its parent rather than promoting it to a sibling of one.
    """
    matches = [(depth, f"({m.group(1)})")
               for depth, pattern in enumerate(LEVEL_PATTERNS)
               if (m := pattern.match(text))]
    if not matches:
        return None

    for depth, label in matches:
        current = path[depth] if depth < len(path) else None
        # A designator cannot open a level whose parent has not been opened -- that is
        # how "(b)" followed by "(i)" collapses into the nonexistent "(b)(i)".
        if depth > len(path):
            continue
        if _is_continuation(depth, label, current):
            return depth, label

    return matches[-1]


# The CFR frequently opens a sub-level inside its parent's paragraph:
#     "(b) Criteria for head protection. (1) The employer must provide ..."
# The (1) never appears on a line of its own, so without this the following (i), (ii),
# (iii) have no depth-1 parent on the stack and every citation below them is wrong.
INLINE_NEXT = {
    0: re.compile(r"^\([a-z]\)\s*(?:[^.()]{0,80}\.\s*)?\((\d+)\)\s"),
    1: re.compile(r"^\(\d+\)\s*(?:[^.()]{0,80}\.\s*)?\(([ivxlc]+)\)\s"),
    2: re.compile(r"^\([ivxlc]+\)\s*(?:[^.()]{0,80}\.\s*)?\(([A-Z])\)\s"),
}


def _inline_child(text: str, depth: int) -> str | None:
    """A nested designator opened inside this same paragraph, if any."""
    pattern = INLINE_NEXT.get(depth)
    if not pattern:
        return None
    m = pattern.match(text)
    if not m:
        return None
    value = m.group(1)
    # Only an opening designator counts. A mid-sentence "(2)" is prose, not structure.
    return f"({value})" if value in ("1", "i", "A") else None


def _authorities() -> dict[str, str]:
    """Authority per section, from the clause map.

    A specific provision governs over a general one -- the canon that a lawyer would
    apply, and the reason 1926.100 should outrank 1926.95 on a helmet query even though
    1926.95 also mentions head protection.
    """
    import yaml
    raw = yaml.safe_load((ROOT / "kb" / "clauses.yaml").read_text(encoding="utf-8"))
    return {cid: c.get("authority", "") for cid, c in raw["clauses"].items()}


AUTHORITY = _authorities()


def parse_section(div8, subpart: str) -> list[Chunk]:
    section = div8.get("N", "")
    head = div8.find("HEAD")
    title = _clean(head).split(" ", 2)[-1].rstrip(".") if head is not None else ""
    topics = SECTION_TOPICS.get(section, [])

    chunks: list[Chunk] = []
    path: list[str] = []          # current designator stack

    for p in div8.findall("P"):
        text = _clean(p)
        if not text:
            continue

        found = _designator(text, path)
        if found:
            depth, label = found
            path = path[:depth] + [label]
            # Follow any sub-level opened inside this same paragraph, so the children
            # that follow on their own lines attach beneath it rather than beside it.
            child = _inline_child(text, depth)
            if child:
                path.append(child)
        # A paragraph with no designator continues the current one.

        paragraph = "".join(path)
        citation = f"29 CFR {section}{paragraph}"
        chunk_id = f"{section}{paragraph}".replace(" ", "")

        # Continuation lines join the chunk they belong to rather than orphaning.
        if not found and chunks and chunks[-1].chunk_id == chunk_id:
            prev = chunks.pop()
            text = f"{prev.text} {text}"

        chunks.append(Chunk(
            chunk_id=chunk_id, section=section, paragraph=paragraph,
            citation=citation, section_title=title, subpart=subpart,
            text=text, topics=list(topics),
            authority=AUTHORITY.get(section, ""),
        ))

    return chunks


def parse_file(path: Path) -> list[Chunk]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    subpart = root.get("N", "")
    return [c for div8 in root.iter("DIV8") for c in parse_section(div8, subpart)]


def build() -> list[Chunk]:
    files = sorted(CORPUS.glob("*.xml"))
    if not files:
        sys.exit(f"no XML in {CORPUS} -- see kb/corpus/README.md for how to fetch it")

    chunks: list[Chunk] = []
    for f in files:
        got = parse_file(f)
        print(f"  {f.name:<26} {len(got):>4} chunks")
        chunks.extend(got)

    # Only sections this system actually cites are worth indexing. 1926.103 respiratory
    # protection is real law, but nothing here detects respirators, and an unfilterable
    # clause is just a chance to retrieve the wrong one.
    keep = [c for c in chunks if c.topics]
    dropped = len(chunks) - len(keep)
    print(f"\n  kept {len(keep)}, dropped {dropped} chunks from sections with no PPE "
          f"class this system detects")
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", action="store_true", help="report without writing")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    print("parsing 29 CFR 1926 ...")
    chunks = build()

    from collections import Counter
    by_section = Counter(c.section for c in chunks)
    by_topic = Counter(t for c in chunks for t in c.topics)
    lengths = sorted(c.char_count for c in chunks)

    print(f"\n{'section':<12} {'chunks':>7}  title")
    print("-" * 60)
    for sec, n in sorted(by_section.items()):
        title = next(c.section_title for c in chunks if c.section == sec)
        print(f"{sec:<12} {n:>7}  {title}")

    print(f"\nchunks per PPE class (a chunk may serve several):")
    for topic, n in by_topic.most_common():
        print(f"  {topic:<9} {n:>4}")

    print(f"\nchunk length: min {lengths[0]}, median {lengths[len(lengths) // 2]}, "
          f"max {lengths[-1]} chars")
    short = [c for c in chunks if c.char_count < 40]
    if short:
        print(f"  {len(short)} chunks under 40 chars -- these embed poorly, check them:")
        for c in short[:3]:
            print(f"    {c.citation}: {c.text[:60]!r}")

    if args.stats:
        print("\n--stats: nothing written")
        return

    Path(args.out).write_text(
        json.dumps([asdict(c) for c in chunks], indent=2), encoding="utf-8")
    print(f"\nwrote {len(chunks)} chunks -> {args.out}")


if __name__ == "__main__":
    main()
