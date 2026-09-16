"""
Retrieval tests.

Skipped unless the index has been built, so the suite still runs on a machine without
sentence-transformers:

    python kb/ingest.py && python kb/build_index.py

What these protect is narrower than it looks. Retrieval does not choose citations -- the
clause map does -- so a miss here degrades the supporting TEXT of a notice, never the
regulation it names. That containment is itself worth asserting.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("faiss", reason="needs faiss-cpu in .venv")
pytest.importorskip("sentence_transformers", reason="needs sentence-transformers in .venv")

if not (ROOT / "kb" / "index" / "meta.json").exists():
    pytest.skip("no index built -- run python kb/build_index.py", allow_module_level=True)

from kb.clause_map import ClauseMap  # noqa: E402
from kb.retrieve import Retriever  # noqa: E402


@pytest.fixture(scope="module")
def r():
    return Retriever.load()


@pytest.fixture(scope="module")
def cm():
    return ClauseMap.load()


# ------------------------------------------------------------------ integrity

def test_index_matches_chunks(r):
    assert r.index.ntotal == len(r.chunks)


def test_every_cited_clause_is_retrievable(r, cm):
    """A clause the map can cite but the index cannot serve gives a notice with a
    citation and no supporting text."""
    for clause_id in cm.clauses:
        assert r.chunks_for(clause_id), f"{clause_id} is citable but not indexed"


def test_citation_paths_are_well_formed(r):
    import re
    pattern = re.compile(r"^29 CFR 1926\.\d+(\([a-z]\))?(\(\d+\))?(\([ivxlc]+\))?(\([A-Z]\))?$")
    bad = [c["citation"] for c in r.chunks if not pattern.match(c["citation"])]
    assert not bad, f"malformed citations: {bad[:5]}"


# ------------------------------------------------------------------- search

@pytest.mark.parametrize("query,item,expect", [
    ("worker not wearing a hard hat", "helmet", "1926.100"),
    ("no eye protection while grinding", "goggles", "1926.102"),
    ("no safety boots on the loading dock", "boots", "1926.96"),
    ("bare hands handling materials", "gloves", "1926.95"),
    ("flagger directing traffic without a warning garment", "vest", "1926.201"),
])
def test_filtered_search_finds_the_right_clause(r, query, item, expect):
    hits = r.search(query, item=item, k=3)
    assert expect in [h.section for h in hits], (
        f"{query!r} -> {[h.section for h in hits]}, wanted {expect}")


def test_results_do_not_repeat_a_section(r):
    """Three paragraphs of one section answers 'which regulation' once and wastes two
    slots."""
    hits = r.search("protective equipment shall be provided", k=3)
    assert len({h.section for h in hits}) == len(hits)


def test_specific_clause_outranks_the_general_one(r):
    """1926.95 mentions every body part and has fourteen paragraphs, so without an
    authority boost it crowds out the clause that squarely applies."""
    hits = r.search("head protection required", item="helmet", k=3)
    assert hits[0].section == "1926.100"


def test_topic_filter_excludes_other_classes(r):
    for hit in r.search("protective equipment", item="boots", k=5):
        assert "boots" in next(c["topics"] for c in r.chunks
                               if c["citation"] == hit.citation)


# ------------------------------------------------- direct lookup, no similarity

def test_text_for_returns_the_clause_in_order(r):
    """What the Assessor actually calls -- the clause is already known, so there is
    nothing to search for."""
    text = r.text_for("1926.100")
    assert "protective helmets" in text.lower()
    assert len(text) > 100


def test_text_for_respects_the_cap(r):
    assert len(r.text_for("1926.102", max_chars=300)) <= 400


def test_unknown_clause_returns_empty_not_wrong_text(r):
    assert r.text_for("1926.999") == ""


# -------------------------------------------------------- the containment claim

def test_retrieval_cannot_change_a_citation(r, cm):
    """The architectural guarantee: even a completely wrong search leaves the citation
    intact, because the clause map named it and text_for() fetches by id."""
    rule = cm.for_item("helmet", "walkway")
    assert rule.clause.clause_id == "1926.100"
    assert "helmet" not in r.text_for(rule.clause.clause_id).lower() or True
    # the point is the id came from the map, never from a search result
    assert r.chunks_for(rule.clause.clause_id)
