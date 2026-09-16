"""
FAISS retrieval over the 1926 chunks.

Retrieval's job is to fetch the TEXT of a clause, never to choose which clause applies.
`kb/clause_map.py` decides that deterministically; this module supplies the words that go
in the notice. If the two ever disagree, the clause map wins.

Filter before ranking. Every chunk carries the PPE classes its section covers, so a glove
query is restricted to glove-relevant sections before similarity is computed at all.
Without that, "the employer shall provide" matches equally well in every clause in the
subpart and the top hit becomes a coin flip.

    python kb/build_index.py        # build it (needs kb/chunks.json)
    python kb/retrieve.py "hand protection"  --item gloves

    from kb.retrieve import Retriever
    r = Retriever.load()
    r.search("head protection requirements", item="helmet", k=3)
    r.text_for("1926.100")          # all chunks of a clause, in order
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
INDEX_DIR = ROOT / "kb" / "index"
CHUNKS = ROOT / "kb" / "chunks.json"

# Small, fast, and ample for a few dozen regulatory paragraphs. Swapping it means
# rebuilding the index; the model name is stored alongside so a mismatch is caught.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Hit:
    rank: int
    score: float
    citation: str
    section: str
    paragraph: str
    section_title: str
    text: str

    def __str__(self) -> str:
        return f"[{self.rank}] {self.score:.3f}  {self.citation}  {self.text[:70]}..."


class Retriever:
    def __init__(self, index, chunks: list[dict], model_name: str) -> None:
        self.index = index
        self.chunks = chunks
        self.model_name = model_name
        self._model = None

    # ---- loading --------------------------------------------------------

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "Retriever":
        import faiss

        meta_path = index_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no index at {index_dir} -- run: python kb/build_index.py")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(index_dir / "clauses.faiss"))
        chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))

        if index.ntotal != len(chunks):
            raise ValueError(
                f"index holds {index.ntotal} vectors but {len(chunks)} chunks were "
                f"stored -- rebuild with: python kb/build_index.py")

        return cls(index, chunks, meta["model"])

    @property
    def model(self):
        """Loaded lazily -- importing sentence-transformers costs seconds."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ---- search ---------------------------------------------------------

    # A specific provision governs over a general one. 1926.95 says "provide PPE"
    # and mentions every body part, so on raw similarity it crowds out 1926.100 on a
    # helmet query and 1926.96 on a boots query -- and it has fourteen paragraphs to
    # their one, so it wins on volume too. This is the canon a lawyer would apply,
    # applied to ranking.
    AUTHORITY_BOOST = {"specific": 0.06, "narrow": 0.04, "general": 0.0}

    def search(self, query: str, item: str | None = None, k: int = 3,
               fetch_k: int = 40, one_per_section: bool = True) -> list[Hit]:
        """Nearest chunks, optionally restricted to one PPE class.

        `item` narrows the candidate set before ranking. Always pass it when you know it
        -- an unfiltered search over a single subpart is close to a coin flip, because
        every clause in it shares the same regulatory phrasing.

        `one_per_section` keeps the best paragraph from each section rather than letting
        a long section fill every slot. The caller wants to know WHICH REGULATION
        applies; three paragraphs of the same one answers that question once and wastes
        two slots.
        """
        vector = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(vector, min(fetch_k, self.index.ntotal))

        scored: list[tuple[float, dict]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if item and item not in chunk["topics"]:
                continue
            boost = self.AUTHORITY_BOOST.get(chunk.get("authority", ""), 0.0)
            scored.append((float(score) + boost, chunk))

        scored.sort(key=lambda pair: -pair[0])

        hits: list[Hit] = []
        seen: set[str] = set()
        for score, chunk in scored:
            if one_per_section and chunk["section"] in seen:
                continue
            seen.add(chunk["section"])
            hits.append(Hit(
                rank=len(hits) + 1, score=score, citation=chunk["citation"],
                section=chunk["section"], paragraph=chunk["paragraph"],
                section_title=chunk["section_title"], text=chunk["text"],
            ))
            if len(hits) == k:
                break
        return hits

    # ---- direct lookup, no similarity involved --------------------------

    def text_for(self, clause_id: str, max_chars: int = 1200) -> str:
        """Every chunk of a clause, in document order.

        This is what the Assessor actually calls: the clause map has already named the
        clause, so there is nothing to search for.
        """
        parts = [c for c in self.chunks if c["section"] == clause_id]
        parts.sort(key=lambda c: c["paragraph"])
        out, total = [], 0
        for c in parts:
            if total + c["char_count"] > max_chars and out:
                break
            out.append(c["text"])
            total += c["char_count"]
        return " ".join(out)

    def chunks_for(self, clause_id: str) -> list[dict]:
        return [c for c in self.chunks if c["section"] == clause_id]

    @property
    def sections(self) -> list[str]:
        return sorted({c["section"] for c in self.chunks})


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Query the clause index")
    ap.add_argument("query")
    ap.add_argument("--item", default=None, help="restrict to a PPE class")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    r = Retriever.load()
    print(f"{len(r.chunks)} chunks across {len(r.sections)} sections\n")
    print(f'query: {args.query!r}' + (f"  filtered to {args.item}" if args.item else ""))
    for hit in r.search(args.query, item=args.item, k=args.k):
        print(f"\n{hit}")
