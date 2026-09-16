"""
Embed kb/chunks.json and write a FAISS index.

    python kb/ingest.py           # first, parse the corpus
    python kb/build_index.py      # then, embed and index it

Writes kb/index/:
    clauses.faiss   the vectors
    chunks.json     the payloads, in the same order as the vectors
    meta.json       model name and counts, so a stale index is detected not used

Inner-product on normalised vectors, which is cosine similarity. The corpus is a few
dozen paragraphs, so a flat index is exact and instant -- no reason for anything cleverer.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from kb.retrieve import CHUNKS, INDEX_DIR, MODEL_NAME  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--chunks", default=str(CHUNKS))
    ap.add_argument("--out", default=str(INDEX_DIR))
    args = ap.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        sys.exit(f"no {chunks_path} -- run: python kb/ingest.py")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not chunks:
        sys.exit("chunks.json is empty")

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print(f"loading {args.model} ...")
    model = SentenceTransformer(args.model)

    # Embed the clause text prefixed with its section title. The title carries the
    # subject ("Head protection") that the paragraph body often assumes rather than
    # states -- 1926.100(b)(1)(i) is just an ANSI reference on its own.
    texts = [f"{c['section_title']}. {c['text']}" for c in chunks]

    print(f"embedding {len(texts)} chunks ...")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])       # cosine, since normalised
    index.add(vectors)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "clauses.faiss"))
    (out / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({
        "model": args.model,
        "dimensions": int(vectors.shape[1]),
        "chunks": len(chunks),
        "sections": sorted({c["section"] for c in chunks}),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")

    print(f"\nindexed {index.ntotal} chunks, {vectors.shape[1]} dimensions -> {out}")
    print(f"sections: {', '.join(sorted({c['section'] for c in chunks}))}")


if __name__ == "__main__":
    main()
