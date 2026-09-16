"""
Measure retrieval before an agent sits on top of it.

Twenty hand-written queries with the clause each one must surface. Fix retrieval here,
where a failure is a number, rather than discovering it as a wrong citation in a notice.

The ground truth is not invented for this file -- it comes from `kb/clauses.yaml`, which
is the same map the Assessor uses to pick a citation. So this measures exactly the thing
that matters: can retrieval find the text of the clause the system has already decided
applies.

    python kb/eval_retrieval.py                # Recall@k and MRR, no LLM needed
    python kb/eval_retrieval.py --ragas        # adds LLM-judged metrics (needs a key)

Recall@k is the headline. The Assessor asks for one clause's text, so if the right
section is not in the top 3 the notice quotes the wrong regulation.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class Case:
    query: str
    expect_section: str
    item: str | None = None
    note: str = ""


# Phrased the way an event description would be, not the way the regulation is. That gap
# is the thing being measured -- "no hard hat" has to find a clause that says "protective
# helmets", and nothing in the corpus uses the words a supervisor would.
CASES: list[Case] = [
    # --- head -----------------------------------------------------------
    Case("worker not wearing a hard hat", "1926.100", "helmet"),
    Case("no helmet in an area with overhead work", "1926.100", "helmet"),
    Case("head protection requirements and ANSI standard", "1926.100", "helmet"),
    Case("bare head near falling objects", "1926.100", "helmet"),

    # --- eye and face ---------------------------------------------------
    Case("no eye protection while using a grinder", "1926.102", "goggles"),
    Case("worker without goggles near flying particles", "1926.102", "goggles"),
    Case("face shield requirements for welding", "1926.102", "goggles"),
    Case("eye protection for employees exposed to chemical splash", "1926.102", "goggles"),

    # --- feet -----------------------------------------------------------
    Case("no safety boots on the loading dock", "1926.96", "boots"),
    Case("foot protection where objects may fall or roll", "1926.96", "boots"),
    Case("worker in trainers on site", "1926.96", "boots"),

    # --- hands ----------------------------------------------------------
    # Construction has no hand-protection section, so these must land on the general
    # criterion. A hit on 1910.138 would mean the corpus was built from the wrong part.
    Case("worker handling materials with bare hands", "1926.95", "gloves"),
    Case("no gloves while operating an abrasive wheel", "1926.95", "gloves"),
    Case("hand protection requirement on a construction site", "1926.95", "gloves"),

    # --- high visibility ------------------------------------------------
    Case("worker without a high visibility vest", "1926.95", "vest"),
    Case("no hi-vis clothing near moving plant", "1926.95", "vest"),
    Case("flagger directing traffic without a warning garment", "1926.201", "vest",
         "the one case where hi-vis IS squarely regulated"),

    # --- general duty ---------------------------------------------------
    Case("who is responsible for providing protective equipment", "1926.28"),
    Case("employer duty to require PPE where hazards exist", "1926.28"),
    Case("protective equipment shall be provided and maintained in sanitary condition",
         "1926.95"),
]


def recall_at_k(hits, expected: str, k: int) -> bool:
    return any(h.section == expected for h in hits[:k])


def reciprocal_rank(hits, expected: str) -> float:
    for i, h in enumerate(hits, 1):
        if h.section == expected:
            return 1.0 / i
    return 0.0


def run(filtered: bool, k: int = 5, verbose: bool = True) -> dict:
    from kb.retrieve import Retriever

    r = Retriever.load()
    results, failures = [], []

    for case in CASES:
        hits = r.search(case.query, item=case.item if filtered else None, k=k)
        rr = reciprocal_rank(hits, case.expect_section)
        row = {
            "case": case,
            "hits": hits,
            "at1": recall_at_k(hits, case.expect_section, 1),
            "at3": recall_at_k(hits, case.expect_section, 3),
            "at5": recall_at_k(hits, case.expect_section, 5),
            "rr": rr,
        }
        results.append(row)
        if not row["at3"]:
            failures.append(row)

    n = len(results)
    summary = {
        "n": n,
        "recall@1": sum(r["at1"] for r in results) / n,
        "recall@3": sum(r["at3"] for r in results) / n,
        "recall@5": sum(r["at5"] for r in results) / n,
        "mrr": sum(r["rr"] for r in results) / n,
        "failures": failures,
    }

    if verbose and failures:
        print(f"\n  {len(failures)} queries missed the right clause in the top 3:")
        for row in failures:
            c = row["case"]
            got = ", ".join(h.section for h in row["hits"][:3]) or "nothing"
            print(f"    {c.query[:52]:<54} want {c.expect_section:<10} got {got}")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ragas", action="store_true",
                    help="also run Ragas LLM-judged metrics (needs OPENAI_API_KEY)")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    print(f"{len(CASES)} queries against the 1926 index\n")

    print("WITHOUT the topic filter (similarity alone):")
    unfiltered = run(filtered=False, k=args.k)
    print(f"  recall@1 {unfiltered['recall@1']:.2f}   recall@3 {unfiltered['recall@3']:.2f}"
          f"   recall@5 {unfiltered['recall@5']:.2f}   MRR {unfiltered['mrr']:.3f}")

    print("\nWITH the topic filter (filter, then rank):")
    filtered = run(filtered=True, k=args.k)
    print(f"  recall@1 {filtered['recall@1']:.2f}   recall@3 {filtered['recall@3']:.2f}"
          f"   recall@5 {filtered['recall@5']:.2f}   MRR {filtered['mrr']:.3f}")

    lift = filtered["recall@3"] - unfiltered["recall@3"]
    print(f"\nfiltering changes recall@3 by {lift:+.2f}")
    if lift > 0:
        print("  Filtering before ranking is doing real work. Keep passing `item`.")

    if filtered["recall@3"] < 0.9:
        print("\n  recall@3 below 0.90. Before blaming the model, check the chunking --")
        print("  short paragraphs that are only an ANSI reference carry no subject and")
        print("  embed badly. Merging them into their parent usually fixes it.")

    if args.ragas:
        run_ragas()


def run_ragas() -> None:
    """LLM-judged metrics, for the syllabus box. Recall@k above is the real measure."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import context_precision, context_recall
    except ImportError:
        print("\nragas not installed:  .venv/Scripts/python.exe -m pip install ragas datasets")
        return

    import os
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("\nno OPENAI_API_KEY in .env -- skipping Ragas")
        return

    from kb.retrieve import Retriever
    r = Retriever.load()

    rows = {"question": [], "contexts": [], "ground_truth": []}
    for case in CASES:
        hits = r.search(case.query, item=case.item, k=3)
        rows["question"].append(case.query)
        rows["contexts"].append([h.text for h in hits])
        rows["ground_truth"].append(r.text_for(case.expect_section, max_chars=600))

    print("\nrunning Ragas (this calls the API once per query) ...")
    scores = evaluate(Dataset.from_dict(rows),
                      metrics=[context_precision, context_recall])
    print(scores)


if __name__ == "__main__":
    main()
