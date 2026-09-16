"""
Ragas evaluation of the clause retriever.

Four metrics, all LLM-judged:

  context_precision   are the retrieved passages actually relevant, and ranked with the
                      relevant ones first
  context_recall      does the retrieved context cover everything the ground-truth
                      answer needs
  faithfulness        is the generated answer supported by the retrieved context, or
                      did the model add things
  answer_relevancy    does the answer address the question that was asked

The first two grade retrieval. The last two grade what an agent would produce on top of
it -- which is the failure this system cares most about, since a notice that states
something the regulation does not say is a false claim about a worker.

    python kb/eval_ragas.py                 # all 20 queries, ~80 API calls
    python kb/eval_ragas.py --limit 5       # cheaper smoke test
    python kb/eval_ragas.py --out eval/figures/ragas.json

Needs OPENAI_API_KEY in .env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from kb.eval_retrieval import CASES  # noqa: E402  -- same 20 queries, one source of truth


def build_dataset(limit: int | None, model: str):
    """Assemble question / contexts / answer / ground_truth for each case.

    The answer is generated from the retrieved context only. That is the point:
    faithfulness measures whether a model asked to summarise a regulation stays inside
    what the regulation actually says.
    """
    from kb.clause_map import ClauseMap
    from kb.retrieve import Retriever
    from openai import OpenAI

    retriever = Retriever.load()
    clause_map = ClauseMap.load()
    client = OpenAI()

    cases = CASES[:limit] if limit else CASES
    rows = {"question": [], "contexts": [], "answer": [], "ground_truth": []}

    for i, case in enumerate(cases, 1):
        hits = retriever.search(case.query, item=case.item, k=3)
        contexts = [h.text for h in hits]

        # Ground truth is the text of the clause the deterministic map says applies --
        # not whatever the search returned. So a retrieval miss shows up as a low score
        # rather than being quietly graded against its own output.
        truth = retriever.text_for(case.expect_section, max_chars=800)

        prompt = (
            "Answer using ONLY the regulation text provided. If it does not contain the "
            "answer, say so. Do not add requirements it does not state.\n\n"
            f"Regulation text:\n" + "\n\n".join(contexts) +
            f"\n\nQuestion: {case.query}"
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = completion.choices[0].message.content.strip()

        rows["question"].append(case.query)
        rows["contexts"].append(contexts)
        rows["answer"].append(answer)
        rows["ground_truth"].append(truth)
        print(f"  [{i}/{len(cases)}] {case.query[:52]:<54} -> {hits[0].section}")

    return rows, cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N cases")
    ap.add_argument("--model", default="gpt-4o-mini", help="generator and judge model")
    ap.add_argument("--out", default="eval/figures/ragas_results.json")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("no OPENAI_API_KEY in .env")

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (answer_relevancy, context_precision,
                                   context_recall, faithfulness)
    except ImportError as e:
        sys.exit(f"{e}\n\n  .venv/Scripts/python.exe -m pip install ragas datasets langchain-openai")

    print(f"generating answers with {args.model} ...")
    rows, cases = build_dataset(args.limit, args.model)

    print(f"\nscoring {len(rows['question'])} cases with Ragas ...")
    result = evaluate(
        Dataset.from_dict(rows),
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    )

    print("\n" + "=" * 58)
    print(f"{'metric':<22} {'score':>8}   grades")
    print("-" * 58)
    grades = {
        "context_precision": "retrieval ranking",
        "context_recall": "retrieval coverage",
        "faithfulness": "answer stays inside the regulation",
        "answer_relevancy": "answer addresses the question",
    }
    scores = {}
    for name, what in grades.items():
        value = result[name] if name in result else None
        if value is not None:
            scores[name] = float(value)
            print(f"{name:<22} {float(value):>8.3f}   {what}")
    print("=" * 58)

    if scores.get("faithfulness", 1) < 0.9:
        print("\n  Faithfulness below 0.90 is the one to act on. It means the model is")
        print("  stating things the retrieved regulation does not say -- in a notice,")
        print("  that is a false claim about a worker's conduct.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "n_cases": len(rows["question"]),
        "scores": scores,
        "per_case": [
            {"question": q, "retrieved": len(c), "answer": a[:400]}
            for q, c, a in zip(rows["question"], rows["contexts"], rows["answer"])
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
