"""
Ragas evaluation of the clause retriever.

Four LLM-judged metrics:

  context_precision   are the retrieved passages relevant, and ranked relevant-first
  context_recall      does the retrieved context cover what the reference answer needs
  faithfulness        is the generated answer supported by the retrieved context
  answer_relevancy    does the answer address the question asked

The first two grade retrieval. The last two grade what an agent would produce on top of
it -- and faithfulness is the one that matters most here, because a notice stating
something the regulation does not say is a false claim about a worker.

    python kb/eval_ragas.py --limit 3        # smoke test first, ~20 calls
    python kb/eval_ragas.py                  # all 20 cases, ~100 calls
    python kb/eval_ragas.py --dry-run        # show the plan and cost, spend nothing

Costs are printed at the end from real token counts, not estimates. On gpt-4o-mini the
full run is a few US cents.

Needs OPENAI_API_KEY in .env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
import warnings
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

# ragas 0.4.3 hard-imports langchain_community.chat_models.vertexai, which that package
# removed during its sunset. Nothing here uses VertexAI, so a stub satisfies the import
# without downgrading langchain-community and disturbing the torch/faiss tree.
_VERTEX = "langchain_community.chat_models.vertexai"
if _VERTEX not in sys.modules:
    _stub = types.ModuleType(_VERTEX)
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[_VERTEX] = _stub

warnings.filterwarnings("ignore", category=DeprecationWarning)

from kb.eval_cases import CASES  # noqa: E402  -- shared with the deterministic eval

# gpt-4o-mini, USD per 1M tokens. Only used to print what a run cost.
PRICE_IN, PRICE_OUT = 0.15, 0.60


def build_samples(limit: int | None, model: str, verbose: bool = True):
    """Retrieve, generate an answer from the retrieved text, and pair it with the truth.

    The reference is the text of the clause the DETERMINISTIC map says applies -- not
    whatever search returned. A retrieval miss therefore shows up as a low score rather
    than being quietly graded against its own output.
    """
    from openai import OpenAI
    from ragas.dataset_schema import SingleTurnSample

    from kb.retrieve import Retriever

    retriever = Retriever.load()
    client = OpenAI()
    cases = CASES[:limit] if limit else CASES

    samples, in_tok, out_tok = [], 0, 0
    for i, case in enumerate(cases, 1):
        # one_per_section=False on purpose. The section-deduplicating mode is right for
        # "WHICH regulation applies" -- it stops one long clause filling every slot. It
        # is wrong for "answer this question", because it discards the other paragraphs
        # of the very clause that holds the answer: a query about helmets returned an
        # ANSI cross-reference and the general duty, but never 1926.100(a), the operative
        # sentence. Question answering wants the best chunks, section be damned.
        hits = retriever.search_for_answer(case.query, item=case.item, k=3, sections=2)
        contexts = [h.text for h in hits]
        # The hand-written ground-truth ANSWER, not the raw clause. Passing the whole
        # clause caps context_recall by construction: retrieval returns one paragraph
        # and the reference would hold eight, so it could never look complete.
        reference = case.reference or retriever.text_for(case.expect_section, max_chars=400)

        prompt = (
            "Answer using ONLY the regulation text provided. If it does not contain the "
            "answer, say so plainly. Do not add requirements it does not state.\n\n"
            "Regulation text:\n" + "\n\n".join(contexts) +
            f"\n\nQuestion: {case.as_question}"
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        in_tok += completion.usage.prompt_tokens
        out_tok += completion.usage.completion_tokens

        samples.append(SingleTurnSample(
            user_input=case.as_question,
            retrieved_contexts=contexts,
            response=completion.choices[0].message.content.strip(),
            reference=reference,
        ))
        if verbose:
            got = hits[0].section if hits else "nothing"
            mark = "ok " if got == case.expect_section else "MISS"
            print(f"  [{i:>2}/{len(cases)}] {mark} {case.as_question[:48]:<50} -> {got}")

    return samples, cases, in_tok, out_tok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N cases")
    ap.add_argument("--model", default="gpt-4o-mini", help="generator and judge model")
    ap.add_argument("--out", default="eval/figures/ragas_results.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and estimated cost, make no API call")
    args = ap.parse_args()

    n = args.limit or len(CASES)
    calls = n + n * 4          # one generation each, plus roughly one judge call per metric
    print(f"plan: {n} cases, ~{calls} API calls on {args.model}")
    if args.dry_run:
        print(f"estimated cost at typical sizes: well under $0.05")
        print("dry run: nothing spent")
        return

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("no OPENAI_API_KEY in .env")

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.cost import get_token_usage_for_openai
    from ragas.dataset_schema import EvaluationDataset
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)

    print(f"\ngenerating answers from retrieved context ...")
    samples, cases, gen_in, gen_out = build_samples(args.limit, args.model)

    print(f"\nscoring {len(samples)} samples with ragas ...")
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=LangchainLLMWrapper(ChatOpenAI(model=args.model, temperature=0)),
        embeddings=LangchainEmbeddingsWrapper(OpenAIEmbeddings()),
        token_usage_parser=get_token_usage_for_openai,
    )

    # result[key] is a LIST of per-sample scores, one per case -- not a single float.
    # Discover the keys from the result rather than assuming ragas' metric names, which
    # change between versions.
    grades = {
        "context_precision": "retrieval ranking",
        "context_recall": "retrieval coverage",
        "faithfulness": "answer stays inside the regulation",
        "answer_relevancy": "answer addresses the question",
    }
    available = list(getattr(result, "_scores_dict", {}) or {})

    scores, per_sample = {}, {}
    print("\n" + "=" * 68)
    print(f"{'metric':<34} {'mean':>6} {'min':>6}   grades")
    print("-" * 68)
    for key in available:
        values = [v for v in result[key] if v is not None and v == v]   # drop None/NaN
        if not values:
            print(f"{key:<34} {'n/a':>6} {'':>6}   all samples failed to score")
            continue
        mean = sum(values) / len(values)
        scores[key] = mean
        per_sample[key] = values
        label = next((w for k, w in grades.items() if k in key), "")
        print(f"{key:<34} {mean:>6.3f} {min(values):>6.3f}   {label}")
    print("=" * 68)
    if not scores:
        print("  no metric produced a score -- check the API key and model access")

    # ---- what it cost -------------------------------------------------
    judge_cost = 0.0
    try:
        usage = result.total_tokens()
        judge_in, judge_out = usage.input_tokens, usage.output_tokens
        judge_cost = judge_in / 1e6 * PRICE_IN + judge_out / 1e6 * PRICE_OUT
    except Exception:
        judge_in = judge_out = 0

    gen_cost = gen_in / 1e6 * PRICE_IN + gen_out / 1e6 * PRICE_OUT
    print(f"\ntokens   generation {gen_in:,} in / {gen_out:,} out")
    if judge_in:
        print(f"         judging    {judge_in:,} in / {judge_out:,} out")
    print(f"cost     ~${gen_cost + judge_cost:.4f} on {args.model}")

    faith = next((v for k, v in scores.items() if "faithful" in k), 1.0)
    if faith < 0.9:
        print("\n  Faithfulness under 0.90 is the one to act on: the model is stating")
        print("  things the retrieved regulation does not say. In a notice that is a")
        print("  false claim about a worker's conduct.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "n_cases": len(samples),
        "scores": scores,
        "per_sample": per_sample,
        "tokens": {"generation_in": gen_in, "generation_out": gen_out,
                   "judge_in": judge_in, "judge_out": judge_out},
        "cost_usd": round(gen_cost + judge_cost, 4),
        "per_case": [
            {"question": s.user_input, "expected": c.expect_section,
             "n_contexts": len(s.retrieved_contexts), "response": s.response[:400]}
            for s, c in zip(samples, cases)
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
