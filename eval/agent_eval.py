"""
Score the agents against labelled cases: Macro-F1, a confusion matrix, critical misses.

Each case runs through the real graph -- intake, Assessor, Adjudicator, Recorder, every
guardrail -- on a fresh database seeded with that worker's history. What is scored is
the action the system recommends, against the labelled action for the same facts.

The deterministic rubric is scored the same way, as a baseline. If the agents score no
better than the band table, the model is adding cost and not judgement, and the report
should say so.

    python eval/agent_eval.py --dry-run      # offline: a stub model, no key, no cost
    python eval/agent_eval.py --limit 2      # live smoke test, about a cent
    python eval/agent_eval.py                # all 30 cases, live

The measures:

    Macro-F1            F1 per action, averaged with every action weighted equally, so
                        the rare actions count as much as the common ones
    confusion matrix    what the system said against the label
    critical miss rate  of the cases labelled serious (escalation or stop_work),
                        the share where the system recommended something gentler. A
                        case the system could not decide counts as a miss -- the
                        conservative reading.
    over-escalation     the share where the system went harsher than the label

Costs are counted from real token usage. Needs OPENAI_API_KEY in .env for a live run.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from agent_cases import (ACTIONS, CASES, LABELS, build_event, load_labels,  # noqa: E402
                         rubric_action, seed_history)

FIGURES = ROOT / "eval" / "figures"
RESULTS = FIGURES / "agent_results.json"
MATRIX = FIGURES / "agent_confusion_matrix.png"
NO_DECISION = "no_decision"
SERIOUS = ("escalation", "stop_work")
PRICE_IN, PRICE_OUT = 0.15, 0.60         # gpt-4o-mini, USD per 1M tokens


class Counting:
    """Wraps an OpenAI client and adds up the tokens every call actually used."""

    def __init__(self, client):
        self._client = client
        self.tokens_in = self.tokens_out = self.calls = 0
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        reply = self._client.chat.completions.create(**kwargs)
        usage = getattr(reply, "usage", None)
        if usage is not None:
            self.tokens_in += usage.prompt_tokens
            self.tokens_out += usage.completion_tokens
        self.calls += 1
        return reply


def stub_client(case):
    """Offline: a scripted model that cites correctly and chooses the rubric's action.
    It proves the harness runs end to end; its scores measure nothing."""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_agents import StubCall, StubClient

    from agents.tools import lookup_clause
    rules = [lookup_clause(item, case.zone) for item in case.missing]
    policy = any(not r.is_regulatory for r in rules)
    return StubClient([
        [StubCall("submit_draft", {
            "summary": "A confirmed PPE finding" + (", in part under site policy."
                                                    if policy else "."),
            "clause_ids": [r.clause_id for r in rules]})],
        [StubCall("get_worker_history", {})],
        [StubCall("final_severity", {})],
        [StubCall("submit_decision", {"action": rubric_action(case),
                                      "rationale": "stub", "draft_body": "stub"})],
    ])


def run_one(case, client, model):
    from agents.graph import run_case
    from app.db import EventStore

    with tempfile.TemporaryDirectory() as tmp:
        db = EventStore(Path(tmp) / "eval.db")
        seed_history(db, case)
        started = time.perf_counter()
        out = run_case(build_event(case), db, client=client, model=model)
        seconds = time.perf_counter() - started
    d = out["case"].decision
    action = d.action.value if d is not None and out["outcome"] in ("done", "alert") \
        else NO_DECISION
    return {
        "case": case.case_id,
        "zone": f"{case.camera}/{case.zone}",
        "missing": list(case.missing),
        "identified": case.identified,
        "priors": case.priors if case.identified else None,
        "evidence": case.evidence,
        "agent": action,
        "rubric": rubric_action(case),
        "outcome": out["outcome"],
        "reason": out.get("reason", ""),
        "severity": d.severity.total if d else None,
        "requires_approval": bool(d and d.requires_approval),
        "blocked_on": out["case"].blocked_on.value if out["case"].blocked_on else None,
        "rationale": d.rationale if d else "",
        "draft_body": d.draft_body if d else "",
        # a case that did not finish is the one somebody will want explained
        "trace": [] if out["outcome"] == "done" else [
            f"{s.agent} {s.tool} {json.dumps(s.arguments)[:80]} -> {s.result_summary[:80]}"
            for s in out["case"].trace],
        "seconds": round(seconds, 2),
    }


def score(truth: list[str], pred: list[str]) -> dict:
    """The measures the proposal named, plus the two that make them readable."""
    from sklearn.metrics import f1_score

    rank = {a: i for i, a in enumerate(ACTIONS)}
    # Averaged over the actions only. A case the system could not decide is an error
    # against its true action (it lowers that action's recall); "no decision" is not
    # averaged in as a class of its own, because it can never be right.
    labels = [a for a in ACTIONS if a in truth or a in pred]
    serious = [i for i, t in enumerate(truth) if t in SERIOUS]
    missed = [i for i in serious
              if pred[i] == NO_DECISION or rank[pred[i]] < rank["escalation"]]
    decided = [i for i, p in enumerate(pred) if p != NO_DECISION]
    return {
        "n": len(truth),
        "macro_f1": round(f1_score(truth, pred, labels=labels, average="macro",
                                   zero_division=0), 3),
        "per_class_f1": dict(zip(labels, (round(x, 3) for x in f1_score(
            truth, pred, labels=labels, average=None, zero_division=0)))),
        "accuracy": round(sum(t == p for t, p in zip(truth, pred)) / len(truth), 3),
        "within_one_step": round(sum(
            p != NO_DECISION and abs(rank[t] - rank[p]) <= 1
            for t, p in zip(truth, pred)) / len(truth), 3),
        "serious_cases": len(serious),
        "critical_misses": len(missed),
        "critical_miss_rate": round(len(missed) / len(serious), 3) if serious else None,
        "critical_missed_cases": missed,
        "over_escalation_rate": round(sum(
            rank[pred[i]] > rank[truth[i]] for i in decided) / len(truth), 3),
        "no_decision": len(truth) - len(decided),
    }


def confusion_figure(truth, agent, rubric, path: Path = MATRIX) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cols = list(ACTIONS) + ([NO_DECISION] if NO_DECISION in agent else [])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, pred, title in ((axes[0], agent, "Agents vs label"),
                            (axes[1], rubric, "Rubric alone vs label")):
        m = confusion_matrix(truth, pred, labels=cols)[: len(ACTIONS)]
        ax.imshow(m, cmap="Blues")
        ax.set_xticks(range(len(cols)), [c.replace("_", "\n") for c in cols], fontsize=9)
        ax.set_yticks(range(len(ACTIONS)), ACTIONS, fontsize=9)
        ax.set_xlabel("system recommended")
        ax.set_ylabel("label")
        ax.set_title(title)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i, j]:
                    ax.text(j, i, m[i, j], ha="center", va="center",
                            color="white" if m[i, j] > m.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="stub model, no key, no cost; proves the harness runs")
    ap.add_argument("--limit", type=int, default=None, help="first N cases only")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score the last run against the current labels; no model call")
    args = ap.parse_args()

    if args.rescore:
        return rescore()
    cases = CASES[: args.limit] if args.limit else CASES
    if LABELS.exists():
        labels = load_labels()
        missing = [c.case_id for c in cases if c.case_id not in labels]
        if missing:
            sys.exit(f"no label for: {', '.join(missing)}")
    elif args.dry_run or args.limit:
        labels = {c.case_id: rubric_action(c) for c in cases}
        print("no labels yet -- scoring against the rubric as a placeholder, which "
              "measures nothing; nothing is written\n")
    else:
        sys.exit(f"no labels at {LABELS}. Label eval/agent_labels.xlsx, then run "
                 f"`python eval/agent_cases.py --csv`.")

    counter = None
    if not args.dry_run:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(ROOT / ".env")
        counter = Counting(OpenAI())

    rows = []
    for case in cases:
        client = stub_client(case) if args.dry_run else counter
        row = run_one(case, client, args.model)
        row["human"] = labels[case.case_id]
        rows.append(row)
        flag = "" if row["agent"] == row["human"] else "   <-- differs"
        print(f"{case.case_id}  label {row['human']:<11} agent {row['agent']:<11} "
              f"rubric {row['rubric']:<11} {row['outcome']}{flag}")

    report = {"model": "stub" if args.dry_run else args.model,
              "run_at": datetime.now().isoformat(timespec="seconds"),
              "n_cases": len(rows), "labels": str(LABELS.relative_to(ROOT))}
    if counter is not None:
        cost = counter.tokens_in / 1e6 * PRICE_IN + counter.tokens_out / 1e6 * PRICE_OUT
        report["tokens"] = {"in": counter.tokens_in, "out": counter.tokens_out,
                            "calls": counter.calls, "usd": round(cost, 4)}
        report["seconds_per_case"] = round(sum(r["seconds"] for r in rows) / len(rows), 1)
        print(f"\n{counter.calls} calls, {counter.tokens_in:,} in / "
              f"{counter.tokens_out:,} out, ~${cost:.4f}")
    finish(report, rows, write=not (args.dry_run or args.limit))


def finish(report: dict, rows: list[dict], write: bool = True) -> None:
    """Score the rows, print the comparison, and write the results and the figure."""
    truth = [r["human"] for r in rows]
    agent = score(truth, [r["agent"] for r in rows])
    rubric = score(truth, [r["rubric"] for r in rows])
    for m in (agent, rubric):
        m["critical_missed_cases"] = [rows[i]["case"] for i in m["critical_missed_cases"]]

    print(f"\n{'':<22}{'agents':>10}{'rubric':>10}")
    for key in ("macro_f1", "accuracy", "within_one_step", "critical_miss_rate",
                "over_escalation_rate", "no_decision"):
        print(f"{key:<22}{str(agent[key]):>10}{str(rubric[key]):>10}")
    print(f"serious cases (label): {agent['serious_cases']}")

    report.update(agents=agent, rubric=rubric, cases=rows)
    if not write:
        print("\n(dry run or partial run: results not written)")
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    confusion_figure(truth, [r["agent"] for r in rows], [r["rubric"] for r in rows])
    print(f"\nwrote {RESULTS.relative_to(ROOT)} and {MATRIX.relative_to(ROOT)}")


def rescore() -> None:
    """Labels changed; the agents' answers did not. Score again without calling a model."""
    report = json.loads(RESULTS.read_text(encoding="utf-8"))
    labels = load_labels()
    rows = report["cases"]
    for row in rows:
        row["human"] = labels[row["case"]]
    report["rescored_at"] = datetime.now().isoformat(timespec="seconds")
    finish(report, rows)


if __name__ == "__main__":
    main()
