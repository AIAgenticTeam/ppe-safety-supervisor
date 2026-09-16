"""
Agent 2 · Adjudicator -- given history, what response is proportionate.

The band table gives a default. The agent's contribution is judgement at the edges and
the wording of the notice -- it may argue that a technically-escalation case deserves a
warning, provided it says why. What it may not do is invent the score, the priors, or the
citation: those come from tools.

It is also the agent that decides when it cannot decide. `get_worker_history` returns
`resolved=False` when identity was never bound, and the right response is to say so and
route to a human, not to treat unknown as zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from agents.state import Action, Blocker, CaseState, Decision, SeverityScore  # noqa: E402
from agents.tools import (assess_confidence, event_facts, final_severity,  # noqa: E402
                          get_worker_history)

MAX_STEPS = 6

SYSTEM = """You decide what response a confirmed PPE finding deserves on a construction site.

Everything you send goes to the SUPERVISOR, never to the worker. The system drafts; a
human sends. You are writing something a supervisor will read and decide whether to act on.

The severity bands are:
    0        compliant
    1-2      log_only     record it, no action
    3-4      warning      supervisor has a word
    5-7      escalation   formal, supervisor must respond
    8+       stop_work    halt the activity

Use the tools to get the score, the history and the evidence strength. Then choose an
action. You may depart from the band the score suggests, but you must say why in your
rationale -- and you may only depart downward on evidence, never upward on suspicion.

Hard rules:
- If get_worker_history returns resolved=false, identity was never confirmed. You do NOT
  know there are no priors; you know you cannot see them. Never claim a first offence on
  that basis, and never claim a repeat one.
- Severity says how bad this is if true. Confidence says how sure we are. They are
  separate. Serious-but-uncertain is a matter for urgent human review, not a downgraded
  action.
- Penalties are SITE DISCIPLINARY POLICY, not OSHA. OSHA fines employers after an
  inspection; it does not penalise workers, and this system is not an inspector. Cite the
  regulation for what the rule is; describe the action as site policy.
- Address the supervisor. Do not write to the worker. Do not name anyone.

When you have decided, call submit_decision exactly once."""

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_worker_history",
        "description": ("Prior confirmed violations for this worker in the window. "
                        "Returns resolved=false when identity was never bound -- that "
                        "means unknown, not zero."),
        "parameters": {"type": "object", "properties": {
            "worker_ref": {"type": ["string", "null"]},
            "window_days": {"type": "integer", "default": 7},
        }}}},
    {"type": "function", "function": {
        "name": "final_severity",
        "description": "Severity from zone weights plus the repeat penalty. Deterministic.",
        "parameters": {"type": "object", "properties": {
            "zone": {"type": "string"},
            "missing": {"type": "array", "items": {"type": "string"}},
            "priors": {"type": "integer"},
            "camera": {"type": "string"},
        }, "required": ["zone", "missing", "priors"]}}},
    {"type": "function", "function": {
        "name": "assess_confidence",
        "description": ("How trustworthy the finding is: detector recall, how many frames "
                        "sustained it, how confidently the person was detected."),
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "submit_decision",
        "description": "Finish. Call once.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string",
                       "enum": ["no_action", "log_only", "warning", "escalation",
                                "stop_work"]},
            "rationale": {"type": "string",
                          "description": "why this action, and why any departure "
                                         "from the band the score suggests"},
            "draft_body": {"type": "string",
                           "description": "the note to the supervisor"},
        }, "required": ["action", "rationale", "draft_body"]}}},
]


def _dispatch(name, args, facts, event, db, captured):
    try:
        if name == "get_worker_history":
            h = get_worker_history(args.get("worker_ref") or facts.get("worker_ref"),
                                   db=db, window_days=args.get("window_days", 7))
            captured["history"] = h
            return json.dumps({"resolved": h.resolved,
                               "prior_violations": h.prior_violations,
                               "note": h.note})
        if name == "final_severity":
            s = final_severity(args["zone"], args["missing"], args["priors"],
                               args.get("camera") or event.get("camera_id"),
                               required=facts.get("required_ppe"))
            captured["severity"] = s
            return json.dumps({"baseline": s.baseline, "final": s.final,
                               "band": s.band, "explain": s.explain()})
        if name == "assess_confidence":
            c = assess_confidence(event)
            captured["confidence"] = c
            return json.dumps({"score": c.score, "band": c.band,
                               "weakest_recall": c.weakest_recall, "note": c.note})
    except Exception as e:
        return f"ERROR: {e}"
    return f"ERROR: unknown tool {name}"


def adjudicate(state: CaseState, db=None, client=None,
               model: str = "gpt-4o-mini") -> CaseState:
    from openai import OpenAI

    client = client or OpenAI()
    facts = event_facts(state.event)
    captured: dict = {}

    context = dict(facts)
    context["assessor_summary"] = state.draft.summary if state.draft else ""
    context["citations"] = [
        {"clause": c.clause_id, "title": c.title, "site_policy": c.is_site_policy}
        for c in (state.draft.citations if state.draft else [])
    ]

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(context, indent=2)},
    ]

    decision: Decision | None = None
    for _ in range(MAX_STEPS):
        reply = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SCHEMAS, temperature=0)
        choice = reply.choices[0].message
        messages.append(choice)

        if not choice.tool_calls:
            messages.append({"role": "user",
                             "content": "Call submit_decision with what you have."})
            continue

        for call in choice.tool_calls:
            args = json.loads(call.function.arguments or "{}")

            if call.function.name == "submit_decision":
                action = Action(args.get("action", "log_only"))
                sev = captured.get("severity")
                conf = captured.get("confidence")
                hist = captured.get("history")

                decision = Decision(
                    action=action,
                    severity=SeverityScore(
                        base=sev.baseline if sev else 0,
                        repeat_count=hist.prior_violations if hist else 0,
                        evidence_strength=conf.weakest_recall if conf else 1.0,
                        total=float(sev.final) if sev else 0.0,
                        rationale=sev.explain() if sev else "",
                    ),
                    prior_violations=hist.prior_violations if hist and hist.resolved else 0,
                    draft_body=args.get("draft_body", "").strip(),
                    recipient_role="supervisor",
                    requires_approval=action in (Action.ESCALATION, Action.STOP_WORK),
                    rationale=args.get("rationale", "").strip(),
                )
                state.log("adjudicator", "submit_decision", {"action": action.value},
                          f"severity {decision.severity.total}")

                # Unknown identity must never read as a clean record.
                if hist is not None and not hist.resolved:
                    state.block(Blocker.WORKER_IDENTITY)
                break

            result = _dispatch(call.function.name, args, facts, state.event, db, captured)
            state.log("adjudicator", call.function.name, args, result[:120])
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        if decision is not None:
            break

    state.decision = decision
    return state
