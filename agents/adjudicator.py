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

from agents.llm import ModelUnavailable, complete  # noqa: E402
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

final_severity already accounts for prior violations; it reads the confirmed count
itself. If you believe a case deserves more than the band it returns, check that you
called get_worker_history first -- the repeat penalty is in that number, not something
you add on top of it.

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
        "description": ("Severity for this finding: the zone's item weights plus the "
                        "repeat penalty. Takes no arguments -- it reads the zone, the "
                        "missing items and the confirmed prior count itself, so the "
                        "number cannot be affected by how you ask."),
        "parameters": {"type": "object", "properties": {}}}},
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
            # Every input is authoritative, none of it is the model's to restate.
            # It once read "prior_violations: 2" and then asked for a score with
            # priors=0, which produced a warning-band number attached to an
            # escalation -- the right call recorded against the wrong figure.
            # The model still decides WHETHER to score and what to do about the
            # answer; it no longer decides what goes in.
            hist = captured.get("history")
            if hist is None:
                # Scored before looking anyone up. Fetch it now rather than
                # defaulting to zero, which would read as a clean record.
                hist = get_worker_history(facts.get("worker_ref"), db=db)
                captured["history"] = hist
            priors = hist.prior_violations if hist.resolved else 0
            s = final_severity(facts.get("zone_name") or "default",
                               facts.get("missing", []), priors,
                               event.get("camera_id"),
                               required=facts.get("required_ppe"))
            captured["severity"] = s
            note = ("prior count is unconfirmed and was scored as zero; do not "
                    "read this as a clean record"
                    if not hist.resolved else f"includes {priors} confirmed prior(s)")
            return json.dumps({"baseline": s.baseline, "final": s.final,
                               "band": s.band, "explain": s.explain(), "note": note})
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

    if client is None:
        try:
            client = OpenAI()
        except Exception as exc:      # noqa: BLE001 -- usually no API key
            # Same outcome as an unreachable model: the finding is kept, the
            # case parks, and a human is told why. A 500 here said nothing.
            state.log("adjudicator", "model", {}, f"unavailable: {exc}")
            state.block(Blocker.MODEL_UNAVAILABLE)
            state.decision = None
            return state
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
        try:
            reply = complete(client, model=model, messages=messages,
                             tools=TOOL_SCHEMAS, temperature=0)
        except ModelUnavailable as exc:
            # A dead API is not a reason to lose a finding. The case parks, the event
            # is still recorded, and a human sees it in the queue.
            state.log("adjudicator", "model", {}, f"unavailable: {exc}")
            state.block(Blocker.MODEL_UNAVAILABLE)
            state.decision = None
            return state
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
