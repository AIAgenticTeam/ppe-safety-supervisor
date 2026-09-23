"""
Agent 1 · Assessor -- what happened, and under what rule.

Genuinely agentic: the model is given tools and decides which to call, in what order,
and when it has enough. There is no scripted sequence. It may look up one clause or
five, fetch text or not, re-check a zone it is unsure about.

That freedom is safe because the guardrails check the OUTPUT, not the process. An
assessor that skips `lookup_clause` simply produces a draft with no citation, and
`gate_citation_present` refuses it. Forcing the call would have been weaker anyway --
an agent made to call a tool can still ignore what came back.

What it may NOT do is invent a citation. `lookup_clause` is deterministic, so the model
cannot choose which regulation applies; it can only choose whether to ask.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from agents.llm import MALFORMED, ModelUnavailable, complete, text, tool_arguments  # noqa: E402
from agents.state import Blocker, CaseState, Citation, Draft  # noqa: E402
from agents.tools import (event_facts, lookup_clause, retrieve_clause_text,  # noqa: E402
                          score_baseline)

MAX_STEPS = 6          # generous: the agent rarely needs more than three

SYSTEM = """You assess construction PPE findings against 29 CFR 1926.

You are given the facts of one finding. Your job is to establish which regulation applies
to each missing item and write one plain sentence describing what happened.

Tools are available. Decide for yourself which to use and in what order. You will
normally want the governing clause for each missing item, and often its text.

Rules you must not break:
- Never state a clause number that did not come from lookup_clause. You cannot know which
  regulation applies; the tool does.
- When a clause comes back with basis "site_policy", say so in your summary using the
  words "site policy". Construction has no hand-protection clause and no general
  high-visibility clause, so gloves and vests rest on the employer's general duty.
  Presenting that as a regulatory requirement overstates it.
- Describe only what the facts say. You cannot see the photograph. Do not describe the
  worker, their appearance, their intent, or what they were doing beyond the zone given.
- Do not state consequences, penalties or actions. That is not your job.

Write the summary in one sentence, factual and neutral. When you are finished, call
submit_draft exactly once."""

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "lookup_clause",
        "description": ("Which 29 CFR 1926 clause covers a missing PPE item in a zone, "
                        "and whether it rests on regulation or site policy. "
                        "Deterministic -- this is the only source of a citation."),
        "parameters": {"type": "object", "properties": {
            "item": {"type": "string",
                     "enum": ["helmet", "goggles", "gloves", "boots", "vest"]},
            "zone": {"type": "string", "description": "zone name, e.g. grinding_station"},
        }, "required": ["item"]}}},
    {"type": "function", "function": {
        "name": "retrieve_clause_text",
        "description": "The text of a clause, fetched by id. Use it to quote accurately.",
        "parameters": {"type": "object", "properties": {
            "clause_id": {"type": "string", "description": "e.g. 1926.100"},
        }, "required": ["clause_id"]}}},
    {"type": "function", "function": {
        "name": "score_baseline",
        "description": ("Severity from the zone's item weights, before history is known. "
                        "Deterministic."),
        "parameters": {"type": "object", "properties": {
            "zone": {"type": "string"},
            "missing": {"type": "array", "items": {"type": "string"}},
            "camera": {"type": "string"},
        }, "required": ["zone", "missing"]}}},
    {"type": "function", "function": {
        "name": "submit_draft",
        "description": "Finish. Call once, when you have what you need.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string",
                        "description": "one factual sentence about what happened"},
            "clause_ids": {"type": "array", "items": {"type": "string"},
                           "description": "clauses from lookup_clause only"},
            "confidence_note": {"type": "string",
                                "description": "optional caveat about the evidence"},
        }, "required": ["summary", "clause_ids"]}}},
]


def _dispatch(name: str, args: dict, facts: dict) -> str:
    """Run a tool the agent chose. Errors come back as text so it can recover."""
    try:
        if name == "lookup_clause":
            zone = args.get("zone") or facts.get("zone_name") or "default"
            return json.dumps(lookup_clause(args["item"], zone).as_dict())
        if name == "retrieve_clause_text":
            return retrieve_clause_text(args["clause_id"])
        if name == "score_baseline":
            result = score_baseline(args["zone"], args["missing"], args.get("camera"),
                                    required=facts.get("required_ppe"))
            return json.dumps({"baseline": result.baseline, "band": result.band,
                               "explain": result.explain()})
    except Exception as e:
        return f"ERROR: {e}"
    return f"ERROR: unknown tool {name}"


def assess(state: CaseState, client=None, model: str = "gpt-4o-mini") -> CaseState:
    """Run the Assessor over one case, appending a Draft to the state."""
    from openai import OpenAI

    if client is None:
        try:
            client = OpenAI()
        except Exception as exc:      # noqa: BLE001 -- usually no API key
            # Same outcome as an unreachable model: the finding is kept, the
            # case parks, and a human is told why. A 500 here said nothing.
            state.log("assessor", "model", {}, f"unavailable: {exc}")
            state.block(Blocker.MODEL_UNAVAILABLE)
            state.draft = None
            return state
    facts = event_facts(state.event)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(facts, indent=2)},
    ]

    draft: Draft | None = None
    for step in range(MAX_STEPS):
        try:
            reply = complete(client, model=model, messages=messages,
                             tools=TOOL_SCHEMAS, temperature=0)
        except ModelUnavailable as exc:
            # A dead API is not a reason to lose a finding. The case parks, the event
            # is still recorded, and a human sees it in the queue.
            state.log("assessor", "model", {}, f"unavailable: {exc}")
            state.block(Blocker.MODEL_UNAVAILABLE)
            state.draft = None
            return state
        choice = reply.choices[0].message
        messages.append(choice)

        if not choice.tool_calls:
            # The agent stopped without submitting. Nudge once, then give up and let
            # the citation guardrail refuse the case rather than invent a draft.
            messages.append({"role": "user",
                             "content": "Call submit_draft with what you have."})
            continue

        for call in choice.tool_calls:
            args = tool_arguments(call)
            if args is None:
                state.log("assessor", call.function.name, {}, "malformed arguments")
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": MALFORMED})
                continue

            if call.function.name == "submit_draft":
                citations = []
                clause_ids = args.get("clause_ids")
                for clause_id in clause_ids if isinstance(clause_ids, list) else []:
                    # Re-derive from the map rather than trusting the model's echo. It
                    # cannot smuggle a clause in through the summary.
                    for item in facts.get("missing", []):
                        rule = lookup_clause(item, facts.get("zone_name") or "default")
                        if rule.clause_id == clause_id:
                            citations.append(Citation(
                                clause_id=rule.clause_id, title=rule.title,
                                is_site_policy=not rule.is_regulatory))
                            break
                draft = Draft(
                    summary=text(args.get("summary")),
                    citations=citations,
                    items_missing=list(facts.get("missing", [])),
                    confidence_note=text(args.get("confidence_note")),
                )
                state.log("assessor", "submit_draft",
                          {"clause_ids": args.get("clause_ids")},
                          f"{len(citations)} citation(s)")
                break

            result = _dispatch(call.function.name, args, facts)
            state.log("assessor", call.function.name, args, result[:120])
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        if draft is not None:
            break

    state.draft = draft
    return state
