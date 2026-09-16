# Agent workflow

Three agents, one LangGraph state machine. Lane C owns the graph; Lane D owns the
database, the roster and the console.

Scoring and identity rules are decided in
[SEVERITY_AND_IDENTITY.md](SEVERITY_AND_IDENTITY.md); this document is the graph.

```
                    Lane A · pipeline.py
                            │
                    ViolationEvent (JSON)
                            │
                            ▼
                 ┌──────────────────────┐
                 │  GATE  is_actionable │  deterministic, no LLM
                 └──────────┬───────────┘
                     no ────┴──── yes
                     │             │
              review queue         ▼
              (human looks)  ┌─────────────────────────────────────┐
                             │  AGENT 1 · ASSESSOR                 │
                             │  What happened, and under what rule │
                             │                                     │
                             │  tools                              │
                             │    lookup_clause(class, zone)       │ ← deterministic
                             │    retrieve_clause_text(clause_id)  │ ← RAG / FAISS
                             │    score_baseline(zone, missing)    │ ← deterministic
                             │                                     │
                             │  emits  Draft + baseline severity   │
                             └──────────────┬──────────────────────┘
                                            │
                             ┌──────────────▼──────────────┐
                             │ GUARDRAIL  citation present │  no citation → park,
                             └──────────────┬──────────────┘  never proceed
                                            │
                             ┌──────────────▼──────────────────────┐
                             │  SUPERVISOR · IDENTIFY              │
                             │  dashboard prompt, urgency set by   │
                             │  baseline severity                  │
                             │                                     │
                             │  picks the worker from the roster   │
                             │  (dropdown, never free text)        │
                             │                                     │
                             │  cannot identify → park as          │
                             │  blocked_on = worker_identity       │
                             └──────────────┬──────────────────────┘
                                            │
                             ┌──────────────▼──────────────────────┐
                             │  AGENT 2 · ADJUDICATOR              │
                             │  Given history, what is proportionate│
                             │                                     │
                             │  tools                              │
                             │    get_worker_history(worker_id)    │
                             │    final_severity(baseline, priors) │ ← deterministic
                             │    assess_confidence(event)         │ ← deterministic
                             │    draft_notice(...)                │
                             │                                     │
                             │  emits  Decision                    │
                             └──────────────┬──────────────────────┘
                                            │
              ┌──────────────┬──────────────┼──────────────┬──────────────┐
              ▼              ▼              ▼              ▼              ▼
          COMPLIANT      LOG_ONLY        WARNING      ESCALATION     STOP_WORK
             0            1–2             3–4            5–7            8+
              │              │              │              │              │
              │              │       draft email    draft letter   draft letter
              │              │              │              │              │
              │              │       ┌──────▼──────────────▼──────────────▼──────┐
              │              │       │ GUARDRAIL  SUPERVISOR APPROVES AND SENDS  │
              │              │       │ the system never sends anything itself    │
              │              │       └──────────────────┬────────────────────────┘
              └──────────────┴──────────────────────────┘
                                            │
                             ┌──────────────▼──────────────────────┐
                             │  AGENT 3 · RECORDER                 │
                             │  What goes on file                  │
                             │                                     │
                             │  tools                              │
                             │    commit_record(payload)           │ ← deterministic
                             │    verify_record(record_id)         │ ← read-back
                             │    attach_identity(event, worker)   │ ← update, later
                             │                                     │
                             │  emits  CommittedRecord             │
                             └──────────────┬──────────────────────┘
                                            │
                             ┌──────────────▼──────────────┐
                             │ GUARDRAIL  read-back check  │  write not confirmed →
                             └──────────────┬──────────────┘  retry, then alert
                                            │
                                         SQLite
                                   (history for Agent 2)
```

**The system never sends anything.** Agents produce drafts and records; a supervisor
decides what leaves the building.

---

## The loop that makes the system work

Agent 3 writes the history that Agent 2 reads on the *next* event. That is the memory
loop, and it is also the single point of failure: **if Agent 3 fails to record, Agent 2
sees a first offence where there was a third.** The system silently under-escalates and
nothing in the output looks wrong.

Two consequences for the build:

1. `commit_record` is a plain SQL write, not a judgement. It is not left to a model to
   remember — the graph verifies it by read-back and refuses to close the run otherwise.
2. The failure must be visible. A `pending_write` that never resolves surfaces in the
   console rather than disappearing.

---

## Identity comes before the action, not after

"Repeat" is a conclusion, not an input. The agent cannot route down the repeat branch
before it knows who the person is — that is the very thing the history lookup determines.

So the order is fixed:

```
1. baseline severity      zone weights only, no history needed
2. supervisor identifies  urgency of the prompt set by the baseline
3. history lookup         by worker_id
4. final severity         baseline + 2 per prior in the last 7 days
5. route to an action
```

Agent 3 records every event immediately with `worker_id = NULL`. Identity arrives later
as an update, which means **unattributed events count toward nobody's history** — you
cannot escalate against someone on the basis of incidents nobody confirmed were them.

---

## Two axes, not one number

| axis | from | answers |
| --- | --- | --- |
| **severity** | zone item weights + priors | how bad is this if true |
| **confidence** | detector recall, frames confirmed, person confidence | how sure are we |

Detector recall is deliberately **not** folded into severity. A missing helmet is equally
dangerous whether the detector is 79% or 99% reliable; treating it otherwise would mean
"we are less sure, therefore it is less serious".

The action needs both. **High severity with low confidence is urgent human review**, not
a downgraded escalation.

---

## State

One object flows through the graph, each node adding to it. Defined in `agents/state.py`.

```python
CaseState:
    event:      ViolationEvent      # from Lane A, never modified
    draft:      Draft | None        # agent 1
    decision:   Decision | None     # agent 2
    record:     CommittedRecord | None   # agent 3
    trace:      list[Step]          # every tool call, for week-6 observability
    blocked_on: str | None          # "human_approval" | "worker_identity" | None
```

Agents append. Nothing overwrites `event` — the finding is evidence and evidence does not
change after the fact.

---

## Two rules that are easy to get wrong

### RAG must not choose the citation

Cosine similarity picking which law a worker allegedly broke yields a confident,
real-looking, wrong citation — worse than none, and exactly what the citation guardrail
exists to catch.

The violation space is small and enumerable: 6 PPE classes × N zones. `kb/clauses.yaml`
maps it deterministically. RAG then retrieves the **text** of the clause you already know
applies.

```
missing "helmet" in any zone  →  1926.100   (deterministic lookup)
                              →  FAISS retrieves the text of 1926.100
```

This also gives Lane B its citation-accuracy ground truth for free.

### OSHA does not penalise workers

OSHA penalties are **federal fines assessed against the employer** after an inspection.
OSHA has no mechanism to fine an individual worker, and this system is not an inspector.

| field | source |
| --- | --- |
| `clause` | 29 CFR 1926 — what rule was breached |
| `severity` | the scored rubric — how serious, given zone and history |
| `action` | **site disciplinary policy** — warning, escalation, stop-work |

Cite the regulation for *what the rule is*; cite site policy for *what happens next*. Say
which is which.

---

## Why three agents rather than one with tools

Worth being able to answer, because it will be asked.

Each agent has a different **job, input and failure mode**:

| agent | decides | fails by |
| --- | --- | --- |
| Assessor | what rule applies | citing the wrong clause |
| Adjudicator | what response is proportionate | over- or under-escalating |
| Recorder | what goes on file | losing history, breaking future escalation |

They also run at different **trust levels**. The Assessor works from one immutable event.
The Adjudicator reads history and recommends action against a named person. The Recorder
mutates shared state. Splitting them puts the guardrails on the edges *between* agents,
where they can actually block — a single agent with five tools has no seam to put a human
approval gate into.

## Where determinism stops and autonomy starts

The line is not drawn by what is easy to automate. It is drawn by asking **what does
being wrong cost, and who could tell?**

Deterministic, because a wrong answer is expensive and invisible:

| Decision | Owner | Why not the model |
|---|---|---|
| Which clause governs a missing item | `kb/clauses.yaml` | A hallucinated citation reads exactly like a real one. Nobody on the review panel will check 29 CFR by hand. |
| What the severity number is | `ClauseMap.score()` | It has to be the same number every time, or Macro-F1 against human labels means nothing. |
| How many priors a worker has | `app/db.py` | Counting is not judgement, and an error here silently turns a third offence into a first. |
| What gets written, and whether it landed | `agents/recorder.py` | An LLM that forgets to call `commit_record` loses the history the next escalation depends on. |

Agentic, because judgement is the actual task:

| Decision | Owner | Why not a rule |
|---|---|---|
| Which tools this finding needs, in what order | Assessor | Some cases need one clause, some need three and the text of each. A fixed sequence would run every lookup on every event and still handle the odd one badly. |
| How to describe what happened | Assessor | One neutral sentence from structured facts is exactly what language models are for. |
| What response is proportionate | Adjudicator | The band table gives a default; the edges are where a human would think. |
| When it cannot tell | Adjudicator | Recognising that identity was never bound, and saying so, is a judgement a rule cannot make on its behalf. |

### The failure that set the boundary

An early run had the Adjudicator call `get_worker_history`, read `prior_violations: 2`,
and then call `final_severity` with `priors=0`. It still chose escalation -- it argued
the case up in prose -- so the *action* was right. The *number* recorded against it was
3.0, a warning-band score sitting underneath an escalation.

The tool was deterministic. Its inputs were not. A deterministic function whose arguments
the model supplies is only as reliable as the model's willingness to retype a figure it
was just handed.

So `final_severity` now takes **no arguments**. It reads the zone, the missing items and
the confirmed prior count itself, fetching history first if the agent scored before
looking anyone up. The agent still decides whether to score and what to do with the
answer; it no longer decides what goes in.

`gate_action_matches_the_score` was added for the same reason. An action gentler than the
band is the agent's to make. An action harsher than the band may well be correct, but it
is the model deciding someone deserves more than the auditable rubric says -- so it goes
out only with a human signature. That gate would have caught the bug above even on the
run where the action happened to be right, because the mismatch is visible in the shape
of the decision, not just in the outcome.

### Why the guardrails sit on the edges

Every gate checks an **output**, never a process. That is what makes the autonomy safe
rather than decorative:

- An Assessor that skips `lookup_clause` produces a draft with no citation, and
  `gate_citation_present` refuses it.
- One that cites general industry gets caught by `gate_citations_are_construction`.
- One that presents gloves as a regulatory requirement fails
  `gate_site_policy_is_labelled`.

Forcing the tool call would have been the weaker design. An agent made to call a tool can
still ignore what came back; an agent whose output is checked cannot.
