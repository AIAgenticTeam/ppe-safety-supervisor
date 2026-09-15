# Agent workflow

Three agents, one LangGraph state machine. Lane C owns the graph; Lane D owns the
database and the console it writes to.

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
                             │                                     │
                             │  emits  Draft                       │
                             └──────────────┬──────────────────────┘
                                            │
                             ┌──────────────▼──────────────┐
                             │ GUARDRAIL  citation present │  no citation → park,
                             └──────────────┬──────────────┘  never proceed
                                            │
                             ┌──────────────▼──────────────────────┐
                             │  AGENT 2 · ADJUDICATOR              │
                             │  Given history, what should happen  │
                             │                                     │
                             │  tools                              │
                             │    get_worker_history(worker_ref)   │
                             │    score_severity(rubric)           │
                             │    draft_warning(...)               │
                             │    draft_escalation(...)            │
                             │                                     │
                             │  emits  Decision                    │
                             └──────────────┬──────────────────────┘
                                            │
                        ┌───────────────────┼───────────────────┐
                        ▼                   ▼                   ▼
                   NO_ACTION            WARNING            ESCALATION
                   log only          draft email        draft letter
                        │                   │                   │
                        │                   │        ┌──────────▼──────────┐
                        │                   │        │ GUARDRAIL           │
                        │                   │        │ HUMAN APPROVAL      │
                        │                   │        │ blocks until signed │
                        │                   │        └──────────┬──────────┘
                        └───────────────────┴──────────────────┘
                                            │
                             ┌──────────────▼──────────────────────┐
                             │  AGENT 3 · RECORDER                 │
                             │  Who was this, and what is on file  │
                             │                                     │
                             │  tools                              │
                             │    resolve_worker(candidate)        │ ← judgement
                             │    commit_record(payload)           │ ← deterministic
                             │    verify_record(record_id)         │ ← read-back
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

Nothing is ever emailed or sent by the system. Agents 2 and 3 produce **drafts and
records**; a human in the console decides what leaves the building.

---

## The loop that makes the system work

Agent 3 writes the history that Agent 2 reads on the *next* event. That is the memory
loop, and it is also the single point of failure: **if Agent 3 fails to record, Agent 2
sees a first offence where there was a third.** The system silently under-escalates, and
nothing in the output looks wrong.

Two consequences for the build:

1. `commit_record` is a plain SQL write, not a judgement. It must not be left to a model
   to remember to call it — the graph verifies the write with a read-back and refuses to
   close the run otherwise.
2. The failure has to be visible. A `pending_write` state that never resolves should
   surface in the console, not disappear.

---

## State

One object flows through the graph, each node adding to it. Defined in `agents/state.py`.

```python
CaseState:
    event:      ViolationEvent      # from Lane A, never modified
    draft:      Draft | None        # agent 1
    decision:   Decision | None     # agent 2
    record:     CommittedRecord | None   # agent 3
    trace:      list[Step]          # every tool call, for observability (week 6)
    blocked_on: str | None          # "human_approval" | "worker_identity" | None
```

Agents append. Nothing overwrites `event` — the finding is evidence and stays immutable.

---

## Three corrections to the original description

### 1. RAG must not choose the citation

The description says Agent 1 "uses RAG to get the OSHA citation that is similar to the
violation." Do not do that. Cosine similarity choosing which law a worker allegedly broke
produces a confident, real-looking, wrong citation — worse than none, and precisely the
failure the citation guardrail exists to catch.

The violation space is small and enumerable: 6 PPE classes × N zones. `kb/clauses.yaml`
maps it deterministically. RAG then retrieves the **text** of the clause you already know
applies.

```
missing "helmet" in any zone  →  1926.100  (deterministic lookup)
                              →  FAISS retrieves the text of 1926.100
```

This also gives Lane B its citation-accuracy ground truth for free.

### 2. OSHA does not penalise workers

The description has Agent 1 assigning "the correct penalty according to the OSHA."
OSHA penalties are **federal monetary fines assessed against the employer**, following an
inspection. OSHA does not fine individual workers, and this system is not an OSHA
inspector.

A system that tells a worker "your OSHA penalty is $X" is stating something false about a
named person's legal liability. Replace it with:

| field | source |
| --- | --- |
| `clause` | 29 CFR 1926 — what rule was breached |
| `severity` | the scored rubric — how serious, given zone and history |
| `action` | **site disciplinary policy** — verbal warning, written warning, stop-work |

Cite the regulation for *what the rule is*; cite site policy for *what happens next*. Say
which is which. That distinction is defensible in front of a panel; conflating them is not.

### 3. Worker identity does not come from the vision system

Both Agent 2 and Agent 3 depend on a worker id, and **Lane A cannot supply one**.
ByteTrack ids do not survive a session, a camera hand-off, or someone walking out of
frame. `subject.worker_ref` is `null` in every event the pipeline produces today.

So the identity has to enter from somewhere else. In order of preference:

1. **A supervisor binds it in the console** when approving — the human-in-the-loop step
   does double duty, and identification stays a human act
2. **A badge or roster integration** — out of scope for two weeks
3. **Fall back to zone-level history** — "the walkway generated five violations this
   week" needs no identity at all, and is arguably the more useful insight

Until one exists, Agent 2's history lookup returns empty and every event looks like a
first offence. `resolve_worker` in Agent 3 is the node where this gets handled: it either
matches a supervisor-supplied id, or marks the case `blocked_on="worker_identity"` and
routes it to the console rather than inventing an identity.

---

## Why three agents rather than one with tools

Worth being able to answer, because it will be asked.

Each agent has a different **job, input and failure mode**:

| agent | decides | fails by |
| --- | --- | --- |
| Assessor | what rule applies | citing the wrong clause |
| Adjudicator | what response is proportionate | over- or under-escalating |
| Recorder | who this was, and what goes on file | losing history, breaking future escalation |

They also run on different **trust levels**. The Assessor works from one immutable event.
The Adjudicator reads history and can recommend action against a named person. The
Recorder mutates shared state. Splitting them means the guardrails sit on the edges
between, where they can actually block — a single agent with five tools has no seam to
put a human approval gate into.
