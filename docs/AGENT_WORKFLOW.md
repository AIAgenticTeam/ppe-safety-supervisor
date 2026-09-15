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
