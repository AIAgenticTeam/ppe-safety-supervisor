# Repo review — 16 Sep 2026

Findings from a full pass over the project, ordered by what will actually hurt. Each
item is verified, not suspected. Tick them off as they are fixed.

---

## 1. Two sources of truth for what a zone requires — FIXED 16 Sep

`zones.json` declares `required_ppe` per zone. `kb/clauses.yaml` declares
`severity_weights` per zone. Nothing keeps them in agreement, and the failure is silent
in the dangerous direction:

```
missing helmet+boots in walkway -> helmet 5 + boots 0 = 5 -> escalation
```

`boots` is scored at **0** because the walkway has no weight for it. A required item that
scores zero is invisible to escalation — the system reports the violation but treats it
as costing nothing. Today five weights exist for items that are not required (harmless);
nothing prevents the reverse.

**Fixed.** `ClauseMap.score()` now raises on an unweighted required item instead of
scoring it zero, and `check_against_zones()` compares the two files directly. Both run in
`validate()` and in `tests/test_clause_map.py::test_zones_and_clauses_agree`, so the drift
cannot return unnoticed.

---

## 2. Zone names are global in one file, per-camera in the other — FIXED 16 Sep

`ClauseMap.weights_for("walkway")` takes a bare zone name, but zone names only exist
inside a camera in `zones.json`:

```
'walkway' used by ['cam_3', 'd_view02']   <-- SAME NAME, DIFFERENT REQUIREMENTS
    cam_3:    helmet          weights: {helmet: 5, vest: 2}
    d_view02: helmet + vest   weights: {helmet: 5, vest: 2}
```

Two different zones share one weight table because they happen to share a name. It is
benign now only because the numbers coincide. A second site with its own "walkway"
silently inherits these weights.

**Fixed.** Weights are keyed `camera/zone`. `cam_3/walkway` and `d_view02/walkway` now
hold different tables, and a test asserts they differ. A bare zone name still resolves to
the defaults -- documented, and unreachable in production because every event carries a
`camera_id`.

---

## 3. `worker_ref` is never produced, so the escalation beat cannot fire — RESOLVED 16 Sep

```
fixtures WITH worker_ref:              4/20   (hand-written by me)
REAL pipeline events with worker_ref:  0/1
```

The pipeline cannot set it — ByteTrack ids do not survive a session. Agent 2's history
lookup will therefore return empty for every real event, every event will look like a
first offence, and demo beat 4 has nothing to escalate on.

This is known and documented, but it is worth stating plainly: **until the console binds
identity, the repeat-offender path only works on fixtures.** That is a demo built on
synthetic data at the exact moment the panel is most attentive.

**Resolved.** The vision system never producing `worker_ref` turned out to be the
correct design, not the gap -- a ByteTrack id is not an identification. Identity is
now supplied by a supervisor and refused without one: `attach_identity` checks the
roster and records who bound it, and `scripts/run_agents.py --bind W-0412 --by khalid`
is the path Lane D's console will call.

The escalation beat fires on real footage, not fixtures: two bound priors took the
live Lane A event to severity 7.0 / escalation, approval required.

Lane D still owns the dropdown. It is no longer on the critical path for the beat to
work at all.

---

## 4. Two incompatible evidence formats — BUG

`pipeline.py` uses two different mechanisms depending on input:

| path | function | produces |
| --- | --- | --- |
| stills | `ppe_compliance.save_evidence` | bare crop, no frame, no annotation |
| video | `EvidenceStore.save` | annotated frame + crop pair, keyed by event_id |

The console will receive events whose `evidence` block means different things. Worse, the
stills path produces an unannotated crop — a reviewer sees a photograph of a worker with
no indication of what was alleged.

**Fix:** route stills through `EvidenceStore` too and delete `save_evidence`.

---

## 5. `event_id` is generated twice and the overwrite is load-bearing — FRAGILE

```python
event_id = make_event_id(camera_id, confirmation.track_id, when)
stored = store.save(event_id, ...)          # evidence filenames use THIS id
event = build_event(..., when=when)         # build_event generates its own id
event.event_id = event_id                   # overwrite -- keeps them in sync
```

Same inputs give the same id, so it works. But the evidence files are already written
under the first id, so if the two ever diverge the event points at filenames that do not
exist. The overwrite is doing real work and looks like a leftover.

**Fix:** pass `event_id` into `build_event` instead of overwriting after.

---

## 6. `severity_multiplier` is a zombie — DOC/CODE DRIFT

`docs/SEVERITY_AND_IDENTITY.md` says it was dropped as a double count. It is still:

- carried in `zones.json` on all five zones
- read and written in `zones.py`, `events.py`, `make_fixtures.py`
- part of the frozen `ViolationEvent` schema
- **asserted on by two tests**

So the documented design and the shipped schema disagree. Whoever writes Agent 2 will
find a multiplier in the event and reasonably use it, double-counting zone context.

**Still open.** Note this is a *different* field from `SeverityScore.zone_multiplier`,
which was removed from `agents/state.py` on 16 Sep -- that one was Lane C's own and
never read. This one is in the frozen event schema, 20 fixtures and `zones.json`, so
removing it is a schema bump and a fixture regeneration.

Contained for now: `event_facts()` withholds it, so no agent can double-count with
it, and `test_the_zombie_multiplier_never_reaches_an_agent` holds that in place.

**Fix:** decide before Lane A is touched again. Either remove it (schema bump,
fixtures regenerated, two tests rewritten) or keep it and correct the doc.

---

## 7. Severity scoring is built but wired to nothing — RESOLVED 16 Sep

`ClauseMap.score()` is called by no production code — only tests. Events carry no
`severity` field, so Lane C must import `ClauseMap` and compute it. That is the intended
design, but nobody has written the glue, and the event schema gives no hint it is
required.

**Resolved** the second way, and more strictly than planned. The Adjudicator calls
`final_severity`, which reads the zone, the missing items and the confirmed prior
count **itself** -- it takes no arguments at all. An early live run had the model
read `prior_violations: 2` and then ask for a score with `priors=0`, recording a
warning-band number underneath an escalation. A deterministic function whose inputs
the model supplies is not deterministic.

`gate_action_matches_the_score` now also refuses to let an action outrun its score
without a human signature.

---

## 8. The video path has no automated test — COVERAGE

`track_video`, `process_video` and `save_evidence` have **zero** test coverage. The video
pipeline has run exactly once, by hand. Everything proving it works is a terminal
screenshot.

229 tests and none of them touch the code that produces real events.

**Fix:** a smoke test over the 4-second `scene_02` clip, marked slow, skipped without
weights. It would have caught items 4 and 5.

---

## 9. The entire graded portion does not exist yet — SCHEDULE

| component | state |
| --- | --- |
| Agent 1 Assessor | not started |
| Agent 2 Adjudicator | not started |
| Agent 3 Recorder | not started |
| SQLite schema | not started |
| FastAPI | not started |
| Streamlit console | not started |
| Severity ground-truth labels | not started |

Lanes A and B are complete and heavily tested. Lanes C and D are empty. The bootcamp
grades agents, memory, guardrails, evaluation and observability — that is, the part that
does not exist — and there are roughly ten days left.

**This is the finding that matters most.** Everything above is a day's work combined;
this is the project.

---

## Smaller things

- `CaseState.event` is an untyped `dict`, so a schema change breaks Lane C at runtime
  rather than at import.
- `_nearest_id` correctly rejects distant boxes (verified), but two people whose boxes
  are within 17px L1 could swap ids. Unlikely; worth a comment.
- `zones.py` still contains `cam_1`/`cam_3`, kept for fixtures. Labelled EXAMPLE, fine,
  but they inflate every zone listing.
- `kb/eval_ragas.py` is written and compiles but has never run.
- No `.env` loader is called anywhere in Lane A or B — only `eval_ragas.py` uses
  `load_dotenv`. Fine today, will surprise someone later.

---

## Suggested order

1. **Item 9** — start Lane C today. Everything else is polish by comparison.
2. **Item 1 and 2** — one test and a key change, half an hour, prevents silent
   under-escalation.
3. **Item 3** — identity binding, because the demo depends on it.
4. **Item 6** — resolve the zombie before Agent 2 is written around it.
5. Items 4, 5, 7, 8 — cleanups, do them when touching those files anyway.
