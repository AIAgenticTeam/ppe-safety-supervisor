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

## 4. Two incompatible evidence formats — FIXED 16 Sep

`pipeline.py` uses two different mechanisms depending on input:

| path | function | produces |
| --- | --- | --- |
| stills | `ppe_compliance.save_evidence` | bare crop, no frame, no annotation |
| video | `EvidenceStore.save` | annotated frame + crop pair, keyed by event_id |

The console will receive events whose `evidence` block means different things. Worse, the
stills path produces an unannotated crop — a reviewer sees a photograph of a worker with
no indication of what was alleged.

**Fixed** exactly that way. `process_image` now calls
`EvidenceStore.save_from_image_file`, and `ppe_compliance.save_evidence` is gone.
Both paths write an annotated frame + crop pair into the same directory under the
same naming scheme, so the console needs no special case and no reviewer is ever
handed a photograph with nothing marking what was alleged.

`tests/test_pipeline_stills.py` covers the path with the detector stubbed, which is
how the second format survived this long -- `process_image` had no test at all.

---

## 5. `event_id` is generated twice and the overwrite is load-bearing — FIXED 16 Sep

```python
event_id = make_event_id(camera_id, confirmation.track_id, when)
stored = store.save(event_id, ...)          # evidence filenames use THIS id
event = build_event(..., when=when)         # build_event generates its own id
event.event_id = event_id                   # overwrite -- keeps them in sync
```

Same inputs give the same id, so it works. But the evidence files are already written
under the first id, so if the two ever diverge the event points at filenames that do not
exist. The overwrite is doing real work and looks like a leftover.

**Fixed** as suggested. `build_event` takes an optional `event_id`, and both pipeline
paths mint it once before any file is written. A test pins the invariant the
overwrite was silently holding up: the event's evidence paths are named for the
event's own id.

---

## 5b. Stills sharing a timestamp overwrote each other — FIXED 16 Sep

Found by running the real detector over a folder of frames, not by reading the code.
The pipeline reported `4 frames -> 3 events` and left **one** file on disk.

Stills take their timestamp from the file mtime and person ids restart at 0 on every
frame, so `make_event_id` produced the same id four times. Events and evidence are both
saved under the id, so three findings were overwritten in silence. Nothing raised, and
the summary line still said three.

**Fixed:** `make_event_id` takes an optional `source`, and the stills path passes the
frame stem. Video is untouched -- ByteTrack ids are already unique within a clip, and a
test pins that its id shape has not drifted.

Worth noting how it was found: this is the class of bug that a stubbed test will not
catch, because the stub controls the inputs that collided.

**Corrected 19 Sep: video was not untouched after all.** A track id is unique per
track, but one track confirms more than once as different items cross the threshold.
The pour clip confirmed track 8 at frame 111 and again at frame 123, both inside one
second, and the second event overwrote the first. The test that "pinned the id shape"
was pinning the bug. Video ids now carry the confirming frame (`source=f"f{frame}"`),
and the whole video path is tested for it (item 8).

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

## 8. The video path has no automated test — PARTLY COVERED 23 Sep

`track_video`, `process_video` and `save_evidence` have **zero** test coverage. The video
pipeline has run exactly once, by hand. Everything proving it works is a terminal
screenshot.

229 tests and none of them touch the code that produces real events.

**Fix:** a smoke test over the 4-second `scene_02` clip, marked slow, skipped without
weights. It would have caught items 4 and 5.

**Partly covered.** `save_evidence` is gone (item 4). `process_video` now has a test
with the tracker stubbed to replay the pour clip's collision -- one track confirming
twice inside a second -- and it fails if the confirming frame drops out of the id
(`test_process_video_keeps_both_confirmations_of_one_track`). `track_video` itself,
the ByteTrack loop, is still untested: that needs the weights and a clip, so it stays
a manual run.

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

---
---

# Second review — 23 Sep 2026

A full pass over the system after the console-pages branch was merged. Every finding
below was **reproduced by running it** before it was fixed, and every fix has a
regression test that fails on the old code (checked by reverting the source and
re-running them). The suite was green throughout the review. It tested what had been
thought of, and none of these had been.

| # | Finding | Status |
| --- | --- | --- |
| 1 | The repository is public: vendored template CSS, commit trailers | **Owner's decision, open** |
| 2 | `/evidence` served any file in the repository, `.env` included | FIXED 23 Sep |
| 3 | A malformed model answer crashed the case and lost the finding | FIXED 23 Sep |
| 4 | Re-judging a finding counted it as its own prior | FIXED 23 Sep |
| 4b | The model chose whose history was read | FIXED 23 Sep |
| 5 | Re-judging erased a supervisor's approval | FIXED 23 Sep |
| 6 | Another website could import a roster or replay an event | FIXED 23 Sep |
| 7 | The pattern-claim guard was beaten by ordinary paraphrase | NARROWED 23 Sep |

---

## 2. `/evidence` served any file in the repository — FIXED 23 Sep

The allowed roots included `Path.cwd()`, and the server runs from the repository root.
An event whose evidence path was `.env` got the OpenAI key back byte for byte,
labelled as a JPEG. The same worked for `demo.db` and the source. The existing tests
only tried paths *outside* the repository, so they passed.

**Fix:** evidence is an image (`.jpg`/`.jpeg`/`.png`) under `events/` or `fixtures/`
(or a root named in `SAFETY_EVIDENCE_ROOTS`), and it has to start with the magic bytes
of that type. Anything else is a 404, never a 403, so the endpoint does not confirm
that a file exists. Tests: `test_a_file_from_the_repository_is_not_evidence`,
`test_an_image_name_on_something_that_is_not_an_image_is_refused`.

## 3. A malformed model answer lost the finding — FIXED 23 Sep

Truncated JSON, JSON that is not an object, and an action outside the enum all raised
out of the agent. The API answered 500 and the event was never written. A dead model
parked correctly; a confused one lost data.

**Fix:** `llm.tool_arguments` parses defensively, and a malformed call is answered with
a tool message saying what was wrong, so the model can correct itself. An invalid
action gets the list of valid ones. Non-string summaries are coerced, and a
`clause_ids` string is not iterated character by character. Behind all of that, the
graph catches anything an agent still raises, parks the case as `agent_error` and keeps
the finding. If the Recorder's write itself fails, that is an alert. Tests:
`test_a_malformed_answer_is_corrected_not_fatal` (5 cases),
`test_a_model_that_never_answers_properly_parks_the_case`,
`test_an_agent_failing_in_a_way_nobody_foresaw_still_keeps_the_finding`.

## 4. Re-judging counted a finding as its own prior — FIXED 23 Sep

Once identified, a finding is part of its worker's history, and the lookup read it
back when the same finding was judged again. Priors went 2 → 3, severity 9 → 11, from
one incident. The demo run sheet called this "not wrong". It was wrong.

**Fix:** `get_worker_history(..., exclude_event=)` leaves out the finding being judged.
Test: `test_judging_a_finding_again_does_not_count_it_as_its_own_prior`.

## 4b. The model chose whose history was read — FIXED 23 Sep

Found while fixing 4. `get_worker_history` took `worker_ref` and `window_days` from the
model. On a finding nobody had identified, the model could name someone else's id, the
lookup "resolved", and the case escalated on another person's record, with no identity
block because the lookup had succeeded. It could also widen the window to pull in old
findings, or narrow it to hide them.

**Fix:** the tool takes no arguments, the same principle as `final_severity`: the
model decides whether to ask, never what goes in. Tests:
`test_the_model_cannot_choose_whose_history_is_read`,
`test_the_model_cannot_widen_the_history_window`.

## 5. Re-judging erased a supervisor's approval — FIXED 23 Sep

`record_decision` used `INSERT OR REPLACE`, which deletes the row and writes a new one.
Replaying an approved escalation put it back in the approval queue, unsigned, and the
record of who signed was gone.

**Fix, twice over:** the store refuses to overwrite a signed decision
(`ON CONFLICT ... WHERE approved_by IS NULL`, and `record_decision` returns whether it
wrote). Intake also stops a signed finding before any model is called: outcome `done`,
reason "already decided and approved by …". Tests:
`test_a_signed_decision_cannot_be_rewritten`,
`test_an_approved_decision_is_not_judged_again`,
`test_a_replay_does_not_erase_an_approval`.

## 6. Another website could write to the console — FIXED 23 Sep

The JSON endpoints were safe by accident: a cross-site JSON POST needs a CORS
preflight, and this app grants none. `/roster/import` took `text/plain` and `/replay`
took no body, so neither needed a preflight. Any page the supervisor visited while the
console was open could plant people on the roster, or replay an event, which chained
with 4 and 5.

**Fix:** middleware refuses any POST/PUT/PATCH/DELETE that a browser marks as
cross-site or same-site (`Sec-Fetch-Site`), or whose `Origin` is `null` or a different
host. The pipeline, the scripts and the tests are not browsers and send neither
header, so they are unaffected. The check covers endpoints that do not exist yet.
Tests: `test_another_website_cannot_import_a_roster`,
`test_another_website_cannot_replay_or_post_an_event`,
`test_the_console_itself_can_still_write`.

## 7. The pattern-claim guard was a short word list — NARROWED 23 Sep

Five of six ordinary paraphrases passed: "has done this before", "not the first time",
"habitually ignores", "persistent non-compliance", "was also caught last week". The
list also fired on phrases any single-incident notice uses, "do not let this happen
again" and "please continue to monitor", and a guard that cries wolf trains people to
click through it.

**Fix, in two parts.**

- The list was rewritten against both sets: the ways a history gets alleged, and
  their look-alikes ("priority", "before resuming work", "multiple violations" in one
  finding). It was checked against 32 claims and 23 single-incident notices, including
  the live model output on file, with no misses and no false alarms. The refusal now
  quotes the words that tripped it.
- Every notice now ends with a **Record:** line that the system writes from the
  history lookup, not from the model: "identity not confirmed. Prior violations are
  unknown, not zero." or "2 confirmed prior violations in the last 7 days." It is added
  *after* the gates have run, so the gate only ever reads the model's words.

**Still a limitation.** It is a word list, and a word list can be paraphrased past.
The Record line is what makes that survivable: whatever the model managed to phrase,
the supervisor reads the record next to it. Tests:
`test_pattern_claims_are_caught_when_paraphrased`,
`test_single_incident_notices_are_not_mistaken_for_pattern_claims`,
`test_the_notice_states_an_unknown_record_as_unknown`,
`test_the_record_line_is_added_after_the_gates_have_read_the_notice`.

---

## Limitations, stated rather than fixed

- **The approver's name is self-declared.** There is no login, so anyone who can reach
  the console can sign as anyone. Correct for a single-machine demo bound to
  `127.0.0.1`, wrong for anything shared. The fix is authentication, not a patch.
- **The pattern-claim guard is a word list** (item 7 above).
- **Two events from one incident can count as each other's priors.** The pour clip
  confirms track 8 twice, twelve frames apart. Once both are bound to the same worker,
  each is the other's prior. Whether "same worker, same camera, within N minutes" is
  one incident is a policy decision, so it is left open deliberately.

## Left deliberately

- **Item 6 of the first review** (`severity_multiplier`) is still contained rather than
  removed. Removing it is a schema bump and a fixture regeneration.
- **`demo.db` in the repository root.** Moving it changes every command on the
  rehearsed run sheet, a week before the demo.
- **Vendored template CSS.** Whether it stays depends on item 1.
- **`restratify.py`** is dataset tooling from the training phase. It is historical, not
  dead.
- **`scripts/get_weights.py`** now points at the organisation's repository and says
  what to do when no release exists. No release has been published; the weights are
  shared with the team directly.
