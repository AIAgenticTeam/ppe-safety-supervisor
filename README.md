# Autonomous Industrial Safety Supervisor

An agentic AI system that judges PPE safety violations **in context** — not just detects them.

A camera-based pipeline detects personal protective equipment, a deterministic layer decides
whether a violation actually occurred given where the worker is standing, and a multi-agent
system judges severity against the governing regulation, remembers patterns over time, and
escalates only what a human approves.

SDA Agentic AI Bootcamp capstone · Saudi Digital Academy 2026

---

## The core idea

A detection is not a violation.

| observation | zone | outcome |
| --- | --- | --- |
| no gloves | walkway | **ignored** — only a helmet is required here |
| no gloves | grinding station | **ticket** — abrasive wheel, hand protection mandatory |
| no gloves, third time this week | grinding station | **stop-work escalation**, pending human approval |

Same detection, three different responses. Context and memory are what a threshold rule
cannot do, and they are why this system needs agents rather than `if` statements.

---

## Architecture

```
01  OFFLINE SETUP        config/zones.json · clause map · OSHA → FAISS index
                              ↓
02  PERCEPTION           frame → YOLO26 → assess + zone → confirm over
    (deterministic)      time → ViolationEvent          ← NO LLM HERE
                              ↓  POST /events
03  AGENT LAYER          Assessor     what happened, under what rule   ← LLM
    (LangGraph)          Adjudicator  what response is proportionate   ← LLM
                         Recorder     commit + read back               ← no LLM
                              ↓  guardrail: escalation needs a human
04  STATE & INTERFACE    SQLite · local evidence store · FastAPI · console pages
```

The boundary between **02** and **03** is the important one. Everything above it is
deterministic and unit-testable; no language model touches a compliance finding. The agents
reason about *severity and response*, never about whether a worker was wearing a helmet.

Two of the three are genuinely agentic: they are given tools and choose which to call, in
what order, and when they have enough. The Recorder is deliberately not — committing a row
is not a judgement call, and a model that might forget to call `commit_record` loses the
history the next escalation depends on.

**There is no weekly analyst agent.** An earlier design had one, and the aggregation it was
to perform turned out to be SQL that already exists (`by_zone`, `repeat_offenders`,
`stats`, served by `GET /report`). Wrapping a language model around a query it cannot
improve would have added a place for numbers to be restated wrongly in the one artefact a
manager reads without checking. The weekly view is deterministic, and says so.

`ViolationEvent` ([schema](docs/EVENT_SCHEMA.md)) is the contract across that boundary.

---

## Quick start

```bash
git clone <this repo>
cd ppe-safety-supervisor
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m perception.zones config/zones.json    # validate the zone config
python scripts/make_fixtures.py --out fixtures  # regenerate 20 sample events
python -m pytest tests -q                       # offline: no key, no weights, no network
```

Run the app — one process, one port, the console and the API together:

```bash
python scripts/run_app.py --seed "fixtures/events/*.json" --no-judge   # http://127.0.0.1:8000
```

Judging events with the agents needs `OPENAI_API_KEY` in `.env`; `--no-judge` records the
findings and spends nothing.

The roster (who a finding can be attributed to) is loaded from a CSV on the **Team** page —
drop a file, review the preview, confirm. Excel exports work as they are: `;` delimiters,
UTF-8 or Arabic Windows encodings, and header names like `Employee ID` / `Full Name`.

Remove someone with the trash button, or tick several and remove them together. A person with
findings on record is hidden from the roster rather than deleted, so their history and name stay
intact; importing their id again restores them.

**You do not need a GPU or the trained model to work on the agent layer.** `fixtures/`
contains 20 realistic events covering every branch — compliant, violation, review,
indeterminate, and a repeat offender across a week. Build against those.

The deterministic core (`perception/zones.py`, `perception/events.py`,
`perception/ppe_compliance.py`) imports without torch or ultralytics — those load lazily
only when a model is actually run.

With weights and footage, the live path produces the same JSON shape:

```bash
python -m perception.pipeline clip.mp4   --camera cam_3 --out events/   # tracked, confirmed
python -m perception.pipeline frame.jpg  --camera cam_3                 # one still
python -m perception.pipeline footage/   --camera cam_3                 # a folder of stills
```

Video runs ByteTrack with N-of-M temporal confirmation, so only sustained absences are
emitted and those events are actionable. Stills have no temporal dimension, so nothing
from them is ever actionable — by design, not omission.

Draw the zones over a real frame and have someone who knows the floor check them
before trusting any event:

```bash
python -m perception.zones config/zones.json --overlay frame.jpg cam_3 zones_overlay.jpg
```

---

## Repository layout

| path | what it is | lane |
| --- | --- | --- |
| `perception/pipeline.py` | **frame → detections → zone-aware assessment → event** | A |
| `perception/zones.py` | Camera zones, point-in-polygon, foot-point location | A |
| `perception/ppe_compliance.py` | Detections → per-person compliance assessment | A |
| `perception/events.py` | **The frozen `ViolationEvent` contract** + JSON Schema | A |
| `perception/tracking.py` | **ByteTrack + N-of-M temporal confirmation** | A |
| `perception/evidence.py` | Local evidence store — annotated frame + crop per finding | A |
| `config/zones.json` | Zone polygons and per-zone PPE requirements | A |
| `config/zones.test.json` | **Test-only** config; never loaded by the pipeline | A |
| `scripts/make_fixtures.py` | Generates sample events with no GPU required | A |
| `scripts/restratify.py` | Re-splits a YOLO dataset so every class is in every split | A |
| `notebooks/` | YOLO26 training and evaluation runbook | A |
| `kb/` | OSHA ingest, chunking, FAISS index, clause map, Ragas | B |
| `agents/` | LangGraph graph, tools, guardrails, memory, severity | C |
| `app/` | FastAPI service, supervisor console (`app/web/`), SQLite, telemetry | D |
| `eval/` | Severity labels and the end-to-end evaluation run | shared |
| `fixtures/` | 20 generated events + index | shared |
| `weights/` | Model weights — fetched from a release, not in git | A |
| `scripts/` | `get_weights.py` and other one-off utilities | shared |
| `docs/` | Event schema, workflow, licensing notes | shared |
| `tests/` | Contract and geometry tests | shared |

No Python lives at the repo root. Each lane owns a package (`perception/`, `kb/`, `agents/`,
`app/`), so two people never edit the same file. Run modules from the repo root, e.g.
`python -m perception.zones`; standalone tools live in `scripts/`.

---

## Detector

YOLO26s @ 960px, six presence classes: `helmet, gloves, vest, boots, goggles, Person`.

| class | mAP50 | recall |
| --- | --- | --- |
| helmet | 0.860 | 0.796 |
| vest | 0.854 | 0.816 |
| boots | 0.795 | 0.742 |
| goggles | 0.789 | 0.723 |
| gloves | 0.775 | 0.733 |
| Person | 0.911 | 0.867 |

**Recall is the number that matters here, not mAP.** The compliance layer uses containment
(IoA ≥ 0.6) plus body-zone bands, so loose boxes pass fine — box tightness is irrelevant to
the decision. Recall determines whether a bare-headed worker gets missed.

At helmet recall 0.796, roughly **one worn helmet in five is not detected**. Single-frame
escalation would therefore mislabel a compliant worker about 20% of the time. That is why
`confirmation` exists in the event schema: a violation must be sustained across a tracked
window before it becomes actionable. The measured per-class recall ships *inside every event*
so the agent can weigh an absence appropriately.

The absence classes (`no_helmet`, `no_gloves`, …) were deliberately dropped — at this data
scale they are unlearnable (`no_boots` scored 0.029 mAP50 on 23 instances). Compliance is
inferred geometrically instead.

---

## Working agreement

Four parallel lanes. See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the branch and review flow.

| lane | owns |
| --- | --- |
| **A · Perception** | zones, tracking, event emission, evidence store |
| **B · Knowledge** | OSHA ingest, chunking, FAISS index, clause map, Ragas |
| **C · Agents** | LangGraph graph, tools, guardrails, memory, severity policy |
| **D · App & docs** | FastAPI, console pages, logging, report, demo script |

**The event schema is frozen.** If you need a new field, bump `SCHEMA_VERSION` in `perception/events.py`
and tell everyone — do not add one quietly. Every other lane codes against it.

---

## Known limitations

Stated here because they belong in the final report too.

- **Worker identity does not persist.** ByteTrack IDs do not survive a session, a camera
  hand-off, or a worker leaving frame. "Third time this week" therefore cannot come from
  tracking alone — `subject.worker_ref` is bound by a supervisor in the console, not claimed
  by the vision system.
- **Domain gap.** The detector is trained on the Ultralytics construction-PPE dataset
  (1,416 images). Performance on other camera angles, lighting, and mounting heights will be
  lower. Published benchmarks show a 4–7 point AP50 drop across sources.
- **No dedicated hi-vis or hand-protection clause in construction.** Scope is 29 CFR 1926
  (see [docs/REGULATION_SCOPE.md](docs/REGULATION_SCOPE.md)); 1926.201 covers flaggers only,
  and there is no equivalent of general industry's 1910.138. Vest and glove findings cite
  site policy supported by 1926.95's hazard-assessment duty, and say so rather than
  inventing a clause.
- **Small-object localisation is capped by label quality.** mAP50-95 sat at ~0.435 across
  four different training configurations, unmoved by resolution or capacity.

---

