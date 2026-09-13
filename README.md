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
01  OFFLINE SETUP        zones.json · clause map · OSHA → FAISS index
                              ↓
02  PERCEPTION           frame → YOLO26 → assess + zone → confirm over
    (deterministic)      time → ViolationEvent          ← NO LLM HERE
                              ↓  POST /events
03  AGENT LAYER          Compliance Agent (per event, seconds)
    (LangGraph)          Weekly Analyst Agent (scheduled, weekly)
                              ↓  guardrail: stop-work needs a human
04  STATE & INTERFACE    SQLite · Cloudinary · FastAPI · Streamlit
```

The boundary between **02** and **03** is the important one. Everything above it is
deterministic and unit-testable; no language model touches a compliance finding. The agents
reason about *severity and response*, never about whether a worker was wearing a helmet.

`ViolationEvent` ([schema](docs/EVENT_SCHEMA.md)) is the contract across that boundary.

---

## Quick start

```bash
git clone <this repo>
cd ppe-safety-supervisor
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python zones.py zones.json               # validate the zone config
python make_fixtures.py --out fixtures   # regenerate 20 sample events
python -m pytest tests -q                # 63 contract + integration tests
```

**You do not need a GPU or the trained model to work on the agent layer.** `fixtures/`
contains 20 realistic events covering every branch — compliant, violation, review,
indeterminate, and a repeat offender across a week. Build against those.

The deterministic core (`zones.py`, `events.py`, `ppe_compliance.py`) imports without
torch or ultralytics — those load lazily only when a model is actually run.

With weights and footage, the live path produces the same JSON shape:

```bash
python pipeline.py frame.jpg  --camera cam_3 --weights runs/.../best.pt
python pipeline.py footage/   --camera cam_3 --out events/
```

Draw the zones over a real frame and have someone who knows the floor check them
before trusting any event:

```bash
python zones.py zones.json --overlay frame.jpg cam_3 zones_overlay.jpg
```

---

## Repository layout

| path | what it is | lane |
| --- | --- | --- |
| `pipeline.py` | **frame → detections → zone-aware assessment → event** | A |
| `zones.py` | Camera zones, point-in-polygon, foot-point location | A |
| `ppe_compliance.py` | Detections → per-person compliance assessment | A |
| `events.py` | **The frozen `ViolationEvent` contract** + JSON Schema | A |
| `make_fixtures.py` | Generates sample events with no GPU required | A |
| `zones.json` | Zone polygons and per-zone PPE requirements | A |
| `restratify.py` | Re-splits a YOLO dataset so every class is in every split | A |
| `notebooks/` | YOLO26 training and evaluation runbook | A |
| `fixtures/` | 20 generated events + index | shared |
| `docs/` | Event schema, workflow, licensing notes | shared |
| `tests/` | Contract and geometry tests | shared |

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
| **A · Perception** | zones, tracking, event emission, evidence crops |
| **B · Knowledge** | OSHA ingest, chunking, FAISS index, clause map, Ragas |
| **C · Agents** | LangGraph graph, tools, guardrails, memory, severity policy |
| **D · App & docs** | FastAPI, Streamlit console, logging, report, demo script |

**The event schema is frozen.** If you need a new field, bump `SCHEMA_VERSION` in `events.py`
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
- **No dedicated hi-vis regulation in general industry.** OSHA 1910 Subpart I has no vest
  clause; 1926.201 covers flaggers only. Vest findings cite site policy supported by
  1910.132(a) hazard assessment, and say so rather than inventing a clause.
- **Small-object localisation is capped by label quality.** mAP50-95 sat at ~0.435 across
  four different training configurations, unmoved by resolution or capacity.

---

## Licensing — decision required

This project depends on **Ultralytics YOLO (AGPL-3.0)** and the **construction-PPE dataset
(AGPL-3.0)**. AGPL-3.0 is a strong copyleft licence: distributing this work, or running it as
a network service, can oblige you to release the whole project under AGPL-3.0 as well.

**Do not make this repository public until the team has decided how to license it.**
See [docs/LICENSING.md](docs/LICENSING.md).
