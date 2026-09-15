# Severity and identity — decided

Settled 16 Sep 2026. Supersedes the sketches in the chat. Lane C implements the scoring,
Lane D implements the roster and the dashboard.

## 1. Everything goes to the supervisor

The system never contacts a worker. It has no authority to issue a disciplinary
communication, and it cannot name someone it did not identify.

- Minor findings appear on the supervisor's dashboard as routine
- Severe findings appear as urgent
- Warning emails are **drafted** by the agent and **sent by the supervisor** from their
  own account, one click

## 2. Identity is asked first, not last

"Repeat" is a conclusion, not an input — the agent cannot route down the repeat branch
before it knows who the person is. So:

```
1. baseline severity     from zone weights only, no history needed
2. ask the supervisor to identify   (urgency of the prompt varies by baseline)
3. look up that worker's history
4. final severity = baseline + repeat factor
5. route: log / warning / escalation / stop-work
```

## 3. Roster, not free text

```
workers     worker_id · name · email · role       entered once
violations  event_id · worker_id (nullable) · ... one row per finding
```

The supervisor picks from a dropdown. History matches on `worker_id`, never on a name
string — `"Ahmed Ali"` vs `"ahmed ali"` is how a third offence silently becomes a first.

Agent 3 records every event immediately with `worker_id = NULL`; identity arrives later
as an update. Unattributed events count toward nobody's history.

## 4. Severity: sum what is missing

```python
missing_score = sum(weight[item] for item in missing)
```

**Not** total-minus-missing. Subtracting from the total breaks on small zones: the walkway
requires helmet (5) + vest (2) = 7, which is below a threshold of 9, so a fully compliant
worker would be flagged severe. Scoring the missing items makes zone totals irrelevant.

Per-zone item weights, set by the safety officer:

```yaml
grinding_station:  helmet: 5   gloves: 4   goggles: 4   vest: 2
walkway:           helmet: 5   vest: 2
work_area:         helmet: 5   vest: 2   gloves: 3
```

Because weights are per zone, the separate `severity_multiplier` is dropped — it would
double-count the same context.

## 5. Bands, not one threshold

| missing_score | action |
| --- | --- |
| 0 | compliant |
| 1–2 | log only |
| 3–4 | warning |
| 5–7 | escalation |
| 8+ | stop-work |

## 6. History is additive

```python
final = missing_score + 2 * priors_in_last_7_days
```

A helmet miss (5) with two priors reaches 9 and lands in stop-work. The escalation demo
beat falls out of arithmetic rather than a model's opinion.

## 7. Detector recall stays out of severity

A missing helmet is equally dangerous whether the detector is 79% or 99% reliable.
Folding recall into severity would mean "we are less sure, therefore it is less serious",
which is bad reasoning.

Two axes:

| axis | from | answers |
| --- | --- | --- |
| severity | zone weights + history | how bad is this if true |
| confidence | detector recall, frames confirmed, person confidence | how sure are we |

The action requires both. High severity with low confidence is **urgent human review**,
not a downgraded escalation.

## Where the weights come from

They are a policy input set by the site safety officer, configurable per zone. The system
implements policy; it does not author it. State that in the report, with whose judgement
the numbers represent — otherwise they read as magic constants.
