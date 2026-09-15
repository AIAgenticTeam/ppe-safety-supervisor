# Footage probe results — 14 Sep 2026

Two Pexels stock clips, probed with `scripts/probe_footage.py` at conf 0.25, 12 sampled
frames each, YOLO26s @960 (`weights/best.pt`). Annotated frames in `probe/`.

Both clips are gitignored (`data/`). Re-download from Pexels if lost.

## clip_a_50fps.mp4 — construction site, 3840×2160, 50 fps, 8.1 s

One main worker plus 1–3 others at frame edges. Wearing blue helmet, **red** hi-vis vest,
dark gloves. Tight crop, no floor visible.

| class | boxes | mean conf | min | max |
| --- | --- | --- | --- | --- |
| Person | 32 | 0.519 | 0.272 | 0.743 |
| helmet | 14 | 0.587 | 0.299 | 0.712 |
| gloves | 2 | 0.535 | 0.340 | 0.729 |
| **vest** | **0** | — | — | — |

**The critical finding: zero vest detections on a worker wearing an unmistakable red hi-vis
vest with reflective stripes.** Vest is nominally the second-best class (test mAP50 0.854,
recall 0.816). Most likely cause: construction-PPE is dominated by yellow/green/orange
vests and barely covers red.

Consequence with the default `Policy(required=("helmet", "vest"))`: this compliant worker
is assessed as `missing: [vest] → violation`. **The system would falsely accuse him.** That
is worse than detecting nothing, and it is the exact failure the architecture exists to
prevent.

Helmet boxes sit correctly on heads, so the IoA association logic works fine here. The
problem is purely detection, not association.

## clip_b_24fps.mp4 — indoor workshop, 3840×2160, 24 fps, 6.7 s

Two men working with lumber, wearing no PPE at all. Wide shot, floor visible, natural
indoor/outdoor split. An unworn white hard hat sits on a bench.

| class | boxes | mean conf | min | max |
| --- | --- | --- | --- | --- |
| Person | 22 | 0.443 | 0.251 | 0.679 |
| boots | 1 | 0.255 | — | — |

People are found accurately (0 empty frames) but at 0.47–0.52. Against the policy gates:

- `person_conf_min = 0.50` → the 0.47 man is dropped entirely
- `violation_person_conf = 0.70` → max across the clip is 0.679, so **no violation is
  reachable on this clip at all**

The PPE absence is correctly reported — they genuinely wear none, so that is a true
negative. The single `boots 0.25` is a false positive on a knee, correctly rejected by
`ppe_conf_min = 0.35`. The unworn hard hat on the bench was **not** detected, so the
`unassigned_ppe` path did not get exercised.

## What neither clip provides

- A worker crossing between zones — the slide-5 thesis
- A repeat offender — that needs days, these are 7–8 seconds
- clip_a has no usable zone geometry (tight crop, no floor)

## Open decision — pick this up first

1. **Enforce helmet only** for the demo (`required=("helmet",)`). Helmet detects reliably.
   Free, honest, documents the vest limitation in the report. *Recommended given the
   timeline.*
2. **Fine-tune on ~200 labelled frames** from these clips. Fixes both properly, costs about
   a day plus labelling, and Lane A is the critical path.
3. **Find footage with yellow/green vests** — cheap to try, but a gamble on matching the
   training distribution.

The red-vest miss is good material for the report either way: concrete, measured evidence
of domain gap, caught before deployment rather than after.

## clip_c_aerial_30fps.mp4 — aerial excavation site, 1280×720, 30 fps, 15.9 s

Drone/overhead shot. Excavator plus ~6 workers. Longest clip, ground visible, people move —
structurally the best of the three. Detection is the worst.

| class | boxes | mean conf | min | max |
| --- | --- | --- | --- | --- |
| Person | 9 | 0.470 | 0.349 | 0.679 |
| helmet | 3 | 0.303 | 0.300 | 0.307 |
| vest | 2 | 0.405 | 0.369 | 0.442 |
| boots | 1 | 0.408 | — | — |

**6 of 12 frames detected nothing at all.** Peak 2 people found in frames containing 6.

Worse than the numbers: the detections are wrong, not merely weak. In `frame_02`, `Person
0.42` is a blue tarpaulin, and `Person 0.62` + `vest 0.37` is a red-and-blue tarpaulin —
while all six real workers go undetected. The model appears to key on bright coloured
fabric for `vest`, firing on a tarp here and failing on an actual red vest in clip_a.

False positives on inanimate objects are the most dangerous failure direction for a
compliance system.

## Conclusion across all three clips

| clip | failure |
| --- | --- |
| a — construction, ground level | people and helmets fine, **vest never detected** → would falsely accuse a compliant worker |
| b — indoor workshop | people found but max 0.68, **under the 0.70 violation gate** → nothing actionable |
| c — aerial site | **half the frames empty**, real workers missed, **false positives on tarpaulins** |

Three independent clips, three severe and distinct failures. This is not clip-specific bad
luck: **the detector does not generalise outside its training distribution** (ground-level
construction stills, upright workers, close to camera). Buying a fourth stock clip is a
lottery ticket.

**Direction:** demo detection on the construction-PPE test set, where the model measurably
scores 0.817 mAP50; film one short ground-level clip in-house for the tracking and zone
demo; keep `fixtures/` driving the agent layer. Revisit fine-tuning if time allows.
