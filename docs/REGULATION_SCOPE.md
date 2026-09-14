# Regulatory scope: 29 CFR 1926 (construction)

**Decided 14 Sep 2026.** All citations come from **29 CFR Part 1926 — construction**,
not Part 1910 (general industry).

## Why

1910 and 1926 cover different workplaces. An inspector at a construction site cites under
1926; a citation issued under 1910 on a construction site is the right topic from the wrong
body of law. Since slide 9 commits the system to *"every decision must cite a real policy
clause,"* a real-looking citation from the wrong jurisdiction is worse than none.

What the system actually looks at is construction:

- the detector is trained on the Ultralytics construction-PPE dataset
- the usable demo clip (`clip_a`) is a construction site
- every frame a reviewer sees will be construction

The only thing pointing at 1910 was the "factory floor" wording on slide 3, written before
there was data or footage. **That slide's wording changes to construction.**

## Clause map

| class | clause | title |
| --- | --- | --- |
| helmet | `1926.100` | Head protection |
| goggles | `1926.102` | Eye and face protection |
| boots | `1926.96` | Occupational foot protection |
| gloves | `1926.95` | Criteria for PPE — no dedicated hand clause in construction |
| vest | `1926.95` | Site policy, supported by the hazard-assessment duty |
| — | `1926.28` | Personal protective equipment (general duty) |
| — | `1926.201` | Signaling — hi-vis, flaggers only |

Known gap: construction has no hand-protection section equivalent to general industry's
`1910.138`, so gloves cites the general criteria instead. Accepted.

Where no clause squarely covers an item, cite **site policy supported by 1926.95's
hazard-assessment duty**, and label it that way. The system must distinguish "the
regulation requires this" from "your site requires this" — never invent a clause number.

## Corpus to ingest (lane B)

29 CFR 1926 Subpart E — §§ 1926.95, 1926.96, 1926.100, 1926.102 — plus 1926.28 and
1926.201. A few hundred paragraph-level chunks.

Sources:
- https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926SubpartE
- https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XVII/part-1926/subpart-E
