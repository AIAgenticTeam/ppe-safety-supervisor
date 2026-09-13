# Licensing — a decision the team has to make

**Do not make this repository public until this is settled.**

## The dependency chain

| component | licence | consequence |
| --- | --- | --- |
| Ultralytics YOLO (`ultralytics`) | **AGPL-3.0** | strong copyleft |
| Construction-PPE dataset | **AGPL-3.0** | strong copyleft |
| Model weights trained from the above | derived work | inherits the obligation |
| SH17 dataset (*if* used) | **CC BY-NC-SA 4.0** | **non-commercial only** |

AGPL-3.0 is not MIT. It is a strong copyleft licence with a network clause: distributing the
software, **or making it available to users over a network**, can oblige you to offer the
complete corresponding source of the whole work under AGPL-3.0. A Streamlit console served to
anyone other than yourselves is plausibly that network use.

## What this means in practice

**For the bootcamp:** entirely fine. Coursework, a private repo, a demo to a panel. No
distribution, no public network service, no obligation triggered. This is the normal case and
it needs no action beyond keeping the repo private.

**If you publish the repo:** license it AGPL-3.0 and say so. That is the straightforward,
honest path, and it costs you nothing for a portfolio project. Add the full AGPL-3.0 text as
`LICENSE` and note the dataset attribution.

**If anyone ever wants to commercialise this:** the Ultralytics AGPL obligation is the thing
to deal with first — they sell a commercial licence for exactly this reason. Do not assume a
capstone can be handed to an employer as-is.

**If you add SH17:** its CC BY-NC-SA 4.0 is non-commercial *and* share-alike. Fine for an
experiment, blocking for anything else. Keep any SH17-derived weights clearly separated from
the main model so the provenance stays traceable.

## Action

1. Team decides: private through the bootcamp (recommended), public after, or public now.
2. If public at any point, drop the full AGPL-3.0 text into `LICENSE` at the repo root.
3. Attribute the construction-PPE dataset and Ultralytics in the README.
4. State the licensing position in the final report — a panel that asks about deployment will
   ask about this, and "we checked, it is AGPL, here is what that implies" is a strong answer.
