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
complete corresponding source of the whole work under AGPL-3.0. The console served to
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

## The UI template

The console's look (`app/web/`) is built on a purchased Laravel template ("erepair"). That is a
separate question from the AGPL one, and the answer is in the licence you bought, which this
repository cannot see -- so check it before the repo is public, because a template licence
usually covers *using* the design in your project and not *redistributing its files*.

What was taken, so the question is small:

- **Copied:** `style.css`, `responsive.css` and three module stylesheets (`footer`,
  `page-header`, `error-page`) -- the template's own layout CSS -- plus `bootstrap.min.css`
  (MIT). They live in `app/web/static/css/`.
- **Deliberately not copied:** the icon fonts (the template ships **Font Awesome Pro 5.8**,
  whose CSS header says *Commercial License*, plus a proprietary icomoon set), its JavaScript
  plugins, and every image. Icons are inline SVG drawn for this project; the logo is a shape
  and text. `tests/test_web.py` fails if a font file is added.
- **Loaded from Google Fonts at runtime:** Archivo and Titillium Web (SIL OFL). Nothing is
  bundled, and the pages fall back to system fonts offline.

If the template's terms do not allow publishing its CSS, the fix is contained: replace the
five template stylesheets with your own, keeping `app.css`.

## Action

1. Team decides: private through the bootcamp (recommended), public after, or public now.
2. If public at any point, drop the full AGPL-3.0 text into `LICENSE` at the repo root.
3. Attribute the construction-PPE dataset and Ultralytics in the README.
4. State the licensing position in the final report — a panel that asks about deployment will
   ask about this, and "we checked, it is AGPL, here is what that implies" is a strong answer.
