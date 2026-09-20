# How we work

Four people, two weeks, one `main` that always runs. Optimised for not blocking each other.

## Branches

```
main                 always runs; protected; PRs only
lane-a/<thing>       perception    e.g. lane-a/bytetrack-confirmation
lane-b/<thing>       knowledge     e.g. lane-b/osha-ingest
lane-c/<thing>       agents        e.g. lane-c/compliance-graph
lane-d/<thing>       app + docs    e.g. lane-d/console-pages
```

```bash
git checkout main && git pull
git checkout -b lane-c/compliance-graph
# work, commit
git push -u origin lane-c/compliance-graph
# open a PR on GitHub, request one reviewer from another lane
```

Small PRs. A PR that touches three lanes is a PR nobody reviews properly with two weeks left.

## Rules that exist for a reason

1. **Never commit datasets, weights, videos, or `.env`.** `.gitignore` covers them. If you
   find yourself using `git add -f`, stop and ask. A 667 MB zip in git history is permanent.
2. **`perception/events.py` is frozen.** Changing `ViolationEvent` breaks all four lanes at once. To
   change it: bump `SCHEMA_VERSION`, update `docs/EVENT_SCHEMA.md`, regenerate fixtures,
   announce it. Never silently.
3. **Tests pass before you open a PR.** `python -m pytest tests -q`
4. **Build against `fixtures/`, not against the GPU.** Lanes C and D should never be blocked
   waiting for a pod.
5. **Secrets go in `.env`** (copy `.env.example`). If a key ever lands in a commit, rotate it
   immediately — removing the commit is not enough, it is already in the reflog and on
   anyone's clone.

## Daily rhythm

- Morning: 10 minutes, each lane says what it shipped and what it is blocked on.
- Anything that changes a shared contract gets announced in the group chat *before* the PR.
- End of day: `main` runs. If it does not, that is the next morning's first item.

## Definition of done

A lane's work is done when it is merged to `main`, has a test, and someone from another lane
has run it on their own machine. "Works on my laptop" is not done — the demo runs on one
machine and it will not be yours.

## Before the final presentation

- [ ] `git clone` into a fresh directory, follow the README, and confirm it works
- [ ] No secrets in history: `git log -p | grep -iE "sk-|api_key|secret"`
- [ ] Recorded demo fallback committed or linked
- [ ] Technical report in `docs/`

## Secrets

`.env.example` is **committed**. `.env` is **gitignored**. Real keys go in `.env`, never
in the template — a key in `.env.example` goes to GitHub the moment anyone commits.

Three layers guard this, because removing a secret from git history is not the same as
it never being there:

1. `.gitignore` excludes `.env`
2. `tests/test_no_secrets.py` fails if a value appears in `.env.example`, or if anything
   credential-shaped lands in a tracked file
3. `.githooks/pre-commit` blocks the commit outright

Enable the hook once per clone:

```bash
git config core.hooksPath .githooks
```

If a key ever does reach a commit: **rotate it first**, then worry about history. A
revoked key in a public repo is a curiosity; a live one is an incident.
