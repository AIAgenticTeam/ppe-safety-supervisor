# Lane B — the knowledge base

Answers one question: **which regulation covers this missing PPE, and what does it say.**

```
kb/corpus/*.xml     29 CFR 1926 source text from the eCFR API
      │
      ▼  ingest.py          paragraph chunks + citation paths
kb/chunks.json
      │
      ▼  build_index.py     embed with all-MiniLM-L6-v2
kb/index/                   FAISS, 48 vectors, 384 dimensions

kb/clauses.yaml     PPE class x zone -> clause, severity weights, bands
      │
      ▼  clause_map.py      deterministic lookup and scoring
```

## The rule that shapes everything here

**The clause map decides the citation. Retrieval only fetches its text.**

Cosine similarity choosing which law a worker allegedly breached produces a confident,
real-looking, wrong citation — worse than none, and exactly what the citation guardrail
exists to catch. So `clauses.yaml` maps the small, enumerable violation space (five PPE
classes across a handful of zones) to clauses by hand, and FAISS is asked only for the
words of a clause already named.

The architectural consequence: a retrieval failure degrades the supporting *text* of a
notice. It can never change the regulation the notice cites.

## Use

```python
from kb.clause_map import ClauseMap
from kb.retrieve import Retriever

cm = ClauseMap.load()
rule = cm.for_item("helmet", zone="walkway")
rule.clause.citation        # "29 CFR 1926.100"
rule.is_regulatory          # True — say "required under"; False means say "site policy"
rule.notice_phrase          # the wording a notice may use

Retriever.load().text_for(rule.clause.clause_id)    # the words, fetched by id

cm.score(["helmet", "vest"], "walkway", priors=2)   # severity, deterministic
```

## Measured

```
20 queries, filter then rank:   recall@1 0.90   recall@3 1.00   MRR 0.95
without the topic filter:       recall@1 0.65   recall@3 0.95   MRR 0.81
```

Filtering by PPE class before ranking is worth +0.25 recall@1. Every clause in a subpart
shares the same regulatory phrasing, so unfiltered similarity is close to a coin flip.

```bash
python kb/eval_retrieval.py            # the numbers above
python kb/eval_retrieval.py --ragas    # adds LLM-judged metrics, needs OPENAI_API_KEY
```

## Rebuild

```bash
python kb/ingest.py           # corpus -> chunks.json
python kb/build_index.py      # chunks -> index/
python kb/eval_retrieval.py   # confirm nothing regressed
```

## Two findings worth keeping

**Construction has no hand-protection clause.** General industry has 1910.138; 1926 has
no equivalent, so gloves cite 1926.95's general criteria as *site policy*, not as a
regulatory requirement. A notice must say which it is.

**1926.201 covers flaggers only.** It is not a general high-visibility requirement, and
citing it for ordinary site hi-vis would be wrong. The `loading_dock` zone override is
the one place it applies, because personnel there direct vehicle movement.
