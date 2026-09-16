# Source text

29 CFR 1926 XML, pulled from the eCFR API. Committed so the build is reproducible
without network access, and so a future reader can see exactly which text the citations
were drawn from.

| file | subpart | sections used |
| --- | --- | --- |
| `1926_subpartC.xml` | C — General safety and health provisions | 1926.28 |
| `1926_subpartE.xml` | E — Personal protective and life saving equipment | 1926.95, .96, .100, .102 |
| `1926_subpartG.xml` | G — Signs, signals and barricades | 1926.201 |

## Refetching

The API requires gzip; without an `Accept-Encoding` header it returns 406.

```bash
curl -s --compressed -o kb/corpus/1926_subpartE.xml \
  -H "Accept: application/xml" \
  "https://www.ecfr.gov/api/versioner/v1/full/2026-09-01/title-29.xml?part=1926&subpart=E"
```

Swap `subpart=E` for `C` or `G`. Then rebuild:

```bash
python kb/ingest.py && python kb/build_index.py && python kb/eval_retrieval.py
```

Retrieved 2026-09-16. Regulations change; if a citation in a notice ever looks wrong,
check the date above before blaming the code.
