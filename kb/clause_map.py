"""
Deterministic clause lookup and severity scoring.

This is the half of Lane B that has no model in it. Given a PPE item and a zone it
returns which regulation applies, on what basis, and how serious the absence is --
reproducibly, with no API call and no variance.

Retrieval never chooses a citation. `kb/retrieve.py` fetches the TEXT of a clause named
here; if those two ever disagree, this file wins.

    from kb.clause_map import ClauseMap
    cm = ClauseMap.load()
    cm.for_item("helmet", "walkway")        # -> ItemRule
    cm.score(["helmet", "vest"], "walkway")  # -> SeverityResult
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

ROOT = Path(__file__).absolute().parents[1]
DEFAULT_PATH = ROOT / "kb" / "clauses.yaml"

Basis = Literal["regulation", "site_policy"]
Band = Literal["compliant", "log_only", "warning", "escalation", "stop_work"]


@dataclass(frozen=True)
class Clause:
    clause_id: str
    title: str
    subpart: str
    authority: str          # specific | general | narrow
    summary: str

    @property
    def citation(self) -> str:
        return f"29 CFR {self.clause_id}"


@dataclass(frozen=True)
class ItemRule:
    """What the notice may claim about one missing item in one zone."""

    item: str
    zone: str
    clause: Clause
    basis: Basis
    notice_phrase: str
    supporting_clause: Clause | None = None
    override_reason: str = ""
    note: str = ""

    @property
    def is_regulatory(self) -> bool:
        """False means the site requires it, not the regulation. The notice must say so."""
        return self.basis == "regulation"


@dataclass(frozen=True)
class SeverityResult:
    missing: tuple[str, ...]
    zone: str
    weights: dict[str, int]
    baseline: int
    priors: int = 0
    final: int = 0
    band: Band = "compliant"

    def explain(self) -> str:
        parts = [f"{i} {self.weights.get(i, 0)}" for i in self.missing] or ["nothing missing"]
        line = f"{' + '.join(parts)} = {self.baseline}"
        if self.priors:
            line += f", +{self.final - self.baseline} for {self.priors} prior(s) = {self.final}"
        return f"{line} -> {self.band}"


class ClauseMap:
    def __init__(self, raw: dict) -> None:
        self._raw = raw
        self._clauses = {
            cid: Clause(cid, c["title"], c.get("subpart", ""),
                        c.get("authority", ""), c.get("summary", "").strip())
            for cid, c in raw["clauses"].items()
        }
        self._items = raw["items"]
        self._overrides = raw.get("zone_overrides", {})
        self._weights = raw["severity_weights"]
        self._bands = raw["bands"]
        self._per_prior = raw.get("repeat_penalty_per_prior", 2)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "ClauseMap":
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    # ---- clause lookup --------------------------------------------------

    @property
    def clauses(self) -> dict[str, Clause]:
        return dict(self._clauses)

    @property
    def known_items(self) -> tuple[str, ...]:
        return tuple(self._items)

    def for_item(self, item: str, zone: str = "default") -> ItemRule:
        """Which clause covers this item here, and on what basis."""
        if item not in self._items:
            raise KeyError(f"no rule for {item!r}; known: {sorted(self._items)}")

        base = self._items[item]
        override = self._overrides.get(zone, {}).get(item, {})

        clause_id = override.get("clause", base["clause"])
        basis = override.get("basis", base["basis"])
        phrase = base["notice_phrase"].strip()

        if override and override.get("clause") != base["clause"]:
            # A zone raised this item to a specific clause because the hazard that
            # clause names is actually present here. Say so in the notice.
            clause = self._clauses[clause_id]
            phrase = f"{clause.title.lower()} required under {clause.citation}"

        supporting = base.get("supporting_clause")
        return ItemRule(
            item=item,
            zone=zone,
            clause=self._clauses[clause_id],
            basis=basis,
            notice_phrase=phrase,
            supporting_clause=self._clauses.get(supporting) if supporting else None,
            override_reason=override.get("reason", ""),
            note=base.get("note", "").strip(),
        )

    def citations_for(self, missing: list[str], zone: str = "default") -> list[ItemRule]:
        return [self.for_item(i, zone) for i in missing]

    # ---- severity -------------------------------------------------------

    def weights_for(self, zone: str, camera: str | None = None) -> dict[str, int]:
        """Weights for a zone, preferring a camera-scoped entry over a bare name.

        Zone names are only unique within a camera -- two sites can each have a
        "walkway" with different requirements -- so a bare name silently shares one
        weight table between them. `camera/zone` keys take precedence where present.
        """
        if camera:
            scoped = self._weights.get(f"{camera}/{zone}")
            if scoped is not None:
                return dict(scoped)
        return dict(self._weights.get(zone, self._weights["default"]))

    def score(self, missing: list[str], zone: str = "default", priors: int = 0,
              camera: str | None = None, required: list[str] | None = None,
              strict: bool = True) -> SeverityResult:
        """Sum the weights of MISSING items.

        Deliberately not zone-total-minus-missing: that makes the threshold mean
        something different in every zone, and any zone whose total falls below it
        reports a fully compliant worker as severe.

        `required` is the zone's requirement list from zones.json. Passing it turns on
        the check that every required item actually has a weight -- without it, an
        unweighted item scores zero, the violation is reported, and it never escalates.
        Nothing looks wrong; the number is just quietly too low. That is the failure
        this guard exists to make loud.
        """
        weights = self.weights_for(zone, camera)

        if strict:
            unweighted = [i for i in (required or missing) if i not in weights]
            if unweighted:
                where = f"{camera}/{zone}" if camera else zone
                raise ValueError(
                    f"zone {where!r} requires {unweighted} but kb/clauses.yaml gives "
                    f"{'it' if len(unweighted) == 1 else 'them'} no severity weight. "
                    f"An unweighted item scores 0 and can never escalate. "
                    f"Add {'it' if len(unweighted) == 1 else 'them'} to severity_weights."
                )

        baseline = sum(weights.get(i, 0) for i in missing)
        final = baseline + self._per_prior * priors
        return SeverityResult(
            missing=tuple(missing), zone=zone, weights=weights,
            baseline=baseline, priors=priors, final=final,
            band=self.band_for(final),
        )

    def band_for(self, score: int) -> Band:
        for name, (lo, hi) in self._bands.items():
            if lo <= score <= hi:
                return name                          # type: ignore[return-value]
        return "stop_work"

    # ---- self-check -----------------------------------------------------

    def check_against_zones(
            self, zones_path: str | Path = ROOT / "config" / "zones.json") -> list[str]:
        """Do zones.json and this file agree about every zone?

        They each hold half the truth -- zones.json says what a zone requires, this file
        says what each item is worth -- and they can drift apart silently. The dangerous
        direction is a required item with no weight: it scores zero, the violation is
        reported, and it never escalates while nothing looks wrong.
        """
        if not Path(zones_path).exists():
            return []

        import json

        problems: list[str] = []
        raw = json.loads(Path(zones_path).read_text(encoding="utf-8"))
        for cam_id, cam in raw.get("cameras", {}).items():
            for zone in cam.get("zones", []):
                required = set(zone.get("required_ppe", []))
                key = f"{cam_id}/{zone['name']}"
                weights = set(self._weights.get(key, {}))

                if key not in self._weights:
                    problems.append(
                        f"{key}: no severity weights -- falls back to a bare zone name "
                        f"or the default, which may not match what this zone requires")
                    continue
                for item in sorted(required - weights):
                    problems.append(
                        f"{key}: requires {item!r} but it has no weight -- it would "
                        f"score 0 and never escalate")
                for item in sorted(weights - required):
                    problems.append(
                        f"{key}: weights {item!r} but the zone does not require it")
                for item in sorted(required - set(self._items)):
                    problems.append(f"{key}: requires {item!r} but no clause covers it")
        return problems

    def validate(self) -> list[str]:
        """Problems that would make a notice cite something wrong. Empty means sane."""
        problems: list[str] = []

        for item, rule in self._items.items():
            if rule["clause"] not in self._clauses:
                problems.append(f"item {item}: unknown clause {rule['clause']}")
            if rule["basis"] not in ("regulation", "site_policy"):
                problems.append(f"item {item}: bad basis {rule['basis']!r}")
            sup = rule.get("supporting_clause")
            if sup and sup not in self._clauses:
                problems.append(f"item {item}: unknown supporting clause {sup}")
            if rule["basis"] == "site_policy" and "site policy" not in rule["notice_phrase"]:
                problems.append(
                    f"item {item}: basis is site_policy but the notice phrase does not "
                    f"say so -- it would read as a regulatory requirement")

        for zone, items in self._overrides.items():
            for item, ov in items.items():
                if item not in self._items:
                    problems.append(f"override {zone}/{item}: no such item")
                if ov.get("clause") not in self._clauses:
                    problems.append(f"override {zone}/{item}: unknown clause")
                if not ov.get("reason"):
                    problems.append(f"override {zone}/{item}: no reason given")

        for zone, weights in self._weights.items():
            for item in weights:
                if item not in self._items:
                    problems.append(f"weights {zone}: unknown item {item}")

        # bands must tile 0..N with no gap and no overlap
        edges = sorted((lo, hi) for lo, hi in self._bands.values())
        if edges[0][0] != 0:
            problems.append("bands do not start at 0")
        for (_, hi), (lo, _) in zip(edges, edges[1:]):
            if lo != hi + 1:
                problems.append(f"band gap or overlap between {hi} and {lo}")

        problems.extend(self.check_against_zones())

        # every clause cited by an item should be reachable and used
        used = {r["clause"] for r in self._items.values()}
        used |= {r["supporting_clause"] for r in self._items.values()
                 if r.get("supporting_clause")}
        # A specification clause is cited too -- 1926.96 says what safety-toe footwear
        # must MEET even though the duty to wear it comes from elsewhere.
        used |= {r["specification_clause"] for r in self._items.values()
                 if r.get("specification_clause")}
        used |= {ov["clause"] for z in self._overrides.values() for ov in z.values()}
        for cid in self._clauses:
            if cid not in used:
                problems.append(f"clause {cid} is defined but never cited")

        return problems


if __name__ == "__main__":
    import sys

    cm = ClauseMap.load()
    problems = cm.validate()

    print(f"{len(cm.clauses)} clauses, {len(cm.known_items)} PPE items\n")
    print(f"{'item':<9} {'zone':<18} {'clause':<12} {'basis':<12} authority")
    print("-" * 66)
    for item in cm.known_items:
        for zone in ("default", "grinding_station", "loading_dock"):
            r = cm.for_item(item, zone)
            print(f"{item:<9} {zone:<18} {r.clause.clause_id:<12} "
                  f"{r.basis:<12} {r.clause.authority}")

    print("\nseverity examples")
    for missing, zone, priors in (
        (["helmet"], "walkway", 0),
        (["helmet", "vest"], "walkway", 0),
        (["gloves"], "grinding_station", 0),
        (["helmet"], "grinding_station", 2),
        ([], "walkway", 0),
    ):
        s = cm.score(missing, zone, priors)
        label = "+".join(missing) or "nothing"
        print(f"  {label:<16} in {zone:<17} priors={priors}  {s.explain()}")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("\nclause map OK")
