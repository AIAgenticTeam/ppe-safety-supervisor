"""
Drift monitoring over the event store, with Evidently.

    python -m app.monitoring --db data/safety.db --out drift.html

The question this answers is narrow and it is the one that matters for this system:
**has the detector walked off the distribution it was trained on?**

That is not a hypothetical. The detector is documented as domain-gap limited -- four
clips produced four different failure modes -- and the risk register puts "domain gap on
demo footage" near the top. A camera that gets moved, a site with different lighting, or
a new contractor in unfamiliar hi-vis all shift the confidence distribution before they
shift anything a human notices. Recall does not announce itself when it drops; findings
simply stop appearing, and an empty queue looks exactly like a safe site.

So the features tracked are the ones that move when perception degrades, plus the ones
that move when judgement does:

    person_confidence     how sure the detector was it saw a person at all
    weakest_recall        the detector's own recall for the items it called missing
    missing_ratio         how much of the confirmation window sustained the absence
    missing_count         how many items at once
    zone, camera_id       where -- a moved camera shows up here first
    status                violation / review / compliant mix
    action, severity      what the agents decided
    attributed            whether anyone is being identified at all

That last one is quiet and important. If identity binding stops happening, the memory
loop stops working, every third offence reads as a first, and nothing in the output
looks wrong. Drift in `attributed` is the only place that becomes visible.

A deliberate refusal: below MIN_ROWS in either window this reports "not enough data"
rather than a drift verdict. Evidently will happily compare four rows to six and report
a confident p-value, and a confident number computed from nothing is worse here than no
number, because somebody will put it in a report.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import DEFAULT_DB_PATH, EventStore  # noqa: E402

# Below this, a drift verdict is noise wearing a p-value.
MIN_ROWS = 20

NUMERICAL = ["person_confidence", "weakest_recall", "missing_ratio", "missing_count",
             "severity"]
CATEGORICAL = ["zone", "camera_id", "status", "action", "attributed"]


# ---------------------------------------------------------------- the frame

def _row(event: dict, decision: dict | None) -> dict:
    """One event flattened into the features worth watching."""
    subject = event.get("subject") or {}
    ppe = event.get("ppe") or {}
    confirmation = event.get("confirmation") or {}
    recalls = (event.get("detector") or {}).get("recall") or {}
    missing = ppe.get("missing") or []

    observed = confirmation.get("frames_observed") or 0
    considered = [recalls.get(item) for item in missing if recalls.get(item)]

    return {
        "captured_at": event.get("captured_at"),
        "person_confidence": subject.get("detection_confidence"),
        # The recall of the item we are least able to see is what limits the finding.
        "weakest_recall": min(considered) if considered else None,
        "missing_ratio": ((confirmation.get("frames_missing") or 0) / observed
                          if observed else None),
        "missing_count": len(missing),
        "zone": (event.get("zone") or {}).get("name") or "unknown",
        "camera_id": event.get("camera_id") or "unknown",
        "status": event.get("status") or "unknown",
        "action": (decision or {}).get("action") or "undecided",
        "severity": (decision or {}).get("severity"),
        "attributed": "yes" if event.get("_worker_id") else "no",
    }


def event_frame(db: EventStore, limit: int = 5000):
    """Every recorded finding as a row, newest first."""
    import pandas as pd

    rows = []
    for summary in db.recent(limit=limit):
        event = db.get_event(summary["event_id"])
        if event is None:
            continue
        event = {**event, "_worker_id": summary.get("worker_id")}
        rows.append(_row(event, db.get_decision(summary["event_id"])))

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["captured_at"] = pd.to_datetime(frame["captured_at"], format="mixed",
                                              utc=True, errors="coerce")
        frame = frame.sort_values("captured_at")
    return frame


def split_by_time(frame, split_at: datetime | None = None, days: int = 7):
    """Reference is what came before, current is what came after.

    Comparing recent events against the fixtures was the alternative, and it is worse:
    the fixtures are hand-built, so any difference would mostly measure the gap between
    synthetic and real rather than any change on the site.
    """
    import pandas as pd

    if frame.empty:
        return frame, frame
    if split_at is None:
        split_at = datetime.now().astimezone() - timedelta(days=days)
    cutoff = pd.Timestamp(split_at).tz_convert("UTC") if pd.Timestamp(
        split_at).tzinfo else pd.Timestamp(split_at, tz="UTC")
    return frame[frame["captured_at"] < cutoff], frame[frame["captured_at"] >= cutoff]


# ---------------------------------------------------------------- the report

@dataclass
class ColumnDrift:
    column: str
    score: float                 # p-value, or a distance where the test has no p
    threshold: float
    drifted: bool
    method: str = ""

    def as_dict(self) -> dict:
        return {"column": self.column, "score": self.score,
                "threshold": self.threshold, "drifted": self.drifted,
                "method": self.method}


@dataclass
class DriftResult:
    ran: bool
    reason: str = ""
    reference_rows: int = 0
    current_rows: int = 0
    columns: list[ColumnDrift] = field(default_factory=list)
    html: str = ""

    @property
    def drifted_columns(self) -> list[str]:
        return [c.column for c in self.columns if c.drifted]

    @property
    def checked_columns(self) -> int:
        return len(self.columns)

    @property
    def share(self) -> float:
        return (len(self.drifted_columns) / self.checked_columns
                if self.checked_columns else 0.0)

    def summary(self) -> dict:
        return {
            "ran": self.ran,
            "reason": self.reason,
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "checked_columns": self.checked_columns,
            "drifted_columns": self.drifted_columns,
            "drift_share": round(self.share, 3),
            "columns": [c.as_dict() for c in self.columns],
        }


def usable_columns(reference, current) -> list[str]:
    """Columns that could actually move.

    Two exclusions, both for the same reason -- a column that cannot drift should not
    be counted as a column that did not drift:

    * nothing recorded in it at all (no decisions yet, so no `severity`);
    * one distinct value across BOTH windows together. A site with a single camera has
      a constant `camera_id`, and including it dilutes the drift share with a column
      that had no way of changing. Constant within one window but different across the
      two is exactly drift, so the test is on the pair, never on either alone.

    It also silences a pile of divide-by-zero warnings from the correlation step, but
    that is the symptom rather than the reason.
    """
    import pandas as pd

    both = pd.concat([reference, current])
    # Only the tracked features. `captured_at` is how the two windows were split in the
    # first place, so of course it "differs" between them -- including it would be
    # measuring the split.
    tracked = [c for c in (*NUMERICAL, *CATEGORICAL) if c in both.columns]
    return [c for c in tracked
            if both[c].notna().any() and both[c].nunique(dropna=True) > 1]


def _definition(reference, current):
    from evidently import DataDefinition

    usable = usable_columns(reference, current)
    return DataDefinition(
        numerical_columns=[c for c in NUMERICAL if c in usable],
        categorical_columns=[c for c in CATEGORICAL if c in usable],
    )


def drift_report(reference, current, html_path: str | Path | None = None) -> DriftResult:
    """Compare two windows of findings and say which features moved."""
    if len(reference) < MIN_ROWS or len(current) < MIN_ROWS:
        return DriftResult(
            ran=False,
            reason=(f"not enough data: {len(reference)} reference and {len(current)} "
                    f"current findings, {MIN_ROWS} needed in each. A drift verdict "
                    f"computed from fewer is noise with a p-value attached."),
            reference_rows=len(reference), current_rows=len(current))

    from evidently import Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = _definition(reference, current)
    keep = usable_columns(reference, current)
    if not keep:
        return DriftResult(
            ran=False,
            reason="every tracked feature holds a single value across both windows, "
                   "so none of them could move. Nothing to compare.",
            reference_rows=len(reference), current_rows=len(current))

    ref = Dataset.from_pandas(reference[keep], data_definition=definition)
    cur = Dataset.from_pandas(current[keep], data_definition=definition)

    run = Report(metrics=[DataDriftPreset()]).run(cur, ref)
    result = DriftResult(
        ran=True, reference_rows=len(reference), current_rows=len(current),
        columns=_column_drift(run.dict()))

    if html_path:
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        run.save_html(str(html_path))
        result.html = str(html_path)
    return result


def _column_drift(raw: dict) -> list[ColumnDrift]:
    """Pull the per-column verdicts out of the snapshot.

    Read from `config`, which carries the column, the test and the threshold, rather
    than by parsing them back out of the display name. An earlier version of this read
    a key that does not exist and quietly reported zero columns checked and zero
    drifted — which looks identical to "everything is fine". Anything this cannot
    parse is therefore left out of the list rather than counted as clean, so a parsing
    failure shows up as nothing checked rather than nothing wrong.
    """
    columns: list[ColumnDrift] = []
    for metric in raw.get("metrics", []):
        if not str(metric.get("metric_name", "")).startswith("ValueDrift"):
            continue
        config = metric.get("config") or {}
        column = config.get("column")
        threshold = config.get("threshold")
        value = metric.get("value")
        if column is None or threshold is None or not isinstance(value, (int, float)):
            continue
        score = float(value)          # numpy floats do not survive JSON
        columns.append(ColumnDrift(
            column=column, score=score, threshold=float(threshold),
            # These tests report a p-value: small means the two windows differ.
            drifted=score < float(threshold),
            method=str(config.get("method", ""))))
    return columns


def run_drift(db: EventStore, days: int = 7,
              html_path: str | Path | None = None) -> DriftResult:
    """Convenience: load, split, compare."""
    frame = event_frame(db)
    if frame.empty:
        return DriftResult(ran=False, reason="no findings recorded yet")
    reference, current = split_by_time(frame, days=days)
    return drift_report(reference, current, html_path=html_path)


# ---------------------------------------------------------------- cli

def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB_PATH)
    ap.add_argument("--days", type=int, default=7,
                    help="findings newer than this are the current window")
    ap.add_argument("--out", default="", help="write the full Evidently report here")
    args = ap.parse_args(argv)

    result = run_drift(EventStore(args.db), days=args.days,
                       html_path=args.out or None)
    print(json.dumps(result.summary(), indent=2))
    if result.html:
        print(f"\nfull report: {result.html}")
    if not result.ran:
        return 2
    return 1 if result.drifted_columns else 0


if __name__ == "__main__":
    raise SystemExit(main())
