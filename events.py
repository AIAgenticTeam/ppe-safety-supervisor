"""
ViolationEvent -- the frozen contract between perception and the agent layer.

THIS IS THE INTERFACE. Lane A produces these; Lanes C and D consume them. Nothing
downstream of this module ever sees a bounding box, a confidence tensor, or a
frame. If you need a new field, bump SCHEMA_VERSION and tell the whole team --
do not add one quietly.

Three fields exist specifically so the agent can reason honestly about what the
vision system does NOT know:

  ppe.indeterminate   items that could not be assessed (occluded, out of frame,
                      subject too small) -- distinct from ppe.missing, which
                      means "looked, and it was not there"
  detector.recall     measured per-class recall. gloves at 0.733 means roughly
                      one worn glove in four is missed, so an absence is weaker
                      evidence than a presence. The agent must weigh this.
  confirmation        how many frames agreed. A single-frame absence is not a
                      violation; sustained absence across a tracked window is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any

SCHEMA_VERSION = "1.0"
RIYADH = timezone(timedelta(hours=3))

# Measured on the held-out test split, YOLO26s @ 960px, ppe-presence-s960-3.
# Shipped inside every event so the agent can size its own uncertainty.
DETECTOR_RECALL: dict[str, float] = {
    "helmet": 0.796,
    "vest": 0.816,
    "boots": 0.742,
    "goggles": 0.723,
    "gloves": 0.733,
    "Person": 0.867,
}

STATUS_VALUES = ("compliant", "violation", "review")


@dataclass
class Subject:
    """The person the event is about."""

    track_id: int                        # stable within one camera session only
    bbox: tuple[int, int, int, int]
    detection_confidence: float
    height_px: int
    worker_ref: str | None = None
    # worker_ref is NOT produced by the vision system. ByteTrack ids do not
    # survive a session, a camera hand-off, or a worker leaving frame, so
    # "third time this week" cannot come from tracking alone. It is bound by a
    # supervisor in the console, or by a badge/roster integration. Left null,
    # history lookups fall back to track_id within the current session.


@dataclass
class ZoneRef:
    name: str
    label: str
    required_ppe: list[str]
    severity_multiplier: float = 1.0


@dataclass
class PPEState:
    present: dict[str, float] = field(default_factory=dict)   # class -> confidence
    missing: list[str] = field(default_factory=list)          # required, looked, absent
    indeterminate: list[str] = field(default_factory=list)    # required, could not assess


@dataclass
class Confirmation:
    """Temporal evidence. Populated by the tracker (Lane A, day 2)."""

    frames_observed: int = 1
    frames_missing: int = 1
    window_seconds: float = 0.0
    rule: str = "single_frame"
    confirmed: bool = False


@dataclass
class Evidence:
    crop_url: str | None = None
    frame_url: str | None = None
    crop_path: str | None = None
    frame_path: str | None = None


@dataclass
class DetectorInfo:
    weights: str = "ppe-presence-s960-3/weights/best.pt"
    imgsz: int = 960
    recall: dict[str, float] = field(default_factory=lambda: dict(DETECTOR_RECALL))


@dataclass
class ViolationEvent:
    event_id: str
    captured_at: str
    camera_id: str
    zone: ZoneRef
    subject: Subject
    ppe: PPEState
    status: str                                   # one of STATUS_VALUES
    confirmation: Confirmation = field(default_factory=Confirmation)
    evidence: Evidence = field(default_factory=Evidence)
    detector: DetectorInfo = field(default_factory=DetectorInfo)
    reasons: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    # ---- helpers the agent layer will actually use ----------------------

    @property
    def is_actionable(self) -> bool:
        """True only for confirmed violations. Everything else goes to a human."""
        return self.status == "violation" and self.confirmation.confirmed

    def weakest_evidence(self) -> float:
        """Lowest detector recall among the missing items -- how much to trust this."""
        if not self.ppe.missing:
            return 1.0
        return min(self.detector.recall.get(c, 0.7) for c in self.ppe.missing)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "ViolationEvent":
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"event schema {d.get('schema_version')!r} != expected {SCHEMA_VERSION!r}")
        return cls(
            event_id=d["event_id"],
            captured_at=d["captured_at"],
            camera_id=d["camera_id"],
            zone=ZoneRef(**d["zone"]),
            subject=Subject(**{**d["subject"], "bbox": tuple(d["subject"]["bbox"])}),
            ppe=PPEState(**d["ppe"]),
            status=d["status"],
            confirmation=Confirmation(**d.get("confirmation", {})),
            evidence=Evidence(**d.get("evidence", {})),
            detector=DetectorInfo(**d.get("detector", {})),
            reasons=d.get("reasons", []),
        )


def make_event_id(camera_id: str, track_id: int, when: datetime) -> str:
    return f"evt_{when.strftime('%Y%m%dT%H%M%S')}_{camera_id}_t{track_id}"


def build_event(assessment, zone, camera_id: str, track_id: int,
                confirmation: Confirmation | None = None,
                when: datetime | None = None,
                evidence: Evidence | None = None,
                event_id: str | None = None) -> ViolationEvent:
    """Assemble an event from a PersonAssessment (ppe_compliance) and a Zone (zones).

    `assessment` is duck-typed so this module stays importable without ultralytics.

    Pass `event_id` when evidence files have already been written under it. Callers used
    to let this function mint its own and then overwrite the attribute afterwards, which
    worked only because the same inputs produce the same id -- an invariant nothing
    enforced, holding together filenames that were already on disk.
    """
    when = when or datetime.now(RIYADH)
    x1, y1, x2, y2 = assessment.bbox

    return ViolationEvent(
        event_id=event_id or make_event_id(camera_id, track_id, when),
        captured_at=when.isoformat(timespec="seconds"),
        camera_id=camera_id,
        zone=ZoneRef(
            name=zone.name,
            label=zone.label,
            required_ppe=list(zone.required_ppe),
            severity_multiplier=zone.severity_multiplier,
        ),
        subject=Subject(
            track_id=track_id,
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            detection_confidence=round(float(assessment.person_confidence), 3),
            height_px=int(y2 - y1),
        ),
        ppe=PPEState(
            present=dict(assessment.present),
            missing=list(assessment.missing),
            indeterminate=list(assessment.indeterminate),
        ),
        status=assessment.status,
        confirmation=confirmation or Confirmation(),
        evidence=evidence or Evidence(),
        reasons=list(assessment.reasons),
    )


# --------------------------------------------------------------------------
# JSON Schema -- hand this to Lane C so they can validate fixtures in tests.
# --------------------------------------------------------------------------

JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ViolationEvent",
    "type": "object",
    "required": ["event_id", "schema_version", "captured_at", "camera_id",
                 "zone", "subject", "ppe", "status"],
    "properties": {
        "event_id": {"type": "string"},
        "schema_version": {"const": SCHEMA_VERSION},
        "captured_at": {"type": "string", "format": "date-time"},
        "camera_id": {"type": "string"},
        "zone": {
            "type": "object",
            "required": ["name", "label", "required_ppe"],
            "properties": {
                "name": {"type": "string"},
                "label": {"type": "string"},
                "required_ppe": {"type": "array", "items": {"type": "string"}},
                "severity_multiplier": {"type": "number", "minimum": 0},
            },
        },
        "subject": {
            "type": "object",
            "required": ["track_id", "bbox", "detection_confidence", "height_px"],
            "properties": {
                "track_id": {"type": "integer"},
                "bbox": {"type": "array", "items": {"type": "integer"},
                         "minItems": 4, "maxItems": 4},
                "detection_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "height_px": {"type": "integer", "minimum": 0},
                "worker_ref": {"type": ["string", "null"]},
            },
        },
        "ppe": {
            "type": "object",
            "properties": {
                "present": {"type": "object", "additionalProperties": {"type": "number"}},
                "missing": {"type": "array", "items": {"type": "string"}},
                "indeterminate": {"type": "array", "items": {"type": "string"}},
            },
        },
        "status": {"enum": list(STATUS_VALUES)},
        "confirmation": {
            "type": "object",
            "properties": {
                "frames_observed": {"type": "integer"},
                "frames_missing": {"type": "integer"},
                "window_seconds": {"type": "number"},
                "rule": {"type": "string"},
                "confirmed": {"type": "boolean"},
            },
        },
        "evidence": {"type": "object"},
        "detector": {"type": "object"},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
}


if __name__ == "__main__":
    print(json.dumps(JSON_SCHEMA, indent=2))
