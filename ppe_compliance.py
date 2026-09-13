"""
Turn YOLO26 PPE detections into auditable per-person compliance records.

This is the deterministic layer that sits between the detector and the agents.
It answers "which person is wearing what" using geometry, applies a site policy,
and emits a structured record. It never decides tone, wording, or escalation --
that is the agents' job, and they should receive only these facts.

Design rule: absence of a detection is NOT proof of absence of PPE. The detector
recalls ~0.78, so a missing box can mean "not worn" or "not seen". Every output
therefore carries an explicit status of compliant / violation / review, and
anything the geometry cannot assess is marked indeterminate rather than guessed.

Usage:
    from ppe_compliance import Policy, assess_image
    report = assess_image("frame.jpg", "/workspace/runs/.../best.pt", Policy())
    print(report.to_json())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

PERSON_CLASS = "Person"

# Vertical band each item may occupy, as a fraction of the person box height
# measured from the top of the box. Generous on purpose -- arms raise, people
# crouch. The containment test does most of the work; these bands only reject
# obvious nonsense, e.g. a helmet carried at waist height rather than worn.
ZONES: dict[str, tuple[float, float]] = {
    "helmet":  (0.00, 0.30),
    "goggles": (0.00, 0.30),
    "vest":    (0.10, 0.70),
    "gloves":  (0.10, 1.00),
    "boots":   (0.55, 1.00),
}

# Which body zone must be visible for the item to be assessable at all.
NEEDS_VISIBLE: dict[str, str] = {
    "helmet":  "top",
    "goggles": "top",
    "vest":    "middle",
    "gloves":  "middle",
    "boots":   "bottom",
}


@dataclass(frozen=True)
class Policy:
    """Site rules and the confidence gates that keep this honest.

    `required` defaults to helmet + vest deliberately. Those are the two classes
    the detector actually performs on (mAP50 0.860 / 0.854) and the two most
    universally mandated. gloves / goggles / boots measure 0.775-0.795 and are
    far more site-specific -- enforce them only where the site genuinely
    requires it, and expect more review-status output when you do.
    """

    required: tuple[str, ...] = ("helmet", "vest")
    zone_label: str = "general site"

    person_conf_min: float = 0.50       # below this, the person is ignored entirely
    ppe_conf_min: float = 0.35          # below this, a PPE box does not count as present
    violation_person_conf: float = 0.70  # below this, a missing item is only "review"

    min_person_height_px: int = 120     # smaller than this, small PPE is unreliable
    containment_min: float = 0.60       # IoA of PPE box inside person box
    edge_margin_px: int = 4             # box within this of the frame edge = truncated


@dataclass
class PersonAssessment:
    person_id: int
    bbox: tuple[int, int, int, int]
    person_confidence: float
    present: dict[str, float] = field(default_factory=dict)   # class -> detection conf
    missing: list[str] = field(default_factory=list)
    indeterminate: list[str] = field(default_factory=list)
    status: str = "compliant"                                  # compliant|violation|review
    reasons: list[str] = field(default_factory=list)


@dataclass
class FrameReport:
    source: str
    captured_at: str
    model: str
    policy: dict
    image_size: tuple[int, int]
    persons: list[PersonAssessment]
    unassigned_ppe: list[dict]

    @property
    def violations(self) -> list[PersonAssessment]:
        return [p for p in self.persons if p.status == "violation"]

    @property
    def needs_review(self) -> list[PersonAssessment]:
        return [p for p in self.persons if p.status == "review"]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)


def _ioa(inner: tuple, outer: tuple) -> float:
    """Intersection over area of `inner`. Used instead of IoU because a glove box
    is tiny next to a person box -- IoU would be near zero even when fully worn."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def _zone_fit(ppe: tuple, person: tuple, cls: str) -> bool:
    lo, hi = ZONES.get(cls, (0.0, 1.0))
    cy = (ppe[1] + ppe[3]) / 2.0
    height = max(1e-9, person[3] - person[1])
    return lo <= (cy - person[1]) / height <= hi


def _visible_zones(person: tuple, img_w: int, img_h: int, margin: int) -> set[str]:
    """Which thirds of the person are actually inside the frame."""
    visible = {"top", "middle", "bottom"}
    if person[1] <= margin:
        visible.discard("top")          # head cut off by frame edge
    if person[3] >= img_h - margin:
        visible.discard("bottom")       # feet cut off
    if person[0] <= margin or person[2] >= img_w - margin:
        visible.discard("middle")       # torso partially out of frame
    return visible


def assess_detections(
    boxes,
    names: dict[int, str],
    img_w: int,
    img_h: int,
    policy: Policy,
) -> tuple[list[PersonAssessment], list[dict]]:
    """Associate PPE detections with people and apply the policy."""
    people, items = [], []
    for b in boxes:
        cls = names[int(b.cls)]
        conf = float(b.conf)
        xyxy = tuple(float(v) for v in b.xyxy[0])
        if cls == PERSON_CLASS:
            if conf >= policy.person_conf_min:
                people.append((xyxy, conf))
        elif conf >= policy.ppe_conf_min:
            items.append((xyxy, conf, cls))

    people.sort(key=lambda p: -(p[0][3] - p[0][1]))     # tallest (nearest) first
    assessments = [
        PersonAssessment(person_id=i, bbox=tuple(int(v) for v in box), person_confidence=conf)
        for i, (box, conf) in enumerate(people)
    ]

    # Each PPE item belongs to at most one person: the best containment score.
    unassigned = []
    for xyxy, conf, cls in items:
        best_i, best_score = None, 0.0
        for i, (pbox, _) in enumerate(people):
            score = _ioa(xyxy, pbox)
            if score >= policy.containment_min and _zone_fit(xyxy, pbox, cls) and score > best_score:
                best_i, best_score = i, score
        if best_i is None:
            unassigned.append({
                "class": cls,
                "confidence": round(conf, 3),
                "bbox": [int(v) for v in xyxy],
                "note": "not contained in any person box, or outside that item's body zone",
            })
        else:
            prev = assessments[best_i].present.get(cls, 0.0)
            assessments[best_i].present[cls] = round(max(prev, conf), 3)

    for a in assessments:
        px1, py1, px2, py2 = a.bbox
        height = py2 - py1
        visible = _visible_zones(a.bbox, img_w, img_h, policy.edge_margin_px)

        too_small = height < policy.min_person_height_px
        if too_small:
            a.reasons.append(
                f"person is {height}px tall, below the {policy.min_person_height_px}px "
                f"threshold for reliable PPE detection"
            )

        for item in policy.required:
            if item in a.present:
                continue
            if NEEDS_VISIBLE.get(item, "middle") not in visible:
                a.indeterminate.append(item)
                a.reasons.append(f"{item}: body zone not fully inside the frame")
            elif too_small:
                a.indeterminate.append(item)
            else:
                a.missing.append(item)

        low_conf = a.person_confidence < policy.violation_person_conf
        if low_conf and a.missing:
            a.reasons.append(
                f"person detected at {a.person_confidence:.2f}, below the "
                f"{policy.violation_person_conf:.2f} gate for asserting a violation"
            )

        if a.missing and not low_conf and not too_small and not a.indeterminate:
            a.status = "violation"
        elif a.missing or a.indeterminate:
            a.status = "review"
        else:
            a.status = "compliant"

    return assessments, unassigned


def assess_image(image_path: str | Path, weights: str | Path, policy: Policy | None = None,
                 imgsz: int | None = None) -> FrameReport:
    """Run the detector on one image and return a structured compliance report."""
    policy = policy or Policy()
    model = YOLO(str(weights))

    kwargs = {"conf": min(policy.person_conf_min, policy.ppe_conf_min), "verbose": False}
    if imgsz:                         # omit to inherit the training resolution
        kwargs["imgsz"] = imgsz
    result = model.predict(str(image_path), **kwargs)[0]

    img_h, img_w = result.orig_shape
    persons, unassigned = assess_detections(result.boxes, model.names, img_w, img_h, policy)

    return FrameReport(
        source=str(image_path),
        captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=str(weights),
        policy=asdict(policy),
        image_size=(img_w, img_h),
        persons=persons,
        unassigned_ppe=unassigned,
    )


def save_evidence(image_path: str | Path, person: PersonAssessment, out_dir: str | Path,
                  pad: float = 0.15) -> str:
    """Crop the person out of the frame so a reviewer can check the claim themselves.

    Every escalated record must carry one of these. A violation report that a human
    cannot independently verify is not auditable.
    """
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path)
    x1, y1, x2, y2 = person.bbox
    dx, dy = int((x2 - x1) * pad), int((y2 - y1) * pad)
    crop = img.crop((max(0, x1 - dx), max(0, y1 - dy),
                     min(img.width, x2 + dx), min(img.height, y2 + dy)))
    dst = out_dir / f"{Path(image_path).stem}_person{person.person_id}.jpg"
    crop.save(dst, quality=92)
    return str(dst)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="PPE compliance assessment for one image")
    ap.add_argument("image")
    ap.add_argument("--weights", default="/workspace/runs/ppe-presence-s960-3/weights/best.pt")
    ap.add_argument("--required", default="helmet,vest")
    ap.add_argument("--zone", default="general site")
    ap.add_argument("--evidence-dir", default="/workspace/evidence")
    args = ap.parse_args()

    pol = Policy(required=tuple(args.required.split(",")), zone_label=args.zone)
    rep = assess_image(args.image, args.weights, pol)

    print(rep.to_json())
    for p in rep.violations + rep.needs_review:
        print(f"person {p.person_id}: {p.status} -> evidence {save_evidence(args.image, p, args.evidence_dir)}")
