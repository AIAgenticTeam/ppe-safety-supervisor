"""
Local evidence store: the crop and frame a human needs to check a finding.

A violation report nobody can independently verify is not auditable. Every confirmed
finding therefore carries two images, keyed by event id:

    frame_<event_id>.jpg   the whole scene, so a reviewer can see WHERE the person was
    crop_<event_id>.jpg    the person, so a reviewer can see WHAT they were wearing

Both are annotated with the detection boxes. That matters: a bare crop asks the
supervisor to take the system's word for it, while an annotated one shows what the
model actually saw and lets them disagree with it.

Storage is the local filesystem on purpose. Object storage would add an account, API
keys, upload latency and a network failure mode in the middle of a live demo, in
exchange for nothing the demo needs. The store returns paths; swapping in a bucket
later means changing this file and nothing else.

    store = EvidenceStore("events/evidence")
    ev = store.save(event_id, frame_bgr, person.bbox, label="missing helmet")
    # ev.crop_path, ev.frame_path -> straight into ViolationEvent.evidence
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from events import Evidence

# Drawn in BGR, since frames arrive from OpenCV.
BOX_PERSON = (200, 90, 240)      # violet
BOX_PPE = (90, 190, 90)          # green
BOX_MISSING = (60, 60, 220)      # red
TEXT = (255, 255, 255)


@dataclass(frozen=True)
class StoredEvidence:
    event_id: str
    frame_path: str
    crop_path: str

    def as_event_evidence(self) -> Evidence:
        """Convert into the Evidence block the ViolationEvent contract expects."""
        return Evidence(frame_path=self.frame_path, crop_path=self.crop_path)


class EvidenceStore:
    """Writes annotated frame + crop pairs for confirmed findings."""

    def __init__(self, root: str | Path = "events/evidence", pad: float = 0.18,
                 jpeg_quality: int = 88, max_frame_width: int = 1600) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pad = pad
        self.jpeg_quality = jpeg_quality
        # Full frames are stored downscaled: a 4K JPEG per violation fills a disk fast,
        # and 1600px is well beyond what a reviewer needs to make a judgement.
        self.max_frame_width = max_frame_width

    # ---- writing --------------------------------------------------------

    def save(self, event_id: str, frame, bbox: tuple[int, int, int, int],
             label: str = "", present_boxes: dict | None = None) -> StoredEvidence:
        """Save annotated frame + crop for one person in one frame.

        `frame` is a BGR numpy array (straight from OpenCV or ultralytics).
        `present_boxes` optionally maps class name -> (x1,y1,x2,y2) for detected PPE,
        drawn so the reviewer can see what the model credited them with.
        """
        import cv2

        annotated = frame.copy()
        x1, y1, x2, y2 = (int(v) for v in bbox)

        for cls, box in (present_boxes or {}).items():
            bx1, by1, bx2, by2 = (int(v) for v in box)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), BOX_PPE, 2)
            _caption(cv2, annotated, cls, bx1, by1, BOX_PPE)

        colour = BOX_MISSING if label else BOX_PERSON
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 3)
        if label:
            _caption(cv2, annotated, label, x1, y1, colour)

        crop = self._crop(annotated, bbox)
        frame_out = self._downscale(annotated)

        frame_path = self.root / f"frame_{event_id}.jpg"
        crop_path = self.root / f"crop_{event_id}.jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        cv2.imwrite(str(frame_path), frame_out, params)
        cv2.imwrite(str(crop_path), crop, params)

        return StoredEvidence(event_id, str(frame_path), str(crop_path))

    def save_from_image_file(self, event_id: str, image_path: str | Path,
                             bbox, label: str = "") -> StoredEvidence:
        """Same, for a frame that already exists on disk (the stills pipeline)."""
        import cv2

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"could not read {image_path}")
        return self.save(event_id, frame, bbox, label)

    # ---- helpers --------------------------------------------------------

    def _crop(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        dx, dy = int((x2 - x1) * self.pad), int((y2 - y1) * self.pad)
        return frame[max(0, y1 - dy):min(h, y2 + dy),
                     max(0, x1 - dx):min(w, x2 + dx)]

    def _downscale(self, frame):
        import cv2

        h, w = frame.shape[:2]
        if w <= self.max_frame_width:
            return frame
        scale = self.max_frame_width / w
        return cv2.resize(frame, (self.max_frame_width, int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    # ---- housekeeping ---------------------------------------------------

    def paths_for(self, event_id: str) -> StoredEvidence | None:
        """Look up stored evidence; None if this event has none."""
        frame_path = self.root / f"frame_{event_id}.jpg"
        crop_path = self.root / f"crop_{event_id}.jpg"
        if frame_path.exists() and crop_path.exists():
            return StoredEvidence(event_id, str(frame_path), str(crop_path))
        return None

    def usage(self) -> dict:
        files = list(self.root.glob("*.jpg"))
        return {
            "files": len(files),
            "events": len(files) // 2,
            "megabytes": round(sum(f.stat().st_size for f in files) / 1e6, 1),
        }

    def prune(self, keep_event_ids: set[str]) -> int:
        """Delete evidence for events no longer in the log. Returns files removed.

        Evidence outlives nothing: if the event is gone, so is its justification for
        holding a photograph of a worker.
        """
        removed = 0
        for f in self.root.glob("*.jpg"):
            event_id = f.stem.split("_", 1)[1] if "_" in f.stem else ""
            if event_id not in keep_event_ids:
                f.unlink()
                removed += 1
        return removed

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)


def _caption(cv2, img, text: str, x: int, y: int, colour) -> None:
    """Readable label that stays inside the frame."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    ty = max(th + 6, y)
    cv2.rectangle(img, (x, ty - th - base - 4), (x + tw + 8, ty), colour, -1)
    cv2.putText(img, text, (x + 4, ty - base - 1), font, scale, TEXT, thick, cv2.LINE_AA)
