"""
Temporal confirmation: a single frame is not evidence enough to accuse anyone.

At helmet recall 0.796, roughly one worn helmet in five goes undetected. Single-frame
escalation would therefore mislabel a compliant worker about 20% of the time. This module
is the fix: track each person across frames and require a missing item to be *sustained*
before the finding hardens from `review` into an actionable `violation`.

Two layers, deliberately separated:

  TemporalConfirmer   pure logic. No model, no video, no I/O. Ingests per-frame
                      assessments keyed by track id and decides what is confirmed.
                      Fully unit-testable.

  track_video()       the ultralytics/ByteTrack loop that feeds it.

The rule is N-of-M: an item must be missing in at least N of the last M frames in which
the person was observed. Defaults are 8 of 10.

Three details that matter:

  * `indeterminate` frames are NOT counted as missing. "I could not see their head" is
    not evidence of a bare head. They still occupy a slot in the window, so an
    unassessable person simply never accumulates enough evidence to be accused.
  * A short track cannot confirm. Fewer than M observations means not enough evidence,
    full stop -- somebody crossing a corner of frame is not judged.
  * Confirmation is emitted ONCE per track per item set, not every frame. Otherwise a
    worker standing at a grinder for ten seconds generates 300 identical tickets.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

DEFAULT_N = 8
DEFAULT_M = 10


@dataclass(frozen=True)
class Observation:
    """One frame's assessment of one tracked person."""

    frame_index: int
    status: str                          # compliant | violation | review
    missing: tuple[str, ...] = ()
    indeterminate: tuple[str, ...] = ()


@dataclass
class TrackHistory:
    """Rolling window of observations for a single track id."""

    track_id: int
    window_size: int = DEFAULT_M
    observations: deque[Observation] = field(default_factory=deque)
    confirmed_items: set[str] = field(default_factory=set)   # already emitted, for debounce

    def __post_init__(self) -> None:
        if not isinstance(self.observations, deque):
            self.observations = deque(self.observations, maxlen=self.window_size)
        else:
            self.observations = deque(self.observations, maxlen=self.window_size)

    def add(self, obs: Observation) -> None:
        self.observations.append(obs)

    @property
    def observed(self) -> int:
        return len(self.observations)

    def missing_count(self, item: str) -> int:
        """Frames in the window where this item was assessably absent."""
        return sum(1 for o in self.observations if item in o.missing)

    def indeterminate_count(self, item: str) -> int:
        return sum(1 for o in self.observations if item in o.indeterminate)

    def candidate_items(self) -> set[str]:
        items: set[str] = set()
        for o in self.observations:
            items.update(o.missing)
        return items


@dataclass
class Confirmation:
    """A finding that has survived the temporal test. Ready to become an event."""

    track_id: int
    items: tuple[str, ...]
    frames_observed: int
    frames_missing: int
    window_seconds: float
    rule: str
    frame_index: int

    @property
    def confirmed(self) -> bool:
        return True


class TemporalConfirmer:
    """Decides when a sustained absence becomes actionable.

    Pure logic -- feed it assessments, it tells you what is confirmed. Knows nothing
    about models, frames, or files.

        confirmer = TemporalConfirmer(n=8, m=10, fps=30)
        for frame_index, track_id, assessment in stream:
            for c in confirmer.ingest(frame_index, track_id, assessment):
                ...   # emit an event; this fires once per track per item set
    """

    def __init__(self, n: int = DEFAULT_N, m: int = DEFAULT_M, fps: float = 30.0) -> None:
        if n > m:
            raise ValueError(f"n ({n}) cannot exceed m ({m})")
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.m = m
        self.fps = fps or 1.0
        self.tracks: dict[int, TrackHistory] = {}

    @property
    def rule(self) -> str:
        return f"{self.n}_of_{self.m}"

    def ingest(self, frame_index: int, track_id: int, status: str,
               missing: tuple[str, ...] = (), indeterminate: tuple[str, ...] = (),
               ) -> Iterator[Confirmation]:
        """Record one frame for one track; yield any newly confirmed findings."""
        history = self.tracks.get(track_id)
        if history is None:
            history = TrackHistory(track_id=track_id, window_size=self.m)
            self.tracks[track_id] = history

        history.add(Observation(frame_index, status, tuple(missing), tuple(indeterminate)))

        # Not enough evidence yet. A person glimpsed for three frames is not judged.
        if history.observed < self.m:
            return

        newly: list[str] = []
        for item in sorted(history.candidate_items()):
            if item in history.confirmed_items:
                continue                      # already emitted; do not re-ticket
            if history.missing_count(item) >= self.n:
                newly.append(item)

        if not newly:
            return

        history.confirmed_items.update(newly)
        yield Confirmation(
            track_id=track_id,
            items=tuple(newly),
            frames_observed=history.observed,
            frames_missing=max(history.missing_count(i) for i in newly),
            window_seconds=round(history.observed / self.fps, 2),
            rule=self.rule,
            frame_index=frame_index,
        )

    def drop(self, track_id: int) -> None:
        """Forget a track that has left the scene."""
        self.tracks.pop(track_id, None)

    def state(self) -> dict[int, dict]:
        """Snapshot for the console / debugging."""
        return {
            tid: {
                "observed": h.observed,
                "confirmed": sorted(h.confirmed_items),
                "pending": {
                    item: f"{h.missing_count(item)}/{self.n}"
                    for item in sorted(h.candidate_items())
                    if item not in h.confirmed_items
                },
            }
            for tid, h in self.tracks.items()
        }


# ---------------------------------------------------------------------------
# The video loop. Everything above this line is testable without a model.
# ---------------------------------------------------------------------------

def track_video(video_path, weights, zone_map, camera_id: str, policy,
                n: int = DEFAULT_N, m: int = DEFAULT_M, stride: int = 1,
                imgsz: int | None = None, tracker: str = "bytetrack.yaml",
                verbose: bool = True):
    """Run ByteTrack over a video and yield (Confirmation, assessment, frame_index).

    Yields only confirmed findings -- the caller turns them into ViolationEvents.
    """
    import cv2
    from ultralytics import YOLO

    from ppe_compliance import assess_detections

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    confirmer = TemporalConfirmer(n=n, m=m, fps=fps / max(stride, 1))

    kwargs = {
        "source": str(video_path),
        "persist": True,
        "tracker": tracker,
        "conf": min(policy.person_conf_min, policy.ppe_conf_min),
        "stream": True,
        "verbose": False,
    }
    if imgsz:
        kwargs["imgsz"] = imgsz

    for frame_index, result in enumerate(model.track(**kwargs)):
        if stride > 1 and frame_index % stride:
            continue
        if result.boxes is None or result.boxes.id is None:
            continue                          # nothing tracked in this frame

        img_h, img_w = result.orig_shape
        people, _ = assess_detections(result.boxes, model.names, img_w, img_h,
                                      policy, zone_map, camera_id)

        # Map each assessment back to its ByteTrack id. assess_detections sorts people
        # by height, so match on bbox rather than position in the list.
        ids = result.boxes.id.int().tolist()
        boxes = [tuple(int(v) for v in b) for b in result.boxes.xyxy.tolist()]
        by_box = dict(zip(boxes, ids))

        for person in people:
            track_id = by_box.get(tuple(person.bbox))
            if track_id is None:              # box was rounded differently; nearest match
                track_id = _nearest_id(person.bbox, boxes, ids)
            if track_id is None:
                continue

            for confirmation in confirmer.ingest(
                frame_index, track_id, person.status,
                tuple(person.missing), tuple(person.indeterminate),
            ):
                if verbose:
                    print(f"  frame {frame_index:>5}  track {track_id:>3}  "
                          f"CONFIRMED missing {'+'.join(confirmation.items)}  "
                          f"({confirmation.frames_missing}/{confirmation.frames_observed} "
                          f"frames, {confirmation.window_seconds}s)")
                yield confirmation, person, frame_index


def _nearest_id(bbox, boxes, ids, tolerance: int = 4):
    """Fall back to the closest box when integer rounding breaks an exact match."""
    best, best_d = None, tolerance * 4 + 1
    for b, i in zip(boxes, ids):
        d = sum(abs(a - c) for a, c in zip(bbox, b))
        if d < best_d:
            best, best_d = i, d
    return best
