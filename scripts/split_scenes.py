"""
Split a PTZ patrol recording into static-view segments.

A PTZ camera on a patrol tour is not one camera. It holds a preset for a few seconds,
pans to the next, and holds again -- and every pan invalidates two things the system
depends on:

  zones      a polygon drawn on one view describes nothing on the next
  tracking   a ByteTrack id cannot survive a view change. Worse than losing identity:
             afterwards the tracker may bind a NEW person to an OLD id, letting one
             worker's history contaminate another's. In a system that escalates on
             repeat offences that is a correctness bug, not a nuisance.

So each dwell has to be treated as its own camera, with its own zones and its own
tracking session. This script finds the dwells and exports them.

    python scripts/split_scenes.py data/footage/clip_d_cctv_25fps.mp4

Writes to data/footage/scenes/<clip>/:
    scene_00_t000-013.mp4      the segment, ready for perception/pipeline.py
    scene_00_t000-013.jpg      a representative still, to draw zones on
    scenes.json                index with durations and suggested camera ids

Detection works by finding STILLNESS, not cuts. A patrol camera that jump-cuts would
show spikes in frame difference, but this one pans continuously between presets, so
there is no spike to find -- instead we look for sustained runs where the mean absolute
frame difference stays low. Those runs are the dwells, and only a dwell can carry zones.

A walking worker barely moves the mean (they occupy a small share of the frame), while a
pan moves every pixel. The default threshold of 3.0 sits between the two.

Expect to discard most of a patrol recording. That is the honest result: if the camera
holds a position for four seconds out of every thirteen, only those four seconds can be
used for zone-based compliance, and the rest is unusable no matter how good the detector.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

# Mean absolute frame difference below which the camera counts as holding still.
# A walking worker sits under this; a pan sits far above it.
DEFAULT_STILL = 3.0
DEFAULT_MIN_SECONDS = 3.0
# How much the view may drift end-to-end and still be called static. Zones drawn on the
# first frame must still describe the last one.
MAX_DRIFT = 12.0


@dataclass
class Scene:
    index: int
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    camera_id: str
    video_path: str = ""
    still_path: str = ""

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame


def frame_diffs(cap, sample_width: int = 160) -> list[float]:
    """Mean absolute difference between consecutive frames, downscaled and gray."""
    import cv2
    import numpy as np

    diffs: list[float] = []
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        small = cv2.cvtColor(
            cv2.resize(frame, (sample_width, int(h * sample_width / w))),
            cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diffs.append(float(np.mean(cv2.absdiff(small, prev))))
        prev = small
    return diffs


def find_still_runs(diffs: list[float], fps: float, still: float,
                    min_seconds: float) -> list[tuple[int, int]]:
    """Runs of consecutive frames where the camera is holding a position."""
    min_frames = max(2, int(fps * min_seconds))
    runs: list[tuple[int, int]] = []
    start = None
    for i, d in enumerate(diffs):
        if d < still:
            if start is None:
                start = i
        elif start is not None:
            if i - start >= min_frames:
                runs.append((start, i))
            start = None
    if start is not None and len(diffs) - start >= min_frames:
        runs.append((start, len(diffs)))
    return runs


def build_scenes(runs: list[tuple[int, int]], fps: float, stem: str) -> list[Scene]:
    scenes: list[Scene] = []
    for i, (start, end) in enumerate(runs):
        scenes.append(Scene(
            index=i,
            start_frame=start,
            end_frame=end,
            start_seconds=round(start / fps, 2),
            end_seconds=round(end / fps, 2),
            duration_seconds=round((end - start) / fps, 2),
            camera_id=f"{stem}_view{i:02d}",
        ))
    return scenes


def drift(video: Path, scene: Scene, sample_width: int = 160) -> float:
    """How different the last frame is from the first. Low means zones stay valid."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video))

    def gray(index):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, f = cap.read()
        if not ok:
            return None
        h, w = f.shape[:2]
        return cv2.cvtColor(cv2.resize(f, (sample_width, int(h * sample_width / w))),
                            cv2.COLOR_BGR2GRAY)

    a, b = gray(scene.start_frame + 1), gray(max(scene.start_frame + 2, scene.end_frame - 2))
    cap.release()
    if a is None or b is None:
        return 999.0
    return float(np.mean(cv2.absdiff(a, b)))


def export(video: Path, scenes: list[Scene], out_dir: Path, fps: float,
           size: tuple[int, int], still_at: float = 0.5) -> None:
    """Write each segment as its own clip plus a still from the middle of it."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for scene in scenes:
        base = f"scene_{scene.index:02d}_t{int(scene.start_seconds):03d}-{int(scene.end_seconds):03d}"
        vid_path = out_dir / f"{base}.mp4"
        jpg_path = out_dir / f"{base}.jpg"

        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, scene.start_frame)
        writer = cv2.VideoWriter(str(vid_path), fourcc, fps, size)

        still_index = int(scene.frames * still_at)
        for i in range(scene.frames):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            if i == still_index:
                cv2.imwrite(str(jpg_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        writer.release()
        cap.release()

        scene.video_path = str(vid_path)
        scene.still_path = str(jpg_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", default=None,
                    help="default: data/footage/scenes/<clip stem>")
    ap.add_argument("--still", type=float, default=DEFAULT_STILL,
                    help=f"frame difference below which the camera is holding still "
                         f"(default {DEFAULT_STILL})")
    ap.add_argument("--min-seconds", type=float, default=DEFAULT_MIN_SECONDS,
                    help="discard dwells shorter than this (default 3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the segments without writing any files")
    args = ap.parse_args()

    import cv2

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"no such video: {video}")

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"{video.name}: {size[0]}x{size[1]}  {fps:.0f} fps  {total} frames  "
          f"{total / fps:.1f}s")
    print(f"scanning for dwells (still < {args.still}, >= {args.min_seconds}s) ...")

    diffs = frame_diffs(cap)
    cap.release()

    runs = find_still_runs(diffs, fps, args.still, args.min_seconds)
    stem = video.stem.split("_")[1] if "_" in video.stem else video.stem
    scenes = build_scenes(runs, fps, stem)

    held = sum(s.duration_seconds for s in scenes)
    print(f"{len(scenes)} dwells, {held:.1f}s of {total / fps:.1f}s usable "
          f"({held / (total / fps):.0%})\n")

    print(f"{'#':>2}  {'start':>7}  {'end':>7}  {'secs':>6}  {'frames':>7}  "
          f"{'drift':>6}  camera id")
    print("-" * 72)
    keep: list[Scene] = []
    for s in scenes:
        d = drift(video, s)
        flag = "" if d <= MAX_DRIFT else "  DRIFTS - skipped"
        print(f"{s.index:>2}  {s.start_seconds:>7.1f}  {s.end_seconds:>7.1f}  "
              f"{s.duration_seconds:>6.1f}  {s.frames:>7}  {d:>6.1f}  "
              f"{s.camera_id}{flag}")
        if d <= MAX_DRIFT:
            keep.append(s)
    scenes = keep

    if not scenes:
        sys.exit("\nno usable static views -- this camera never holds a position long "
                 "enough for zone-based compliance")

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    out_dir = Path(args.out) if args.out else ROOT / "data" / "footage" / "scenes" / video.stem
    print(f"\nexporting to {out_dir} ...")
    export(video, scenes, out_dir, fps, size)

    index = {
        "source": str(video),
        "fps": fps,
        "frame_size": list(size),
        "still_threshold": args.still,
        "scenes": [asdict(s) for s in scenes],
    }
    (out_dir / "scenes.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    longest = max(scenes, key=lambda s: s.duration_seconds)
    print(f"\nwrote {len(scenes)} segments + stills + scenes.json")
    print(f"\nEach segment is a separate camera: its own zones, its own tracking session.")
    print(f"Longest view is #{longest.index} at {longest.duration_seconds}s "
          f"({longest.frames} frames).")
    print(f"\nNext: pick a segment, draw zones on its still, then\n"
          f"  python -m perception.pipeline {longest.video_path} --camera {longest.camera_id}")


if __name__ == "__main__":
    main()
