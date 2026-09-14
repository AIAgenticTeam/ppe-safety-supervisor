"""
Does the detector actually work on this footage?

Before building zones around a clip, find out whether the model sees anything in it.
Samples frames from a video, runs detection, saves annotated images, and reports what
was found per class.

    python scripts/probe_footage.py data/footage/clip_b_24fps.mp4

This is a diagnostic, not the pipeline. It emits no events and applies no policy -- it
answers one question: how badly does the domain gap bite on this clip?

Read the output like this:
  - Person recall is the load-bearing number. Every assessment hangs off a person box,
    so if people are being missed, nothing downstream can work.
  - Compare mean confidence against the test-split numbers in the README. A large drop
    means this footage sits outside the training distribution.
  - Then LOOK at the annotated frames. The numbers cannot tell you whether a helmet box
    is on a head or on a bucket.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

# Measured on the held-out test split -- the bar this footage is compared against.
TEST_RECALL = {
    "helmet": 0.796, "vest": 0.816, "boots": 0.742,
    "goggles": 0.723, "gloves": 0.733, "Person": 0.867,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="path to an mp4/mov/avi")
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--frames", type=int, default=12,
                    help="how many evenly spaced frames to sample (default 12)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="detection threshold for this diagnostic (default 0.25)")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="omit to inherit the checkpoint's training resolution (960)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: data/footage/probe/<clip name>)")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"no such video: {video}")

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"no checkpoint at {weights}\nrun: python scripts/get_weights.py")

    out = Path(args.out) if args.out else ROOT / "data" / "footage" / "probe" / video.stem
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{video.name}: {w}x{h}  {fps:.0f} fps  {total} frames  {total / fps:.1f}s")
    print(f"sampling {args.frames} frames  |  conf >= {args.conf}\n")

    model = YOLO(str(weights))

    confs: dict[str, list[float]] = defaultdict(list)
    per_frame_people: list[int] = []
    empty_frames = 0

    indices = [int(total * i / args.frames) for i in range(args.frames)]
    for n, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        kwargs = {"conf": args.conf, "verbose": False}
        if args.imgsz:
            kwargs["imgsz"] = args.imgsz
        result = model.predict(frame, **kwargs)[0]

        found: dict[str, int] = defaultdict(int)
        for b in result.boxes:
            cls = model.names[int(b.cls)]
            found[cls] += 1
            confs[cls].append(float(b.conf))

        people = found.get("Person", 0)
        per_frame_people.append(people)
        if not found:
            empty_frames += 1

        cv2.imwrite(str(out / f"frame_{n:02d}_t{idx / fps:05.2f}s.jpg"), result.plot(),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

        summary = "  ".join(f"{k}x{v}" for k, v in sorted(found.items())) or "nothing detected"
        print(f"  t={idx / fps:5.2f}s  {summary}")

    cap.release()

    # ---- report -------------------------------------------------------
    print(f"\n{'class':<10} {'boxes':>6} {'mean conf':>10} {'min':>6} {'max':>6}")
    print("-" * 42)
    for cls in sorted(confs, key=lambda c: -len(confs[c])):
        v = confs[cls]
        print(f"{cls:<10} {len(v):>6} {sum(v) / len(v):>10.3f} {min(v):>6.3f} {max(v):>6.3f}")
    if not confs:
        print("  NOTHING DETECTED IN ANY FRAME")

    sampled = len(per_frame_people)
    print(f"\nframes sampled      : {sampled}")
    print(f"frames with nothing : {empty_frames}"
          f"{'   <-- the domain gap is biting hard' if empty_frames > sampled / 3 else ''}")
    if per_frame_people:
        print(f"people per frame    : min {min(per_frame_people)}, "
              f"max {max(per_frame_people)}, "
              f"mean {sum(per_frame_people) / sampled:.1f}")

    person_conf = confs.get("Person", [])
    if person_conf:
        mean = sum(person_conf) / len(person_conf)
        print(f"\nPerson mean confidence {mean:.3f}")
        if mean < 0.6:
            print("  Low. Person detection anchors every assessment -- if this is weak,")
            print("  nothing downstream can be trusted on this clip.")
    else:
        print("\nNO PEOPLE DETECTED. Nothing downstream can work on this footage.")

    print(f"\nannotated frames -> {out}")
    print("Open them. The numbers cannot tell you whether a helmet box is on a head")
    print("or on a bucket, and that distinction is the whole system.")


if __name__ == "__main__":
    main()
