"""
Re-split a YOLO dataset so every class is represented in train, val AND test.

Why this exists: if a class is absent from your val/test labels, Ultralytics simply
does not score it -- `metrics.box.ap_class_index` only covers classes present in the
evaluation set, so the mean is taken over the remaining classes. A dataset missing
its hardest class from val/test will report a *higher* mAP than the same dataset with
that class included. The score is not good; it is incomplete.

The fix is not to re-download. Pool every image across the existing splits and
re-partition with multi-label stratification, so each split gets a proportional share
of every class's instances.

Usage:
    python restratify.py /workspace/datasets/ppe-presence \
        --out /workspace/datasets/ppe-strat --ratios 0.70 0.15 0.15

    # then train against /workspace/datasets/ppe-strat/data.yaml
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan(root: Path) -> tuple[dict[Path, Counter], list[str]]:
    """Pool every image across existing splits -> {image path: Counter(class_id)}."""
    cfg = yaml.safe_load((root / "data.yaml").read_text())
    names = cfg["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)

    pooled: dict[Path, Counter] = {}
    for split in SPLITS:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            counts = Counter()
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                for ln in lbl.read_text().splitlines():
                    if ln.strip():
                        counts[int(ln.split()[0])] += 1
            pooled[img.resolve()] = counts
    return pooled, names


def stratify(pooled: dict[Path, Counter], ratios: tuple[float, float, float], seed: int = 0):
    """Greedy iterative stratification: place rare-class images first, always into
    whichever split is furthest below its quota for that rare class."""
    total = Counter()
    for c in pooled.values():
        total.update(c)

    rarity = dict(total)
    n_imgs = len(pooled)

    want_cls = {s: {k: v * r for k, v in total.items()} for s, r in zip(SPLITS, ratios)}
    want_img = {s: n_imgs * r for s, r in zip(SPLITS, ratios)}

    assigned: dict[str, list[Path]] = {s: [] for s in SPLITS}
    have_cls: dict[str, Counter] = {s: Counter() for s in SPLITS}

    rng = random.Random(seed)
    items = list(pooled.items())
    rng.shuffle(items)                       # break ties without positional bias
    # rarest class first, then most-annotated images -- hardest to place go first
    items.sort(key=lambda it: (min((rarity[k] for k in it[1]), default=10**9),
                               -sum(it[1].values())))

    for path, counts in items:
        if counts:
            rare = min(counts, key=lambda k: rarity[k])
            best = max(SPLITS, key=lambda s: (want_cls[s][rare] - have_cls[s][rare],
                                              want_img[s] - len(assigned[s])))
        else:                                # background image: balance by image count
            best = max(SPLITS, key=lambda s: want_img[s] - len(assigned[s]))
        assigned[best].append(path)
        have_cls[best].update(counts)

    return assigned, have_cls, total


def materialise(src_root: Path, out: Path, assigned, names: list[str], copy: bool):
    if out.exists():
        shutil.rmtree(out)

    for split, paths in assigned.items():
        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in paths:
            dst = img_dir / img.name
            if copy:
                shutil.copy2(img, dst)
            else:
                dst.symlink_to(img)

            # find the label wherever it currently lives
            for s in SPLITS:
                cand = src_root / "labels" / s / f"{img.stem}.txt"
                if cand.exists():
                    shutil.copy2(cand, lbl_dir / f"{img.stem}.txt")
                    break
            else:
                (lbl_dir / f"{img.stem}.txt").write_text("")

    cfg = {
        "path": str(out),
        **{s: f"images/{s}" for s in SPLITS},
        "names": {i: n for i, n in enumerate(names)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def report(names: list[str], have_cls, total: Counter, assigned) -> bool:
    w = max(len(n) for n in names)
    print(f"\n{'class':<{w}} {'total':>7} {'train':>7} {'val':>7} {'test':>7}   status")
    print("-" * (w + 45))

    ok = True
    for i, name in enumerate(names):
        row = [have_cls[s][i] for s in SPLITS]
        missing = [s for s, v in zip(SPLITS, row) if v == 0]
        status = "OK" if not missing else f"ABSENT from {', '.join(missing)}"
        ok &= not missing
        print(f"{name:<{w}} {total[i]:>7} {row[0]:>7} {row[1]:>7} {row[2]:>7}   {status}")

    print("-" * (w + 45))
    print(f"{'images':<{w}} {sum(len(v) for v in assigned.values()):>7} "
          f"{len(assigned['train']):>7} {len(assigned['val']):>7} {len(assigned['test']):>7}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset root containing data.yaml, images/, labels/")
    ap.add_argument("--out", required=True, help="where to write the re-split dataset")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking (use if the source may move)")
    args = ap.parse_args()

    ratios = tuple(r / sum(args.ratios) for r in args.ratios)
    root, out = Path(args.root), Path(args.out)

    pooled, names = scan(root)
    print(f"pooled {len(pooled)} images from {root}")

    assigned, have_cls, total = stratify(pooled, ratios, args.seed)
    materialise(root, out, assigned, names, args.copy)
    ok = report(names, have_cls, total, assigned)

    print(f"\nwrote {out / 'data.yaml'}")
    if not ok:
        print("\nWARNING: a class is still absent from a split -- it has too few instances "
              "to divide three ways. Either merge it into another class, drop it, or "
              "collect more of it. Do NOT report mAP as if that class were measured.")


if __name__ == "__main__":
    main()
