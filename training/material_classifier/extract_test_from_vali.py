"""Patch script: extract our test split from RSCD's flat vali_20k/.

train/ is already populated by curate.py. vali_20k/ uses a flat layout where
labels are embedded in filenames like '202201252342483-dry-asphalt-smooth.jpg'.
This script handles that layout and writes data/curated/test/ + test_labels.csv.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

MATERIALS = ("asphalt", "concrete", "mud", "gravel")
UNEVENNESS = ("smooth", "slight", "severe")
FRICTIONS = ("dry", "wet", "water", "ice", "snow")
KEEP_FRICTION = "dry"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_filename_labels(stem: str) -> tuple[str, str, str | None] | None:
    """Extract (friction, material, unevenness) from a filename stem like
    '202201252342483-dry-asphalt-smooth' or '...-dry-mud'."""
    parts = stem.lower().split("-")
    friction = material = None
    unevenness: str | None = None
    for p in parts:
        if p in FRICTIONS:
            friction = p
        elif p in MATERIALS:
            material = p
        elif p in UNEVENNESS:
            unevenness = p
    if friction != KEEP_FRICTION:
        return None
    if material not in MATERIALS:
        return None
    if material in ("asphalt", "concrete") and unevenness not in UNEVENNESS:
        return None
    return friction, material, unevenness


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-material", type=int, default=1250)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f"opening {args.zip}")
    with zipfile.ZipFile(args.zip, "r") as zf:
        # Group flat vali_20k files by class
        buckets: dict[tuple[str, str | None], list[str]] = defaultdict(list)
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if Path(name).suffix.lower() not in IMG_EXTS:
                continue
            parts = [p for p in name.split("/") if p]
            if len(parts) != 2:  # we want depth-1 entries: vali_20k/<file>
                continue
            if parts[0] != "vali_20k":
                continue
            parsed = parse_filename_labels(Path(parts[1]).stem)
            if parsed is None:
                continue
            _f, material, unevenness = parsed
            buckets[(material, unevenness)].append(name)

        print("\nvali_20k bucket sizes (material, unevenness):")
        for (m, u), files in sorted(buckets.items()):
            print(f"  {m:9s} {(u or '-'):8s} {len(files):>8,}")

        # Sample per material
        chosen: list[tuple[str, str, str]] = []
        for material in MATERIALS:
            if material in ("asphalt", "concrete"):
                base = args.per_material // 3
                rem = args.per_material - base * 3
                targets = [(u, base) for u in UNEVENNESS]
                for i in range(rem):
                    targets[i] = (targets[i][0], targets[i][1] + 1)
            else:
                targets = [(None, args.per_material)]
            for unevenness, want in targets:
                pool = list(buckets.get((material, unevenness), []))
                if len(pool) < want:
                    print(f"  WARNING: ({material}, {unevenness}) has {len(pool)}; requested {want}")
                    want = len(pool)
                rng.shuffle(pool)
                for fn in pool[:want]:
                    chosen.append((fn, material, unevenness or ""))

        rng.shuffle(chosen)
        print(f"\nchosen for test: {len(chosen):,}")

        # Extract
        split_dir = args.out / "test"
        split_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.out / "test_labels.csv"
        print("extracting test...")
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["filename", "material", "unevenness"])
            for i, (src, material, unevenness) in enumerate(chosen):
                tag = unevenness or "na"
                target_name = f"{material}_{tag}_{i:06d}{Path(src).suffix.lower()}"
                target_path = split_dir / target_name
                with zf.open(src) as zin, open(target_path, "wb") as fout:
                    fout.write(zin.read())
                w.writerow([target_name, material, unevenness])
                if (i + 1) % 500 == 0:
                    print(f"    {i+1:,}/{len(chosen):,}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
