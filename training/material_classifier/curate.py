"""Stream-extract a balanced curated subset from the RSCD zip.

Layout inside the zip (verified):
    train/<friction>_<material>[_<unevenness>]/<image>.jpg     (~959k)
    vali_20k/<friction>_<material>[_<unevenness>]/<image>.jpg  (~20k)
    test_50k/<image>.jpg  (flat, labels in filename — ignored)

What this script does:
  - Filter to friction == 'dry'  (skip wet/water/snow/ice — irrelevant for our app)
  - Materials kept: asphalt, concrete, mud, gravel
  - Sample N per material from `train/`  → split into our train/val (80/20)
  - Sample M per material from `vali_20k/` → our held-out test set
  - Stream-extract only the chosen files (no full unpack)

Output structure:
    <out>/
      train/<flat_image>.jpg
      val/<flat_image>.jpg
      test/<flat_image>.jpg
      train_labels.csv  (columns: filename, material, unevenness)
      val_labels.csv
      test_labels.csv
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
KEEP_FRICTION = "dry"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_folder(folder_name: str) -> tuple[str, str, str | None] | None:
    """Map e.g. 'dry_asphalt_smooth' -> (friction, material, unevenness_or_None)."""
    s = folder_name.lower()
    parts = s.split("_")
    if len(parts) < 2:
        return None
    friction = parts[0]
    if friction != KEEP_FRICTION:
        return None
    material = parts[1]
    if material not in MATERIALS:
        return None
    unevenness: str | None = None
    if len(parts) >= 3 and parts[2] in UNEVENNESS:
        unevenness = parts[2]
    elif material in ("asphalt", "concrete"):
        # asphalt/concrete need an unevenness suffix; if missing, skip
        return None
    return friction, material, unevenness


def collect(zf: zipfile.ZipFile) -> dict[str, dict[tuple[str, str | None], list[str]]]:
    """Returns: {source: {(material, unevenness): [filenames]}}."""
    out: dict[str, dict[tuple[str, str | None], list[str]]] = {
        "train": defaultdict(list),
        "vali": defaultdict(list),
    }
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        if Path(name).suffix.lower() not in IMG_EXTS:
            continue
        parts = [p for p in name.split("/") if p]
        if len(parts) < 2:
            continue
        top = parts[0]
        if top == "train":
            source = "train"
        elif top == "vali_20k":
            source = "vali"
        else:
            continue
        parsed = parse_folder(parts[1])
        if parsed is None:
            continue
        _friction, material, unevenness = parsed
        out[source][(material, unevenness)].append(name)
    return out


def even_split(per_material: int, has_unevenness_breakdown: bool) -> list[tuple[str | None, int]]:
    """Return list of (unevenness_label_or_None, count) for one material's sampling target."""
    if not has_unevenness_breakdown:
        return [(None, per_material)]
    base = per_material // 3
    rem = per_material - base * 3
    targets = [(u, base) for u in UNEVENNESS]
    for i in range(rem):
        targets[i] = (targets[i][0], targets[i][1] + 1)
    return targets


def sample(
    pool_by_class: dict[tuple[str, str | None], list[str]],
    rng: random.Random,
    per_material: int,
) -> dict[str, list[tuple[str, str, str]]]:
    """For each material, sample images per the unevenness breakdown.
    Returns: {material: [(filename_in_zip, material, unevenness_or_empty)]}.
    """
    out: dict[str, list[tuple[str, str, str]]] = {m: [] for m in MATERIALS}
    for material in MATERIALS:
        breakdown = material in ("asphalt", "concrete")
        targets = even_split(per_material, breakdown)
        for unevenness, want in targets:
            pool = list(pool_by_class.get((material, unevenness), []))
            if len(pool) < want:
                print(
                    f"  WARNING: only {len(pool)} files for ({material}, {unevenness}); requested {want}",
                    file=sys.stderr,
                )
                want = len(pool)
            rng.shuffle(pool)
            chosen = pool[:want]
            u_str = unevenness or ""
            for fn in chosen:
                out[material].append((fn, material, u_str))
    return out


def write_split(zf, items, split_dir, csv_path, *, status_every=500):
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["filename", "material", "unevenness"])
        for i, (src, material, unevenness) in enumerate(items):
            tag = unevenness or "na"
            target_name = f"{material}_{tag}_{i:06d}{Path(src).suffix.lower()}"
            target_path = split_dir / target_name
            with zf.open(src) as zin, open(target_path, "wb") as fout:
                fout.write(zin.read())
            w.writerow([target_name, material, unevenness])
            if (i + 1) % status_every == 0:
                print(f"    {i+1:,}/{len(items):,}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-material", type=int, default=10000, help="train+val images per material")
    ap.add_argument("--test-per-material", type=int, default=1250, help="test images per material")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out: Path = args.out

    print(f"opening {args.zip}")
    with zipfile.ZipFile(args.zip, "r") as zf:
        print("indexing entries...")
        sources = collect(zf)
        print("\nbucket sizes (source / material / unevenness / count):")
        for src in ("train", "vali"):
            for (m, u), files in sorted(sources[src].items()):
                print(f"  {src:5s}  {m:8s}  {(u or '-'):8s}  {len(files):>8,}")

        # Sample
        print("\nsampling train+val (per_material =", args.per_material, ")")
        train_pool = sample(sources["train"], rng, args.per_material)
        print("sampling test (per_material =", args.test_per_material, ")")
        test_pool = sample(sources["vali"], rng, args.test_per_material)

        # Split each material's train_pool into train/val 80/20, mix across materials, then shuffle
        train_items = []
        val_items = []
        cut_frac = 1.0 - args.val_frac
        for material, items in train_pool.items():
            rng.shuffle(items)
            cut = int(len(items) * cut_frac)
            train_items.extend(items[:cut])
            val_items.extend(items[cut:])
        rng.shuffle(train_items)
        rng.shuffle(val_items)

        test_items: list[tuple[str, str, str]] = []
        for material, items in test_pool.items():
            test_items.extend(items)
        rng.shuffle(test_items)

        print(f"\nfinal counts: train={len(train_items):,}  val={len(val_items):,}  test={len(test_items):,}")

        print("\nextracting train...")
        write_split(zf, train_items, out / "train", out / "train_labels.csv")
        print("extracting val...")
        write_split(zf, val_items, out / "val", out / "val_labels.csv")
        print("extracting test...")
        write_split(zf, test_items, out / "test", out / "test_labels.csv")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
