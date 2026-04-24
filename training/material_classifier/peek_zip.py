"""Peek into the RSCD zip without extracting to learn the directory layout
and confirm the actual class folder names before we run curation.

Usage:
    python peek_zip.py path/to/rscd.zip
"""
from __future__ import annotations

import sys
import zipfile
from collections import Counter
from pathlib import Path


def peek(zip_path: Path, max_examples_per_top: int = 5) -> None:
    if not zip_path.exists():
        print(f"missing: {zip_path}", file=sys.stderr)
        sys.exit(1)

    print(f"opening {zip_path} ({zip_path.stat().st_size/1e9:.2f} GB)")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    print(f"total entries: {len(names):,}")

    # Show the first few entries verbatim
    print("\nfirst 20 entries:")
    for n in names[:20]:
        print(f"  {n}")

    # Count top-level dirs (one level after a possible single root prefix)
    top1 = Counter()
    top2 = Counter()
    for n in names:
        parts = [p for p in n.split("/") if p]
        if len(parts) >= 1:
            top1[parts[0]] += 1
        if len(parts) >= 2:
            top2[parts[1]] += 1

    print(f"\ntop-level (depth 1) dir counts (top 30):")
    for k, v in top1.most_common(30):
        print(f"  {v:>10,}  {k}")

    print(f"\ndepth-2 dir counts (top 60):")
    for k, v in top2.most_common(60):
        print(f"  {v:>10,}  {k}")

    # Sample full paths by depth
    print("\nsample paths by depth:")
    by_depth: dict[int, list[str]] = {}
    for n in names:
        d = n.count("/")
        by_depth.setdefault(d, []).append(n)
    for d in sorted(by_depth):
        print(f"  depth {d}: {len(by_depth[d]):,} entries; e.g.")
        for ex in by_depth[d][:max_examples_per_top]:
            print(f"     {ex}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: peek_zip.py <rscd.zip>")
        sys.exit(2)
    peek(Path(sys.argv[1]))
