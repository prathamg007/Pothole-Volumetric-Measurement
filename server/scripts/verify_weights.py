"""Phase 1 verification: load every model weight and print its metadata.

Run:
    python scripts/verify_weights.py
"""
from pathlib import Path
import sys

SERVER_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = SERVER_ROOT / "weights"

EXPECTED = {
    "pothole_seg.pt": {"task": "segment", "expect_classes": {"Manhole", "Pothole", "Unmarked Bump"}},
    "crack_typology.pt": {"task": "detect", "expect_classes": {"Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Potholes"}},
}


def main() -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. pip install -r requirements.txt first.", file=sys.stderr)
        return 1

    all_ok = True
    for name, spec in EXPECTED.items():
        path = WEIGHTS / name
        print(f"--- {name}")
        if not path.exists():
            print(f"  MISSING: {path}")
            all_ok = False
            continue
        try:
            model = YOLO(str(path))
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            all_ok = False
            continue

        task = model.task
        names = set(model.names.values()) if hasattr(model, "names") else set()
        task_ok = task == spec["task"]
        classes_ok = names == spec["expect_classes"]
        print(f"  task: {task} ({'ok' if task_ok else 'MISMATCH expected ' + spec['task']})")
        print(f"  classes: {sorted(names)}")
        if not classes_ok:
            missing = spec["expect_classes"] - names
            extra = names - spec["expect_classes"]
            if missing:
                print(f"  MISSING CLASSES: {missing}")
            if extra:
                print(f"  UNEXPECTED CLASSES: {extra}")
            all_ok = False
    print()
    print("ALL OK" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
