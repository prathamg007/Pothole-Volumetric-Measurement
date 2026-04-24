"""Phase 3 smoke test: run the video pipeline on the bundled road_test.mp4.

Usage:
    cd server
    python scripts/smoke_test_phase3.py [input.mp4] [output.mp4]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_ROOT))

from app.utils.config import load_config
from app.utils.logger import get_logger
from app.worker.models_registry import ModelRegistry
from app.worker.pipeline import process_video

log = get_logger("smoke_phase3")


def main() -> int:
    default_in = SERVER_ROOT.parent / "legacy" / "InfraSight" / "road_test.mp4"
    default_out = SERVER_ROOT / "data" / "results" / "phase3_smoke.mp4"

    input_video = Path(sys.argv[1]) if len(sys.argv) > 1 else default_in
    output_video = Path(sys.argv[2]) if len(sys.argv) > 2 else default_out

    if not input_video.exists():
        log.error(f"Input not found: {input_video}")
        return 1

    cfg = load_config()
    models = ModelRegistry(cfg)
    models.load_all()
    report = process_video(input_video, output_video, cfg, models=models)

    # Print summary
    print()
    print("=" * 78)
    print("VIDEO REPORT")
    print("=" * 78)
    print(f"  Input:         {input_video}")
    print(f"  Output:        {output_video}")
    v = report["video"]
    p = report["processing"]
    print(
        f"  Video:         {v['width']}x{v['height']} @ {v['fps']:.1f} fps, "
        f"{v['frames']} frames ({v['duration_s']}s)"
    )
    print(f"  Wall time:     {p['wall_s']:.1f}s  (inference {p['inference_s']:.1f}s, stride {p['frame_stride']})")
    print()

    cracks = report["cracks"]
    if cracks:
        print("  Crack detections (across all frames):")
        for name, count in cracks.items():
            print(f"    {name:25s} {count}")
        print()

    potholes = report["potholes"]
    print(f"  Tracked potholes: {len(potholes)}")
    print("-" * 78)
    for pot in potholes:
        print(
            f"  #{pot['track_id']:2d}  "
            f"t={pot['first_time_s']:5.1f}s  "
            f"obs={pot['observations']:3d}  "
            f"area={pot['area_cm2']:6.0f} cm2  "
            f"d_avg={pot['avg_depth_cm']:5.2f} cm  "
            f"d_max={pot['max_depth_cm']:5.2f} cm  "
            f"vol={pot['volume_cm3']:6.0f} cm3  "
            f"{pot['severity_level']:8s}  "
            f"Rs {pot['repair_cost']:.0f}"
        )
    print("-" * 78)

    s = report["summary"]
    print(
        f"  SUMMARY:  {s['num_potholes']} potholes, "
        f"total area {s['total_area_cm2']:.0f} cm^2, "
        f"volume {s['total_volume_cm3']:.0f} cm^3, "
        f"material {s['total_material_kg']:.1f} kg, "
        f"cost Rs {s['total_cost']:.0f}"
    )
    print(f"  Cracks:   {s['total_cracks_detected']} total detections")
    print()

    # Also write full JSON next to the MP4
    json_path = output_video.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2))
    log.info(f"Report JSON: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
