"""Phase 2 smoke test: run the single-image pipeline end-to-end and print measurements.

Usage:
    cd server
    python scripts/smoke_test_phase2.py [image_path]

If no path is given, uses ../legacy/InfraSight/test8.png.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

SERVER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_ROOT))

from app.models.crack_classifier import CrackClassifier
from app.models.depth_metric import MetricDepthEstimator
from app.models.pothole_segmenter import PotholeSegmenter
from app.physics.ground_plane import fit_ground_plane
from app.physics.intrinsics import backproject, compute_K
from app.physics.repair_advisor import RepairAdvisor, format_currency
from app.physics.severity import SeverityClassifier
from app.physics.volumetric import measure_pothole
from app.utils.config import load_config, resolve_path
from app.utils.logger import get_logger

log = get_logger("smoke_phase2")


def main(image_path: Path) -> int:
    cfg = load_config()
    if not image_path.exists():
        log.error(f"Image not found: {image_path}")
        return 1

    log.info(f"Loading image: {image_path}")
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        log.error("cv2.imread returned None")
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    log.info(f"Image size: {W}x{H}")

    log.info("Loading pothole segmenter...")
    seg_cfg = cfg["models"]["pothole_segmenter"]
    segmenter = PotholeSegmenter(
        weights_path=resolve_path(seg_cfg["weights"]),
        conf_threshold=seg_cfg["conf_threshold"],
    )

    log.info("Loading crack classifier...")
    crk_cfg = cfg["models"]["crack_classifier"]
    cracks = CrackClassifier(
        weights_path=resolve_path(crk_cfg["weights"]),
        conf_threshold=crk_cfg["conf_threshold"],
        ignore_classes=crk_cfg.get("ignore_classes", []),
    )

    log.info("Loading metric depth model (may download weights on first run)...")
    depth_cfg = cfg["models"]["depth"]
    depth = MetricDepthEstimator(model_name=depth_cfg["model_name"], device=depth_cfg["device"])

    # ---------- Inference ----------
    t0 = time.time()
    log.info("Segmenting potholes...")
    pothole_dets = segmenter.detect_potholes(rgb)
    log.info(f"  -> {len(pothole_dets)} pothole(s) in {time.time()-t0:.2f}s")

    t0 = time.time()
    log.info("Classifying cracks...")
    crack_dets = cracks.detect(rgb)
    log.info(f"  -> {len(crack_dets)} crack(s) in {time.time()-t0:.2f}s")
    for c in crack_dets:
        log.info(f"     {c.class_name} conf={c.confidence:.2f} bbox={c.bbox}")

    t0 = time.time()
    log.info("Estimating metric depth...")
    depth_m = depth.predict(rgb)
    log.info(
        f"  -> depth map {depth_m.shape}, range [{np.nanmin(depth_m):.2f}, {np.nanmax(depth_m):.2f}] m"
        f" in {time.time()-t0:.2f}s"
    )

    # ---------- Geometry ----------
    log.info("Computing camera intrinsics K...")
    intrinsics_cfg = cfg["intrinsics"]
    device_key = intrinsics_cfg["fallback_device"]
    device_cfg = intrinsics_cfg["devices"][device_key]
    K = compute_K((H, W), image_path=image_path, device_cfg=device_cfg)
    log.info(f"  K:\n{K}")

    log.info("Back-projecting to 3D point cloud...")
    points = backproject(depth_m, K)

    # Exclude all pothole masks from ground plane fit
    exclude = np.zeros((H, W), dtype=bool)
    for d in pothole_dets:
        exclude |= d.mask.astype(bool)

    log.info("Fitting ground plane via RANSAC...")
    gp_cfg = cfg["pipeline"]["ground_plane"]
    plane = fit_ground_plane(
        points,
        exclude_mask=exclude,
        iterations=gp_cfg["ransac_iterations"],
        threshold_m=gp_cfg["ransac_threshold_m"],
        min_inlier_ratio=gp_cfg["min_inliers"],
        max_depth_m=gp_cfg.get("max_depth_m"),
        debug=True,
    )
    if plane is None:
        log.error("Ground plane fit failed")
        return 1
    log.info(
        f"  plane normal={plane.normal.tolist()}, d={plane.d:.3f}, "
        f"inliers={plane.inlier_ratio:.1%}, est camera height={plane.camera_height_m*100:.1f} cm"
    )

    # ---------- Measurements per pothole ----------
    sev = SeverityClassifier(cfg["severity"])
    advisor = RepairAdvisor(cfg["repair"])

    print()
    print("=" * 78)
    print(f"POTHOLES: {len(pothole_dets)}")
    print("=" * 78)

    totals = {"area_cm2": 0.0, "volume_cm3": 0.0, "material_kg": 0.0, "cost": 0.0}

    for i, d in enumerate(pothole_dets, 1):
        print(f"\n--- Pothole #{i}  (conf={d.confidence:.2f}  bbox={d.bbox}) ---")
        m = measure_pothole(d.mask, points, plane)
        if m is None:
            print("  measurement failed (mask had no valid 3D points)")
            continue
        print(
            f"  Area:        {m.area_cm2:8.1f} cm^2"
            f"\n  Depth (avg): {m.avg_depth_cm:8.2f} cm"
            f"\n  Depth (max): {m.max_depth_cm:8.2f} cm"
            f"\n  Volume:      {m.volume_cm3:8.1f} cm^3"
        )
        s = sev.classify(m.avg_depth_cm, m.area_cm2, m.volume_cm3)
        print(f"  Severity:    {s.level} (score {s.score}/10) -- {s.priority}")

        r = advisor.recommend(
            volume_cm3=m.volume_cm3,
            depth_cm=m.avg_depth_cm,
            area_cm2=m.area_cm2,
            severity_level=s.level,
        )
        print(
            f"  Repair:      {r.method} using {r.material_name}"
            f"\n  Material:    {r.material_kg:.2f} kg"
            f"\n  Cost:        {format_currency(r.total_cost, r.currency)}"
            f" (material {format_currency(r.material_cost, r.currency)}"
            f" + labor {format_currency(r.labor_cost, r.currency)})"
            f"\n  Durability:  {r.durability_months} months"
        )
        if r.notes:
            print(f"  Notes:       {r.notes}")

        totals["area_cm2"] += m.area_cm2
        totals["volume_cm3"] += m.volume_cm3
        totals["material_kg"] += r.material_kg
        totals["cost"] += r.total_cost

    if len(pothole_dets) > 0:
        print()
        print("=" * 78)
        print("SUMMARY")
        print("=" * 78)
        print(f"  Total potholes:     {len(pothole_dets)}")
        print(f"  Total damaged area: {totals['area_cm2']:.1f} cm^2")
        print(f"  Total volume:       {totals['volume_cm3']:.1f} cm^3")
        print(f"  Total material:     {totals['material_kg']:.2f} kg")
        print(f"  Total repair cost:  {format_currency(totals['cost'], cfg['repair']['currency'])}")
        print()

    return 0


if __name__ == "__main__":
    default_image = SERVER_ROOT.parent / "legacy" / "InfraSight" / "test8.png"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_image
    sys.exit(main(path))
