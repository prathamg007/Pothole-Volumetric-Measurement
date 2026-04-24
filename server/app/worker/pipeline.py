"""End-to-end video pipeline: frame extract -> per-frame inference -> track -> annotate -> stitch."""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.models.crack_classifier import CrackClassifier
from app.models.depth_metric import MetricDepthEstimator
from app.models.pothole_segmenter import PotholeSegmenter
from app.physics.ground_plane import Plane, fit_ground_plane
from app.physics.intrinsics import backproject, compute_K
from app.physics.repair_advisor import RepairAdvisor
from app.physics.severity import SeverityClassifier
from app.physics.volumetric import measure_pothole
from app.utils.config import resolve_path
from app.utils.logger import get_logger
from app.utils.video_io import make_writer, probe_video
from app.worker.annotator import annotate_frame
from app.worker.tracker import PotholeTracker

log = get_logger("pipeline")


def process_video(
    input_video: Path,
    output_video: Path,
    cfg: dict,
    *,
    exif_reference_image: Optional[Path] = None,
    progress_every: int = 30,
) -> dict:
    """Run the full pipeline on a video.

    Returns a structured report dict with per-pothole measurements + summary.
    """
    info = probe_video(input_video)
    log.info(
        f"Video: {info.width}x{info.height} @ {info.fps:.1f} fps, "
        f"{info.frame_count} frames ({info.duration_s:.1f}s)"
    )

    seg_cfg = cfg["models"]["pothole_segmenter"]
    crk_cfg = cfg["models"]["crack_classifier"]
    depth_cfg = cfg["models"]["depth"]
    intr_cfg = cfg["intrinsics"]
    gp_cfg = cfg["pipeline"]["ground_plane"]
    stride = int(cfg["pipeline"].get("frame_stride", 1))

    log.info("Loading models...")
    t0 = time.time()
    segmenter = PotholeSegmenter(
        weights_path=resolve_path(seg_cfg["weights"]),
        conf_threshold=seg_cfg["conf_threshold"],
    )
    crack_clf = CrackClassifier(
        weights_path=resolve_path(crk_cfg["weights"]),
        conf_threshold=crk_cfg["conf_threshold"],
        ignore_classes=crk_cfg.get("ignore_classes", []),
    )
    depth_model = MetricDepthEstimator(model_name=depth_cfg["model_name"], device=depth_cfg["device"])
    log.info(f"Models loaded in {time.time()-t0:.1f}s")

    device_key = intr_cfg["fallback_device"]
    device_cfg = intr_cfg["devices"][device_key]
    K = compute_K((info.height, info.width), image_path=exif_reference_image, device_cfg=device_cfg)

    tracker = PotholeTracker(iou_threshold=0.3, max_gap_frames=int(info.fps))
    sev = SeverityClassifier(cfg["severity"])
    advisor = RepairAdvisor(cfg["repair"])

    cap = cv2.VideoCapture(str(input_video))
    writer = make_writer(output_video, info.fps, (info.width, info.height))

    # Cached per-stride state
    last_potholes_annot: list[dict] = []
    last_cracks_annot: list[dict] = []
    plane: Optional[Plane] = None
    plane_refit_every = max(int(info.fps), 5)  # refit ~once per second

    crack_counts: dict[str, int] = defaultdict(int)
    frame_idx = 0
    t_start = time.time()
    inference_time = 0.0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            t_inf = time.time()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            pothole_dets = segmenter.detect_potholes(frame_rgb)
            crack_dets = crack_clf.detect(frame_rgb)

            pot_with_meas: list = []
            if pothole_dets:
                depth_m = depth_model.predict(frame_rgb)
                points = backproject(depth_m, K)

                if plane is None or (frame_idx % plane_refit_every == 0):
                    exclude = np.zeros((info.height, info.width), dtype=bool)
                    for d in pothole_dets:
                        exclude |= d.mask.astype(bool)
                    new_plane = fit_ground_plane(
                        points,
                        exclude_mask=exclude,
                        iterations=gp_cfg["ransac_iterations"],
                        threshold_m=gp_cfg["ransac_threshold_m"],
                        min_inlier_ratio=gp_cfg["min_inliers"],
                        max_depth_m=gp_cfg.get("max_depth_m"),
                    )
                    if new_plane is not None:
                        plane = new_plane

                for det in pothole_dets:
                    m = measure_pothole(det.mask, points, plane) if plane is not None else None
                    pot_with_meas.append((det, m))
            else:
                for det in pothole_dets:
                    pot_with_meas.append((det, None))

            track_ids = tracker.update(frame_idx, pot_with_meas)

            last_potholes_annot = []
            for (det, m), tid in zip(pot_with_meas, track_ids):
                sev_res = sev.classify(m.avg_depth_cm, m.area_cm2, m.volume_cm3) if m is not None else None
                last_potholes_annot.append(
                    {
                        "track_id": tid,
                        "bbox": det.bbox,
                        "mask": det.mask,
                        "measurement": m,
                        "severity_level": sev_res.level if sev_res else None,
                        "severity_score": sev_res.score if sev_res else None,
                    }
                )
            last_cracks_annot = [
                {"class_name": c.class_name, "confidence": c.confidence, "bbox": c.bbox}
                for c in crack_dets
            ]
            for c in crack_dets:
                crack_counts[c.class_name] += 1

            inference_time += time.time() - t_inf

        running = {"potholes": len(tracker.tracks), "cracks": sum(crack_counts.values())}
        annotated = annotate_frame(
            frame_bgr,
            potholes=last_potholes_annot,
            cracks=last_cracks_annot,
            frame_idx=frame_idx,
            total_frames=info.frame_count,
            running_totals=running,
        )
        writer.write(annotated)

        frame_idx += 1
        if progress_every and frame_idx % progress_every == 0:
            elapsed = time.time() - t_start
            log.info(
                f"  frame {frame_idx}/{info.frame_count}  "
                f"active_tracks={len([t for t in tracker.tracks if frame_idx - t.last_frame <= int(info.fps)])}  "
                f"wall={elapsed:.1f}s  infer={inference_time:.1f}s"
            )

    cap.release()
    writer.release()

    # Aggregate per-track
    tracks_summary = tracker.finalize(min_observations=2, min_valid_measurements=1)
    potholes_report: list[dict] = []
    totals = {"area_cm2": 0.0, "volume_cm3": 0.0, "material_kg": 0.0, "cost": 0.0}

    for t in tracks_summary:
        sev_res = sev.classify(t["avg_depth_cm"], t["area_cm2"], t["volume_cm3"])
        rec = advisor.recommend(
            volume_cm3=t["volume_cm3"],
            depth_cm=t["avg_depth_cm"],
            area_cm2=t["area_cm2"],
            severity_level=sev_res.level,
        )
        potholes_report.append(
            {
                "track_id": t["track_id"],
                "first_frame": t["first_frame"],
                "last_frame": t["last_frame"],
                "first_time_s": t["first_frame"] / info.fps if info.fps > 0 else 0,
                "observations": t["observations"],
                "valid_measurements": t["valid_measurements"],
                "confidence": round(t["confidence"], 2),
                "area_cm2": round(t["area_cm2"], 1),
                "avg_depth_cm": round(t["avg_depth_cm"], 2),
                "max_depth_cm": round(t["max_depth_cm"], 2),
                "volume_cm3": round(t["volume_cm3"], 1),
                "severity_level": sev_res.level,
                "severity_score": sev_res.score,
                "repair_method": rec.method,
                "material_name": rec.material_name,
                "material_kg": rec.material_kg,
                "repair_cost": rec.total_cost,
                "currency": rec.currency,
                "durability_months": rec.durability_months,
            }
        )
        totals["area_cm2"] += t["area_cm2"]
        totals["volume_cm3"] += t["volume_cm3"]
        totals["material_kg"] += rec.material_kg
        totals["cost"] += rec.total_cost

    elapsed = time.time() - t_start
    report = {
        "video": {
            "fps": info.fps,
            "width": info.width,
            "height": info.height,
            "frames": frame_idx,
            "duration_s": round(info.duration_s, 2),
        },
        "processing": {
            "wall_s": round(elapsed, 2),
            "inference_s": round(inference_time, 2),
            "frame_stride": stride,
        },
        "cracks": dict(crack_counts),
        "potholes": potholes_report,
        "summary": {
            "num_potholes": len(potholes_report),
            "total_area_cm2": round(totals["area_cm2"], 1),
            "total_volume_cm3": round(totals["volume_cm3"], 1),
            "total_material_kg": round(totals["material_kg"], 2),
            "total_cost": round(totals["cost"], 2),
            "currency": cfg["repair"]["currency"],
            "total_cracks_detected": sum(crack_counts.values()),
        },
    }
    log.info(f"Done. {frame_idx} frames in {elapsed:.1f}s (inference {inference_time:.1f}s)")
    return report
