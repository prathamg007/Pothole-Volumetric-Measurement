"""End-to-end video pipeline: frame extract -> per-frame inference -> track -> annotate -> stitch."""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.physics.ground_plane import Plane, fit_ground_plane
from app.physics.intrinsics import backproject, compute_K
from app.physics.repair_advisor import RepairAdvisor
from app.physics.severity import SeverityClassifier
from app.physics.volumetric import measure_pothole
from app.utils.imu import angle_between_deg, summarize as summarize_imu
from app.utils.logger import get_logger
from app.utils.video_io import make_writer, normalize_input, probe_video, transcode_for_web
from app.worker.annotator import annotate_frame
from app.worker.models_registry import ModelRegistry
from app.worker.tracker import BBoxTracker, PotholeTracker

log = get_logger("pipeline")


def process_video(
    input_video: Path,
    output_video: Path,
    cfg: dict,
    models: ModelRegistry,
    *,
    exif_reference_image: Optional[Path] = None,
    progress_every: int = 30,
) -> dict:
    """Run the full pipeline on a video.

    Returns a structured report dict with per-pothole measurements + summary.

    The caller owns the ModelRegistry — models are loaded once and reused across jobs.
    """
    # Normalize input first: re-encode to H.264 MP4 at a clean 30 fps so OpenCV
    # gets reliable metadata. Without this, MediaRecorder WebM files claim
    # 1000 fps and break tracker timestamps + plane-refit cadence.
    output_video.parent.mkdir(parents=True, exist_ok=True)
    normalized_input = output_video.parent / "input_normalized.mp4"
    log.info(f"Normalizing input via ffmpeg ({input_video.name} -> input_normalized.mp4)...")
    try:
        normalize_input(input_video, normalized_input, target_fps=30.0)
        active_input = normalized_input
    except Exception as e:
        log.warning(f"normalize_input failed: {e}; falling back to raw input")
        active_input = input_video

    info = probe_video(active_input)
    log.info(
        f"Video: {info.width}x{info.height} @ {info.fps:.1f} fps, "
        f"{info.frame_count} frames ({info.duration_s:.1f}s)"
    )

    intr_cfg = cfg["intrinsics"]
    gp_cfg = cfg["pipeline"]["ground_plane"]
    stride = int(cfg["pipeline"].get("frame_stride", 1))
    material_every = int(cfg["pipeline"].get("material_inference_every_n_frames", 30))

    if not models.is_ready():
        models.load_all()
    segmenter = models.segmenter
    crack_clf = models.crack_clf
    depth_model = models.depth
    material_clf = models.material  # may be None if weights missing

    device_key = intr_cfg["fallback_device"]
    device_cfg = intr_cfg["devices"][device_key]
    K = compute_K((info.height, info.width), image_path=exif_reference_image, device_cfg=device_cfg)

    pot_track_cfg = cfg["pipeline"].get("pothole_tracking", {})
    crk_track_cfg = cfg["pipeline"].get("crack_tracking", {})
    tracker = PotholeTracker(
        iou_threshold=pot_track_cfg.get("iou_threshold", 0.15),
        max_gap_frames=pot_track_cfg.get("max_gap_frames", int(info.fps)),
    )
    crack_tracker = BBoxTracker(
        iou_threshold=crk_track_cfg.get("iou_threshold", 0.15),
        max_gap_frames=crk_track_cfg.get("max_gap_frames", int(info.fps)),
    )
    sev = SeverityClassifier(cfg["severity"])
    advisor = RepairAdvisor(cfg["repair"])

    # IMU sanity check. Read sensors.json if it was uploaded alongside the video.
    sensors_path = input_video.parent / "sensors.json"
    imu_summary = summarize_imu(sensors_path)
    imu_gravity = None
    if imu_summary.gravity_camera is not None:
        imu_gravity = np.asarray(imu_summary.gravity_camera, dtype=np.float32)
        log.info(
            f"IMU: {imu_summary.samples_count} samples, "
            f"gravity_cam={[round(x, 3) for x in imu_summary.gravity_camera]}"
        )
    elif imu_summary.sensors_present:
        log.info(f"IMU sensors.json present but no usable samples (source={imu_summary.source})")
    else:
        log.info("IMU sensors.json not present; measurements rely on depth-only plane fit")

    imu_agreement_threshold_deg = 15.0
    plane_imu_angles: list[float] = []

    cap = cv2.VideoCapture(str(active_input))
    writer = make_writer(output_video, info.fps, (info.width, info.height))

    # Cached per-stride state
    last_potholes_annot: list[dict] = []
    last_cracks_annot: list[dict] = []
    plane: Optional[Plane] = None
    plane_refit_every = max(int(info.fps), 5)  # refit ~once per second

    raw_crack_detections: dict[str, int] = defaultdict(int)
    # Material accumulator: each entry is the per-frame predict() dict
    material_predictions: list[dict] = []
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
                        # IMU sanity check: plane normal points from ground toward camera,
                        # so the angle vs gravity (which points down) should be ~180°.
                        # We compare plane_normal to (-gravity) so the expected angle is ~0°.
                        if imu_gravity is not None:
                            angle = angle_between_deg(plane.normal, -imu_gravity)
                            plane_imu_angles.append(angle)

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
            crack_tracker.update(frame_idx, crack_dets)
            for c in crack_dets:
                raw_crack_detections[c.class_name] += 1

            # Material classifier — sparser cadence. Crop to road region first
            # so the classifier sees mostly road surface, not sky/vehicles.
            if material_clf is not None and frame_idx % material_every == 0:
                try:
                    road_crop = _crop_road_region(frame_rgb)
                    mat_pred = material_clf.predict(road_crop)
                    material_predictions.append(mat_pred)
                except Exception as e:
                    log.warning(f"material classifier failed on frame {frame_idx}: {e}")

            inference_time += time.time() - t_inf

        running = {"potholes": len(tracker.tracks), "cracks": sum(raw_crack_detections.values())}
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

    # Transcode the OpenCV-produced mp4v file to browser-friendly H.264
    # in place. Without this, Chrome on Android refuses to play it.
    log.info("Transcoding output to web-friendly H.264...")
    try:
        transcode_for_web(output_video)
    except Exception as e:
        log.warning(f"transcode_for_web failed: {e}; keeping mp4v output (may not play in browser)")

    # Cleanup normalized input
    if normalized_input.exists() and active_input == normalized_input:
        try:
            normalized_input.unlink()
        except Exception:
            pass

    # Aggregate IMU plane-agreement stats (if IMU was present)
    imu_check = {
        "sensors_present": imu_summary.sensors_present,
        "samples_count": imu_summary.samples_count,
        "source": imu_summary.source,
        "gravity_camera_frame": imu_summary.gravity_camera,
        "gravity_magnitude_ms2": (
            round(imu_summary.gravity_magnitude_ms2, 3)
            if imu_summary.gravity_magnitude_ms2 is not None else None
        ),
        "plane_normal_agreement": None,
    }
    if plane_imu_angles:
        angles = np.asarray(plane_imu_angles, dtype=np.float32)
        agreeing = float((angles < imu_agreement_threshold_deg).mean() * 100)
        imu_check["plane_normal_agreement"] = {
            "n_refits": len(plane_imu_angles),
            "threshold_deg": imu_agreement_threshold_deg,
            "mean_angle_deg": round(float(angles.mean()), 2),
            "max_angle_deg": round(float(angles.max()), 2),
            "percent_within_threshold": round(agreeing, 1),
        }

    # Aggregate material classifier predictions across frames
    road_surface = _aggregate_road_surface(material_predictions)
    repair_surface_type = _map_to_repair_surface(road_surface.get("material") if road_surface else None)

    # Aggregate per-track
    tracks_summary = tracker.finalize(
        min_observations=int(pot_track_cfg.get("min_observations", 5)),
        min_valid_measurements=1,
        min_avg_depth_cm=float(pot_track_cfg.get("min_avg_depth_cm", 0.5)),
        min_area_cm2=float(pot_track_cfg.get("min_area_cm2", 50)),
    )
    crack_tracks = crack_tracker.finalize(
        min_observations=int(crk_track_cfg.get("min_observations", 3)),
    )
    crack_counts_unique: dict[str, int] = defaultdict(int)
    for ct in crack_tracks:
        crack_counts_unique[ct["class_name"]] += 1

    potholes_report: list[dict] = []
    totals = {"area_cm2": 0.0, "volume_cm3": 0.0, "material_kg": 0.0, "cost": 0.0}

    for t in tracks_summary:
        sev_res = sev.classify(t["avg_depth_cm"], t["area_cm2"], t["volume_cm3"])
        rec = advisor.recommend(
            volume_cm3=t["volume_cm3"],
            depth_cm=t["avg_depth_cm"],
            area_cm2=t["area_cm2"],
            severity_level=sev_res.level,
            surface_type=repair_surface_type,
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
        "cracks": dict(crack_counts_unique),
        "crack_detections_raw": dict(raw_crack_detections),
        "road_surface": road_surface,
        "imu_check": imu_check,
        "potholes": potholes_report,
        "summary": {
            "num_potholes": len(potholes_report),
            "total_area_cm2": round(totals["area_cm2"], 1),
            "total_volume_cm3": round(totals["volume_cm3"], 1),
            "total_material_kg": round(totals["material_kg"], 2),
            "total_cost": round(totals["cost"], 2),
            "currency": cfg["repair"]["currency"],
            "total_cracks_detected": sum(crack_counts_unique.values()),
            "road_material": road_surface.get("material") if road_surface else None,
            "road_unevenness": road_surface.get("unevenness") if road_surface else None,
        },
    }
    log.info(f"Done. {frame_idx} frames in {elapsed:.1f}s (inference {inference_time:.1f}s)")
    return report


def _aggregate_road_surface(predictions: list[dict]) -> dict | None:
    """Combine per-frame material/unevenness predictions into a single video-level summary."""
    if not predictions:
        return None
    # Sum per-class probabilities, then pick the argmax — this is more robust than
    # majority-voting because it weights confident predictions higher.
    mat_sums: dict[str, float] = defaultdict(float)
    uneven_sums: dict[str, float] = defaultdict(float)
    for p in predictions:
        for name, prob in p.get("all_materials", {}).items():
            mat_sums[name] += float(prob)
        for name, prob in p.get("all_unevenness", {}).items():
            uneven_sums[name] += float(prob)
    n = len(predictions)
    mat_avg = {k: v / n for k, v in mat_sums.items()}
    uneven_avg = {k: v / n for k, v in uneven_sums.items()}
    top_mat = max(mat_avg, key=mat_avg.get) if mat_avg else None
    top_uneven = max(uneven_avg, key=uneven_avg.get) if uneven_avg else None
    return {
        "material": top_mat,
        "material_confidence": round(mat_avg.get(top_mat, 0.0), 3) if top_mat else None,
        "unevenness": top_uneven,
        "unevenness_confidence": round(uneven_avg.get(top_uneven, 0.0), 3) if top_uneven else None,
        "frames_used": n,
        "material_distribution": {k: round(v, 3) for k, v in sorted(mat_avg.items(), key=lambda kv: -kv[1])},
        "unevenness_distribution": {k: round(v, 3) for k, v in sorted(uneven_avg.items(), key=lambda kv: -kv[1])},
    }


def _map_to_repair_surface(predicted_material: str | None) -> str:
    """Map RSCD material classes -> RepairAdvisor surface_type domain.
    RepairAdvisor knows 'asphalt' and 'concrete'. mud/gravel/None default to asphalt
    (common case in Indian roads; advisor handles the rest)."""
    if predicted_material == "concrete":
        return "concrete"
    return "asphalt"


def _crop_road_region(frame_rgb: np.ndarray) -> np.ndarray:
    """Crop the frame to the area most likely to contain road surface.

    For phone-shot road videos held naturally (camera tilted slightly down),
    the road occupies roughly the bottom 60% of the frame, middle 80% width.
    Cropping to this region before material classification dramatically
    improves accuracy compared to feeding the whole frame, which dilutes
    the signal with sky / vehicles / scenery.
    """
    h, w = frame_rgb.shape[:2]
    y0 = int(h * 0.40)
    x0 = int(w * 0.10)
    x1 = int(w * 0.90)
    return frame_rgb[y0:, x0:x1]
