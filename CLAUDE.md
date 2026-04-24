# Project: Road Anomaly Analysis (UGP_amarb)

Mobile app records road video → uploads to a laptop-side FastAPI server → server processes through a pipeline that segments potholes, classifies cracks by type, measures potholes in cm²/cm/cm³, classifies road material, and produces repair recommendations in INR. Processed video + structured report returned to the app.

## Layout

- `server/` — FastAPI + Python worker pipeline (the brain)
- `mobile/` — Flutter Android app (thin client; records video + IMU, uploads, displays results)
- `training/` — One-off training scripts (material classifier)
- `legacy/` — Original `InfraSight/` and `road-anomaly-detection/` repos, preserved read-only for reference
- `weights lookup: `server/weights/` holds the shipping models; training outputs land in `training/.../runs/`

## Models in use

| Role | Source | File |
|---|---|---|
| Pothole instance segmentation | InfraSight (3 classes: Manhole, Pothole, Unmarked Bump) | `server/weights/pothole_seg.pt` |
| Crack typology (Longitudinal, Transverse, Alligator; Potholes class ignored) | road-anomaly YOLOv8_Small_2nd_Model | `server/weights/crack_typology.pt` |
| Metric depth | HuggingFace Depth Anything V2 Metric (downloaded on first run) | cache |
| Road material | Trained in Phase 5 | `server/weights/material_classifier.pt` (not yet present) |

Dropped: `road-anomaly-detection` Model 1 (`RoadDetectionModel/.../best.pt`). Redundant and its crack-severity classes are weak.

## Key design decisions (firm)

- **Pothole** gets full cm² / cm / cm³ measurement. **Cracks** get type + location only — no area or volume.
- **No magic calibration constant.** Depth comes from a metric depth model (outputs meters). Ground plane comes from RANSAC on the 3D point cloud derived from K+depth. Camera height is the perpendicular distance from camera origin to that plane — computed per frame, not input by the user.
- **Camera intrinsics K** are computed per-video from EXIF/video metadata. OnePlus 12 fallback values live in `config.yaml`.
- **IMU role**: sanity-check the ground plane normal against gravity; fall back to IMU pitch when plane detection fails; reject sky-facing frames. Height is NOT integrated from IMU.
- **Currency is INR**, rates config-driven in `server/config/config.yaml` under `repair.materials` and `repair.labor`.
- **Upload-to-server architecture**: phone and laptop on the same WiFi for the demo. No cloud.
- **Target accuracy**: ±15% on pothole depth/volume. Documented and acceptable.

## Running the server (dev)

```bash
cd D:\UGP_amarb\codebase\server
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_server.py
# http://localhost:8000/docs
```

## Verify weights load (Phase 1 smoke test)

```bash
cd D:\UGP_amarb\codebase\server
python scripts/verify_weights.py
```

## Conventions

- Always use `pathlib.Path` for filesystem; no string concatenation, no hardcoded Windows separators.
- All tunable numbers live in `server/config/config.yaml`. If you find yourself writing a magic number, move it to config.
- Module organization: `app/models/` wraps neural networks; `app/physics/` handles measurement math; `app/worker/` orchestrates the pipeline; `app/routes/` is HTTP-only.
- Legacy code stays in `legacy/` untouched. Copy and refactor into `server/` rather than editing in place.

## Memory

Session memory at `C:\Users\prath\.claude\projects\D--UGP-amarb-codebase\memory\` — durable decisions + codebase facts live there across sessions.
