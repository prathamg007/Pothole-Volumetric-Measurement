# Project: Road Anomaly Analysis (UGP_amarb)

A mobile-to-server road condition analysis system. The user records a road video on a phone (PWA in any browser), uploads it over local WiFi to a FastAPI server running on a laptop. The server runs a multi-stage pipeline (pothole instance segmentation, crack typology, monocular metric depth, RANSAC ground plane, road material + unevenness classification) and returns an annotated MP4 plus a structured JSON report containing per-pothole measurements (cm² area, cm depth, cm³ volume), severity, and repair recommendations in INR.

## Layout

```
codebase/
├── CLAUDE.md             ← this file (auto-loaded by Claude Code)
├── CONVERSATION.md       ← full chat transcript (re-generated on demand)
├── project_memory/       ← decisions + facts portable to a fresh AI session
│   ├── MEMORY.md
│   ├── project_scope.md
│   ├── codebase_facts.md
│   └── project_decisions.md
├── server/               ← FastAPI brain (Python)
│   ├── app/
│   │   ├── main.py                 FastAPI app + lifespan
│   │   ├── routes/
│   │   │   ├── analyze.py          POST /analyze, GET /jobs/*, /jobs/{id}/result, /jobs/{id}/video
│   │   │   └── health.py           GET /health
│   │   ├── models/                 thin wrappers around the neural networks
│   │   │   ├── pothole_segmenter.py    YOLOv8 seg (3 classes)
│   │   │   ├── crack_classifier.py     YOLOv8 det (4 classes; Potholes ignored)
│   │   │   ├── depth_metric.py         Depth Anything V2 Metric Indoor (HuggingFace)
│   │   │   └── material_classifier.py  Multi-head MobileNetV3 (4 mat + 3 unev)
│   │   ├── physics/                measurement math (no deep nets here)
│   │   │   ├── intrinsics.py           K from EXIF / OnePlus 12 fallback, back-projection
│   │   │   ├── ground_plane.py         RANSAC plane fit on point cloud
│   │   │   ├── volumetric.py           per-pothole area/depth/volume from plane + 3D pts
│   │   │   ├── severity.py             LOW/MEDIUM/HIGH/CRITICAL scoring
│   │   │   └── repair_advisor.py       method, material kg, cost INR (config-driven)
│   │   ├── worker/                 orchestration, no models
│   │   │   ├── models_registry.py      singleton holder; loaded once at server startup
│   │   │   ├── job_store.py            SQLite-backed job persistence
│   │   │   ├── job_runner.py           async task that executes the pipeline
│   │   │   ├── pipeline.py             end-to-end video orchestrator
│   │   │   ├── tracker.py              IoU-based PotholeTracker + BBoxTracker
│   │   │   └── annotator.py            draws masks, bboxes, labels, HUD on frames
│   │   ├── utils/
│   │   │   ├── config.py               YAML loader
│   │   │   ├── logger.py               stdout logger
│   │   │   ├── video_io.py             OpenCV + ffmpeg wrappers
│   │   │   └── imu.py                  parse sensors.json sidecar, gravity vector math
│   │   ├── schemas.py              Pydantic API models
│   │   └── routes/, worker/, utils/, models/, physics/  (subdirs above)
│   ├── config/config.yaml          all tunables: weights paths, thresholds, INR rates
│   ├── weights/                    .pt files (gitignored; copy from legacy/)
│   ├── data/
│   │   ├── uploads/                incoming videos (gitignored)
│   │   ├── results/                processed videos + reports (gitignored)
│   │   ├── dev_cert/               self-signed HTTPS cert (gitignored)
│   │   └── jobs.db                 SQLite (gitignored)
│   ├── static/app/                 Progressive Web App (HTML/CSS/JS)
│   │   ├── index.html              4 views: home / recorder / upload / results
│   │   ├── manifest.json           PWA manifest
│   │   ├── icon-192.png, icon-512.png
│   │   ├── css/style.css           dark mobile-first theme
│   │   └── js/{api,recorder,results,app}.js
│   ├── tools/gen_dev_cert.py       generate self-signed cert with LAN IPs in SAN
│   ├── scripts/                    smoke tests + one-off utilities
│   │   ├── verify_weights.py
│   │   ├── smoke_test_phase2.py    single-image pipeline check
│   │   ├── smoke_test_phase3.py    video pipeline check
│   │   └── smoke_test_phase4.py    full HTTP flow check
│   ├── run_server.py               HTTP on :8000 (dev)
│   ├── run_server_https.py         HTTPS on :8443 (required for phone camera/IMU)
│   └── requirements.txt
├── training/material_classifier/   one-off training scripts
│   ├── peek_zip.py                 inspect RSCD layout
│   ├── curate.py                   stream-extract balanced 32k+8k subset from train/
│   ├── extract_test_from_vali.py   handle vali_20k flat layout for test set
│   ├── train.py                    multi-head MobileNetV3 trainer
│   ├── eval.py                     test-set metrics (precision/recall/F1/confusion)
│   ├── runs/v1/                    checkpoints + metrics.json + test_metrics.json
│   └── data/curated/               train/val/test images + *_labels.csv
├── mobile/                         empty (we pivoted from Flutter to PWA)
├── legacy/                         original repos preserved read-only
│   ├── InfraSight/                 source of pothole_seg.pt, dataset, design ideas
│   └── road-anomaly-detection/     source of crack_typology.pt + Model 1 (dropped)
└── .gitignore
```

## Models in use

| Role | Source | File | Notes |
|---|---|---|---|
| Pothole instance segmentation | InfraSight (3 classes: Manhole, Pothole, Unmarked Bump) | `server/weights/pothole_seg.pt` | 6.5 MB |
| Crack typology (Longitudinal, Transverse, Alligator) | road-anomaly `YOLOv8_Small_2nd_Model.pt` | `server/weights/crack_typology.pt` | 86 MB; 4 classes, "Potholes" class ignored |
| Metric depth | HuggingFace `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | HF cache | downloaded on first run; ~100 MB |
| Material + unevenness | Trained by us on RSCD subset | `server/weights/material_classifier.pt` | 5 MB; 4 + 3 outputs |

**Dropped:** road-anomaly Model 1 (`RoadDetectionModel/.../best.pt`, YOLOv8m 7-class). Crack mAP@.5:.95 was 0.24-0.27, weak; pothole detection redundant with InfraSight's segmenter.

## Architecture at a glance

```
Phone (PWA in Chrome)               Laptop (FastAPI server)
─────────────────────               ────────────────────────
record video + IMU samples
                ↓ HTTPS multipart upload (LAN, self-signed cert)
                                    POST /analyze → job_id
                                            ↓ BackgroundTasks + asyncio.Lock
                                    ffmpeg normalize input  (fixes WebM 1000 fps quirk)
                                            ↓
                                    For each frame at stride N:
                                       pothole seg (YOLOv8)
                                       crack det (YOLOv8 small)
                                       metric depth (Depth Anything V2 Metric)
                                       intrinsics K → 3D point cloud
                                       RANSAC ground plane (refit ~1×/sec)
                                       per-pothole measurement (area/depth/volume)
                                       material classifier (every ~30 frames, road-region crop)
                                            ↓
                                    IoU tracker for potholes
                                    IoU tracker for cracks (dedup)
                                    severity + repair (config-driven INR rates)
                                            ↓
                                    Annotate frames; ffmpeg → H.264 MP4 (browser-friendly)
                                            ↓
                                    Write SQLite job state + report.json + annotated.mp4
                ← poll GET /jobs/{id} every 2s
                ← GET /jobs/{id}/result   (JSON)
                ← GET /jobs/{id}/video    (MP4)
display annotated video + per-pothole list + summary
```

## Key design decisions (locked in, with rationale)

- **Pothole** gets full cm² / cm / cm³ measurement. **Cracks** get type + location only — no area or volume. (Crack models give bboxes, not masks; segmentation would need new training data.)
- **No magic calibration constant.** The legacy InfraSight code had `calibration_constant=30.0` as a hardcoded scalar to convert relative depth differences to cm. We replaced this with a metric depth model that returns meters directly, then derived camera height from RANSAC ground plane geometry. Self-calibrating.
- **Indoor depth model**, not Outdoor. The KITTI-trained Outdoor variant floor-clips at ~4 m and inflates close-range areas ~18×. The Indoor variant (HyperSim/NYU) handles 0–10 m, which covers handheld pothole shots cleanly.
- **Camera intrinsics K** computed per-video from EXIF / OnePlus 12 fallback, not hardcoded. Falls back to a 60° HFOV guess if neither is available.
- **IMU role:** sanity-check the ground plane normal against gravity; flag mismatches >15°. Not used for height (the depth+plane gives that more reliably than IMU integration which drifts).
- **Currency is INR**, all rates config-driven in `server/config/config.yaml`.
- **Upload-to-server architecture**: phone and laptop on the same WiFi for the demo. No cloud.
- **PWA over Flutter**: pivoted in Phase 6 because installing Flutter SDK + Android Studio + Android SDK would have cost ~2 hours of user time. PWA gets the same demo workflow with zero install.
- **HTTPS required**: browsers refuse `getUserMedia` and `DeviceMotionEvent` over plain HTTP except on localhost. We ship a self-signed cert; user accepts the warning once per device.
- **Target accuracy**: ±15% on pothole depth/volume. Documented and acceptable for an undergrad project.

## Running the server (dev)

First time only:
```bash
cd D:\UGP_amarb\codebase\server
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python tools/gen_dev_cert.py    # one-time; valid for 1 year
```

Each session:
```bash
python run_server_https.py      # HTTPS on :8443 (use this for phone)
# or
python run_server.py            # HTTP on :8000 (laptop-only testing)
```

Phone opens `https://<your-wifi-ip>:8443/app/`. Tap Advanced → Proceed once. Allow camera. Record. Process.

## Verification scripts

| Script | What it checks |
|---|---|
| `server/scripts/verify_weights.py` | Both YOLO weight files load cleanly |
| `server/scripts/smoke_test_phase2.py` | Single-image pipeline end-to-end on `legacy/InfraSight/test8.png` |
| `server/scripts/smoke_test_phase3.py` | Video pipeline on `legacy/InfraSight/road_test.mp4`; produces annotated.mp4 + report.json |
| `server/scripts/smoke_test_phase4.py` | Full HTTP flow against a running server |

## Conventions

- `pathlib.Path` everywhere; no string path concatenation, no hardcoded Windows separators.
- All tunable numbers live in `server/config/config.yaml`. Magic numbers are a smell — move them to config.
- Module organization: `app/models/` wraps neural networks; `app/physics/` is measurement math; `app/worker/` orchestrates; `app/routes/` is HTTP-only.
- Legacy code stays in `legacy/` untouched. Copy and refactor into `server/` rather than editing in place.
- Heavy artifacts (model weights, datasets, runtime data, dev cert, jobs.db) are gitignored. Code, config, and small documentation files are tracked.

## Phase log

- **Phase 1+2**: Repo scaffold, FastAPI skeleton, ported pipeline core with metric depth + RANSAC + EXIF intrinsics. (commits `56c34b0`)
- **Phase 3**: Video pipeline with multi-frame pothole tracking. (`8b38fd3`)
- **Phase 4**: HTTP API + async job queue with SQLite. (`8607daf`)
- **Phase 5**: Multi-head MobileNetV3 material+unevenness classifier trained on RSCD subset. (`9a14073`)
- **Phase 6**: Progressive Web App + HTTPS for camera/IMU access. (`3fe1c19`)
- **Phase 7**: Polish — codec/fps fixes, tracker dedup, IMU sanity check, road-region crop for material classifier, PWA file-upload feature, comprehensive handoff docs.

`git log` has detailed per-commit messages explaining what changed and why.

## Known limitations (Phase 7+)

- **Material classifier accuracy is high on RSCD (91.18%) but degrades on dashcam-style frames where road occupies <50% of the image.** The road-region crop helps; for further improvement, mask the inference region using the depth-derived ground plane.
- **OnePlus 12-only intrinsic fallback.** Other phones will have proportionally off cm² values when EXIF is absent (which is most of the time for video). Counts and types remain correct.
- **Monocular depth ±15%.** Inherent to the approach; would require a depth sensor or stereo to improve.
- **Pothole tracker can still split same-pothole into multiple tracks under very fast motion.** Lowering IoU below 0.15 starts merging different real potholes; a Kalman filter for bbox motion prediction would be the principled fix.

## Memory + handoff

Persistent decisions live in `project_memory/` (copied from Claude Code's per-account memory). A new AI session can read these files to bootstrap context. The full chat history is in `CONVERSATION.md`.
