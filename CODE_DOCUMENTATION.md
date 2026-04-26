# Code Documentation — Pothole Volumetric Measurement and Repair Cost Estimation from Smartphone Video

This document is a tour of the codebase. It is meant to be read end-to-end
by anyone who wants to (a) understand the structure of the repository,
(b) know what each file is for, (c) configure or extend the system, or
(d) install everything from scratch and run the system on their own
machine.

The accompanying technical report (`report/main.pdf`) covers the
methodology, the experimental results, and the design rationale; this
document covers the code itself.

---

## 1. Repository overview

```
codebase/
├── server/                 ← FastAPI inference server + Progressive Web App
├── training/               ← scripts that produced the road-surface classifier
├── legacy/                 ← reference repositories (read-only, not used at runtime)
├── report/                 ← LaTeX report + figure generators
├── CLAUDE.md               ← short architectural summary
└── CODE_DOCUMENTATION.md   ← this file
```

Three things actually run: the **server** (where the pipeline lives),
the **mobile client** (the Progressive Web App, served by the server),
and the **training scripts** (used once to train the road-surface
classifier; not needed at deployment time).

The **legacy** subtree contains two earlier open-source repositories
that produced some of the pretrained weights that we initialise from
(see Section 4 below). They are preserved read-only for provenance and
are not used during normal operation.

The **report** subtree contains the LaTeX source of the technical
report and small Python scripts that regenerate every figure in the
report from data on disk.

---

## 2. The server

The server is a single Python process that owns four neural networks,
a SQLite job database, and a small REST API.

```
server/
├── app/
│   ├── main.py                       FastAPI app + lifespan
│   ├── schemas.py                    Pydantic request/response models
│   ├── routes/
│   │   ├── analyze.py                POST /analyze, GET /jobs/*, /result, /video, /mesh
│   │   └── health.py                 GET /health
│   ├── models/                       thin wrappers around the four neural networks
│   │   ├── pothole_segmenter.py        YOLOv8-seg (3 classes)
│   │   ├── crack_classifier.py         YOLOv8-det (4 classes; "Potholes" ignored)
│   │   ├── depth_metric.py             Depth Anything V2 Metric Indoor (HuggingFace)
│   │   └── material_classifier.py      Multi-head MobileNetV3 (4 mat + 3 unevenness)
│   ├── physics/                      pure-math measurement code; no neural networks
│   │   ├── intrinsics.py               K from EXIF / device fallback; back-projection
│   │   ├── ground_plane.py             RANSAC plane fitter
│   │   ├── volumetric.py               per-pothole area / depth / volume on the plane
│   │   ├── severity.py                 LOW/MEDIUM/HIGH/CRITICAL scoring
│   │   └── repair_advisor.py           method, kg, INR cost from config
│   ├── worker/                       orchestration; no neural networks
│   │   ├── models_registry.py          singleton holding the loaded models
│   │   ├── job_store.py                SQLite-backed job persistence
│   │   ├── job_runner.py               background task that runs the pipeline
│   │   ├── pipeline.py                 end-to-end video orchestrator
│   │   ├── tracker.py                  PotholeTracker + BBoxTracker
│   │   └── annotator.py                draws masks/boxes/HUD on each frame
│   ├── visualization/
│   │   └── mesh_engine.py              3D Plotly pothole tomography
│   └── utils/
│       ├── config.py                   YAML loader
│       ├── logger.py                   stdout logger
│       ├── video_io.py                 OpenCV + ffmpeg wrappers
│       └── imu.py                      sensors.json parser; gravity in camera frame
├── config/
│   └── config.yaml                   ALL tunables: weights paths, thresholds, INR rates
├── weights/                          .pt files (gitignored; copy from legacy/ once)
├── data/
│   ├── uploads/                      incoming videos (gitignored)
│   ├── results/                      processed videos + reports + 3D meshes (gitignored)
│   ├── dev_cert/                     self-signed HTTPS cert (gitignored)
│   └── jobs.db                       SQLite job store (gitignored)
├── static/app/                       the Progressive Web App
│   ├── index.html                    home / recorder / upload / results views
│   ├── manifest.json                 PWA manifest
│   ├── icon-192.png, icon-512.png
│   ├── css/style.css                 mobile-first dark theme
│   └── js/{api,recorder,results,app}.js
├── tools/
│   └── gen_dev_cert.py               generate self-signed TLS cert with LAN IPs in SAN
├── scripts/                          smoke-test scripts (see §6 below)
│   ├── verify_weights.py
│   ├── smoke_test_phase2.py          single-image pipeline check
│   ├── smoke_test_phase3.py          full video pipeline check (no HTTP)
│   └── smoke_test_phase4.py          full HTTP flow check (server must be running)
├── run_server.py                     uvicorn launcher, HTTP on :8000 (dev)
├── run_server_https.py               uvicorn launcher, HTTPS on :8443 (mobile demo)
└── requirements.txt
```

### 2.1 Entry points

There are **two ways to start the server**:

```bash
python server/run_server.py        # plain HTTP on :8000 (laptop-only testing)
python server/run_server_https.py  # HTTPS on :8443 (required for phone)
```

The HTTPS variant requires a self-signed certificate at
`server/data/dev_cert/{cert,key}.pem`. Generate it once with
`python server/tools/gen_dev_cert.py`; the cert is valid for one year
and covers `localhost` plus every active LAN IPv4 address. Re-run the
generator if the laptop's LAN IP changes (e.g. on a new Wi-Fi
network).

### 2.2 Module-by-module walkthrough

#### `app/main.py`

The FastAPI application object plus the lifespan context manager. The
lifespan loads the YAML config, opens the SQLite job store, builds the
`ModelRegistry` and runs `models.load_all()` inside an executor (so
the ~12 second first-time model load does not block the event loop),
and stores everything on `app.state`. It also mounts the PWA at
`/app/`, redirects `/` to `/app/`, and includes the two route modules.

#### `app/schemas.py`

Pydantic models that correspond exactly to the JSON shape returned by
the API. The notable ones are:

- `JobCreated` — returned by `POST /analyze`
- `JobState` — returned by `GET /jobs/{id}`
- `PotholeResult` — one entry in the `potholes[]` array of the report
- `AnalysisReport` — full structure returned by `GET /jobs/{id}/result`

#### `app/routes/analyze.py`

The seven analysis endpoints (see Table 2 in the report).
`POST /analyze` streams the multipart body to disk in 1 MB chunks,
enforcing the upload size limit during the stream so a malicious
client cannot exhaust disk by sending a large body before validation.
On success it inserts a row in the `jobs` table with status `queued`
and registers a `BackgroundTasks.add_task(run_job, ...)` that fires
after the HTTP response is returned.

The polling endpoints (`GET /jobs`, `GET /jobs/{id}`) read directly
from the SQLite store. The terminal endpoints (`/result`, `/video`,
`/mesh/{tid}`, `/mesh/{tid}.png`) return 409 if the job is not yet
complete and 404 if the job ID is unknown.

#### `app/routes/health.py`

Returns 200 plus a small JSON object reporting whether the two
required `.pt` weight files exist. Used by the smoke-test wait-loop
and by the PWA's status pill.

#### `app/worker/models_registry.py`

Singleton that owns the four loaded models. `load_all()` is
thread-safe via an internal `threading.Lock` and is idempotent. The
material classifier is loaded only if its weight file exists on disk
and the `material_classifier.optional` flag in `config.yaml` is true
(the default); otherwise the server starts without it and the pipeline
silently skips the material/unevenness branch.

#### `app/worker/job_store.py`

Thin wrapper around the single-table `jobs` SQLite schema. All writes
go through a process-wide `threading.Lock`. Each row carries the job
ID (UUID), status, created/started/completed timestamps in ISO 8601
UTC, the input file path, the output paths, and an error message
column for failed jobs.

#### `app/worker/job_runner.py`

The async function that executes a single job. It acquires
`app.state.pipeline_lock` (the `asyncio.Lock` that serialises pipeline
runs across simultaneous uploads), marks the job `processing`, runs
`process_video` on the default thread executor (because the pipeline
is GPU/CPU-bound and synchronous), writes the JSON report, and marks
the job `completed`. Any exception is captured into the
`error_message` column and the job is marked `failed`.

#### `app/worker/pipeline.py`

The orchestrator. The flow:

1. Re-encode the input video via `ffmpeg` to a clean H.264 MP4 at 30 fps.
2. Probe video metadata (width, height, fps, frame count) via OpenCV
   and clamp out-of-range fps values.
3. Compute the camera intrinsic matrix `K` from EXIF or the per-device
   fallback in config.
4. Build a `PotholeTracker`, a `BBoxTracker` for cracks, the severity
   classifier, and the repair advisor. Read inertial data from
   `sensors.json` if present.
5. Open the video and an OpenCV `VideoWriter`. Loop over frames at
   `frame_stride`. On each strided frame:
   - Run pothole segmentation and crack typology detection.
   - If at least one pothole was detected: run the metric depth
     network, back-project to the 3D point cloud, refit the ground
     plane every ~1 second.
   - For each pothole: compute area/avg-depth/max-depth/volume on the
     fitted plane via the convex-hull integration in
     `physics/volumetric.py`.
   - Update the pothole tracker and the crack tracker.
   - Save the highest-confidence per-track depth and mask crops for
     later 3D mesh rendering.
   - Every 30 frames, run the material classifier on a road-region
     crop of the frame.
   - Annotate the frame and write it to the output video.
6. Transcode the output MP4 to H.264 baseline so mobile browsers can
   play it.
7. Aggregate inertial agreement statistics, material/unevenness
   distributions, and per-track measurements.
8. For each finalised pothole track, render a 3D Plotly mesh from its
   best-observation crop. Save the HTML and (if `kaleido` is
   installed) the PNG to `data/results/<job_id>/meshes/`.
9. Assemble and return the report dict, which is then JSON-serialised
   to disk by `job_runner.py`.

#### `app/worker/tracker.py`

`PotholeTracker` is a greedy IoU matcher with mask + measurement
storage per track and a method `maybe_update_best_obs` that keeps the
highest-confidence observation's cropped depth/mask/image arrays for
later 3D mesh rendering. `BBoxTracker` is the lighter version used for
crack detections; it matches same-class boxes only and stores no
measurements. Both have a `finalize()` method that aggregates per-
track values, applies the persistence filter
(`min_observations`), and applies plausibility floors
(`min_avg_depth_cm`, `min_area_cm2`).

#### `app/worker/annotator.py`

The per-frame OpenCV drawing logic: semi-transparent severity-coloured
mask fills, severity-coloured bounding boxes, per-track text labels,
and the top-left HUD with running counters of active pothole tracks
and accumulated raw crack detections.

#### `app/models/pothole_segmenter.py`

Wraps Ultralytics' `YOLO` class. Validates that the loaded model has
`task == "segment"`. Runs inference, resizes the mask to the input
resolution, thresholds at 0.5, and packages each detection into a
`PotholeDetection` dataclass. `detect_potholes()` filters the result
to only the `Pothole` class.

#### `app/models/crack_classifier.py`

Wraps Ultralytics' `YOLO` class for the crack detector. Validates that
the loaded model has `task == "detect"`. Reads `ignore_classes` from
config and drops any detection whose class ID is in that set (this is
how we avoid double-counting potholes from this model).

#### `app/models/depth_metric.py`

Wraps the HuggingFace `transformers.pipeline("depth-estimation", ...)`
for Depth Anything V2 Metric Indoor. Returns a per-pixel depth map in
metres at the input resolution. Includes a colorize helper for
visualisation.

#### `app/models/material_classifier.py`

Loads the multi-head MobileNetV3-Small checkpoint produced by
`training/material_classifier/train.py`. Re-instantiates the same
two-head architecture in-process and applies ImageNet normalisation.
Returns `{material, material_confidence, unevenness,
unevenness_confidence, all_materials, all_unevenness}` for a single
image.

#### `app/physics/intrinsics.py`

Builds the `3 × 3` intrinsic matrix `K`. Tries EXIF
`ExifIFD.FocalLength` via `piexif` first; falls back to a per-device
config (default `oneplus_12`); falls back to a 60° HFOV assumption.
`backproject(depth_m, K)` returns the dense `(H, W, 3)` 3D point cloud
in OpenCV camera coordinates (X right, Y down, Z forward).

#### `app/physics/ground_plane.py`

`Plane` dataclass and `fit_ground_plane()`. Vanilla 3-point RANSAC
over 1000 iterations, then SVD refinement on the inliers, then
acceptance check on the refined plane's inlier ratio. Orients the
normal so the camera origin is on the positive side, so
`Plane.camera_height_m` returns the perpendicular distance from camera
to plane.

#### `app/physics/volumetric.py`

`measure_pothole(mask, point_cloud, plane)` computes per-pothole area,
average and maximum depth, and volume. The signed distance of every
pothole pixel to the plane gives depth; orthogonal projection of the
points onto the plane plus a 2D convex hull gives area on the actual
ground surface (correcting for image-plane foreshortening). Volume is
area × mean depth.

#### `app/physics/severity.py`

Depth, area, and volume each map onto a 0–10 score by a monotone
non-linear curve (log, sqrt, log10 respectively). The composite score
is a weighted sum (default 0.4 / 0.3 / 0.3) and is bracketed to
LOW/MEDIUM/HIGH/CRITICAL with metadata (priority, repair days, risk
description) per bracket.

#### `app/physics/repair_advisor.py`

Selects a repair method (throw-and-roll / semi-permanent / full-depth)
from severity and dimensions; selects a material (cold mix asphalt /
hot mix asphalt / concrete patch) from the surface type predicted by
the material classifier; computes the required mass and the total
cost in INR. All densities, prices, labour rates, durabilities, and
the per-method procedural step list are read from `config.yaml`.

#### `app/visualization/mesh_engine.py`

The 3D Plotly tomography renderer. Takes a per-pothole cropped depth
map, a cropped binary mask, and (optionally) a cropped RGB image and
produces a self-contained HTML file showing the pothole as a
depression below a flat road plane, with edge-blending via the
`cv2.distanceTransform` of the mask, a depth-intuitive colour scale,
and a red overlay on the deepest 20% of the interior. The PNG export
is best-effort and requires the optional `kaleido` package.

#### `app/utils/config.py`

`lru_cache`'d YAML loader. `resolve_path()` makes any relative path in
config absolute against the server root.

#### `app/utils/logger.py`

Stdout-only logger with a clean format
`%(asctime)s [%(levelname)s] %(name)s: %(message)s`. Tracks already-
configured names to avoid duplicate handlers.

#### `app/utils/video_io.py`

Three functions:

- `probe_video(path)` — uses OpenCV to read width/height/fps/frame
  count, clamps unrealistic fps values to 30.
- `normalize_input(src, dst, target_fps=30.0)` — re-encodes via
  `ffmpeg` to clean H.264 MP4 at exactly the target fps.
- `transcode_for_web(src, dst=None)` — re-encodes the OpenCV-written
  MP4 to H.264 baseline 3.1 with `+faststart` so mobile browsers can
  play it; atomically replaces the original file.

#### `app/utils/imu.py`

Parses `sensors.json` (the inertial sidecar uploaded with the video),
averages `accelerationIncludingGravity` over the recording, applies
the back-camera-portrait device→camera rotation, and returns a unit
gravity vector in the camera frame. Also exposes
`angle_between_deg(v1, v2)` used to compare the plane normal against
gravity at every plane refit.

### 2.3 The Progressive Web App (`server/static/app/`)

A single-page application with four DOM views.

- `index.html` — view shells, loads CSS and four JS modules.
- `manifest.json` — PWA manifest (name, icons, theme).
- `css/style.css` — dark mobile-first theme, severity-tinted pothole
  rows, full-screen recorder UI.
- `js/api.js` — `Api` namespace, fetch + XHR wrapper around the
  server's REST endpoints. The `upload()` method uses XHR specifically
  so the upload progress can drive a `<progress>` bar.
- `js/recorder.js` — `Recorder` namespace. Calls
  `navigator.mediaDevices.getUserMedia` for the rear camera, attaches
  a `MediaRecorder` to the stream, and a `devicemotion` listener that
  buffers inertial samples timestamped relative to the recorder's
  start.
- `js/results.js` — `Results.render(report, jobId)` renders the
  finished job's JSON into the results view, including the
  per-pothole rows with the *View 3D Tomography* button per row.
- `js/app.js` — top-level shell: view router, server URL persistence
  in `localStorage`, the home/recorder/upload/results flow, and the
  job-polling loop.

### 2.4 The configuration file

Every tunable in the system lives in `server/config/config.yaml`. The
sections are:

- `server.{host, port, upload_dir, results_dir, max_upload_mb}`
- `models.pothole_segmenter` / `crack_classifier` /
  `material_classifier` / `depth` — weights paths, confidence
  thresholds, class lists.
- `pipeline.{frame_stride, ground_plane.*, pothole_tracking.*,
  crack_tracking.*}` — frame stride, RANSAC parameters, tracker
  parameters.
- `intrinsics.{fallback_device, devices.<name>}` — per-device focal
  length and sensor dimensions.
- `severity.weights` — depth/area/volume weights for the composite
  severity score.
- `repair.{currency, materials, labor, durability_months}` — repair
  method recipes and cost rates.

The full file is reproduced verbatim in Appendix A of the report.

---

## 3. The training pipeline

```
training/material_classifier/
├── peek_zip.py                  inspect RSCD's directory layout
├── curate.py                    stream-extract balanced subset from train/
├── extract_test_from_vali.py    handle vali_20k's flat layout for the test set
├── train.py                     multi-head MobileNetV3 trainer + masked loss
├── eval.py                      held-out test-set evaluation
└── runs/v1/
    ├── material_classifier_best.pt
    ├── metrics.json             per-epoch training log
    └── test_metrics.json        per-class precision/recall/F1/confusion
```

### Pipeline order

```
peek_zip.py          → confirm RSCD directory layout
curate.py            → extract 32k train + 8k val from RSCD train/  (~15 min)
extract_test_from_vali.py → extract 4140 test images from vali_20k  (~5 min)
train.py             → multi-head MobileNetV3, 25 epochs            (~37 min on RTX 4050)
eval.py              → held-out test-set per-class metrics           (~1 min)
```

### Each script

#### `peek_zip.py`

Inspects an RSCD zip without extracting. Counts top-level directories,
class folders, and entries by depth. Useful sanity check before
running curation.

#### `curate.py`

Stream-extracts a balanced subset from `train/`. For each of the four
materials (asphalt, concrete, mud, gravel) it samples 10 000 images
with a per-unevenness breakdown for asphalt and concrete (which carry
unevenness labels) and no breakdown for mud/gravel (which do not). The
default 80/20 train/val split gives ~32 000 train + ~8 000 val. The
script writes flat directories `train/`, `val/` plus matching
`*_labels.csv` files. Stream extraction means we do not have to fully
unpack the 14 GB archive.

#### `extract_test_from_vali.py`

A patch script for the test set. RSCD's `vali_20k/` directory uses a
flat layout where labels are encoded in filenames like
`<datestamp>-<friction>-<material>-<unevenness>.jpg`, which `curate.py`
does not handle. This script parses those filenames and extracts a
balanced 4140-image test set into `test/` plus `test_labels.csv`.

#### `train.py`

The multi-head MobileNetV3-Small training script. Five-epoch frozen-
backbone warm-up at learning rate $10^{-3}$ for the heads only; then
unfreeze and switch to a two-parameter-group AdamW (backbone at
$10^{-4}$, heads at $10^{-3}$) on a fresh cosine schedule. Loss is
material cross-entropy plus 0.5 × masked unevenness cross-entropy.
ImageNet normalisation; standard image-classification augmentation
(`RandomResizedCrop`, horizontal flip, `ColorJitter`). Saves the best-
val-material-accuracy checkpoint.

The output checkpoint contains the model state dict plus the
`materials` and `unevenness` class lists, so the inference wrapper
can reconstruct everything from the file alone. Copy the final
`material_classifier_best.pt` to
`server/weights/material_classifier.pt`.

#### `eval.py`

Loads a trained checkpoint, runs inference on the test set, and
writes per-class precision, recall, F1, and confusion matrices for
both heads to `test_metrics.json`. The numbers in this file are the
ones quoted in the report's Experiments section.

---

## 4. The legacy reference repositories

`legacy/` contains two earlier open-source repositories that were used
as starting points and are kept read-only for provenance.

- `legacy/InfraSight/` — a pothole-volumetric-measurement project that
  used a relative-depth network plus a hardcoded scalar calibration
  constant. Its YOLOv8-seg pothole weights
  (`weights/phase1_segmentation_v1.pt`) were copied to
  `server/weights/pothole_seg.pt`. Its Roboflow dataset is used in the
  classifier training pipeline for context only.
- `legacy/road-anomaly-detection/` — a YOLOv8-detect crack typology
  project. Its weights file (`YOLOv8_Small_2nd_Model.pt`) was copied
  to `server/weights/crack_typology.pt`. It also contains a 7-class
  YOLOv8m alternative (`RoadDetectionModel/.../weights/best.pt`) that
  was considered and rejected; the report's Experiments section
  reproduces its training metrics for context.

Nothing in `legacy/` is loaded at runtime. The two `.pt` files
copied across are the only artefacts that flow from `legacy/` into
the running server.

---

## 5. Running the system

### 5.1 First-time setup

You need Python 3.11 or 3.12, plus `ffmpeg` on PATH, plus a CUDA
toolkit if you want GPU inference (CPU also works, ~10× slower).

```bash
cd server

# Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Install PyTorch with the right CUDA wheels for your machine
# (the requirements.txt has only the CPU-default torch; replace it)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy the .pt model weights into server/weights/
copy ..\legacy\InfraSight\weights\phase1_segmentation_v1.pt weights\pothole_seg.pt
copy ..\legacy\road-anomaly-detection\YOLOv8_Small_2nd_Model.pt weights\crack_typology.pt
copy ..\training\material_classifier\runs\v1\material_classifier_best.pt weights\material_classifier.pt

# Verify all weights load cleanly
python scripts\verify_weights.py

# Generate the self-signed TLS certificate (one-time, valid 1 year)
python tools\gen_dev_cert.py
```

The HuggingFace metric depth model (~100 MB) is downloaded on first
use and cached under `~/.cache/huggingface/`.

### 5.2 Running the server

Two ways:

```bash
# Plain HTTP on :8000  (laptop-only, no phone access)
python run_server.py

# HTTPS on :8443       (phone-accessible; required for camera/IMU APIs)
python run_server_https.py
```

Both load the four neural networks at startup (~12 seconds first time)
and then expose the same set of endpoints.

### 5.3 Using the system from a phone

1. Make sure the phone and the laptop are on the same Wi-Fi.
2. The HTTPS server's startup output will print the URL, typically
   `https://<your-laptop's-Wi-Fi-IP>:8443/app/`.
3. Open that URL in mobile Chrome on the phone.
4. Tap *Advanced → Proceed* on the cert warning (one-time per device).
5. Allow camera permission when prompted.
6. Either tap *Record New Video* and record some road footage, or tap
   *Upload existing video* to pick a video from the gallery.
7. Tap *Process video* and wait. Status updates as it goes from
   `queued` → `processing` → `completed`.
8. The results screen shows the annotated playback, road material/
   unevenness card, summary card, and per-pothole list. Each pothole
   row has a *View 3D Tomography* button that opens the interactive
   3D mesh in a new tab.

---

## 6. Smoke tests

`server/scripts/` contains four smoke tests at progressively
increasing levels:

#### `verify_weights.py`

Loads each `.pt` file via Ultralytics and asserts the expected `task`
(segment vs detect) and the expected class list. Run this first after
checkout to confirm the weight files are valid.

#### `smoke_test_phase2.py`

Runs the single-image pipeline (segmentation, depth, plane fit,
volumetric) on `legacy/InfraSight/test8.png` and prints a
per-pothole summary. Validates the geometric reasoning stack in
isolation (no tracking, no annotation, no HTTP).

#### `smoke_test_phase3.py`

Runs the full video pipeline (with tracking, annotation, transcoding,
and 3D mesh rendering) on `legacy/InfraSight/road_test.mp4`. Writes
the annotated MP4 and a JSON report to `server/data/results/`. Does
not exercise the HTTP server.

#### `smoke_test_phase4.py`

Assumes a server is already running on `http://127.0.0.1:8000`.
Exercises the full HTTP flow: upload, poll until completed, fetch
result JSON, download annotated MP4. Run this last to confirm the
client-server integration.

---

## 7. Where to look when something goes wrong

| Symptom | Where to look |
|---|---|
| Server fails to start, "weights file missing" | `server/weights/*.pt` paths and `config.yaml` paths |
| `verify_weights.py` complains about task or class mismatch | `.pt` file is corrupted or wrong file copied |
| Server starts but `/health` doesn't respond | Check stdout for tracebacks during model load |
| Phone can't reach the server | Wi-Fi: are phone and laptop on the same network and subnet? Cert: did you regenerate after a Wi-Fi change? |
| Phone shows cert warning even after Proceed | Cert SAN doesn't include the laptop's current IP; re-run `tools/gen_dev_cert.py` |
| Pipeline crashes inside ffmpeg | `ffmpeg` not on PATH, or input video is malformed; check `data/results/<job_id>/` for partial output |
| Mesh HTML loads but is blank | Browser blocked the Plotly CDN; try opening with internet access |
| Mesh PNG endpoint returns 404 | `kaleido` is not installed; `pip install kaleido==0.2.1` and re-run the failed job |
| Material classifier section missing from report | `material_classifier.pt` not present and `optional: true`; copy the file or set `optional: false` to fail loudly |
| Video plays in VLC but not in mobile Chrome | Output transcoding failed; check `data/results/<job_id>/` and look for `annotated.mp4` vs `annotated.h264.mp4` |
| Tracker over-counting | Lower `min_observations`; tighten `min_avg_depth_cm` / `min_area_cm2`; or accept and post-process |

---

## 8. Tips for extending the system

- **New mobile device intrinsics.** Add an entry to
  `intrinsics.devices` in `config.yaml` with the device's
  `focal_length_mm`, `sensor_width_mm`, and `sensor_height_mm`. No
  code change needed.
- **New currency.** Edit `repair.currency` and the per-material
  `price_per_kg` and per-method `labor` values in `config.yaml`.
- **Different severity thresholds.** Edit
  `severity.thresholds.{low,medium,high}` in `config.yaml`.
- **Higher accuracy at the cost of more inference.** Reduce
  `pipeline.frame_stride` from 3 to 1 (every frame), increase
  `pipeline.ground_plane.ransac_iterations`, or change the depth model
  to `Depth-Anything-V2-Metric-Indoor-Base-hf`.
- **Different language for the front-end.** All user-visible strings
  live in `server/static/app/index.html` and `js/results.js`. There is
  no i18n framework; replace the strings directly.
- **A new repair material.** Add an entry to `repair.materials` with
  density, price, and compaction factor. Update `repair_advisor.py`'s
  `_select_material` if you want it to be auto-selected.

---

## 9. Project file inventory

A complete listing of every meaningful file is below for reference.
Files marked (gen.) are auto-generated and should not be hand-edited.

```
server/
  app/__init__.py                      empty package marker
  app/main.py                          FastAPI app + lifespan
  app/schemas.py                       Pydantic API models
  app/routes/__init__.py
  app/routes/analyze.py                analysis endpoints
  app/routes/health.py                 liveness endpoint
  app/models/__init__.py
  app/models/pothole_segmenter.py      YOLO seg wrapper
  app/models/crack_classifier.py       YOLO det wrapper
  app/models/depth_metric.py           HuggingFace depth wrapper
  app/models/material_classifier.py    multi-head MobileNetV3 wrapper
  app/physics/__init__.py
  app/physics/intrinsics.py            K from EXIF; back-projection
  app/physics/ground_plane.py          RANSAC plane fitter
  app/physics/volumetric.py            per-pothole geometry
  app/physics/severity.py              severity scoring
  app/physics/repair_advisor.py        repair method + cost
  app/worker/__init__.py
  app/worker/models_registry.py        4-model singleton
  app/worker/job_store.py              SQLite persistence
  app/worker/job_runner.py             async runner
  app/worker/pipeline.py               video pipeline orchestrator
  app/worker/tracker.py                IoU trackers
  app/worker/annotator.py              per-frame drawing
  app/visualization/__init__.py
  app/visualization/mesh_engine.py     3D Plotly tomography
  app/utils/__init__.py
  app/utils/config.py                  YAML loader
  app/utils/logger.py                  stdout logger
  app/utils/video_io.py                ffmpeg + OpenCV wrappers
  app/utils/imu.py                     sensors.json parser
  config/config.yaml                   all tunables
  weights/.gitkeep                     placeholder
  weights/pothole_seg.pt               (copy from legacy/)
  weights/crack_typology.pt            (copy from legacy/)
  weights/material_classifier.pt       (copy from training/)
  data/uploads/.gitkeep
  data/results/.gitkeep
  data/dev_cert/                       (gen.) self-signed TLS
  data/jobs.db                         (gen.) SQLite job store
  static/app/index.html                PWA shell
  static/app/manifest.json             PWA manifest
  static/app/icon-192.png
  static/app/icon-512.png
  static/app/css/style.css             dark theme
  static/app/js/api.js                 REST API wrapper
  static/app/js/recorder.js            camera + IMU capture
  static/app/js/results.js             results renderer
  static/app/js/app.js                 view router + flow
  tools/gen_dev_cert.py                self-signed cert generator
  scripts/verify_weights.py            sanity check
  scripts/smoke_test_phase2.py         single-image smoke test
  scripts/smoke_test_phase3.py         full video smoke test
  scripts/smoke_test_phase4.py         full HTTP smoke test
  run_server.py                        HTTP launcher
  run_server_https.py                  HTTPS launcher
  requirements.txt                     pip dependencies

training/material_classifier/
  peek_zip.py                          inspect RSCD zip
  curate.py                            extract train+val
  extract_test_from_vali.py            extract test set
  train.py                             multi-head MobileNetV3 trainer
  eval.py                              test-set evaluation
  runs/v1/material_classifier_best.pt  (gen.) trained weights
  runs/v1/metrics.json                 (gen.) per-epoch training log
  runs/v1/test_metrics.json            (gen.) test-set metrics
  data/curated/                        (gen.) extracted images + CSVs

legacy/                                 read-only reference; not at runtime
  InfraSight/                           pothole-volumetric reference
  road-anomaly-detection/               crack-typology reference

report/
  main.tex                             IEEE document root
  references.bib                       bibliography
  sections/                            10 section files + 4 appendices
  figures/                             PNG figures
  generate_figures.py                  regenerate data-driven figures
  generate_extra_figures.py            additional figures
  build.sh, build.bat                  compile pdflatex
  README.md
  main.pdf                             (gen.) compiled report
```
