# Pothole Volumetric Measurement & Repair Cost Estimation from Smartphone Video

> Undergraduate Project (UGP), IIT Kanpur — Even Semester 2025–26

A system that takes a smartphone video of a road, detects potholes and cracks in real time, estimates the physical volume of each pothole in cubic centimeters, classifies the road surface material, and produces a per-pothole repair cost estimate in INR — all from a single handheld phone recording. No LiDAR, no special hardware.

---

## What it does

You open the app on your phone, record a short video of the road (or upload one from your gallery), hit "Process", and get back:

- An **annotated video** with colour-coded severity overlays on every pothole and bounding boxes on every crack
- **Per-pothole measurements**: area (cm²), average depth (cm), max depth (cm), volume (cm³)
- **Severity classification** (Low / Medium / High / Critical) based on a weighted depth–area–volume composite score
- **Repair recommendations**: method (throw-and-roll / semi-permanent / full-depth), material quantity in kg, and cost in INR
- An interactive **3D tomography mesh** for each pothole, viewable in the browser
- **Road surface classification**: material (asphalt / concrete / mud / gravel) and unevenness (smooth / slight / severe)
- **Crack typology**: longitudinal, transverse, and alligator crack counts

Everything runs locally on a laptop with a GPU — no cloud, no API keys, no internet required after the initial setup.

---

## Architecture at a glance

The system stacks four neural networks in a per-frame pipeline:

| Model | Task | Architecture |
|---|---|---|
| Pothole segmenter | Instance segmentation | YOLOv8-seg, 3 classes |
| Crack typology detector | Object detection | YOLOv8-det, 4 classes |
| Metric depth estimator | Monocular depth | Depth Anything V2 (HuggingFace) |
| Road surface classifier | Multi-label classification | MobileNetV3-Small, dual-head |

After the neural network stage, a geometry pipeline takes over:
1. Back-project the depth map to a 3D point cloud using camera intrinsics
2. Fit a ground plane via RANSAC (1000 iterations, SVD-refined)
3. Compute per-pothole volume by integrating signed distances to the plane over the segmentation mask
4. Track potholes across frames using greedy mask-IoU matching
5. Classify severity and compute repair cost from configurable material densities, prices, and labour rates

The server is FastAPI + SQLite, the client is a Progressive Web App that works on any phone with a browser.

---

## Getting started

### Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (tested on RTX 4050 Laptop, RTX 3060; CPU-only will work but expect ~10x slower inference)
- **ffmpeg** on your PATH (used for video normalization and transcoding)
- A phone on the same Wi-Fi as your laptop (for the mobile demo)

### 1. Clone and set up the environment

```bash
git clone https://github.com/prathamg007/Pothole-Volumetric-Measurement.git
cd Pothole-Volumetric-Measurement

# Create a virtualenv (or use conda, whatever you prefer)
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

# Install PyTorch with CUDA first (adjust the URL for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install everything else
pip install -r server/requirements.txt
```

### 2. Get the model weights

The `.pt` weight files are too large for git. You need three files in `server/weights/`:

| File | Size | Source |
|---|---|---|
| `pothole_seg.pt` | ~6.5 MB | YOLOv8-seg trained on [Roboflow pothole-segmentation-g6hbh v14](https://universe.roboflow.com/pothol/pothole-segmentation-g6hbh) |
| `crack_typology.pt` | ~85 MB | YOLOv8-det for crack classification |
| `material_classifier.pt` | ~5 MB | MobileNetV3-Small dual-head, trained on the RSCD dataset |

Download them from the [Releases](https://github.com/prathamg007/Pothole-Volumetric-Measurement/releases) page and place them in `server/weights/`. Or if you have access to the original training artifacts, copy them from there.

You can verify the weights loaded correctly:

```bash
python server/scripts/verify_weights.py
```

### 3. Generate the HTTPS certificate

The phone needs HTTPS to access the camera (browser security policy). Generate a self-signed cert:

```bash
python server/tools/gen_dev_cert.py
```

This creates `server/data/dev_cert/cert.pem` and `key.pem`, valid for one year. The cert covers `localhost` and every LAN IP on your machine. Re-run this if you switch Wi-Fi networks.

### 4. Start the server

```bash
# For phone access (HTTPS on port 8443):
python server/run_server_https.py

# For local-only testing (HTTP on port 8000):
python server/run_server.py
```

First startup takes ~15 seconds while the models load. You'll see:

```
20:11:32 [INFO] models: Model registry ready
20:11:32 [INFO] main: Ready to accept requests
INFO:     Uvicorn running on https://0.0.0.0:8443 (Press CTRL+C to quit)
```

### 5. Open on your phone

1. Make sure the phone and laptop are on the **same Wi-Fi**
2. Open `https://<your-laptop-ip>:8443/app/` in Chrome on the phone
3. Tap through the certificate warning (one time per device)
4. Allow camera permission
5. Record a video of some road, or upload an existing one
6. Tap **Process Video** and wait for results

---

## Project structure

```
├── server/
│   ├── app/
│   │   ├── main.py                 FastAPI app + lifespan
│   │   ├── schemas.py              Pydantic request/response models
│   │   ├── routes/                 REST API endpoints
│   │   ├── models/                 Neural network wrappers
│   │   │   ├── pothole_segmenter.py
│   │   │   ├── crack_classifier.py
│   │   │   ├── depth_metric.py
│   │   │   └── material_classifier.py
│   │   ├── physics/                Geometric measurement code
│   │   │   ├── intrinsics.py         Camera K matrix + back-projection
│   │   │   ├── ground_plane.py       RANSAC plane fitting
│   │   │   ├── volumetric.py         Area / depth / volume computation
│   │   │   ├── severity.py           Severity scoring
│   │   │   └── repair_advisor.py     Repair cost estimation
│   │   ├── worker/                 Pipeline orchestration
│   │   │   ├── pipeline.py           End-to-end video processor
│   │   │   ├── tracker.py            Multi-frame pothole tracking
│   │   │   ├── annotator.py          Frame annotation + HUD
│   │   │   ├── job_store.py          SQLite job persistence
│   │   │   └── job_runner.py         Async job executor
│   │   ├── visualization/
│   │   │   └── mesh_engine.py        3D Plotly tomography renderer
│   │   └── utils/                  Config, logging, video I/O, IMU
│   ├── static/app/                 Progressive Web App (HTML/CSS/JS)
│   ├── config/config.yaml          All tunables in one place
│   ├── weights/                    Model weights (gitignored)
│   ├── scripts/                    Smoke tests
│   ├── tools/                      Certificate generator
│   ├── run_server.py               HTTP launcher
│   ├── run_server_https.py         HTTPS launcher
│   └── requirements.txt
│
├── training/
│   └── material_classifier/        Training scripts for the road surface classifier
│       ├── train.py                  MobileNetV3 dual-head training loop
│       ├── eval.py                   Evaluation + confusion matrices
│       └── curate.py                 RSCD dataset curation
│
└── CODE_DOCUMENTATION.md           Detailed module-by-module walkthrough
```

---

## Smoke tests

Run these in order after setup to make sure everything works:

```bash
# 1. Check that weight files are valid
python server/scripts/verify_weights.py

# 2. Single-image pipeline (no tracking, no HTTP)
python server/scripts/smoke_test_phase2.py

# 3. Full video pipeline (tracking + annotation, no HTTP)
python server/scripts/smoke_test_phase3.py

# 4. Full HTTP flow (start the server first)
python server/scripts/smoke_test_phase4.py
```

---

## Configuration

Everything is in `server/config/config.yaml`. The main things you might want to tweak:

- **`pipeline.frame_stride`**: Process every Nth frame. Default is 3 (every 3rd frame). Increase for faster processing, decrease for denser tracking.
- **`intrinsics.fallback_device`**: Camera calibration. Default is `oneplus_12`. Add your own phone's focal length and sensor size for better depth accuracy.
- **`severity.weights`**: The depth/area/volume weighting for the composite severity score.
- **`repair.materials`**: Material densities and per-kg prices. Update these if prices change.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze` | Upload a video (multipart), returns a job ID |
| `GET` | `/jobs` | List all jobs |
| `GET` | `/jobs/{id}` | Poll job status |
| `GET` | `/jobs/{id}/result` | Get the full JSON report |
| `GET` | `/jobs/{id}/video` | Stream the annotated video |
| `GET` | `/jobs/{id}/mesh/{track_id}` | Get the 3D mesh HTML for a pothole |
| `GET` | `/health` | Server health check |

---

## Training the material classifier

The road surface classifier was trained on the [Road Surface Classification Dataset (RSCD)](https://github.com/tsinghua-feit-vehicle-dynamics/RSCD) from Tsinghua University. If you want to retrain it:

```bash
# The training scripts are in training/material_classifier/
# You'll need to download the RSCD dataset separately and place it in
# training/material_classifier/data/

python training/material_classifier/train.py
python training/material_classifier/eval.py
```

The trained checkpoint goes to `server/weights/material_classifier.pt`.

---

## Known limitations

- Depth estimation uses the **indoor** variant of Depth Anything V2 (better for close-range <10m pothole shots). It may underperform on highway-speed captures where potholes are farther away.
- Camera intrinsics default to OnePlus 12 parameters. For best accuracy, add your phone's actual focal length and sensor dimensions to `config.yaml`.
- The system is designed for **walking-speed** or **slow-driving** recordings. Fast motion blur will degrade both segmentation and depth accuracy.
- 3D mesh PNG export requires the optional `kaleido` package. The interactive HTML meshes always work.

---

## Acknowledgements

- Pothole segmentation trained on the [Roboflow pothole-segmentation-g6hbh v14](https://universe.roboflow.com/pothol/pothole-segmentation-g6hbh) dataset (CC BY 4.0)
- Road surface classification trained on the [RSCD dataset](https://github.com/tsinghua-feit-vehicle-dynamics/RSCD) (CC BY-NC)
- Depth estimation via [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf)
- Built on [Ultralytics](https://github.com/ultralytics/ultralytics), [HuggingFace Transformers](https://github.com/huggingface/transformers), [FastAPI](https://fastapi.tiangolo.com/), [OpenCV](https://opencv.org/), and [Plotly](https://plotly.com/)

---

## License

This project was developed as part of an Undergraduate Project at IIT Kanpur. See individual dataset licenses above for data usage terms.
