---
name: UGP_amarb codebase facts
description: Verified durable facts about both InfraSight and road-anomaly-detection codebases as of 2026-04-21 analysis
type: project
originSessionId: 70bde1f0-f541-435a-bc1c-5a78507a2792
---
Verified by full codebase analysis on 2026-04-21. Durable facts (model classes, blockers) worth keeping; transient bugs will be fixed so not saved here.

**road-anomaly-detection/ — TWO trained models with different class sets (verified by torch.load on 2026-04-21):**
- **Model 1** (`RoadDetectionModel/RoadModel_yolov8m.pt_rounds120_b9/weights/best.pt`, YOLOv8m, detect, 7 classes): Heavy-Vehicle, Light-Vehicle, Pedestrian, Crack, Crack-Severe, Pothole, Speed-Bump. Crack classification is BINARY severity.
- **Model 2** (`YOLOv8_Small_2nd_Model.pt`, detect, 4 classes): Longitudinal Crack, Transverse Crack, Alligator Crack, Potholes. This is the one with fine-grained crack typology.
- Model 1 has training metrics (mAP@.5:.95 cracks 0.24–0.27, pothole 0.20). Model 2 has no on-disk metrics — quality unknown.
- Test mAP@.5:.95 overall 0.448; crack/pothole classes weakest (0.20–0.27)
- Dataset NOT in repo — retraining requires re-sourcing
- Three redundant entry points: main.py (Streamlit, primary), run.py, run2model.py; also Flask alt at interface-app/app.py
- Original developer: Navneet Sharma (absolute Windows paths in args.yaml)

**InfraSight/**
- YOLOv8 segmentation, weights: `weights/phase1_segmentation_v1.pt` (6.5 MB)
- 3 classes: Manhole, Pothole, Unmarked Bump (Roboflow pothole-segmentation-g6hbh v14)
- Dataset IS in repo (2690 train / 300 val / 354 test)
- Full pipeline: YOLO → Depth Anything V2 → homography → volumetric → material classifier → severity → repair advisor → 3D Plotly
- Streamlit UI in webapp/app.py (~824 lines, fairly complete)
- Costs in IDR (Indonesian Rupiah), 2024–2025 rates — needs conversion to INR for user
- Config-driven via config/config.yaml

**Critical blockers for final "mobile + server" goal:**
1. Material classifier weights missing at `models/weights/road_material/material_classifier_v1.pt` — predictions are random silently
2. Calibration constant 30.0 in volumetric.py is a magic number — cm measurements unverified
3. Camera intrinsics hardcoded (focal 700px, 640×640, height 50cm, pitch 20°) — breaks for any phone camera
4. Depth Anything V2 returns RELATIVE depth [0,1] — cm conversion requires either reference object per frame OR known camera extrinsics
5. No server/API layer exists anywhere
6. No unified video pipeline that runs detection + segmentation + volumetric together
7. Volume in cm³ is physically ambiguous without per-frame calibration — honest limitation, not a bug

**Why:** User wants mobile app → upload video to server → processed video back with detection, classification, area/depth/volume in cm, material type, repair recommendations.

**How to apply:** When planning the rewrite, treat InfraSight as the measurement backbone (keep segmentation + volumetric + severity + repair), use Model 2 (`YOLOv8_Small_2nd_Model.pt`) for crack typology since it has longitudinal/transverse/alligator classes, likely drop Model 1 (its crack classes are weak and its pothole is redundant with InfraSight's segmentation). Build a FastAPI server layer on top. Be upfront that cm-accurate volume requires either reference object, known camera extrinsics, or ARCore/ARKit device-side depth — Depth Anything V2 alone gives relative depth only.
