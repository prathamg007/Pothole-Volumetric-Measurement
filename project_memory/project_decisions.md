---
name: UGP_amarb technical decisions
description: Locked-in technical decisions for the merged mobile-app + server project, as of 2026-04-21
type: project
originSessionId: 70bde1f0-f541-435a-bc1c-5a78507a2792
---
User confirmed these decisions on 2026-04-21 after codebase analysis. These are firm unless user explicitly revises.

**Scope:**
- Pothole: full measurement (cm² area, cm depth, cm³ volume, severity, repair material + cost)
- Cracks: detection + typology only (Longitudinal / Transverse / Alligator). NO crack area or volume for now. No exceptions.
- Road material classification: asphalt / concrete / paving-block (will train ourselves — weights don't exist)
- Repair cost/material recommendations in INR

**Model lineup:**
- InfraSight `phase1_segmentation_v1.pt` — pothole segmentation (masks)
- road-anomaly `YOLOv8_Small_2nd_Model.pt` — crack typology (3 classes used, Potholes class ignored)
- road-anomaly `best.pt` (Model 1) — DROPPED
- New material classifier — to be trained

**Architecture:** upload-to-server
- Mobile: **Flutter** (Android-only target — OnePlus 12)
- Backend: **FastAPI**, runs on user's laptop
- Mobile uploads video + IMU sidecar JSON over WiFi on local network
- Server processes async, mobile polls for result
- Demo scenario: laptop + phone on same WiFi, no cloud deployment

**Measurement strategy:**
- Replace Depth Anything V2 Relative → **metric depth model** (Depth Anything V2 Metric or ZoeDepth) to kill the `30.0` magic calibration constant
- Compute camera intrinsic matrix K per-video from EXIF / video metadata, not hardcoded
- **Height is NOT a user input and NOT integrated from IMU.** It's derived from the metric depth map: back-project pixels to 3D using K+depth → RANSAC ground plane → perpendicular distance from camera origin to that plane IS the camera height at that instant. Self-calibrating per frame.
- IMU role: sanity-check ground plane normal ≈ gravity; fall back to IMU pitch when plane detection fails; reject sky-facing frames.
- Area calculation: project pothole mask pixels to 3D world coords using K+depth, compute polygon area on the detected ground plane. No separate homography path needed once depth is metric.
- Skip reference-object approach (UX is bad)
- Skip ARCore depth (too much native mobile work)
- Target accuracy: ±15% on pothole depth/volume — acceptable and documented

**Result output format:**
- Individual per-pothole list: id, timestamp in video, area cm², depth cm, volume cm³, severity level, repair method, material type + kg, cost INR
- Summary: total potholes, total damaged area, total volume, total repair cost INR, overall road condition rating

**Why:** User explicitly accepted ±15% accuracy, wants simple and accessible, prefers local-laptop demo, no cloud. Mobile framework choice is open but Flutter is simplest given "any simple option is fine."

**How to apply:** When implementing, preserve the clean module boundaries from InfraSight (models/, core/, visualization/). Delete Model 1 paths. Migrate repair_advisor.py constants to config.yaml. Don't waste time on reference-object workflow or ARCore.
