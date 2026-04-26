"""Read the sensors.json sidecar captured by the mobile PWA and derive
the gravity direction in the camera's coordinate frame.

The PWA captures DeviceMotionEvent samples during recording. Each sample
has `accelerationIncludingGravity` in *device* coordinates (Android/iOS
DOM convention):
    +X = to the right of the screen
    +Y = toward the top of the phone
    +Z = out of the screen toward the user

Because `accelerationIncludingGravity` reports the *specific force* on the
device (the reaction force opposite to gravity), the averaged vector points
"up" in the world when the phone is nearly stationary. Gravity direction is
its negation.

For the back camera held in portrait pointing forward, the device → OpenCV-
camera frame rotation is:
    camera +X = device +X      (right)
    camera +Y = -device +Y     (camera Y points DOWN, device Y points UP)
    camera +Z = -device +Z     (camera Z points INTO scene, device Z points toward user)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Device → camera rotation for back camera held in portrait mode
_R_DEVICE_TO_CAMERA = np.array(
    [[1, 0, 0],
     [0, -1, 0],
     [0, 0, -1]],
    dtype=np.float32,
)


@dataclass
class ImuSummary:
    sensors_present: bool
    samples_count: int = 0
    gravity_camera: Optional[list[float]] = None  # unit vector in camera frame, "down"
    gravity_magnitude_ms2: Optional[float] = None
    duration_ms: Optional[float] = None
    source: Optional[str] = None  # e.g. "uploaded_file" if no live IMU


def _load_sensors(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize(sensors_path: Path) -> ImuSummary:
    data = _load_sensors(sensors_path)
    if data is None:
        return ImuSummary(sensors_present=False)

    samples = data.get("samples") or []
    source = data.get("source")
    duration_ms = data.get("duration_ms")
    if not samples:
        return ImuSummary(sensors_present=True, samples_count=0, duration_ms=duration_ms, source=source)

    vecs: list[tuple[float, float, float]] = []
    for s in samples:
        ax = s.get("ax_g")
        ay = s.get("ay_g")
        az = s.get("az_g")
        if ax is None or ay is None or az is None:
            continue
        vecs.append((float(ax), float(ay), float(az)))

    if len(vecs) < 5:
        return ImuSummary(sensors_present=True, samples_count=len(vecs), duration_ms=duration_ms, source=source)

    arr = np.asarray(vecs, dtype=np.float32)              # (N, 3) device frame
    avg_device_up = arr.mean(axis=0)                      # average "up" reaction vector
    avg_camera_up = _R_DEVICE_TO_CAMERA @ avg_device_up
    gravity_camera = -avg_camera_up                       # world "down" in camera frame
    magnitude = float(np.linalg.norm(gravity_camera))
    if magnitude < 1e-6:
        return ImuSummary(sensors_present=True, samples_count=len(vecs), duration_ms=duration_ms, source=source)
    unit = (gravity_camera / magnitude).tolist()
    return ImuSummary(
        sensors_present=True,
        samples_count=len(vecs),
        gravity_camera=[float(x) for x in unit],
        gravity_magnitude_ms2=magnitude,
        duration_ms=duration_ms,
        source=source,
    )


def angle_between_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two 3D vectors in degrees."""
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    cos_a = float(np.dot(a, b) / (na * nb))
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))
