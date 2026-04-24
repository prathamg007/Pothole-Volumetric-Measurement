"""Camera intrinsic matrix K from EXIF or device fallback, plus back-projection to 3D."""
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import piexif
except ImportError:
    piexif = None


def _focal_mm_from_exif(image_path: Path) -> Optional[float]:
    if piexif is None:
        return None
    try:
        exif = piexif.load(str(image_path))
        fl = exif.get("Exif", {}).get(piexif.ExifIFD.FocalLength)
        if fl is None:
            return None
        num, den = fl
        return float(num) / float(den) if den else None
    except Exception:
        return None


def compute_K(
    image_shape: tuple[int, int],
    image_path: Optional[Path] = None,
    device_cfg: Optional[dict] = None,
) -> np.ndarray:
    """
    Build 3x3 intrinsic matrix K.

    image_shape: (H, W)
    image_path: optional path to a JPEG/PNG with EXIF (for focal length)
    device_cfg: dict with keys focal_length_mm, sensor_width_mm (used as fallback / sensor dims)
    """
    img_h, img_w = image_shape
    focal_mm = _focal_mm_from_exif(image_path) if image_path else None

    if device_cfg is not None:
        sensor_w_mm = float(device_cfg["sensor_width_mm"])
        if focal_mm is None:
            focal_mm = float(device_cfg["focal_length_mm"])
        fx = (focal_mm / sensor_w_mm) * img_w
        fy = fx
    elif focal_mm is not None:
        # EXIF focal but no sensor size — assume 35mm full-frame as weak guess
        fx = (focal_mm / 36.0) * img_w
        fy = fx
    else:
        # Last resort: assume 60 degree horizontal FOV
        hfov = np.radians(60.0)
        fx = img_w / (2 * np.tan(hfov / 2))
        fy = fx

    cx = img_w / 2.0
    cy = img_h / 2.0

    return np.array(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        dtype=np.float32,
    )


def backproject(depth_map_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert metric depth map (meters, H, W) + K into a 3D point cloud (H, W, 3) in camera coords.

    Uses OpenCV convention: X right, Y down, Z forward (into scene).
    Pixels with depth <= 0 or NaN become NaN in the point cloud.
    """
    H, W = depth_map_m.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    Z = depth_map_m.astype(np.float32)
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    points = np.stack([X, Y, Z], axis=-1)
    invalid = ~np.isfinite(Z) | (Z <= 0)
    points[invalid] = np.nan
    return points
