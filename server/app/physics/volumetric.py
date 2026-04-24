"""Metric pothole measurements from depth map + ground plane. No magic constants."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.physics.ground_plane import Plane


@dataclass
class VolumetricResult:
    area_cm2: float
    avg_depth_cm: float
    max_depth_cm: float
    volume_cm3: float
    n_pothole_pixels: int


def measure_pothole(
    mask: np.ndarray,
    point_cloud: np.ndarray,
    plane: Plane,
) -> Optional[VolumetricResult]:
    """
    mask: (H, W) uint8/bool; 1 inside pothole
    point_cloud: (H, W, 3) camera coords in meters; NaN for invalid
    plane: ground plane (normal points toward camera; signed_distance > 0 for camera side)

    Returns None if the mask has no valid 3D points.
    """
    mask_b = mask.astype(bool)
    pot_pts = point_cloud[mask_b]
    valid = np.isfinite(pot_pts).all(axis=1)
    pot_pts = pot_pts[valid]
    if len(pot_pts) < 10:
        return None

    signed = pot_pts @ plane.normal + plane.d
    depths_m = np.clip(-signed, a_min=0.0, a_max=None)
    avg_depth_m = float(depths_m.mean())
    max_depth_m = float(depths_m.max())

    # Project pothole points onto the ground plane
    proj = pot_pts - signed[:, None] * plane.normal

    # Build a 2D basis on the plane
    n = plane.normal
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)

    centroid = proj.mean(axis=0)
    uv = np.stack([(proj - centroid) @ u, (proj - centroid) @ v], axis=-1)

    area_m2 = _convex_hull_area(uv)
    if area_m2 is None or area_m2 <= 0:
        return None

    volume_m3 = area_m2 * avg_depth_m

    return VolumetricResult(
        area_cm2=float(area_m2 * 1e4),
        avg_depth_cm=float(avg_depth_m * 100.0),
        max_depth_cm=float(max_depth_m * 100.0),
        volume_cm3=float(volume_m3 * 1e6),
        n_pothole_pixels=int(len(pot_pts)),
    )


def _convex_hull_area(points_2d: np.ndarray) -> Optional[float]:
    if len(points_2d) < 3:
        return None
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points_2d)
        return float(hull.volume)  # For 2D, .volume is area
    except Exception:
        return None
