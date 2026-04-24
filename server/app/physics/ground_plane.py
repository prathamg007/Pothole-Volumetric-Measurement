"""RANSAC ground plane fitting on a 3D point cloud."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Plane:
    """Plane equation: n · p + d = 0, unit normal n, oriented so camera (origin) has signed distance > 0."""

    normal: np.ndarray
    d: float
    inlier_ratio: float

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """points: (..., 3). Returns signed distance to plane (camera side = positive, below = negative)."""
        orig_shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        dist = flat @ self.normal + self.d
        return dist.reshape(orig_shape)

    @property
    def camera_height_m(self) -> float:
        """Perpendicular distance from camera origin to plane, in meters."""
        return abs(float(self.d))


def fit_ground_plane(
    points: np.ndarray,
    exclude_mask: Optional[np.ndarray] = None,
    iterations: int = 1000,
    threshold_m: float = 0.05,
    min_inlier_ratio: float = 0.15,
    max_depth_m: Optional[float] = 15.0,
    rng_seed: int = 42,
    debug: bool = False,
) -> Optional[Plane]:
    """
    RANSAC plane fit on a (H, W, 3) point cloud.

    points: camera-coord point cloud (NaN for invalid pixels).
    exclude_mask: (H, W) boolean/uint8, pixels to exclude (e.g. pothole masks).
    threshold_m: inlier distance threshold in meters.
    min_inlier_ratio: minimum inliers/candidates to accept.
    max_depth_m: pixels with depth beyond this are excluded (reduces sky/far-field noise).

    Returns a Plane with normal pointing from the ground TOWARD the camera.
    """
    H, W, _ = points.shape
    flat = points.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1)

    if exclude_mask is not None:
        flat_excl = np.asarray(exclude_mask).reshape(-1).astype(bool)
        valid &= ~flat_excl

    if max_depth_m is not None:
        valid &= flat[:, 2] < max_depth_m

    pts = flat[valid]
    N = len(pts)
    if debug:
        import sys
        print(f"[ground_plane] candidates: {N} pts (valid+near)", file=sys.stderr)
    if N < 100:
        return None

    rng = np.random.default_rng(rng_seed)
    best_inliers = 0
    best = None

    for _ in range(iterations):
        idx = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = pts[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, p1))

        dists = np.abs(pts @ normal + d)
        n_in = int(np.sum(dists < threshold_m))
        if n_in > best_inliers:
            best_inliers = n_in
            best = (normal, d, pts[dists < threshold_m])

    if best is None:
        return None

    # Refine with SVD on inliers
    _, _, inlier_pts = best
    centroid = inlier_pts.mean(axis=0)
    centered = inlier_pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    refined_normal = vh[-1]
    refined_normal = refined_normal / np.linalg.norm(refined_normal)
    refined_d = -float(np.dot(refined_normal, centroid))

    # Recompute inliers on refined plane
    dists = np.abs(pts @ refined_normal + refined_d)
    final_in = int(np.sum(dists < threshold_m))
    ratio = final_in / N
    if debug:
        import sys
        print(f"[ground_plane] best inliers={best_inliers}/{N} ({best_inliers/N:.1%}); refined inliers={final_in} ({ratio:.1%})", file=sys.stderr)
    if ratio < min_inlier_ratio:
        return None

    # Orient normal so camera (origin) is on the positive side (signed distance at origin = d > 0)
    if refined_d < 0:
        refined_normal = -refined_normal
        refined_d = -refined_d

    return Plane(normal=refined_normal.astype(np.float32), d=float(refined_d), inlier_ratio=float(ratio))
