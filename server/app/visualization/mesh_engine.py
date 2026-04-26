"""3D pothole tomography via Plotly.

Renders each finalized pothole as a depression below a flat road plane.
Ported and refactored from legacy/InfraSight/src/visualization/mesh_engine.py
with two changes:

1. The render function accepts already-cropped depth/mask/image, since the
   pipeline pre-crops to bbox+padding before saving (avoids holding the full
   1080x1920 depth map in memory per track).
2. Depth values are in meters (from the metric depth model) instead of the
   legacy normalized [0,1]. Plotly auto-scales the Z axis, so the visual
   result is comparable; the absolute Z values are now physically meaningful.

The output is a self-contained HTML file (Plotly bundled from CDN) that can
be opened directly or served via the FastAPI route.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter


# Severity → highlight color used in the corner annotation
_SEVERITY_COLOR = {
    "CRITICAL": "#ef4444",
    "HIGH": "#f97316",
    "MEDIUM": "#eab308",
    "LOW": "#22c55e",
}

# Depth-intuitive colorscale: green/light at road level, transitioning to
# red/orange as depth grows (more negative Z).
_DEPTH_COLORSCALE = [
    [0.00, "#1a9641"],   # deepest (most negative Z)
    [0.15, "#d7191c"],
    [0.30, "#fdae61"],
    [0.50, "#fee08b"],
    [0.70, "#d9ef8b"],
    [0.85, "#a6d96a"],
    [1.00, "#66bd63"],   # road level (Z ≈ 0)
]


def _crop_to_mask(
    depth_map: np.ndarray,
    pothole_mask: np.ndarray,
    image_rgb: Optional[np.ndarray],
    padding: int,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Crop depth_map / pothole_mask / image_rgb to the mask's bbox + padding."""
    coords = np.where(pothole_mask == 1)
    if len(coords[0]) == 0:
        raise ValueError("pothole_mask is empty; nothing to render")

    h, w = depth_map.shape[:2]
    y_min = max(0, int(coords[0].min()) - padding)
    y_max = min(h, int(coords[0].max()) + padding)
    x_min = max(0, int(coords[1].min()) - padding)
    x_max = min(w, int(coords[1].max()) + padding)

    depth_c = depth_map[y_min:y_max, x_min:x_max].astype(np.float32)
    mask_c = pothole_mask[y_min:y_max, x_min:x_max].astype(np.uint8)
    img_c = None
    if image_rgb is not None:
        raw = image_rgb[y_min:y_max, x_min:x_max]
        if raw.size > 0:
            img_c = cv2.resize(raw, (depth_c.shape[1], depth_c.shape[0]))
    return depth_c, mask_c, img_c


def render_premium_mesh(
    depth_cropped: np.ndarray,
    mask_cropped: np.ndarray,
    image_cropped: Optional[np.ndarray] = None,
    metrics: Optional[dict] = None,
) -> go.Figure:
    """Render the 3D pothole mesh, given pre-cropped inputs.

    Args:
        depth_cropped:  (h, w) float32, depth in metres (or any unit — Plotly
                        auto-scales Z, but the contour/highlight thresholds
                        are relative so the choice doesn't matter).
        mask_cropped:   (h, w) uint8 binary, 1 inside pothole.
        image_cropped:  (h, w, 3) uint8, optional (currently unused; reserved
                        for future texture overlay).
        metrics:        Optional dict with keys ``depth`` (cm), ``area``
                        (cm²), ``severity`` (LOW/MEDIUM/HIGH/CRITICAL).
                        Rendered as a corner annotation.

    Returns:
        Plotly Figure ready for ``write_html`` or ``show``.
    """
    h_c, w_c = depth_cropped.shape

    # ── 1. Ground plane = median of pixels OUTSIDE the mask ──────────────────
    outside = depth_cropped[mask_cropped == 0]
    ground_level = float(np.median(outside)) if outside.size > 0 else float(np.median(depth_cropped))

    # Depth-Anything-style convention: larger depth = farther from camera. A
    # pothole's bottom is farther → larger depth value. We want the hole to go
    # DOWNWARD (negative Z), so:  Z = -(depth - ground)
    z_raw = -(depth_cropped - ground_level)

    # ── 2. Smooth out per-pixel sensor noise ────────────────────────────────
    z_smooth = gaussian_filter(z_raw, sigma=0.8)

    # ── 3. Edge-blend at the mask boundary ──────────────────────────────────
    # Smoothly transition from road (Z=0) to pothole depth at the mask edge.
    blend_width = max(3.0, min(h_c, w_c) * 0.06)
    dist_inside = cv2.distanceTransform(mask_cropped.astype(np.uint8), cv2.DIST_L2, 5)
    blend_factor = np.clip(dist_inside / blend_width, 0.0, 1.0)

    z_values = np.where(
        mask_cropped == 1,
        z_smooth * blend_factor,    # inside: real depth, blended at edges
        0.0,                        # outside: flat road (Z = 0)
    )

    # Subtle road-level noise so the "outside" region doesn't look artificial
    rng = np.random.default_rng(42)
    road_noise = gaussian_filter(rng.normal(0, 0.002, (h_c, w_c)), sigma=3.0)
    z_values = np.where(mask_cropped == 0, z_values + road_noise, z_values)

    # ── 4. Build figure ─────────────────────────────────────────────────────
    fig = go.Figure()

    # Main pothole surface
    fig.add_trace(go.Surface(
        z=z_values,
        surfacecolor=z_values,
        colorscale=_DEPTH_COLORSCALE,
        lighting=dict(ambient=0.45, diffuse=0.85, roughness=0.5, specular=0.4, fresnel=0.2),
        contours=dict(z=dict(
            show=True, usecolormap=False,
            color="rgba(255,255,255,0.15)", width=1,
            highlightcolor="rgba(255,255,255,0.3)", project_z=False,
        )),
        colorbar=dict(
            title=dict(text="Depth", font=dict(size=12, color="white")),
            thickness=15, len=0.6, x=1.05, tickfont=dict(color="white"),
        ),
        name="Pothole Surface",
    ))

    # Semi-transparent road-level reference plane at Z = 0
    fig.add_trace(go.Surface(
        z=np.full((2, 2), 0.001),
        x=[0, w_c - 1],
        y=[0, h_c - 1],
        surfacecolor=np.zeros((2, 2)),
        colorscale=[[0, "rgba(150,150,150,0.15)"], [1, "rgba(150,150,150,0.15)"]],
        showscale=False, opacity=0.3,
        name="Road Level (Z=0)", hoverinfo="name",
    ))

    # Highlight the deepest 20% with a red overlay
    pothole_z = z_values.copy()
    pothole_z[mask_cropped == 0] = np.nan
    if not np.all(np.isnan(pothole_z)):
        min_z = float(np.nanmin(pothole_z))
        if min_z < 0:
            threshold = min_z * 0.80
            deep_mask = (mask_cropped == 1) & (z_values <= threshold)
            z_deep = z_values.copy()
            z_deep[~deep_mask] = np.nan
            fig.add_trace(go.Surface(
                z=z_deep - 0.003,    # offset slightly below to overlay
                surfacecolor=np.full_like(z_values, 1.0),
                colorscale=[[0, "#ef4444"], [1, "#dc2626"]],
                showscale=False, opacity=0.85,
                name="Deepest Zone", hoverinfo="name+z",
            ))

    # Layout / camera
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(
                title=dict(text="Depth ↓", font=dict(size=11, color="#94a3b8")),
                showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                zeroline=True, zerolinecolor="rgba(100,200,100,0.4)", zerolinewidth=2,
            ),
            camera=dict(
                eye=dict(x=2.0, y=2.0, z=1.6),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.05),
            ),
            aspectratio=dict(x=1, y=1, z=0.7),    # exaggerate depth for visibility
        ),
        margin=dict(l=0, r=0, b=0, t=45),
        title=dict(text="3D Pothole Tomography", x=0.5, font=dict(color="white", size=16)),
        template="plotly_dark",
        height=520,
        legend=dict(font=dict(color="white", size=10), bgcolor="rgba(0,0,0,0.4)", x=0.01, y=0.01),
    )

    # Corner annotation with measurements
    if metrics:
        sev_color = _SEVERITY_COLOR.get(str(metrics.get("severity", "")).upper(), "#ef4444")
        ann_text = (
            f"⬇ DEPTH: {metrics.get('depth', 0):.1f} cm<br>"
            f"📐 AREA: {metrics.get('area', 0):.1f} cm²<br>"
            f"⚠ SEVERITY: <span style='color:{sev_color};font-weight:700'>"
            f"{metrics.get('severity', 'N/A')}</span>"
        )
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98,
            text=ann_text, showarrow=False,
            font=dict(size=13, color="white", family="Inter, sans-serif"),
            align="left",
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor=sev_color, borderwidth=2, borderpad=10,
        )

    return fig


def render_pothole_mesh_to_html(
    depth_cropped: np.ndarray,
    mask_cropped: np.ndarray,
    out_path: Path,
    image_cropped: Optional[np.ndarray] = None,
    metrics: Optional[dict] = None,
) -> Path:
    """Convenience wrapper: render the mesh and write a self-contained HTML.

    Plotly is loaded from the public CDN so the HTML stays small (~50 KB);
    requires internet at view time.
    """
    fig = render_premium_mesh(depth_cropped, mask_cropped, image_cropped, metrics)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return out_path


def render_pothole_mesh_to_png(
    depth_cropped: np.ndarray,
    mask_cropped: np.ndarray,
    out_path: Path,
    image_cropped: Optional[np.ndarray] = None,
    metrics: Optional[dict] = None,
    width: int = 1000,
    height: int = 700,
) -> Optional[Path]:
    """Best-effort PNG export via kaleido. Returns None if kaleido is missing
    or the export fails (which is fine — HTML is the primary output).
    """
    try:
        fig = render_premium_mesh(depth_cropped, mask_cropped, image_cropped, metrics)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(out_path), width=width, height=height, scale=2)
        return out_path
    except Exception:
        return None
