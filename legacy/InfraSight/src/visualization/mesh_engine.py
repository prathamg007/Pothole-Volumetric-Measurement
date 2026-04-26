"""
mesh_engine.py — Premium 3D Pothole Visualizer
Uses actual per-pothole depth data so each visualization reflects
the real shape of that specific pothole.
"""
import numpy as np
import plotly.graph_objects as go
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image
from typing import Optional
from src.utils.logger import setup_logger

logger = setup_logger("MeshEngine")


class Mesh3DVisualizer:
    """Create interactive 3D mesh visualizations of potholes using Plotly."""

    # ------------------------------------------------------------------
    # Basic mesh (full depth map, optional mask)
    # ------------------------------------------------------------------
    @staticmethod
    def create_pothole_mesh(
        depth_map: np.ndarray,
        pothole_mask: Optional[np.ndarray] = None,
        image_rgb: Optional[np.ndarray] = None,
        title: str = "Pothole 3D Profile",
        colorscale: str = "Viridis"
    ) -> go.Figure:
        """
        Create an interactive 3D surface plot of a pothole.

        Args:
            depth_map:     Depth map (H, W) — inverted internally to show depression.
            pothole_mask:  Optional binary mask; non-pothole pixels set to NaN.
            image_rgb:     Optional RGB image for texture mapping.
            title:         Plot title.
            colorscale:    Plotly colorscale name.

        Returns:
            Plotly Figure.
        """
        z_values = -depth_map.copy().astype(float)

        if pothole_mask is not None and image_rgb is None:
            z_values[pothole_mask == 0] = np.nan

        if image_rgb is not None:
            img_pil = Image.fromarray(image_rgb)
            img_quant = img_pil.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
            palette = np.array(img_quant.getpalette()[:256 * 3]).reshape(-1, 3)
            custom_colorscale = [
                [i / 255.0, f"rgb({r},{g},{b})"]
                for i, (r, g, b) in enumerate(palette)
            ]
            surfacecolor = np.array(img_quant).astype(float)
            active_colorscale = custom_colorscale
            cmin, cmax = 0, 255
            showscale = False
        else:
            surfacecolor = z_values
            active_colorscale = colorscale
            cmin, cmax = None, None
            showscale = True

        fig = go.Figure(data=[go.Surface(
            z=z_values,
            surfacecolor=surfacecolor,
            colorscale=active_colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=showscale,
            lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.9, specular=0.1),
            colorbar=dict(title="Depth (relative)") if showscale else None
        )])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (pixels)",
                yaxis_title="Y (pixels)",
                zaxis_title="Depth (relative)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            width=700,
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig

    # ------------------------------------------------------------------
    # Cropped mesh (zoomed to pothole bbox)
    # ------------------------------------------------------------------
    @staticmethod
    def create_pothole_mesh_cropped(
        depth_map: np.ndarray,
        pothole_mask: np.ndarray,
        image_rgb: Optional[np.ndarray] = None,
        padding: int = 20
    ) -> go.Figure:
        """
        Create a 3D mesh cropped and zoomed to the pothole region.

        Args:
            depth_map:    Full depth map.
            pothole_mask: Binary mask of the pothole.
            image_rgb:    Optional RGB image for texture.
            padding:      Pixels to pad around the pothole bounding box.

        Returns:
            Plotly Figure with cropped view.
        """
        coords = np.where(pothole_mask == 1)
        if len(coords[0]) == 0:
            raise ValueError("Pothole mask is empty.")

        y_min = max(0, coords[0].min() - padding)
        y_max = min(depth_map.shape[0], coords[0].max() + padding)
        x_min = max(0, coords[1].min() - padding)
        x_max = min(depth_map.shape[1], coords[1].max() + padding)

        depth_cropped = depth_map[y_min:y_max, x_min:x_max]
        mask_cropped  = pothole_mask[y_min:y_max, x_min:x_max]

        img_cropped = None
        if image_rgb is not None:
            raw = image_rgb[y_min:y_max, x_min:x_max]
            if raw.size > 0:
                img_cropped = cv2.resize(
                    raw,
                    (depth_cropped.shape[1], depth_cropped.shape[0])
                )

        return Mesh3DVisualizer.create_pothole_mesh(
            depth_cropped,
            mask_cropped,
            image_rgb=img_cropped,
            title="Pothole 3D Profile (Zoomed)",
            colorscale="Inferno"
        )

    # ------------------------------------------------------------------
    # Premium mesh — data-driven, unique per pothole
    # ------------------------------------------------------------------
    @staticmethod
    def create_premium_pothole_mesh(
        depth_map: np.ndarray,
        pothole_mask: np.ndarray,
        image_rgb: Optional[np.ndarray] = None,
        metrics: Optional[dict] = None,
        padding: int = 50
    ) -> go.Figure:
        """
        Premium 3D Visualizer — ground-relative depth rendering.

        The pothole is rendered as a **depression going downward** from
        a visible flat road surface (Z = 0).  Uses the actual depth map
        data so every pothole has a unique, realistic shape.

        Key design decisions:
        - Z = 0 is the road surface (ground plane).
        - Z < 0 is the pothole interior (deeper = more negative).
        - The surrounding road is shown as a flat surface giving context.
        - Edge-blend via distanceTransform avoids hard cliffs at the mask boundary.
        - Colorscale: green (road) → yellow → red (deepest).

        Args:
            depth_map:    Full relative depth map (H, W), float32, [0–1].
            pothole_mask: Binary mask (H, W) of this specific pothole.
            image_rgb:    Unused (reserved for future texture overlay).
            metrics:      Dict with keys 'depth', 'area', 'severity' for annotation.
            padding:      Pixels to pad around the pothole bounding box.

        Returns:
            Plotly Figure.
        """
        # ── 1. Crop to pothole region ──────────────────────────────────
        coords = np.where(pothole_mask == 1)
        if len(coords[0]) == 0:
            raise ValueError("Pothole mask is empty.")

        y_min = max(0, coords[0].min() - padding)
        y_max = min(depth_map.shape[0], coords[0].max() + padding)
        x_min = max(0, coords[1].min() - padding)
        x_max = min(depth_map.shape[1], coords[1].max() + padding)

        depth_cropped = depth_map[y_min:y_max, x_min:x_max].astype(np.float32)
        mask_cropped  = pothole_mask[y_min:y_max, x_min:x_max]
        h_c, w_c = depth_cropped.shape

        # ── 2. Ground-relative Z transform ─────────────────────────────
        #    Ground plane = median depth of pixels OUTSIDE the mask
        #    (the surrounding road surface in the padded crop).
        outside = depth_cropped[mask_cropped == 0]
        if outside.size > 0:
            ground_level = np.median(outside)
        else:
            # Fallback: use overall median
            ground_level = np.median(depth_cropped)

        # Depth Anything V2: higher value = farther from camera.
        # For a pothole, the bottom is farther → higher depth value.
        # We want the hole to go DOWNWARD (negative Z), so:
        #   Z = -(depth - ground)
        # Road pixels: depth ≈ ground → Z ≈ 0
        # Pothole pixels: depth > ground → Z < 0 (goes down)
        z_raw = -(depth_cropped - ground_level)

        # ── 3. Smooth to reduce sensor noise ───────────────────────────
        z_smooth = gaussian_filter(z_raw, sigma=0.8)

        # ── 4. Edge-blend at mask boundary ─────────────────────────────
        #    Smoothly transition from road (Z=0) to pothole depth.
        #    Outside the mask → force Z to 0 (flat road).
        #    Near the edge → gradual blend.
        blend_width = max(3.0, min(h_c, w_c) * 0.06)
        dist_inside = cv2.distanceTransform(
            mask_cropped.astype(np.uint8), cv2.DIST_L2, 5
        )
        blend_factor = np.clip(dist_inside / blend_width, 0.0, 1.0)

        # Road = 0, pothole interior = actual negative depth
        z_values = np.where(
            mask_cropped == 1,
            z_smooth * blend_factor,       # inside: real depth, blended at edges
            0.0                            # outside: flat road (Z = 0)
        )

        # Add subtle road-level noise so the flat road doesn't look artificial
        road_noise = gaussian_filter(
            np.random.default_rng(42).normal(0, 0.002, (h_c, w_c)),
            sigma=3.0
        )
        z_values = np.where(mask_cropped == 0, z_values + road_noise, z_values)

        # ── 5. Depth-intuitive colorscale ──────────────────────────────
        #    Green/gray = road level (Z ≈ 0)
        #    Yellow = shallow
        #    Orange → Red = deep
        depth_colorscale = [
            [0.00, "#1a9641"],   # deep green  (deepest — most negative Z)
            [0.15, "#d7191c"],   # red
            [0.30, "#fdae61"],   # orange
            [0.50, "#fee08b"],   # yellow
            [0.70, "#d9ef8b"],   # light green (shallow)
            [0.85, "#a6d96a"],   # green
            [1.00, "#66bd63"],   # road level  (Z ≈ 0)
        ]

        # ── 6. Build figure ────────────────────────────────────────────
        fig = go.Figure()

        # Main surface — real pothole shape as depression
        fig.add_trace(go.Surface(
            z=z_values,
            surfacecolor=z_values,
            colorscale=depth_colorscale,
            lighting=dict(
                ambient=0.45,
                diffuse=0.85,
                roughness=0.5,
                specular=0.4,
                fresnel=0.2
            ),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=False,
                    color="rgba(255,255,255,0.15)",
                    width=1,
                    highlightcolor="rgba(255,255,255,0.3)",
                    project_z=False
                )
            ),
            colorbar=dict(
                title=dict(text="Depth", font=dict(size=12, color="white")),
                thickness=15,
                len=0.6,
                x=1.05,
                tickfont=dict(color="white")
            ),
            name="Pothole Surface"
        ))

        # ── 7. Road-level reference plane (semi-transparent) ───────────
        #    A flat plane at Z = 0 to clearly mark where the road is.
        road_plane_z = np.full((2, 2), 0.001)  # tiny offset above 0
        fig.add_trace(go.Surface(
            z=road_plane_z,
            x=[0, w_c - 1],
            y=[0, h_c - 1],
            surfacecolor=np.zeros((2, 2)),
            colorscale=[[0, "rgba(150,150,150,0.15)"], [1, "rgba(150,150,150,0.15)"]],
            showscale=False,
            opacity=0.3,
            name="Road Level (Z=0)",
            hoverinfo="name"
        ))

        # ── 8. Red highlight for deepest area ──────────────────────────
        pothole_z = z_values.copy()
        pothole_z[mask_cropped == 0] = np.nan

        if not np.all(np.isnan(pothole_z)):
            min_z = np.nanmin(pothole_z)  # most negative = deepest
            if min_z < 0:
                threshold = min_z * 0.80  # top 20% deepest
                deep_mask = (mask_cropped == 1) & (z_values <= threshold)
                z_deep = z_values.copy()
                z_deep[~deep_mask] = np.nan

                fig.add_trace(go.Surface(
                    z=z_deep - 0.003,  # slightly below to overlay
                    surfacecolor=np.full_like(z_values, 1.0),
                    colorscale=[[0, "#ef4444"], [1, "#dc2626"]],
                    showscale=False,
                    opacity=0.85,
                    name="Deepest Zone",
                    hoverinfo="name+z"
                ))

        # ── 9. Layout & camera ─────────────────────────────────────────
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(
                    title=dict(text="Depth ↓", font=dict(size=11, color="#94a3b8")),
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.06)",
                    zeroline=True,
                    zerolinecolor="rgba(100,200,100,0.4)",
                    zerolinewidth=2,
                ),
                camera=dict(
                    eye=dict(x=2.0, y=2.0, z=1.6),  # wider view, not too zoomed
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=-0.05)   # look slightly down
                ),
                aspectratio=dict(x=1, y=1, z=0.7)    # exaggerate depth
            ),
            margin=dict(l=0, r=0, b=0, t=45),
            title=dict(
                text="3D Pothole Tomography",
                x=0.5,
                font=dict(color="white", size=16)
            ),
            template="plotly_dark",
            height=520,
            legend=dict(
                font=dict(color="white", size=10),
                bgcolor="rgba(0,0,0,0.4)",
                x=0.01, y=0.01
            )
        )

        # ── 10. Metrics annotation ─────────────────────────────────────
        if metrics:
            sev_color = {
                "CRITICAL": "#ef4444",
                "HIGH":     "#f97316",
                "MEDIUM":   "#eab308",
                "LOW":      "#22c55e",
            }.get(str(metrics.get("severity", "")).upper(), "#ef4444")

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
                bordercolor=sev_color, borderwidth=2, borderpad=10
            )

        return fig


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    h, w = 300, 400

    # Simulate Depth Anything V2 output:
    # Higher values = farther from camera.
    # Road surface ≈ 0.5, pothole bottom = higher (farther away).
    depth_map = rng.random((h, w)).astype(np.float32) * 0.05 + 0.50  # road ~0.50

    # Irregular pothole shape
    pothole_mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt(((y - 150) / 60) ** 2 + ((x - 200) / 90) ** 2)
            if dist < 1.0:
                pothole_mask[y, x] = 1
                # Pothole is DEEPER → higher depth value
                depth_map[y, x] += 0.15 * max(0.0, 1.0 - dist)

    viz = Mesh3DVisualizer()
    fig = viz.create_premium_pothole_mesh(
        depth_map, pothole_mask,
        metrics={"depth": 5.2, "area": 320.0, "severity": "HIGH"}
    )
    fig.show()
