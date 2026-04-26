import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
import cv2
from typing import Optional

def create_premium_3d_plot(
    depth_map: np.ndarray,
    pothole_mask: np.ndarray,
    image_rgb: Optional[np.ndarray] = None,
    metrics: dict = None
):
    """
    Prototipe Visualisasi 3D Pothole dengan standar 'Premium'.
    """
    # 1. Smoothing (Gaussian Blur) - Mencegah "kulit jeruk"
    z_smooth = gaussian_filter(depth_map, sigma=1.5)
    z_values = -z_smooth  # Invert for depression
    
    # 2. Highlight Area Pothole (Red Contour)
    # Mencari area paling dalam di dalam mask
    pothole_depths = z_values.copy()
    pothole_depths[pothole_mask == 0] = np.nan
    max_depth_val = np.nanmin(pothole_depths) # deepest is most negative
    
    # Buat surface color berdasarkan depth (Turbo colormap)
    surfacecolor = z_values
    
    # 3. Create Figure
    fig = go.Figure()

    # Base Surface
    fig.add_trace(go.Surface(
        z=z_values,
        surfacecolor=surfacecolor,
        colorscale='Turbo', # Premium colormap
        lighting=dict(
            ambient=0.5,
            diffuse=0.9,
            fresnel=0.5,
            roughness=0.4,
            specular=0.3
        ),
        colorbar=dict(
            title="Relative Depth",
            thickness=15,
            len=0.7,
            x=1.1
        )
    ))

    # 4. Highlight Deepest Area (Contour line at 90% depth)
    threshold = max_depth_val * 0.95
    contour_mask = (pothole_mask == 1) & (z_values <= threshold)
    z_contour = z_values.copy()
    z_contour[~contour_mask] = np.nan
    
    fig.add_trace(go.Surface(
        z=z_contour + 0.01, # slightly above to avoid clipping
        surfacecolor=np.full_like(z_values, 100), # Red
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False,
        name="Deepest Point"
    ))

    # 5. Camera Angle (Top-down + Slight Tilt)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title='Depth'),
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=0.8), # Slight tilt
                up=dict(x=0, y=0, z=1)
            ),
            aspectratio=dict(x=1, y=1, z=0.4)
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        title=dict(
            text="Premium 3D Tomography",
            font=dict(size=24, color="white"),
            x=0.5
        ),
        template="plotly_dark"
    )

    # 6. Add Metrics Annotation
    if metrics:
        ann_text = (
            f"MAX DEPTH: {metrics.get('depth', 0):.1f} cm<br>"
            f"AREA: {metrics.get('area', 0):.2f} m²<br>"
            f"SEVERITY: <span style='color:red'>{metrics.get('severity', 'HIGH')}</span>"
        )
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            text=ann_text,
            showarrow=False,
            font=dict(size=16, family="Inter, sans-serif"),
            align="left",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="red",
            borderwidth=2,
            borderpad=10
        )

    return fig

if __name__ == "__main__":
    # Generate Dummy Data for Demo
    size = 100
    x = np.linspace(0, 5, size)
    y = np.linspace(0, 5, size)
    X, Y = np.meshgrid(x, y)
    
    # Plane with a hole
    depth = np.full((size, size), 0.2)
    mask = np.zeros((size, size))
    
    # Create a complex pothole shape
    center = (50, 50)
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i-center[0])**2 + (j-center[1])**2)
            if dist < 30:
                depth[i, j] += 0.5 * (1 - dist/30) + (np.random.rand()*0.05)
                mask[i, j] = 1
                
    results_fig = create_premium_3d_plot(
        depth, 
        mask, 
        metrics={'depth': 8.3, 'area': 0.42, 'severity': 'HIGH'}
    )
    
    # Save to HTML to let user preview
    results_fig.write_html("test_premium_3d.html")
    print("✓ Prototipe disimpan ke test_premium_3d.html")
