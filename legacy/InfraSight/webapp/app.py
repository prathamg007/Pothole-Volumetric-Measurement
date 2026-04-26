"""
InfraSight — AI-Powered Pothole Volumetric Analysis
Streamlit dashboard for real-time pothole detection, 3D tomography,
severity classification, and repair cost estimation.

Run with:
    streamlit run webapp/app.py

    Or from the project root:
    python -m streamlit run webapp/app.py
"""
import os
import sys
import yaml
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from pathlib import Path
from datetime import datetime

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent.absolute()
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"
sys.path.insert(0, str(ROOT_DIR))

from src.models.yolo_segmentation  import PotholeSegmenter
from src.models.depth_estimation    import DepthEstimator
from src.models.material_classifier import MaterialClassifier
from src.core.volumetric            import VolumetricCalculator
from src.core.severity              import SeverityClassifier
from src.core.repair_advisor        import RepairAdvisor
from src.visualization.mesh_engine  import Mesh3DVisualizer
from src.utils.logger               import setup_logger

logger = setup_logger("WebApp")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InfraSight Analytics",
    page_icon="assets/icon.png" if (ROOT_DIR / "assets" / "icon.png").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: #0b0f19;
    color: #e2e8f0;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid rgba(99,102,241,0.2);
}

[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

/* Glass card */
.glass-card {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
}

/* Page header */
.page-header {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 60%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
    line-height: 1.2;
}

.page-subtitle {
    color: #64748b;
    font-size: 0.95rem;
    font-weight: 400;
    margin-bottom: 28px;
}

/* Section labels */
.section-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #475569;
    margin-bottom: 8px;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}

.metric-card {
    background: rgba(30,41,59,0.8);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 10px;
    padding: 16px 18px;
}

.metric-card .label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
}

.metric-card .value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-top: 2px;
    line-height: 1.2;
}

.metric-card .unit {
    font-size: 0.8rem;
    color: #94a3b8;
    font-weight: 400;
}

/* Severity badge */
.severity-badge {
    display: inline-flex;
    align-items: center;
    padding: 5px 14px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    gap: 6px;
}

/* Primary button override */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: #fff;
    border: none;
    padding: 11px 26px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    transition: opacity 0.2s ease, transform 0.2s ease;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3);
}

.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* Expander override */
[data-testid="stExpander"] {
    background: rgba(17,24,39,0.6);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 12px;
    margin-bottom: 12px;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div > div {
    background: #6366f1;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #6366f1, #c084fc);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 10px;
}

/* Info/warning boxes */
.stInfo, .stWarning, .stError, .stSuccess {
    border-radius: 10px;
}

/* Metric delta override */
[data-testid="stMetricValue"] { font-weight: 700; color: #f1f5f9; }
[data-testid="stMetricLabel"] { color: #64748b; font-size: 0.8rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(15,23,42,0.5);
    padding: 4px;
    border-radius: 10px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.15);
    color: #818cf8;
}

/* Divider */
hr { border-color: rgba(99,102,241,0.1); }

/* Sidebar logo block */
.sidebar-brand {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sidebar-tagline {
    font-size: 0.78rem;
    color: #475569 !important;
    margin-top: -2px;
}

.sidebar-nav-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #334155 !important;
    margin-top: 16px;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached — loaded ONCE, never reloaded on slider change) ───────
@st.cache_resource
def get_models():
    """
    Load all AI models into memory once and cache them for the session.
    conf_threshold is NOT a parameter here — it is applied dynamically
    in run_analysis() so that slider changes take effect immediately
    without triggering an expensive model reload.
    """
    if not CONFIG_PATH.exists():
        st.error(f"Configuration file not found: {CONFIG_PATH}")
        return None, None, None, None

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    yolo_cfg  = config["models"]["yolo"]
    weights   = yolo_cfg.get("weights_path", "")
    if not weights or not (ROOT_DIR / weights).exists():
        weights = yolo_cfg.get("weights_fallback", "")

    full_path = ROOT_DIR / weights
    if not full_path.exists():
        st.error(f"YOLO weights not found at: {full_path}")
        return None, None, None, None

    logger.info(f"Loading YOLO — {full_path.name}")
    # Load with default threshold; it will be overridden at inference time
    segmenter = PotholeSegmenter(
        str(full_path),
        conf_threshold=yolo_cfg.get("conf_threshold", 0.25),
    )

    logger.info("Loading Depth Anything V2 (first run may take ~30 s)...")
    depth_est = DepthEstimator(config["models"]["depth"]["model_name"])

    logger.info("Loading Material Classifier...")
    mat_clf = MaterialClassifier(config_path=str(CONFIG_PATH))

    return segmenter, depth_est, mat_clf, config


# ── Core analysis pipeline ─────────────────────────────────────────────────────
def run_analysis(
    image_np,
    segmenter,
    depth_estimator,
    material_classifier,
    config,
    conf_threshold: float = 0.25,
    cam_height_cm: float = 50.0,
):
    """
    Run the full InfraSight pipeline on a single image.

    conf_threshold and cam_height_cm are applied here (not at load-time)
    so that UI slider changes always take effect on the next run without
    reloading the heavy neural-network weights.

    Returns a result dict or None if no potholes are detected.
    """
    # Apply the current slider value directly on the cached model object
    segmenter.conf_threshold = conf_threshold

    logger.info(
        f"Running detection | conf={conf_threshold} | cam_height={cam_height_cm} cm"
    )

    seg_results = segmenter.detect(image_np, visualize=True)
    detections  = seg_results.get("detections", [])

    logger.info(f"Detection returned {len(detections)} total detection(s)")

    if not detections:
        return None

    potholes = [d for d in detections if d.class_id == 0]
    logger.info(f"{len(potholes)} pothole detection(s) after class filter")
    if not potholes:
        return None

    depth_map = depth_estimator.predict(image_np)

    # Initialise physics engines — use the cam_height from the UI slider
    vol_calc = VolumetricCalculator(
        calibration_constant=config["volumetric"]["calibration_constant"],
        cam_height_cm=cam_height_cm,
        cam_pitch_deg=20.0,
    )
    mat_conf_threshold = (
        config.get("models", {}).get("material", {}).get("confidence_threshold", 0.6)
    )

    results_list = []
    for p_det in potholes:
        # Volumetric analysis (homography area + relative depth)
        vol_res = vol_calc.calculate_volume(
            pothole_mask=p_det.mask,
            pothole_bbox=p_det.bbox,
            depth_map=depth_map,
        )

        # Material classification on the pothole crop
        x1, y1, x2, y2 = p_det.bbox
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(image_np.shape[1], int(x2)); y2 = min(image_np.shape[0], int(y2))
        crop = image_np[y1:y2, x1:x2]

        mat_res      = None
        surface_type = "asphalt"
        if crop.size > 0:
            mat_res = material_classifier.predict(crop)
            if mat_res["confidence"] >= mat_conf_threshold:
                surface_type = mat_res["class"]

        # Severity and repair
        sev_res = SeverityClassifier().classify(
            depth_cm=vol_res.avg_depth_cm,
            area_cm2=vol_res.area_cm2,
            volume_cm3=vol_res.volume_cm3,
        )
        rep_res = RepairAdvisor().recommend(
            volume_cm3=vol_res.volume_cm3,
            depth_cm=vol_res.avg_depth_cm,
            area_cm2=vol_res.area_cm2,
            severity_level=sev_res.level,
            surface_type=surface_type,
        )

        results_list.append({
            "volumetric":   vol_res,
            "severity":     sev_res,
            "repair":       rep_res,
            "surface_type": surface_type,
            "surface_conf": mat_res["confidence"] if mat_res else 0.0,
            "pothole_mask": p_det.mask,
            "bbox":         p_det.bbox,
            "confidence":   p_det.confidence,
        })

    # Sort by severity score descending
    results_list.sort(key=lambda x: x["severity"].score, reverse=True)
    top = results_list[0]

    return {
        "annotated":    seg_results.get("annotated_image", image_np),
        "depth_viz":    depth_estimator.visualize_depth(depth_map),
        "potholes":     results_list,
        "summary": {
            "area_cm2":           sum(p["volumetric"].area_cm2  for p in results_list),
            "volume_cm3":         sum(p["volumetric"].volume_cm3 for p in results_list),
            "severity_level":     top["severity"].level,
            "severity_score":     top["severity"].score,
            "repair_method":      top["repair"].method if len(results_list) == 1 else "Multiple",
            "repair_cost_idr":    sum(p["repair"].total_cost_idr for p in results_list),
            "repair_material_kg": sum(p["repair"].material_kg    for p in results_list),
        },
        "depth_raw":    depth_map,
        "original_rgb": image_np,
    }


# ── Metric card helper ─────────────────────────────────────────────────────────
def metric_card(label: str, value: str, unit: str = ""):
    st.markdown(
        f"""<div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}<span class="unit"> {unit}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )


def severity_badge(level: str, color: str):
    st.markdown(
        f'<span class="severity-badge" style="background:{color}22;'
        f'color:{color};border:1px solid {color}55">'
        f'SEVERITY: {level}</span>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Analyze
# ══════════════════════════════════════════════════════════════════════════════
def page_analyze():
    st.markdown('<div class="page-header">Pothole Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Upload road surface images to run volumetric analysis, '
        'severity classification, and repair cost estimation.</div>',
        unsafe_allow_html=True,
    )

    # Threshold controls
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            conf_t = st.slider(
                "Detection Confidence Threshold",
                min_value=0.05, max_value=1.0, value=0.25, step=0.05,
                help="Lower values detect more potholes at the cost of more false positives.",
            )
        with col2:
            cam_height = st.slider(
                "Camera Height (cm)",
                min_value=20, max_value=300, value=50, step=5,
                help="Approximate height of the camera lens above the road surface."
                     " Used for homography-based area estimation.",
            )

    # Models are loaded once from cache — threshold + height applied at inference
    with st.spinner("Loading AI models (first run only)..."):
        segmenter, depth_est, mat_clf, config = get_models()
    if segmenter is None:
        st.stop()

    # File uploader
    st.markdown("---")
    files = st.file_uploader(
        "Upload Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload one or more pothole images. JPEG and PNG supported.",
    )

    if files:
        if st.button("Run Batch Analysis", use_container_width=False):
            progress_bar = st.progress(0)
            status_text  = st.empty()
            batch_results = []

            for i, file in enumerate(files):
                status_text.markdown(
                    f"Processing **{file.name}** ({i + 1} / {len(files)})..."
                )
                img_np = np.array(Image.open(file).convert("RGB"))

                try:
                    res = run_analysis(
                        img_np,
                        segmenter,
                        depth_est,
                        mat_clf,
                        config,
                        conf_threshold=conf_t,
                        cam_height_cm=float(cam_height),
                    )
                    if res:
                        # Save annotated image
                        save_dir = ROOT_DIR / "output" / "analysis_results"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_path = save_dir / f"annotated_{ts}_{file.name}"
                        cv2.imwrite(
                            str(img_path),
                            cv2.cvtColor(res["annotated"], cv2.COLOR_RGB2BGR),
                        )
                        res["annotated_path"] = str(img_path)
                        res["image_name"]     = file.name
                        batch_results.append(res)
                    else:
                        st.warning(
                            f"{file.name}: No potholes detected "
                            f"(confidence threshold: {conf_t}). "
                            "Try lowering the threshold or using a clearer image."
                        )
                except Exception as exc:
                    st.error(f"{file.name}: Analysis failed — {exc}")
                    logger.exception(f"Analysis error for {file.name}")

                progress_bar.progress((i + 1) / len(files))

            if batch_results:
                status_text.success(
                    f"Analysis complete: {len(batch_results)} / {len(files)} images processed successfully."
                )
                st.session_state["batch_results"] = batch_results
            else:
                status_text.error(
                    "No images were processed successfully. "
                    "Ensure that the images clearly show road damage, "
                    "or lower the Confidence Threshold."
                )

    # Display results
    if "batch_results" not in st.session_state:
        return

    st.markdown("---")
    for i, res in enumerate(st.session_state["batch_results"]):
        n_holes = len(res["potholes"])
        sev     = res["summary"]["severity_level"]
        label   = (
            f"{res['image_name']}  |  "
            f"{n_holes} pothole(s) detected  |  "
            f"Severity: {sev}"
        )

        with st.expander(label, expanded=(i == 0)):
            # Annotated image + depth map
            img_col, depth_col = st.columns(2)
            with img_col:
                st.markdown('<div class="section-label">Detection Result</div>', unsafe_allow_html=True)
                st.image(res["annotated"], use_container_width=True)
            with depth_col:
                st.markdown('<div class="section-label">Depth Map (Depth Anything V2)</div>', unsafe_allow_html=True)
                st.image(res["depth_viz"], use_container_width=True)

            st.markdown("---")

            # Per-pothole tabs
            tab_labels = [f"Pothole {p + 1}" for p in range(n_holes)]
            tabs = st.tabs(tab_labels)

            for p_idx, (tab, p_res) in enumerate(zip(tabs, res["potholes"])):
                with tab:
                    col_3d, col_info = st.columns([1, 1])

                    # 3D visualization
                    with col_3d:
                        st.markdown('<div class="section-label">3D Tomography</div>', unsafe_allow_html=True)
                        show_3d = st.toggle(
                            "Enable 3D Model",
                            key=f"toggle_3d_{i}_{p_idx}",
                            help="3D rendering is compute-intensive. Enable only when needed.",
                        )
                        if show_3d:
                            with st.spinner("Rendering 3D model..."):
                                viz = Mesh3DVisualizer()
                                fig = viz.create_premium_pothole_mesh(
                                    res["depth_raw"],
                                    p_res["pothole_mask"],
                                    metrics={
                                        "depth":    p_res["volumetric"].avg_depth_cm,
                                        "area":     p_res["volumetric"].area_cm2,
                                        "severity": p_res["severity"].level,
                                    },
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"plotly_{i}_{p_idx}")
                        else:
                            st.info(
                                "3D tomography rendering is disabled by default to conserve memory. "
                                "Use the toggle above to enable it."
                            )

                    # Metrics and repair
                    with col_info:
                        sev = p_res["severity"]
                        st.markdown('<div class="section-label">Measurements</div>', unsafe_allow_html=True)
                        severity_badge(sev.level, sev.color)
                        st.write("")

                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Severity Score",  f"{sev.score}/10")
                        mc2.metric("Avg. Depth",      f"{p_res['volumetric'].avg_depth_cm:.1f} cm")
                        mc3.metric("Volume",          f"{p_res['volumetric'].volume_cm3:.0f} cm\u00b3")

                        mc4, mc5 = st.columns(2)
                        mc4.metric("Surface Area",   f"{p_res['volumetric'].area_cm2:.1f} cm\u00b2")
                        mc5.metric("Max Depth",      f"{p_res['volumetric'].max_depth_cm:.1f} cm")

                        st.markdown("---")
                        st.markdown('<div class="section-label">Repair Recommendation</div>', unsafe_allow_html=True)

                        rep = p_res["repair"]
                        rc1, rc2 = st.columns(2)
                        rc1.metric("Method",         rep.method)
                        rc2.metric("Est. Cost (IDR)", f"Rp {rep.total_cost_idr:,.0f}")

                        rd1, rd2 = st.columns(2)
                        rd1.metric("Material",       f"{rep.material_kg:.2f} kg")
                        rd2.metric("Est. Time",      f"{rep.estimated_time_hours:.1f} h")

                        st.metric("Durability",      f"{rep.durability_months} months")

                        st.markdown("---")
                        st.markdown('<div class="section-label">Road Surface</div>', unsafe_allow_html=True)
                        mat_conf = p_res["surface_conf"]
                        st.write(
                            f"**Type:** {p_res['surface_type'].capitalize()}  \n"
                            f"**Classifier Confidence:** {mat_conf:.2%}  \n"
                            f"**Detection Confidence:** {p_res['confidence']:.2%}"
                        )

                        if sev.description:
                            st.info(sev.description)

                        if rep.notes:
                            st.warning(rep.notes)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: About
# ══════════════════════════════════════════════════════════════════════════════
def page_about():
    st.markdown('<div class="page-header">About InfraSight</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Technical overview of the AI analysis pipeline.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Detection Pipeline")
        st.markdown("""
**Stage 1 — Pothole Segmentation**  
YOLOv8 segmentation model trained on annotated road damage datasets.
Produces pixel-accurate masks for each detected pothole.

**Stage 2 — Monocular Depth Estimation**  
Depth Anything V2 (Small) processes the image and produces a
relative depth map. Higher depth values indicate regions farther
from the camera (deeper into the pothole).

**Stage 3 — Volumetric Calculation**  
A homography engine converts pixel-space area to real-world cm².
Relative depth difference between the surrounding road and the
pothole interior is calibrated to produce average depth in cm.
Volume = Area × Depth.
        """)

    with col2:
        st.markdown("### Classification & Reporting")
        st.markdown("""
**Stage 4 — Severity Classification**  
A weighted scoring model (depth 40%, area 30%, volume 30%)
produces a 1–10 severity score that maps to LOW / MEDIUM / HIGH / CRITICAL.
Thresholds follow FHWA and IRI-inspired engineering standards.

**Stage 5 — Road Material Detection**  
MobileNetV3 classifies the road surface (asphalt, concrete, paving).
The result disambiguates material costs and repair methods.

**Stage 6 — Repair Advisory**  
Repair method, material quantity (kg), labor cost, and estimated
repair time are derived from severity, volume, and surface type.
Cost estimates are in IDR, based on 2024-2025 market rates.
        """)

    st.markdown("---")
    st.markdown("### How to Run")

    st.code("""# 1. Activate your virtual environment (if applicable)
source venv/bin/activate

# 2. Install dependencies (first-time setup)
pip install -r requirements.txt

# 3. Launch the Streamlit app from the project root
streamlit run webapp/app.py

# Optional: specify a custom port
streamlit run webapp/app.py --server.port 8502

# Optional: run without opening the browser automatically
streamlit run webapp/app.py --server.headless true
""", language="bash")

    st.markdown("### Configuration")
    st.markdown("""
All model paths, thresholds, and cost parameters are controlled via
`config/config.yaml`. Key settings:

| Parameter | Key | Default |
|---|---|---|
| YOLO weights | `models.yolo.weights_path` | `weights/phase1_segmentation_v1.pt` |
| Detection confidence | `models.yolo.conf_threshold` | `0.25` |
| Depth model | `models.depth.model_name` | `depth-anything/Depth-Anything-V2-Small-hf` |
| Calibration constant | `volumetric.calibration_constant` | `30.0` |
| Camera height | `app.default_camera_height` | `150 cm` |
    """)

    st.markdown("### System Requirements")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
**Minimum (CPU-only)**  
- Python 3.10+
- 8 GB RAM
- 4-core CPU
- ~5 GB disk (models)
        """)
    with col4:
        st.markdown("""
**Recommended (GPU)**  
- Python 3.10+
- 16 GB RAM
- NVIDIA GPU with CUDA 11.8+
- 8 GB VRAM
        """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    with st.sidebar:
        # Brand
        st.markdown(
            '<div class="sidebar-brand">InfraSight</div>'
            '<div class="sidebar-tagline">AI Road Infrastructure Analysis</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Navigation
        st.markdown('<div class="sidebar-nav-label">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            ["Analyze", "About"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Quick reference
        st.markdown('<div class="sidebar-nav-label">Quick Reference</div>', unsafe_allow_html=True)
        st.markdown("""
<small style="color:#475569;line-height:1.7">
1. Upload one or more road images.<br>
2. Adjust the confidence threshold if needed.<br>
3. Click <b>Run Batch Analysis</b>.<br>
4. Review metrics, 3D model, and repair recommendations.<br>
5. Lower the threshold if no potholes are detected.
</small>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Severity legend
        st.markdown('<div class="sidebar-nav-label">Severity Scale</div>', unsafe_allow_html=True)
        severity_items = [
            ("LOW",      "#4CAF50", "Score 1–3, scheduled repair"),
            ("MEDIUM",   "#FF9800", "Score 4–5, priority repair"),
            ("HIGH",     "#F44336", "Score 6–7, urgent repair"),
            ("CRITICAL", "#9C27B0", "Score 9–10, emergency"),
        ]
        for lvl, color, desc in severity_items:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0">'
                f'<span style="width:8px;height:8px;border-radius:50%;'
                f'background:{color};flex-shrink:0"></span>'
                f'<span style="font-size:0.75rem;color:#94a3b8"><b>{lvl}</b> — {desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if page == "Analyze":
        page_analyze()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()