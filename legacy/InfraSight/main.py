import cv2
import sys
from pathlib import Path
from src.utils.logger import setup_logger

# AI Models
from src.models.yolo_segmentation import PotholeSegmenter
from src.models.depth_estimation import DepthEstimator

# Core Physics & Civil Engineering Logic
from src.core.volumetric import VolumetricCalculator
from src.core.severity import SeverityClassifier
from src.core.repair_advisor import RepairAdvisor

# Visualization
from src.visualization.mesh_engine import Mesh3DVisualizer

# Initialize our custom terminal logger
logger = setup_logger("MasterPipeline")

def main():
    # ---------------------------------------------------------
    # 1. SETUP PATHS & HARDWARE CONFIG
    # ---------------------------------------------------------
    test_image_path = "test8.png"  # Change this to whatever image you are testing
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------
    # 2. INITIALIZE NEURAL NETWORKS
    # ---------------------------------------------------------
    logger.info("Booting up Vision AI Phase...")
    # Load your custom trained model with the lower 0.10 threshold we discussed
    segmenter = PotholeSegmenter("weights/phase1_segmentation_v1.pt", conf_threshold=0.10)
    depth_estimator = DepthEstimator(model_name="depth-anything/Depth-Anything-V2-Small-hf")

    # ---------------------------------------------------------
    # 3. INITIALIZE PHYSICS & LOGIC ENGINES
    # ---------------------------------------------------------
    logger.info("Booting up Physics & Logic Engines...")
    # Assuming Camera Height = 50cm, Pitch = 20 degrees for a standard rover
    vol_calc = VolumetricCalculator(calibration_constant=30.0, cam_height_cm=50.0, cam_pitch_deg=20.0)
    sev_classifier = SeverityClassifier()
    rep_advisor = RepairAdvisor()

    # ---------------------------------------------------------
    # 4. LOAD IMAGE
    # ---------------------------------------------------------
    logger.info(f"Loading image target: {test_image_path}")
    img = cv2.imread(test_image_path)
    if img is None:
        logger.error(f"Image not found! Please place {test_image_path} in your main folder.")
        sys.exit(1)
    
    # OpenCV loads in BGR, AI models expect RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # 5. INFERENCE (THE EYES)
    # ---------------------------------------------------------
    logger.info("Running Phase 1: Segmentation Engine...")
    seg_results = segmenter.detect(img_rgb, visualize=True)
    detections = seg_results["detections"]

    if not detections:
        logger.warning("No potholes detected in this image based on current confidence threshold.")
        sys.exit(0)

    logger.info(f"Found {len(detections)} pothole(s). Running Phase 2: Z-Axis Depth Estimation...")
    depth_map = depth_estimator.predict(img_rgb)

    # ---------------------------------------------------------
    # 6. MATHEMATICAL ANALYSIS (THE BRAIN)
    # ---------------------------------------------------------
    for i, det in enumerate(detections):
        logger.info(f"\n--- Analyzing Pothole #{i+1} ---")

        # A. Volumetric Math (Homography Area + Relative Depth)
        vol_res = vol_calc.calculate_volume(
            pothole_mask=det.mask,
            pothole_bbox=det.bbox,
            depth_map=depth_map
        )
        logger.info(f"Geometry -> Area: {vol_res.area_cm2:.2f} cm2 | Depth: {vol_res.avg_depth_cm:.2f} cm | Volume: {vol_res.volume_cm3:.2f} cm3")

        # B. Severity Classification
        sev_res = sev_classifier.classify(
            depth_cm=vol_res.avg_depth_cm,
            area_cm2=vol_res.area_cm2,
            volume_cm3=vol_res.volume_cm3
        )
        logger.info(f"Severity -> {sev_res.level} (Score: {sev_res.score}/10)")

        # C. Repair Recommendation
        rep_res = rep_advisor.recommend(
            volume_cm3=vol_res.volume_cm3,
            depth_cm=vol_res.avg_depth_cm,
            area_cm2=vol_res.area_cm2,
            severity_level=sev_res.level,
            surface_type="asphalt"  # Hardcoded since we aren't using the material classifier
        )
        logger.info(f"Repair   -> Method: {rep_res.method} | Asphalt Required: {rep_res.material_kg:.2f} kg | Est. Cost: Rp {rep_res.total_cost_idr:,.0f}")

        # D. Generate 3D Interactive Mesh
        logger.info("Generating 3D interactive mesh...")
        viz = Mesh3DVisualizer()
        fig = viz.create_pothole_mesh(
            depth_map=depth_map,
            pothole_mask=det.mask,
            image_rgb=img_rgb,
            title=f"Pothole #{i+1} Topology"
        )
        # Pop open the default web browser to show the 3D Plotly graph
        fig.show()

    # ---------------------------------------------------------
    # 7. SAVE OUTPUTS
    # ---------------------------------------------------------
    output_img_path = output_dir / f"annotated_{test_image_path}"
    # Convert back to BGR for OpenCV saving
    cv2.imwrite(str(output_img_path), cv2.cvtColor(seg_results["annotated_image"], cv2.COLOR_RGB2BGR))
    
    logger.info(f"\nPipeline Complete! Annotated image saved to {output_img_path}")

if __name__ == "__main__":
    main()