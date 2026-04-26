"""
Volumetric calculation using Homography Area and Relative Depth Heuristics
CRITICAL MODULE: Implements the core depth estimation and volume calculation
"""
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from src.utils.logger import setup_logger
from src.core.homography import HomographyEngine

logger = setup_logger("Volumetric")


@dataclass
class VolumetricResult:
    """Container for volumetric measurement results"""
    area_cm2: float
    avg_depth_cm: float
    max_depth_cm: float
    volume_cm3: float


class VolumetricCalculator:
    """
    Calculate pothole volume using Camera Homography for Area
    and Surface vs Bottom heuristic for Depth.
    """
    
    def __init__(self, calibration_constant: float = 30.0, cam_height_cm: float = 50.0, cam_pitch_deg: float = 20.0):
        """
        Initialize calculator with physical camera parameters.
        
        Args:
            calibration_constant: Empirical constant to convert normalized depth to cm
            cam_height_cm: Height of the camera lens from the ground.
            cam_pitch_deg: Downward tilt of the camera.
        """
        self.calibration_constant = calibration_constant
        
        # Initialize the physics engine
        logger.info(f"Initializing Homography: Height={cam_height_cm}cm, Pitch={cam_pitch_deg}deg")
        self.homography_engine = HomographyEngine(
            height_cm=cam_height_cm, 
            pitch_deg=cam_pitch_deg
        )
    
    def calculate_volume(
        self,
        pothole_mask: np.ndarray,
        pothole_bbox: Tuple[int, int, int, int],
        depth_map: np.ndarray
    ) -> VolumetricResult:
        """
        Calculate volumetric measurements using Homography and Depth maps.
        
        Args:
            pothole_mask: Binary mask (H, W) of pothole
            pothole_bbox: Bounding box (x1, y1, x2, y2) of pothole
            depth_map: Relative depth map (H, W) - normalized [0, 1]
            
        Returns:
            VolumetricResult with physical measurements
        """
        # Step 1: GROUND PLANE ESTIMATION
        # We sample the healthy asphalt immediately surrounding the pothole
        d_surface = self._estimate_ground_plane(
            pothole_mask,
            pothole_bbox,
            depth_map
        )
        
        # Step 2: POTHOLE BOTTOM ESTIMATION
        d_bottom = self._estimate_pothole_bottom(
            pothole_mask,
            depth_map,
            percentile=10  # Use bottom 10% deepest pixels
        )
        
        # Step 3: RELATIVE DEPTH DIFFERENCE
        depth_diff_normalized = abs(d_bottom - d_surface)
        
        # Step 4: CALIBRATION TO REAL-WORLD DEPTH
        real_depth_cm = depth_diff_normalized * self.calibration_constant
        
        # Calculate max depth (single deepest pixel)
        pothole_depths = depth_map[pothole_mask == 1]
        if len(pothole_depths) > 0:
            max_depth_normalized = abs(np.min(pothole_depths) - d_surface)
            max_depth_cm = max_depth_normalized * self.calibration_constant
        else:
            max_depth_cm = 0.0
            
        # Step 5: AREA CALCULATION (THE HOMOGRAPHY UPGRADE)
        # We pass the exact polygon mask into the homography engine
        pothole_area_cm2 = self.homography_engine.calculate_physical_area(pothole_mask)
        
        # Step 6: VOLUME ESTIMATION
        volume_cm3 = pothole_area_cm2 * real_depth_cm
        
        return VolumetricResult(
            area_cm2=pothole_area_cm2,
            avg_depth_cm=real_depth_cm,
            max_depth_cm=max_depth_cm,
            volume_cm3=volume_cm3
        )
    
    def _estimate_ground_plane(
        self,
        pothole_mask: np.ndarray,
        pothole_bbox: Tuple[int, int, int, int],
        depth_map: np.ndarray
    ) -> float:
        """
        Estimate ground plane depth by sampling healthy asphalt 
        in an expanded bounding box around the pothole.
        """
        x1, y1, x2, y2 = pothole_bbox
        
        # Expand the bounding box slightly to ensure we capture healthy asphalt
        pad = 20
        h, w = depth_map.shape
        x1 = max(0, int(x1) - pad)
        y1 = max(0, int(y1) - pad)
        x2 = min(w, int(x2) + pad)
        y2 = min(h, int(y2) + pad)
        
        bbox_mask = np.zeros_like(pothole_mask)
        bbox_mask[y1:y2, x1:x2] = 1
        
        # Healthy asphalt = inside expanded bbox but OUTSIDE the exact pothole mask
        healthy_asphalt_mask = (bbox_mask == 1) & (pothole_mask == 0)
        asphalt_depths = depth_map[healthy_asphalt_mask]
        
        if len(asphalt_depths) == 0:
            # Fallback if the mask consumes the entire image
            return float(np.median(depth_map))
            
        # Use median for robustness against sensor noise
        d_surface = float(np.median(asphalt_depths))
        
        return d_surface
    
    def _estimate_pothole_bottom(
        self,
        pothole_mask: np.ndarray,
        depth_map: np.ndarray,
        percentile: int = 10
    ) -> float:
        """
        Estimate pothole bottom depth using bottom percentile of deepest pixels
        """
        pothole_depths = depth_map[pothole_mask == 1]
        
        if len(pothole_depths) == 0:
            raise ValueError("Pothole mask is empty")
        
        # Depth Anything V2 typical convention: lower values = closer to camera, 
        # higher values = farther away (deeper into the hole).
        percentile_threshold = np.percentile(pothole_depths, 100 - percentile)
        deepest_pixels = pothole_depths[pothole_depths >= percentile_threshold]
        
        if len(deepest_pixels) == 0:
            return float(np.mean(pothole_depths))
            
        d_bottom = float(np.mean(deepest_pixels))
        
        return d_bottom
        

if __name__ == "__main__":
    # Sanity Check Usage
    h, w = 640, 640
    
    # Create simulated pothole mask
    pothole_mask = np.zeros((h, w), dtype=np.uint8)
    pothole_mask[300:400, 250:400] = 1  
    
    # Create simulated depth map
    depth_map = np.random.rand(h, w) * 0.3 + 0.5 
    depth_map[pothole_mask == 1] += 0.2  # Simulate a physical depression
    
    bbox = (250, 300, 400, 400)
    
    # Initialize with default robotics parameters
    calculator = VolumetricCalculator(
        calibration_constant=30.0,
        cam_height_cm=50.0,
        cam_pitch_deg=20.0
    )
    
    result = calculator.calculate_volume(
        pothole_mask=pothole_mask,
        pothole_bbox=bbox,
        depth_map=depth_map
    )
    
    print("\n--- Physical Measurements ---")
    print(f"Pothole Area: {result.area_cm2:.2f} cm²")
    print(f"Average Depth: {result.avg_depth_cm:.2f} cm")
    print(f"Max Depth: {result.max_depth_cm:.2f} cm")
    print(f"Volume: {result.volume_cm3:.2f} cm³")