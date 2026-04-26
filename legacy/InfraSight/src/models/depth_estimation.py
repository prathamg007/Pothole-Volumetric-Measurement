"""
Depth Anything V2 for monocular depth estimation
"""
import cv2
import numpy as np
import torch
from typing import Optional
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger("DepthEstimation")


class DepthEstimator:
    """Monocular depth estimation using Depth Anything V2"""
    
    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small",
        device: Optional[str] = None
    ):
        """
        Initialize Depth Anything V2 model from HuggingFace
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading Depth Anything V2 on {device}...")
        
        self.device = device
        self.pipe = pipeline(
            task="depth-estimation",
            model=model_name,
            device=0 if device == "cuda" else -1
        )
        
        logger.info("Depth Anything V2 loaded successfully")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Generate depth map from RGB image.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            depth_map: Relative depth map (H, W) - normalized [0, 1]
        """
        import torch
        from PIL import Image
        
        # Convert numpy to PIL
        if image.dtype == np.uint8:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Get depth prediction with optimization
        with torch.inference_mode():
            result = self.pipe(pil_image)
        
        # Extract depth map
        depth_pil = result["depth"]
        depth_map = np.array(depth_pil, dtype=np.float32)
        
        # Normalize to [0, 1]
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        # Resize to match input image size
        if depth_map.shape[:2] != image.shape[:2]:
            depth_map = cv2.resize(
                depth_map,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        return depth_map
    
    def visualize_depth(
        self,
        depth_map: np.ndarray,
        colormap: int = cv2.COLORMAP_VIRIDIS
    ) -> np.ndarray:
        """
        Convert depth map to colormap for visualization
        
        Args:
            depth_map: Normalized depth map [0, 1]
            colormap: OpenCV colormap (VIRIDIS, INFERNO, JET, etc.)
            
        Returns:
            RGB colorized depth map
        """
        # Convert to uint8
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(depth_uint8, colormap)
        
        # Convert BGR to RGB
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        return colored
    
    def extract_depth_at_mask(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract depth values within a binary mask
        
        Args:
            depth_map: Depth map (H, W)
            mask: Binary mask (H, W) with 0 and 1
            
        Returns:
            Array of depth values within mask
        """
        return depth_map[mask == 1]


if __name__ == "__main__":
    # Example usage
    estimator = DepthEstimator()
    
    # Load test image
    image = cv2.imread("test_image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Predict depth
    depth_map = estimator.predict(image)
    
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
    
    # Visualize
    colored_depth = estimator.visualize_depth(depth_map, cv2.COLORMAP_INFERNO)
    
    cv2.imshow("Depth Map", cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
