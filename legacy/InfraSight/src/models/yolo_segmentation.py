"""
YOLOv8 Detection/Segmentation for pothole and reference object detection.
Supports both segmentation models (mask output) and detection-only models
(bbox-to-mask fallback).
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.utils.logger import setup_logger

logger = setup_logger("YOLOSegmentation")


@dataclass
class Detection:
    """Detection result container"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: np.ndarray  # Binary mask


class PotholeSegmenter:
    """YOLO inference for pothole and reference object detection.
    Supports segmentation models (preferred) and detection-only models
    (bbox used as rectangular mask fallback).
    """
    
    def __init__(self, weights_path: str, conf_threshold: float = 0.25):
        """
        Initialize YOLO model (auto-detects seg vs detect task).
        
        Args:
            weights_path: Path to trained weights (.pt file)
            conf_threshold: Confidence threshold for detections
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")
        
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.class_names = {0: 'pothole', 1: 'reference_object'}
        
        self.is_detection_only = (self.model.task == 'detect')
        logger.info(f"Model loaded — task: {'detection (bbox mask fallback)' if self.is_detection_only else 'segmentation'}")
    
    def _bbox_to_mask(self, bbox: Tuple[int,int,int,int], img_h: int, img_w: int) -> np.ndarray:
        """Create a filled rectangular mask from a bounding box."""
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        mask[max(0,y1):min(img_h,y2), max(0,x1):min(img_w,x2)] = 1
        return mask

    def detect(
        self,
        image: np.ndarray,
        visualize: bool = False
    ) -> Dict[str, any]:
        """
        Detect potholes (and optionally reference objects) in an image.
        
        Args:
            image: RGB image (H, W, 3)
            visualize: If True, return annotated image
            
        Returns:
            {
                'detections': List[Detection],
                'pothole_masks': List[np.ndarray],
                'reference_masks': List[np.ndarray],
                'annotated_image': np.ndarray (if visualize=True)
            }
        """
        img_h, img_w = image.shape[:2]
        
        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            verbose=False
        )[0]
        
        detections = []
        pothole_masks = []
        reference_masks = []
        
        has_masks = (results.masks is not None) and not self.is_detection_only
        
        if has_masks:
            # Segmentation model: use predicted masks
            for i, (box, mask_data, cls) in enumerate(zip(
                results.boxes.xyxy,
                results.masks.data,
                results.boxes.cls
            )):
                class_id = int(cls.item())
                confidence = float(results.boxes.conf[i].item())
                
                # Convert box to integers
                x1, y1, x2, y2 = map(int, box.tolist())
                
                # Convert mask to binary numpy array (resize to original image size)
                mask = mask_data.cpu().numpy()
                mask = cv2.resize(
                    mask,
                    (img_w, img_h),
                    interpolation=cv2.INTER_LINEAR
                )
                mask = (mask > 0.5).astype(np.uint8)
                
                detection = Detection(
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, 'unknown'),
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    mask=mask
                )
                
                detections.append(detection)
                
                # Separate by class
                if class_id == 0:  # Pothole
                    pothole_masks.append(mask)
                elif class_id == 1:  # Reference object
                    reference_masks.append(mask)
        else:
            # Detection-only model: fill bbox as mask approximation
            for i, (box, cls) in enumerate(zip(
                results.boxes.xyxy,
                results.boxes.cls
            )):
                class_id = int(cls.item())
                confidence = float(results.boxes.conf[i].item())
                
                # Convert box to integers
                x1, y1, x2, y2 = map(int, box.tolist())
                
                # Create mask from bbox
                mask = self._bbox_to_mask((x1, y1, x2, y2), img_h, img_w)
                
                detection = Detection(
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, 'unknown'),
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    mask=mask
                )
                detections.append(detection)
                
                # Separate by class
                if class_id == 0:  # Pothole
                    pothole_masks.append(mask)
                elif class_id == 1:  # Reference object
                    reference_masks.append(mask)
        
        result = {
            'detections': detections,
            'pothole_masks': pothole_masks,
            'reference_masks': reference_masks
        }
        
        # Visualization
        if visualize:
            annotated = image.copy()
            
            pothole_idx = 0
            for det in detections:
                # Draw mask
                color = (0, 255, 0) if det.class_id == 0 else (255, 0, 0)
                colored_mask = np.zeros_like(annotated)
                colored_mask[det.mask == 1] = color
                annotated = cv2.addWeighted(annotated, 1.0, colored_mask, 0.4, 0)
                
                # Draw bounding box
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Label with ID for potholes
                if det.class_id == 0:
                    pothole_idx += 1
                    label = f"#{pothole_idx} {det.class_name} {det.confidence:.2f}"
                else:
                    label = f"{det.class_name} {det.confidence:.2f}"

                cv2.putText(
                    annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
            
            result['annotated_image'] = annotated
        
        return result
    
    def get_largest_detection(
        self,
        detections: List[Detection],
        class_id: int
    ) -> Optional[Detection]:
        """Get largest detection of specific class (by mask area)."""
        filtered = [d for d in detections if d.class_id == class_id]
        
        if not filtered:
            return None
        
        # Sort by mask area
        filtered.sort(key=lambda d: np.sum(d.mask), reverse=True)
        return filtered[0]


if __name__ == "__main__":
    # Example usage
    segmenter = PotholeSegmenter("models/weights/pothole_det/best.pt")
    
    # Load test image
    image = cv2.imread("test_image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect
    results = segmenter.detect(image, visualize=True)
    
    print(f"Found {len(results['pothole_masks'])} potholes")
    
    # Show annotated image
    if 'annotated_image' in results:
        cv2.imshow("Detections", cv2.cvtColor(results['annotated_image'], cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
