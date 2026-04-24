"""YOLOv8 segmentation for pothole/manhole/bump masks."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PotholeDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    mask: np.ndarray  # (H, W) uint8, 1 inside


class PotholeSegmenter:
    """Wrapper around Ultralytics YOLO segmentation model.
    Classes: 0=Manhole, 1=Pothole, 2=Unmarked Bump.
    By default, only Pothole class (1) is returned via detect_potholes().
    """

    def __init__(self, weights_path: str | Path, conf_threshold: float = 0.25):
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        if self.model.task != "segment":
            raise ValueError(f"Expected a segmentation model, got task={self.model.task}")
        self.conf_threshold = float(conf_threshold)
        self.class_names: dict[int, str] = dict(self.model.names)

    def detect_all(self, image_rgb: np.ndarray) -> list[PotholeDetection]:
        H, W = image_rgb.shape[:2]
        results = self.model.predict(image_rgb, conf=self.conf_threshold, verbose=False)[0]
        if results.masks is None:
            return []
        out: list[PotholeDetection] = []
        for i, (box, mask_data, cls) in enumerate(
            zip(results.boxes.xyxy, results.masks.data, results.boxes.cls)
        ):
            cls_id = int(cls.item())
            conf = float(results.boxes.conf[i].item())
            x1, y1, x2, y2 = map(int, box.tolist())
            mask = mask_data.cpu().numpy()
            if mask.shape != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.5).astype(np.uint8)
            out.append(
                PotholeDetection(
                    class_id=cls_id,
                    class_name=self.class_names.get(cls_id, str(cls_id)),
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    mask=mask,
                )
            )
        return out

    def detect_potholes(self, image_rgb: np.ndarray) -> list[PotholeDetection]:
        return [d for d in self.detect_all(image_rgb) if d.class_name == "Pothole"]
