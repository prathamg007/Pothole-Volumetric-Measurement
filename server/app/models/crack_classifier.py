"""YOLOv8 crack typology detector. Classes: Longitudinal / Transverse / Alligator / Potholes.
The 'Potholes' class is ignored — pothole segmentation is done by the dedicated segmenter.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CrackDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


class CrackClassifier:
    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.30,
        ignore_classes: list[int] | None = None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        if self.model.task != "detect":
            raise ValueError(f"Expected a detection model, got task={self.model.task}")
        self.conf_threshold = float(conf_threshold)
        self.ignore_classes: set[int] = set(ignore_classes or [])
        self.class_names: dict[int, str] = dict(self.model.names)

    def detect(self, image_rgb: np.ndarray) -> list[CrackDetection]:
        results = self.model.predict(image_rgb, conf=self.conf_threshold, verbose=False)[0]
        out: list[CrackDetection] = []
        if results.boxes is None:
            return out
        for i, (box, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.cls)):
            cls_id = int(cls.item())
            if cls_id in self.ignore_classes:
                continue
            conf = float(results.boxes.conf[i].item())
            x1, y1, x2, y2 = map(int, box.tolist())
            out.append(
                CrackDetection(
                    class_id=cls_id,
                    class_name=self.class_names.get(cls_id, str(cls_id)),
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                )
            )
        return out
