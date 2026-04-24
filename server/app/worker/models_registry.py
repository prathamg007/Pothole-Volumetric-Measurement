"""Singleton that holds the loaded YOLO + depth models so the server reuses them across jobs."""
import threading

from app.models.crack_classifier import CrackClassifier
from app.models.depth_metric import MetricDepthEstimator
from app.models.pothole_segmenter import PotholeSegmenter
from app.utils.config import resolve_path
from app.utils.logger import get_logger

log = get_logger("models")


class ModelRegistry:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.segmenter: PotholeSegmenter | None = None
        self.crack_clf: CrackClassifier | None = None
        self.depth: MetricDepthEstimator | None = None
        self._lock = threading.Lock()

    def load_all(self) -> None:
        with self._lock:
            if self.segmenter is not None and self.crack_clf is not None and self.depth is not None:
                return
            seg_cfg = self.cfg["models"]["pothole_segmenter"]
            crk_cfg = self.cfg["models"]["crack_classifier"]
            depth_cfg = self.cfg["models"]["depth"]

            log.info("Loading pothole segmenter...")
            self.segmenter = PotholeSegmenter(
                weights_path=resolve_path(seg_cfg["weights"]),
                conf_threshold=seg_cfg["conf_threshold"],
            )
            log.info("Loading crack classifier...")
            self.crack_clf = CrackClassifier(
                weights_path=resolve_path(crk_cfg["weights"]),
                conf_threshold=crk_cfg["conf_threshold"],
                ignore_classes=crk_cfg.get("ignore_classes", []),
            )
            log.info("Loading metric depth model (may take a moment on first run)...")
            self.depth = MetricDepthEstimator(model_name=depth_cfg["model_name"], device=depth_cfg["device"])
            log.info("Model registry ready")

    def is_ready(self) -> bool:
        return all([self.segmenter, self.crack_clf, self.depth])
