"""Metric monocular depth. Uses Depth Anything V2 Metric variant (outputs meters)."""
from typing import Optional

import cv2
import numpy as np


class MetricDepthEstimator:
    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        device: str = "auto",
    ):
        import torch
        from transformers import pipeline

        resolved_device = device
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = resolved_device
        self._torch = torch

        self.pipe = pipeline(
            task="depth-estimation",
            model=model_name,
            device=0 if resolved_device == "cuda" else -1,
        )
        self.model_name = model_name

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return metric depth map in meters, shape (H, W), float32, resized to input dims."""
        from PIL import Image

        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb * 255, 0, 255).astype(np.uint8)
        pil = Image.fromarray(image_rgb)

        with self._torch.inference_mode():
            result = self.pipe(pil)

        depth: Optional[np.ndarray] = None
        # Prefer the raw predicted_depth tensor (metric models return meters here)
        if "predicted_depth" in result:
            t = result["predicted_depth"]
            if hasattr(t, "detach"):
                depth = t.detach().squeeze().cpu().numpy().astype(np.float32)
            else:
                depth = np.array(t, dtype=np.float32).squeeze()

        if depth is None and "depth" in result:
            # Fallback: result["depth"] is PIL; for metric models it may be the metric map
            depth = np.array(result["depth"], dtype=np.float32)

        if depth is None:
            raise RuntimeError(f"Depth pipeline returned no usable field. Keys: {list(result.keys())}")

        H, W = image_rgb.shape[:2]
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        return depth.astype(np.float32)

    @staticmethod
    def colorize(depth_m: np.ndarray, colormap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
        """Normalize and colorize depth for visualization. Returns RGB uint8."""
        d = depth_m.copy()
        finite = np.isfinite(d)
        if not finite.any():
            return np.zeros((*d.shape, 3), dtype=np.uint8)
        d_min = float(d[finite].min())
        d_max = float(d[finite].max())
        denom = max(d_max - d_min, 1e-6)
        norm = np.clip((d - d_min) / denom, 0.0, 1.0)
        norm[~finite] = 0.0
        u8 = (norm * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(u8, colormap)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
