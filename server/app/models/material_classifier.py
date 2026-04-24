"""Multi-head road-surface classifier wrapper.

Loads the weights produced by training/material_classifier/train.py and exposes
predict(image_rgb) returning material + unevenness with per-class confidences.

Architecture must mirror the training script (RoadSurfaceModel).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class _RoadSurfaceModel:
    """Lazily constructed; defined inside the class init to keep torch import deferred."""


class MaterialClassifier:
    def __init__(self, weights_path: str | Path, device: str = "auto"):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F  # noqa: F401  (used in predict)
        from torchvision import transforms
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        self.materials: list[str] = list(ckpt["materials"])
        self.unevenness: list[str] = list(ckpt["unevenness"])

        class RoadSurfaceModel(nn.Module):
            def __init__(self, num_materials: int, num_unevenness: int):
                super().__init__()
                backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
                feat_dim = backbone.classifier[0].in_features
                self.features = backbone.features
                self.avgpool = backbone.avgpool
                self.dropout = nn.Dropout(0.2)
                self.material_head = nn.Sequential(
                    nn.Linear(feat_dim, 256),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_materials),
                )
                self.unevenness_head = nn.Sequential(
                    nn.Linear(feat_dim, 256),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_unevenness),
                )

            def forward(self, x):
                f = self.features(x)
                f = self.avgpool(f)
                f = self._flat(f)
                f = self.dropout(f)
                return self.material_head(f), self.unevenness_head(f)

            @staticmethod
            def _flat(x):
                import torch as _torch
                return _torch.flatten(x, 1)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = RoadSurfaceModel(len(self.materials), len(self.unevenness))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # Keep a torch reference for predict()
        self._torch = torch
        self._softmax = F.softmax

    def predict(self, image_rgb: np.ndarray) -> dict:
        """Run classifier on an RGB image (H, W, 3) uint8.

        Returns:
            {
              'material': str,
              'material_confidence': float,
              'unevenness': str,
              'unevenness_confidence': float,
              'all_materials':  {name: prob, ...},
              'all_unevenness': {name: prob, ...},
            }
        """
        from PIL import Image

        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb * 255, 0, 255).astype(np.uint8)
        pil = Image.fromarray(image_rgb)
        x = self.transform(pil).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            m_logits, u_logits = self.model(x)
        m_probs = self._softmax(m_logits, dim=-1).squeeze(0).cpu().numpy()
        u_probs = self._softmax(u_logits, dim=-1).squeeze(0).cpu().numpy()
        m_idx = int(np.argmax(m_probs))
        u_idx = int(np.argmax(u_probs))
        return {
            "material": self.materials[m_idx],
            "material_confidence": float(m_probs[m_idx]),
            "unevenness": self.unevenness[u_idx],
            "unevenness_confidence": float(u_probs[u_idx]),
            "all_materials": {n: float(p) for n, p in zip(self.materials, m_probs)},
            "all_unevenness": {n: float(p) for n, p in zip(self.unevenness, u_probs)},
        }
