"""Monocular depth (Depth Anything V2 Small, Apache-2.0) for 2.5D parallax."""
from __future__ import annotations

import gc

import numpy as np
from PIL import Image

from .core import MODELS, torch_device


class DepthEstimator:
    def __init__(self, log):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        path = MODELS / "depth-anything-v2-small"
        self.device = torch_device()
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(path)
        self.model = AutoModelForDepthEstimation.from_pretrained(path).to(self.device).eval()
        log.info(f"Depth model loaded on {self.device}")

    def estimate(self, image_path, out_path):
        img = Image.open(image_path).convert("RGB")
        inputs = {k: v.to(self.device) for k, v in self.proc(images=img, return_tensors="pt").items()}
        with self.torch.no_grad():
            pred = self.model(**inputs).predicted_depth
        pred = self.torch.nn.functional.interpolate(pred.unsqueeze(1), size=img.size[::-1], mode="bicubic", align_corners=False)
        d = pred.squeeze().float().cpu().numpy()
        d = (d - d.min()) / max(1e-6, d.max() - d.min())  # 1 = near, 0 = far
        Image.fromarray((d * 255).astype(np.uint8)).save(out_path)

    def close(self):
        del self.model
        gc.collect()
