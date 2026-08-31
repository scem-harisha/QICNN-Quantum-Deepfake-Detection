"""
Inference engine for QICNN deepfake detection.
"""

from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn

from data.dataset import get_val_transforms
from utils.metrics import compute_metrics


class Evaluator:
    """
    Run inference with a trained QICNN model.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: str,
        image_size: int = 224,
        device: torch.device = None,
        threshold: float = 0.5,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.transform = get_val_transforms(image_size)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model

    @torch.no_grad()
    def predict_image(self, image_path: Union[str, Path]) -> dict:
        """
        Predict on a single image file.

        Returns:
            {"label": 0|1, "prob_fake": float, "prob_real": float}
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=img)["image"].unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        label = int(probs[1] >= self.threshold)
        return {"label": label, "prob_real": float(probs[0]), "prob_fake": float(probs[1])}

    @torch.no_grad()
    def evaluate_loader(self, loader) -> dict:
        """Run evaluation on a DataLoader and return aggregate metrics."""
        self.model.eval()
        all_logits, all_labels = [], []
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            logits = self.model(images)
            all_logits.append(logits.cpu())            all_labels.append(labels)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        return compute_metrics(all_logits, all_labels, threshold=self.threshold)
