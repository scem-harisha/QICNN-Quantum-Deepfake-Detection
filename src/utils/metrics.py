"""
Evaluation metrics for deepfake detection.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    """
    Compute classification metrics from raw logits.

    Args:
        logits: (N, num_classes) raw output from the model.
        labels: (N,) integer ground-truth labels.
        threshold: Decision threshold for binary classification.

    Returns:
        Dictionary with accuracy, auc_roc, f1, precision, recall.
    """
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = (probs[:, 1] >= threshold).astype(int)
    y_true = labels.cpu().numpy()

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }

    # AUC-ROC (requires both classes present)
    if len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, probs[:, 1]))
    else:
        metrics["auc_roc"] = float("nan")

    return metrics


def compute_confusion_matrix(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    """Return confusion matrix as a numpy array."""
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = (probs[:, 1] >= threshold).astype(int)
    return confusion_matrix(labels.cpu().numpy(), preds)
