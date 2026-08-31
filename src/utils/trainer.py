"""
Training loop for QICNN deepfake detection.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from utils.metrics import compute_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Manages the training, validation, and checkpointing of a QICNN model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        t_cfg = config.get("training", {})
        self.epochs = t_cfg.get("epochs", 50)
        self.gradient_clip = t_cfg.get("gradient_clip", 1.0)
        self.mixed_precision = t_cfg.get("mixed_precision", True) and self.device.type == "cuda"
        self.checkpoint_dir = Path(config.get("paths", {}).get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = AdamW(
            model.parameters(),
            lr=t_cfg.get("learning_rate", 1e-4),
            weight_decay=t_cfg.get("weight_decay", 1e-5),
        )

        warmup_epochs = t_cfg.get("warmup_epochs", 5)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
                CosineAnnealingLR(self.optimizer, T_max=self.epochs - warmup_epochs),
            ],
            milestones=[warmup_epochs],
        )

        self.scaler = GradScaler(enabled=self.mixed_precision)
        self.best_val_auc = 0.0
        self.patience = t_cfg.get("early_stopping_patience", 10)
        self._no_improve = 0
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [], "val_auc": [], "val_f1": [], "val_acc": []
        }

    # ------------------------------------------------------------------

    def train_epoch(self) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            with autocast(enabled=self.mixed_precision):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item() * images.size(0)
            n += images.size(0)
        return total_loss / n

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss, n = 0.0, 0
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with autocast(enabled=self.mixed_precision):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item() * images.size(0)
            n += images.size(0)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_logits, all_labels)
        metrics["val_loss"] = total_loss / n
        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], tag: str = "best"):
        path = self.checkpoint_dir / f"qicnn_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        logger.info("Saved checkpoint → %s", path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("Loaded checkpoint from %s (epoch %d)", path, ckpt["epoch"])
        return ckpt

    # ------------------------------------------------------------------

    def fit(self):
        """Run full training loop with early stopping."""
        logger.info("Starting training on %s for %d epochs", self.device, self.epochs)
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_auc"].append(val_metrics["auc_roc"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            elapsed = time.time() - t0
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
                "auc=%.4f | f1=%.4f | acc=%.4f | %.1fs",
                epoch, self.epochs,
                train_loss, val_metrics["val_loss"],
                val_metrics["auc_roc"], val_metrics["f1"], val_metrics["accuracy"],
                elapsed,
            )

            # Checkpoint best model
            if val_metrics["auc_roc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc_roc"]
                self.save_checkpoint(epoch, val_metrics, tag="best")
                self._no_improve = 0
            else:
                self._no_improve += 1

            # Periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_metrics, tag=f"epoch{epoch}")

            # Early stopping
            if self._no_improve >= self.patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, self.patience)
                break

        logger.info("Training complete. Best val AUC: %.4f", self.best_val_auc)
        return self.history
