"""
Entry-point script: train or evaluate the QICNN model.

Usage:
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --mode evaluate --checkpoint checkpoints/qicnn_best.pt
"""

import argparse
import sys
from pathlib import Path

import torch

# Allow relative imports from src/
sys.path.insert(0, str(Path(__file__).parent))

from model.qicnn import QICNN
from data.dataset import DeepfakeDataset, build_dataloader
from utils.config import load_config
from utils.trainer import Trainer
from utils.evaluator import Evaluator
from utils.logger import get_logger

logger = get_logger(__name__)


def build_model(cfg: dict) -> QICNN:
    m = cfg.get("model", {})
    q = cfg.get("quantum", {})
    return QICNN(
        num_classes=m.get("num_classes", 2),
        image_size=m.get("image_size", 224),
        channels=m.get("cnn_channels", [32, 64, 128, 256]),
        n_qubits=q.get("n_qubits", 8),
        n_quantum_layers=q.get("n_layers", 3),
        encoding=q.get("encoding", "angle"),
        dropout=m.get("dropout", 0.4),
        device_name=q.get("device", "default.qubit"),
    )


def build_loaders(cfg: dict):
    d_cfg = cfg.get("data", {})
    t_cfg = cfg.get("training", {})
    image_size = cfg.get("model", {}).get("image_size", 224)
    batch_size = t_cfg.get("batch_size", 32)
    num_workers = d_cfg.get("num_workers", 4)
    datasets_cfg = d_cfg.get("datasets", [])

    train_datasets, val_datasets = [], []
    for ds in datasets_cfg:
        ds_path = ds.get("path", "")
        if not Path(ds_path).exists():
            logger.warning("Dataset path does not exist, skipping: %s", ds_path)
            continue
        train_datasets.append(DeepfakeDataset(ds_path, split="train", image_size=image_size))
        val_datasets.append(DeepfakeDataset(ds_path, split="val", image_size=image_size))

    if not train_datasets:
        raise RuntimeError(
            "No valid dataset paths found. Please populate data directories as described in README."
        )

    train_loader = build_dataloader(train_datasets, batch_size=batch_size, num_workers=num_workers)
    val_loader = build_dataloader(val_datasets, batch_size=batch_size, num_workers=num_workers, balanced=False)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="QICNN Deepfake Detection")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    if args.mode == "train":
        train_loader, val_loader = build_loaders(cfg)
        trainer = Trainer(model, train_loader, val_loader, cfg)
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        trainer.fit()

    elif args.mode == "evaluate":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for evaluate mode")
        _, val_loader = build_loaders(cfg)
        evaluator = Evaluator(model, args.checkpoint)
        metrics = evaluator.evaluate_loader(val_loader)
        for k, v in metrics.items():
            logger.info("%s: %.4f", k, v)


if __name__ == "__main__":
    main()
