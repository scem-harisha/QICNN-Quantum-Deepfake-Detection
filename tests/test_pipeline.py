"""
Unit tests for QICNN pipeline components.
Tests run on CPU with minimal qubits to stay fast.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantum.quantum_preprocessing import (
    AmplitudeEncoder,
    AngleEncoder,
    VariationalQuantumLayer,
    QuantumFeatureMap,
)
from model.qicnn import (
    ChannelAttention,
    SpatialAttention,
    ResidualBlock,
    CNNBackbone,
    QICNN,
)
from utils.metrics import compute_metrics, compute_confusion_matrix
from utils.config import load_config, merge_configs


# ---------------------------------------------------------------------------
# Quantum preprocessing tests
# ---------------------------------------------------------------------------

class TestAmplitudeEncoder:
    def test_output_shape(self):
        enc = AmplitudeEncoder(n_qubits=4)
        feat = torch.randn(16)
        out = enc.encode(feat)
        assert out.shape == (4,), f"Expected (4,), got {out.shape}"

    def test_output_range(self):
        enc = AmplitudeEncoder(n_qubits=4)
        feat = torch.randn(16)
        out = enc.encode(feat)
        assert (out >= -1.0).all() and (out <= 1.0).all()


class TestAngleEncoder:
    def test_output_shape(self):
        enc = AngleEncoder(n_qubits=4)
        feat = torch.randn(8)
        out = enc.encode(feat)
        assert out.shape == (4,)

    def test_output_range(self):
        enc = AngleEncoder(n_qubits=4)
        out = enc.encode(torch.zeros(8))
        assert (out >= -1.0).all() and (out <= 1.0).all()


class TestVariationalQuantumLayer:
    def test_forward_shape(self):
        vql = VariationalQuantumLayer(n_qubits=4, n_layers=1)
        x = torch.randn(2, 4)
        out = vql(x)
        assert out.shape == (2, 4), f"Expected (2,4), got {out.shape}"

    def test_gradients(self):
        vql = VariationalQuantumLayer(n_qubits=4, n_layers=1)
        x = torch.randn(1, 4)
        out = vql(x)
        loss = out.sum()
        loss.backward()
        assert vql.weights.grad is not None


class TestQuantumFeatureMap:
    def test_output_shape(self):
        qfm = QuantumFeatureMap(input_dim=16, n_qubits=4, n_layers=1)
        x = torch.randn(2, 16)
        out = qfm(x)
        assert out.shape == (2, 8), f"Expected (2,8), got {out.shape}"


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

class TestAttentionBlocks:
    def test_channel_attention(self):
        ca = ChannelAttention(channels=32)
        x = torch.randn(2, 32, 8, 8)
        out = ca(x)
        assert out.shape == x.shape

    def test_spatial_attention(self):
        sa = SpatialAttention()
        x = torch.randn(2, 64, 8, 8)
        out = sa(x)
        assert out.shape == x.shape


class TestResidualBlock:
    def test_same_channels(self):
        block = ResidualBlock(32, 32)
        x = torch.randn(2, 32, 16, 16)
        assert block(x).shape == (2, 32, 16, 16)

    def test_downsample(self):
        block = ResidualBlock(32, 64, stride=2)
        x = torch.randn(2, 32, 16, 16)
        assert block(x).shape == (2, 64, 8, 8)


class TestCNNBackbone:
    def test_output_dim(self):
        backbone = CNNBackbone(channels=[8, 16, 32])
        x = torch.randn(2, 3, 64, 64)
        out = backbone(x)
        assert out.shape == (2, 32), f"Expected (2,32), got {out.shape}"


class TestQICNN:
    """Full end-to-end forward pass with tiny config."""

    def _make_model(self):
        return QICNN(
            num_classes=2,
            image_size=32,
            channels=[8, 16],
            n_qubits=4,
            n_quantum_layers=1,
            encoding="angle",
            dropout=0.1,
        )

    def test_forward_shape(self):
        model = self._make_model()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 2), f"Expected (2,2), got {out.shape}"

    def test_extract_features(self):
        model = self._make_model()
        x = torch.randn(2, 3, 32, 32)
        feat = model.extract_features(x)
        # cnn_dim=16 + n_qubits*2=8 = 24
        assert feat.shape[0] == 2
        assert feat.shape[1] > 0

    def test_no_nan(self):
        model = self._make_model()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def _sample_data(self, n=20):
        labels = torch.tensor([0] * 10 + [1] * 10)
        logits = torch.randn(n, 2)
        # Force correct predictions by bumping the right class
        for i in range(n):
            logits[i, labels[i]] += 2.0
        return logits, labels

    def test_keys_present(self):
        logits, labels = self._sample_data()
        metrics = compute_metrics(logits, labels)
        for key in ["accuracy", "f1", "precision", "recall", "auc_roc"]:
            assert key in metrics

    def test_perfect_prediction(self):
        logits = torch.tensor([[10.0, 0.0], [10.0, 0.0], [0.0, 10.0], [0.0, 10.0]])
        labels = torch.tensor([0, 0, 1, 1])
        m = compute_metrics(logits, labels)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["auc_roc"] == pytest.approx(1.0)

    def test_confusion_matrix_shape(self):
        logits = torch.randn(10, 2)
        labels = torch.randint(0, 2, (10,))
        cm = compute_confusion_matrix(logits, labels)
        assert cm.shape == (2, 2)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_load_config(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("model:\n  num_classes: 2\ntraining:\n  epochs: 10\n")
        cfg = load_config(str(cfg_file))
        assert cfg["model"]["num_classes"] == 2
        assert cfg["training"]["epochs"] == 10

    def test_merge_configs(self):
        base = {"a": 1, "b": {"x": 10, "y": 20}}
        override = {"b": {"y": 99}, "c": 3}
        result = merge_configs(base, override)
        assert result["b"]["y"] == 99
        assert result["b"]["x"] == 10
        assert result["c"] == 3
