"""
Hybrid Quantum-Classical CNN for Deepfake Detection.

Architecture:
  - Classical CNN backbone with residual blocks and channel attention
  - Quantum feature map branch (QuantumFeatureMap)
  - Late fusion of classical and quantum features
  - Binary classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantum.quantum_preprocessing import QuantumFeatureMap


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx = self.fc(self.max_pool(x).view(b, c))
        scale = torch.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention from CBAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))
        return x * attn


class ResidualBlock(nn.Module):
    """Pre-activation residual block with CBAM attention."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.ca(out)
        out = self.sa(out)
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


# ---------------------------------------------------------------------------
# CNN Backbone
# ---------------------------------------------------------------------------

class CNNBackbone(nn.Module):
    """Multi-stage CNN backbone producing a flat feature vector."""

    def __init__(self, channels: list[int] = None, input_channels: int = 3):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        layers = [
            nn.Conv2d(input_channels, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        ]

        in_ch = channels[0]
        for out_ch in channels[1:]:
            layers.append(ResidualBlock(in_ch, out_ch, stride=2))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)


# ---------------------------------------------------------------------------
# Hybrid QICNN
# ---------------------------------------------------------------------------

class QICNN(nn.Module):
    """
    Quantum-Inspired CNN for deepfake detection.

    Classical CNN path and a quantum feature map path are fused before the
    final classification head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        image_size: int = 224,
        channels: list[int] = None,
        n_qubits: int = 8,
        n_quantum_layers: int = 3,
        encoding: str = "angle",
        dropout: float = 0.4,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        self.backbone = CNNBackbone(channels)
        cnn_dim = self.backbone.out_dim          # e.g. 256

        # Quantum branch: takes CNN features → quantum-enhanced features
        self.quantum_map = QuantumFeatureMap(
            input_dim=cnn_dim,
            n_qubits=n_qubits,
            n_layers=n_quantum_layers,
            encoding=encoding,
            device_name=device_name,
        )
        quantum_out_dim = n_qubits * 2           # from QuantumFeatureMap

        # Domain adaptation: feature normalisation
        self.domain_norm = nn.LayerNorm(cnn_dim + quantum_out_dim)

        # Classification head
        fusion_dim = cnn_dim + quantum_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.backbone(x)               # (B, cnn_dim)
        q_feat = self.quantum_map(cnn_feat)       # (B, n_qubits*2)
        fused = torch.cat([cnn_feat, q_feat], dim=1)
        fused = self.domain_norm(fused)
        return self.classifier(fused)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return fused feature vector before the classifier."""
        cnn_feat = self.backbone(x)
        q_feat = self.quantum_map(cnn_feat)
        return self.domain_norm(torch.cat([cnn_feat, q_feat], dim=1))
