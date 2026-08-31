"""
Quantum Preprocessing Module for QICNN Deepfake Detection.

This module implements quantum circuits for feature encoding using PennyLane,
including amplitude encoding, angle encoding, and variational quantum layers.
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


def _build_device(n_qubits: int, device_name: str = "default.qubit"):
    """Create a PennyLane quantum device."""
    return qml.device(device_name, wires=n_qubits)


class AmplitudeEncoder:
    """Encode a normalised feature vector into qubit amplitudes."""

    def __init__(self, n_qubits: int, device_name: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.device = _build_device(n_qubits, device_name)
        self._build_circuit()

    def _build_circuit(self):
        n_qubits = self.n_qubits

        @qml.qnode(self.device, interface="torch", diff_method="best")
        def circuit(features):
            # Pad or truncate features to match 2^n_qubits amplitudes
            size = 2 ** n_qubits
            padded = torch.zeros(size, dtype=features.dtype, device=features.device)
            length = min(features.shape[0], size)
            padded[:length] = features[:length]
            norm = torch.norm(padded)
            padded = padded / (norm + 1e-8)
            qml.AmplitudeEmbedding(padded, wires=range(n_qubits), normalize=False)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Return expectation values as an encoded feature vector."""
        return torch.stack(self.circuit(features))


class AngleEncoder:
    """Encode features using rotation-angle embedding."""

    def __init__(self, n_qubits: int, device_name: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.device = _build_device(n_qubits, device_name)
        self._build_circuit()

    def _build_circuit(self):
        n_qubits = self.n_qubits

        @qml.qnode(self.device, interface="torch", diff_method="best")
        def circuit(features):
            qml.AngleEmbedding(features[:n_qubits], wires=range(n_qubits), rotation="Y")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack(self.circuit(features))


class VariationalQuantumLayer(nn.Module):
    """
    A trainable variational quantum layer with strongly entangling ansatz.
    Acts as a drop-in nn.Module, differentiable via PennyLane's torch interface.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
        self.weights = nn.Parameter(torch.randn(weight_shape) * 0.1)

        dev = _build_device(n_qubits, device_name)

        @qml.qnode(dev, interface="torch", diff_method="best")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_qubits) float tensor.
        Returns:
            (batch, n_qubits) expectation values (float32).
        """
        outputs = []
        for sample in x:
            out = self.circuit(sample, self.weights)
            outputs.append(torch.stack(out))
        return torch.stack(outputs).float()


class QuantumFeatureMap(nn.Module):
    """
    Full quantum feature map: classical projection → VQL → output.

    Maps an arbitrary input dimension to n_qubits quantum features,
    enabling dimensionality reduction through quantum operations.
    """

    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 8,
        n_layers: int = 3,
        encoding: str = "angle",
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoding = encoding

        # Classical projection to qubit-compatible dimension
        self.projection = nn.Sequential(
            nn.Linear(input_dim, n_qubits * 2),
            nn.LayerNorm(n_qubits * 2),
            nn.Tanh(),
            nn.Linear(n_qubits * 2, n_qubits),
            nn.Tanh(),
        )

        self.vql = VariationalQuantumLayer(n_qubits, n_layers, device_name)

        # Post-quantum projection
        self.post_proj = nn.Sequential(
            nn.Linear(n_qubits, n_qubits * 2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) tensor.
        Returns:
            (batch, n_qubits * 2) quantum-enhanced feature tensor.
        """
        projected = self.projection(x)           # (batch, n_qubits)
        quantum_out = self.vql(projected)         # (batch, n_qubits)
        return self.post_proj(quantum_out)        # (batch, n_qubits*2)
