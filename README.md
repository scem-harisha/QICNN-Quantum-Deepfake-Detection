# QICNN – Quantum-Inspired CNN for Deepfake Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.36+-green.svg)](https://pennylane.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

A hybrid **Quantum-Inspired CNN** pipeline that leverages PennyLane variational quantum circuits as a preprocessing stage before classical convolutional layers, enabling cross-dataset generalisation for deepfake detection.

---

## Architecture Overview

```
Input Image (224×224)
       │
  CNN Backbone (ResNet-style with CBAM attention)
       │
  Classical Feature Vector (256-d)
       │
  ┌─────────────────────────────┐
  │  Quantum Feature Map         │
  │  ─ Classical projection      │
  │  ─ Variational Quantum Layer │   ← PennyLane StronglyEntanglingLayers
  │  ─ Post-quantum projection   │
  └─────────────────────────────┘
       │
  Fused Feature Vector  [classical || quantum]
       │
  Classification Head → Real / Fake
```

**Key components:**
| Component | Description |
|---|---|
| `AmplitudeEncoder` | Encodes feature vectors into qubit amplitudes |
| `AngleEncoder` | Encodes features via rotation angles |
| `VariationalQuantumLayer` | Trainable StronglyEntanglingLayers circuit |
| `QuantumFeatureMap` | Full classical→quantum→classical module |
| `CNNBackbone` | Multi-stage CNN with residual + CBAM blocks |
| `QICNN` | Hybrid model fusing CNN and quantum branches |

---

## Installation

```bash
git clone https://github.com/scem-harisha/QICNN-Quantum-Deepfake-Detection.git
cd QICNN-Quantum-Deepfake-Detection
pip install -r requirements.txt
# or editable install:
pip install -e .
```

### Requirements

- Python ≥ 3.9
- PennyLane ≥ 0.36
- PyTorch ≥ 2.0
- See `requirements.txt` for the full list

---

## Dataset Setup

Organise each dataset in the following layout:

```
data/
  ff++/
    real/   *.jpg ...
    fake/   *.jpg ...
  dfdc/
    real/   ...
    fake/   ...
  celebdf/
    real/   ...
    fake/   ...
```

Update `configs/config.yaml` with the actual paths.

---

## Training

```bash
python src/train.py --config configs/config.yaml
```

Resume from a checkpoint:

```bash
python src/train.py --config configs/config.yaml --checkpoint checkpoints/qicnn_best.pt
```

### Configuration

Edit `configs/config.yaml` to tune:

| Key | Default | Description |
|---|---|---|
| `quantum.n_qubits` | 8 | Number of qubits in the VQL |
| `quantum.n_layers` | 3 | VQL depth |
| `quantum.encoding` | `"angle"` | `"angle"` or `"amplitude"` |
| `model.cnn_channels` | `[32,64,128,256]` | Per-stage channel widths |
| `training.epochs` | 50 | Max training epochs |
| `training.learning_rate` | 1e-4 | Initial LR |

---

## Evaluation

```bash
python src/train.py --config configs/config.yaml \
  --mode evaluate \
  --checkpoint checkpoints/qicnn_best.pt
```

**Reported metrics:** Accuracy, AUC-ROC, F1-score, Precision, Recall.

---

## Cross-Dataset Generalisation

The model achieves cross-dataset generalisation through:

1. **Multi-dataset training** – Combine FaceForensics++, DFDC, CelebDF in a single loader
2. **Class-balanced sampling** – `WeightedRandomSampler` corrects label imbalance
3. **Domain-normalisation layer** – `LayerNorm` on the fused feature space
4. **Strong augmentation** – Gaussian blur, JPEG compression, random crops, colour jitter
5. **Quantum feature maps** – Variational circuits add non-linear feature transformations that are dataset-agnostic

---

## Project Structure

```
QICNN-Quantum-Deepfake-Detection/
├── configs/
│   └── config.yaml         # Hyperparameters and dataset paths
├── notebooks/
│   └── QICNN_Demo.ipynb    # End-to-end walkthrough
├── src/
│   ├── train.py            # CLI entry point
│   ├── quantum/
│   │   └── quantum_preprocessing.py
│   ├── model/
│   │   └── qicnn.py
│   ├── data/
│   │   └── dataset.py
│   └── utils/
│       ├── trainer.py
│       ├── evaluator.py
│       ├── metrics.py
│       ├── logger.py
│       └── config.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Results

| Dataset | Accuracy | AUC-ROC | F1 |
|---|---|---|---|
| FaceForensics++ | — | — | — |
| DFDC | — | — | — |
| CelebDF | — | — | — |

*(Populate after training with your dataset)*

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Open a pull request

---

## License

MIT License – see `LICENSE` for details.
