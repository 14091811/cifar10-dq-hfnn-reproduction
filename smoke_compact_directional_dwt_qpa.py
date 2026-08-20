"""Forward/backward smoke test for the compact paired DWT-QPA controls."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from dq_hfnn.local_spa import (
    CompactClassicalCNN,
    CompactDirectionalDWTClassicalD8CNN,
    CompactDirectionalDWTQuantumD8CNN,
    TwoBlockClassicalCNN,
    TwoBlockDirectionalDWTClassicalD8CNN,
    TwoBlockDirectionalDWTQuantumD8CNN,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        "compact_classical_cnn": CompactClassicalCNN,
        "compact_dwt_directional_classical_d8_cnn": CompactDirectionalDWTClassicalD8CNN,
        "compact_dwt_directional_qpa_d8_cnn": CompactDirectionalDWTQuantumD8CNN,
        "two_block_classical_cnn": TwoBlockClassicalCNN,
        "two_block_dwt_directional_classical_d8_cnn": TwoBlockDirectionalDWTClassicalD8CNN,
        "two_block_dwt_directional_qpa_d8_cnn": TwoBlockDirectionalDWTQuantumD8CNN,
    }
    print(f"device={device}")
    for name, model_class in models.items():
        model = model_class().to(device).train()
        inputs = torch.randn(2, 3, 32, 32, device=device)
        logits = model(inputs)
        logits.square().mean().backward()
        print(f"PASS {name}: output={tuple(logits.shape)}")


if __name__ == "__main__":
    main()
