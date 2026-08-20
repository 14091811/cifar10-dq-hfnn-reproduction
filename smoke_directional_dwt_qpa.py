"""Smoke test and timing check for the standalone directional DWT-QPA models."""

import time

import torch

from src.dq_hfnn.local_spa import (
    DirectionalDWTClassicalCNN,
    DirectionalDWTClassicalD8CNN,
    DirectionalDWTQuantumCNN,
    DirectionalDWTQuantumD8CNN,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    inputs = torch.randn(batch_size, 3, 32, 32, device=device)
    print(f"device={device} batch_size={batch_size}", flush=True)

    for name, model_class in (
        ("dwt_directional_classical_cnn", DirectionalDWTClassicalCNN),
        ("dwt_directional_qpa_cnn", DirectionalDWTQuantumCNN),
        ("dwt_directional_classical_d8_cnn", DirectionalDWTClassicalD8CNN),
        ("dwt_directional_qpa_d8_cnn", DirectionalDWTQuantumD8CNN),
    ):
        model = model_class().to(device).train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        output = model(inputs)
        output.mean().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        print(
            f"PASS {name}: output={tuple(output.shape)} "
            f"time={elapsed:.2f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
