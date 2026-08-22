"""Validate the three learned-D16 QPA circuit variants."""

import time

import torch

from train_medmnist import MODELS


MODELS_TO_TEST = (
    "two_block_dwt_directional_qpa_learned_d16_circuit_baseline_cnn",
    "two_block_dwt_directional_qpa_learned_d16_circuit_no_entanglement_cnn",
    "two_block_dwt_directional_qpa_learned_d16_circuit_rz_cnn",
    "two_block_dwt_directional_qpa_learned_d16_circuit_single_cnot_cnn",
    "two_block_dwt_directional_qpa_learned_d16_circuit_symmetric_ry_cnn",
    "two_block_dwt_directional_qpa_learned_d16_circuit_rz_single_cnot_cnn",
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = torch.randn(2, 3, 32, 32, device=device)
    labels = torch.tensor([0, 1], device=device)
    print(f"device={device} configurations={len(MODELS_TO_TEST)}", flush=True)
    for name in MODELS_TO_TEST:
        started = time.perf_counter()
        model = MODELS[name](num_classes=2, hidden_dim=128).to(device)
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        print(
            f"PASS {name} params={sum(p.numel() for p in model.parameters()):,} "
            f"loss={loss.item():.4f} time={time.perf_counter() - started:.2f}s",
            flush=True,
        )
        del model, logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("ALL CIRCUIT CONFIGURATIONS PASS", flush=True)


if __name__ == "__main__":
    main()
