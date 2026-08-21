"""Validate every learned-gate configuration before launching the full study."""

import time

import torch

from train_medmnist import MODELS


DIMENSIONS = (8, 16, 32)
MODEL_NAMES = tuple(
    name
    for gate in ("", "standard_")
    for dimension in DIMENSIONS
    for name in (
        f"two_block_dwt_directional_classical_{gate}learned_partial_d{dimension}_cnn",
        f"two_block_dwt_directional_qpa_{gate}learned_partial_d{dimension}_cnn",
    )
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = torch.randn(2, 3, 32, 32, device=device)
    labels = torch.tensor([0, 1], device=device)
    print(f"device={device} configurations={len(MODEL_NAMES)}", flush=True)
    for name in MODEL_NAMES:
        started = time.perf_counter()
        model = MODELS[name](num_classes=2, hidden_dim=128).to(device)
        logits = model(images)
        if logits.shape != (2, 2):
            raise RuntimeError(f"{name}: unexpected output shape {tuple(logits.shape)}")
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
    print("ALL CONFIGURATIONS PASS", flush=True)


if __name__ == "__main__":
    main()
