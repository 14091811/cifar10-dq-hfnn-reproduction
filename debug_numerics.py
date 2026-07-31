"""Find the first non-finite value in the DQ-HFNN forward/backward path."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dq_hfnn.model import DQHFNN


def report_tensor(name, value):
    finite = torch.isfinite(value)
    print(
        f"{name}: shape={tuple(value.shape)} "
        f"finite={finite.all().item()} "
        f"min={value.nan_to_num().min().item():.6g} "
        f"max={value.nan_to_num().max().item():.6g}"
    )


def main():
    torch.manual_seed(42)
    model = DQHFNN(
        total_pairs=153,
        random_pair_ratio=0.3,
        circuit_variant="strong_ent",
        pair_source="pixels",
    )
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))

    pairs = model.pair_source(images)
    memberships = model.memberships(images)
    classical = model.classical(images)
    quantum = model.quantum_features(images)
    logits = model.classifier(classical + quantum)
    loss = torch.nn.CrossEntropyLoss()(logits, labels)

    report_tensor("pairs", pairs)
    report_tensor("memberships", memberships)
    report_tensor("classical_features", classical)
    report_tensor("quantum_features", quantum)
    report_tensor("logits", logits)
    print(f"loss: {loss.item():.6g}")

    loss.backward()
    non_finite = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        is_finite = torch.isfinite(parameter.grad).all().item()
        maximum = parameter.grad.nan_to_num().abs().max().item()
        print(f"gradient {name}: finite={is_finite} max_abs={maximum:.6g}")
        if not is_finite:
            non_finite.append(name)
    print(f"non_finite_gradients: {non_finite}")


if __name__ == "__main__":
    main()
