"""Detect optimizer-step instability without downloading or reading a dataset."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dq_hfnn.model import DQHFNN


def finite_model(model):
    return all(torch.isfinite(parameter).all() for parameter in model.parameters())


def main():
    torch.manual_seed(42)
    model = DQHFNN(total_pairs=153, random_pair_ratio=0.3, circuit_variant="strong_ent")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.012)
    loss_fn = torch.nn.CrossEntropyLoss()
    for step in range(1, 101):
        images = torch.randn(128, 3, 32, 32)
        labels = torch.randint(0, 10, (128,))
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            print(f"non_finite_loss_before_step={step}")
            return
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if not finite_model(model):
            print(f"non_finite_parameter_after_step={step}")
            return
        if step == 1 or step % 10 == 0:
            print(f"step={step} loss={loss.item():.6f}")
    print("completed_100_steps_without_non_finite_values")


if __name__ == "__main__":
    main()
