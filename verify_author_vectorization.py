"""Check that class-vectorized author circuits match the original execution."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from dq_hfnn.model import DQHFNN


def build(vectorized):
    return DQHFNN(
        num_classes=10,
        hidden_dim=256,
        total_pairs=460,
        random_pair_ratio=1.0,
        circuit_variant="no_ent",
        sampler_seed=42,
        pair_source="pixels",
        pairing_layout="author_cifar",
        evaluation_pairing="fixed",
        vectorize_class_circuits=vectorized,
    ).eval()


def main():
    torch.manual_seed(42)
    original, vectorized = build(False), build(True)
    vectorized.load_state_dict(original.state_dict())
    images = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        original_logits, vectorized_logits = original(images), vectorized(images)
    max_error = (original_logits - vectorized_logits).abs().max().item()
    print(f"max_abs_logit_error={max_error:.8g}")
    if not torch.allclose(original_logits, vectorized_logits, rtol=1e-5, atol=1e-6):
        raise SystemExit("Vectorized author circuit is not numerically equivalent")
    print("author_class_vectorization=equivalent")


if __name__ == "__main__":
    main()
