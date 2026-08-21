"""Inspect RetinaMNIST class balance and latest run confusion matrices."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


MODELS = (
    "two_block_classical_cnn",
    "two_block_dwt_directional_classical_learned_partial_d16_cnn",
    "two_block_dwt_directional_qpa_learned_partial_d16_cnn",
    "two_block_dwt_directional_classical_partial_d16_cnn",
    "two_block_dwt_directional_qpa_partial_d16_cnn",
)


def class_counts(root, dataset_name):
    from medmnist import INFO
    import medmnist

    dataset_class = getattr(medmnist, INFO[dataset_name]["python_class"])
    labels_by_split = {}
    for split in ("train", "val", "test"):
        dataset = dataset_class(split=split, root=str(root), download=False)
        labels = np.asarray(dataset.labels).reshape(-1).astype(int).tolist()
        labels_by_split[split] = labels
    return labels_by_split


def latest_result(root, experiment, seed):
    seed_dir = root / "runs" / experiment / f"seed_{seed}"
    pointer = seed_dir / "latest.txt"
    if not pointer.exists():
        return None
    attempt = pointer.read_text(encoding="utf-8").strip()
    result_path = seed_dir / attempt / "result.json"
    if not result_path.exists():
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))


def print_confusion(matrix):
    total = sum(sum(row) for row in matrix)
    print("confusion_matrix (rows=true, cols=pred):")
    for row in matrix:
        print("  " + " ".join(f"{value:5d}" for value in row))
    print("per-class recall:")
    recalls = []
    for index, row in enumerate(matrix):
        support = sum(row)
        recall = row[index] / support if support else 0.0
        recalls.append(recall)
        print(f"  class {index}: support={support:4d} recall={recall * 100:6.2f}%")
    print(f"macro recall from matrix: {sum(recalls) / len(recalls) * 100:.2f}%")
    print(f"total samples: {total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", default="retinamnist")
    parser.add_argument("--data-root", default="medmnist_data")
    parser.add_argument("--seeds", default="42,456,789,5050,6060,123,1024,2024,3030,4040")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_root = root / args.data_root
    labels_by_split = class_counts(data_root, args.dataset)
    print(f"Dataset={args.dataset}")
    print(f"Root={data_root}")
    for split, labels in labels_by_split.items():
        counts = Counter(labels)
        print(f"{split}: n={len(labels)} counts={dict(sorted(counts.items()))}")
    print(f"Uniform random accuracy baseline: {100 / len(set(labels_by_split['train'])):.2f}%")

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    for model in MODELS:
        experiment = f"{args.dataset}_{model}"
        print("\n" + "=" * 96)
        print(f"MODEL={model}")
        found = 0
        for seed in seeds:
            result = latest_result(root, experiment, seed)
            if result is None:
                continue
            found += 1
            print(
                f"Seed={seed} best_epoch={result.get('best_epoch')} "
                f"test_acc={result.get('test_accuracy', 0) * 100:.2f}% "
                f"test_f1={result.get('test_macro_f1', 0) * 100:.2f}% "
                f"test_auc={result.get('test_auc', 0) * 100:.2f}%"
            )
            matrix = result.get("test_confusion_matrix")
            if matrix:
                print_confusion(matrix)
        print(f"runs_found={found}/{len(seeds)}")


if __name__ == "__main__":
    main()
