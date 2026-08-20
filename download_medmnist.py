"""Download and inspect the MedMNIST v2 datasets used by H-QDCT studies."""

from __future__ import annotations

import argparse
from pathlib import Path


DATASETS = (
    "breastmnist",
    "retinamnist",
    "pneumoniamnist",
    "dermamnist",
    "bloodmnist",
    "pathmnist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="./medmnist_data",
        help="Directory in which MedMNIST files are stored.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated dataset names, or 'all'.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=28,
        choices=(28, 64, 128),
        help="MedMNIST image size to download.",
    )
    return parser.parse_args()


def load_dataset_class(name: str):
    from medmnist import INFO
    import medmnist

    if name not in INFO:
        valid = ", ".join(DATASETS)
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {valid}")
    return getattr(medmnist, INFO[name]["python_class"])


def main() -> None:
    args = parse_args()
    names = DATASETS if args.datasets.strip().lower() == "all" else tuple(
        item.strip().lower() for item in args.datasets.split(",") if item.strip()
    )
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    try:
        import medmnist  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "medmnist is not installed. Run: python -m pip install medmnist"
        ) from exc

    print(f"Download root: {root}")
    print(f"Image size: {args.size}x{args.size}")
    for name in names:
        dataset_class = load_dataset_class(name)
        print(f"\n[{name}] downloading train/val/test")
        for split in ("train", "val", "test"):
            dataset = dataset_class(
                split=split,
                root=str(root),
                download=True,
                size=args.size,
            )
            labels = getattr(dataset, "labels", None)
            class_count = len(set(int(label[0]) for label in labels)) if labels is not None else "?"
            print(f"  {split}: n={len(dataset)} classes={class_count}")

    print("\nMedMNIST download complete.")


if __name__ == "__main__":
    main()
