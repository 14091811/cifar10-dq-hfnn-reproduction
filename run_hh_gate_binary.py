"""Screen the HH-inclusive high-frequency gate on two binary datasets."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEEDS = (42, 456, 789)
DATASETS = {
    "breastmnist": ROOT / "configs" / "breastmnist_paper_protocol.json",
    "pneumoniamnist": ROOT / "configs" / "pneumoniamnist_2x2_fixed50.json",
}
MODELS = (
    "two_block_dwt_directional_classical_hh_gate_d16_cnn",
    "two_block_dwt_directional_qpa_hh_gate_d16_cnn",
)


def parse_csv(value):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("value must not be empty")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()

    datasets = parse_csv(args.datasets)
    seeds = tuple(int(value) for value in parse_csv(args.seeds))
    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset: {dataset}")
        base_config = json.loads(DATASETS[dataset].read_text(encoding="utf-8"))
        output_dir = ROOT / "configs" / f"generated_{dataset}_hh_gate"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"dataset={dataset} seeds={seeds}", flush=True)
        for seed in seeds:
            for model_type in MODELS:
                cfg = dict(base_config)
                cfg["seed"] = seed
                cfg["model_type"] = model_type
                cfg["run_name"] = f"{dataset}_{model_type}"
                generated = output_dir / f"{model_type}_seed{seed}.json"
                generated.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                print(f"starting model={model_type} seed={seed}", flush=True)
                subprocess.run(
                    [sys.executable, "-u", "train_medmnist.py", "--config", str(generated)],
                    cwd=ROOT,
                    check=True,
                )
    print("HH gate binary screening complete", flush=True)


if __name__ == "__main__":
    main()
