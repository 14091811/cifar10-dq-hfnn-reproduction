"""Run the complete PathMNIST 10-seed 2x2 fixed-50 comparison."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "pathmnist_2x2_fixed50.json"
SEEDS = (42, 456, 789, 5050, 6060, 123, 1024, 2024, 3030, 4040)
MODELS = (
    "two_block_classical_cnn",
    "two_block_dwt_directional_classical_learned_partial_d16_cnn",
    "two_block_dwt_directional_qpa_learned_partial_d16_cnn",
    "two_block_dwt_directional_classical_partial_d16_cnn",
    "two_block_dwt_directional_qpa_partial_d16_cnn",
)


def parse_seeds(value):
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("seed list must not be empty")
    return seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    if base_config.get("dataset") != "pathmnist":
        raise ValueError(f"expected pathmnist config, got {base_config.get('dataset')!r}")

    seeds = parse_seeds(args.seeds)
    output_dir = ROOT / "configs" / "generated_pathmnist"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"dataset={base_config['dataset']} epochs={base_config['epochs']} seeds={seeds}", flush=True)

    for seed in seeds:
        for model_type in MODELS:
            cfg = dict(base_config)
            cfg["seed"] = seed
            cfg["model_type"] = model_type
            cfg["run_name"] = f"{cfg['dataset']}_{model_type}"
            generated = output_dir / f"{model_type}_seed{seed}.json"
            generated.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            print(f"starting model={model_type} seed={seed}", flush=True)
            subprocess.run(
                [sys.executable, "-u", "train_medmnist.py", "--config", str(generated)],
                cwd=ROOT,
                check=True,
            )

    print("PathMNIST 10-seed study complete", flush=True)


if __name__ == "__main__":
    main()
