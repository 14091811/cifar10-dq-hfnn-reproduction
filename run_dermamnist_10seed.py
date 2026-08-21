"""Complete the DermaMNIST 10-seed 2x2 fixed-50 study.

The first five seeds for the DWT-D16 models were already run.  This runner
adds the remaining five seeds for those models and runs the no-DWT baseline
for all ten seeds.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "dermamnist_2x2_fixed50.json"

INITIAL_SEEDS = (42, 456, 789, 5050, 6060)
REMAINING_SEEDS = (123, 1024, 2024, 3030, 4040)

BASELINE = "two_block_classical_cnn"
DWT_MODELS = (
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


def run_model(base_config, output_dir, model_type, seed):
    cfg = dict(base_config)
    cfg["seed"] = seed
    cfg["model_type"] = model_type
    cfg["run_name"] = f"{cfg['dataset']}_{model_type}"
    config_path = output_dir / f"{model_type}_seed{seed}.json"
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"starting model={model_type} seed={seed}", flush=True)
    subprocess.run(
        [sys.executable, "-u", "train_medmnist.py", "--config", str(config_path)],
        cwd=ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--initial-seeds", default=",".join(map(str, INITIAL_SEEDS)))
    parser.add_argument("--remaining-seeds", default=",".join(map(str, REMAINING_SEEDS)))
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    if base_config.get("dataset") != "dermamnist":
        raise ValueError(f"expected dermamnist config, got {base_config.get('dataset')!r}")

    initial_seeds = parse_seeds(args.initial_seeds)
    remaining_seeds = parse_seeds(args.remaining_seeds)
    all_seeds = initial_seeds + tuple(seed for seed in remaining_seeds if seed not in initial_seeds)
    output_dir = ROOT / "configs" / "generated_dermamnist"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset={base_config['dataset']} epochs={base_config['epochs']}", flush=True)
    print(f"baseline seeds={all_seeds}", flush=True)
    for seed in all_seeds:
        run_model(base_config, output_dir, BASELINE, seed)

    if not args.baseline_only:
        print(f"DWT remaining seeds={remaining_seeds}", flush=True)
        for seed in remaining_seeds:
            for model_type in DWT_MODELS:
                run_model(base_config, output_dir, model_type, seed)

    print("DermaMNIST 10-seed study complete", flush=True)


if __name__ == "__main__":
    main()
