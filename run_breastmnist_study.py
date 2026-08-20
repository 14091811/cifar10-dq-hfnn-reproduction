"""Run the BreastMNIST single-head D64 comparison."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "breastmnist_paper_protocol.json"
MODELS = (
    "two_block_dwt_directional_classical_d64_cnn",
    "two_block_dwt_directional_qpa_d64_cnn",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,456,789,5050,6060")
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = set(models) - set(MODELS)
    if unknown:
        raise ValueError(f"Unsupported models: {sorted(unknown)}")
    output = ROOT / "configs" / "generated_breastmnist"
    output.mkdir(exist_ok=True)
    for seed in seeds:
        for model_type in models:
            cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
            cfg["seed"] = seed
            cfg["model_type"] = model_type
            cfg["run_name"] = f"breastmnist_{model_type}"
            config_path = output / f"{model_type}_seed{seed}.json"
            config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            print(f"starting model={model_type} seed={seed}", flush=True)
            subprocess.run(
                [sys.executable, "-u", "train_medmnist.py", "--config", str(config_path)],
                cwd=ROOT,
                check=True,
            )


if __name__ == "__main__":
    main()
