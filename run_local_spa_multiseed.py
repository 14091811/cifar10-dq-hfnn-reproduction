"""Serial CIFAR-10 comparison for one local SP&A enhancement."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "cifar10_local_spa.json"
MODELS = (
    "classical_cnn",
    "yang_spa_cnn",
    "dwt_classical_spa_cnn",
    "dwt_qpa_spa_cnn",
    "dwt_directional_classical_cnn",
    "dwt_directional_qpa_cnn",
    "dwt_directional_classical_d8_cnn",
    "dwt_directional_qpa_d8_cnn",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,456,789")
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    generated = ROOT / "configs" / "generated_local_spa"
    generated.mkdir(exist_ok=True)

    for model_type in models:
        if model_type not in MODELS:
            raise ValueError(f"Unsupported model: {model_type}")
        for seed in seeds:
            cfg = json.loads(CONFIG.read_text())
            cfg["model_type"] = model_type
            cfg["seed"] = seed
            cfg["run_name"] = f"{model_type}_seed{seed}"
            path = generated / f"{model_type}_seed{seed}.json"
            path.write_text(json.dumps(cfg, indent=2))
            print(f"starting model={model_type} seed={seed}", flush=True)
            subprocess.run(
                [sys.executable, "train_cifar10.py", "--config", str(path)],
                cwd=ROOT,
                check=True,
            )


if __name__ == "__main__":
    main()
