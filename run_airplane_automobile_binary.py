"""Matched train/validation-only CIFAR-10 Airplane vs Automobile study."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "cifar10_airplane_automobile_binary.json"
MODELS = (
    "compact_classical_cnn",
    "compact_dwt_directional_classical_d8_cnn",
    "compact_dwt_directional_qpa_d8_cnn",
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
    generated = ROOT / "configs" / "generated_airplane_automobile"
    generated.mkdir(exist_ok=True)
    for seed in seeds:
        for model_type in models:
            cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
            cfg["model_type"] = model_type
            cfg["seed"] = seed
            cfg["run_name"] = f"{cfg['binary_task_name']}_{model_type}"
            path = generated / f"{model_type}_seed{seed}.json"
            path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            print(f"starting model={model_type} seed={seed}", flush=True)
            subprocess.run(
                [sys.executable, "train_cifar10.py", "--config", str(path)],
                cwd=ROOT,
                check=True,
            )


if __name__ == "__main__":
    main()
