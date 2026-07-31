"""Run the author source baseline and V215-local fusion for matched CIFAR seeds."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(config_path, seed):
    config = json.loads(config_path.read_text())
    config["seed"] = seed
    config["run_name"] = f"{config['run_name']}_seed{seed}"
    generated_dir = ROOT / "configs" / "generated"
    generated_dir.mkdir(exist_ok=True)
    generated_path = generated_dir / f"{config_path.stem}_seed{seed}.json"
    generated_path.write_text(json.dumps(config, indent=2))
    print(f"\n{'=' * 72}\nseed={seed} config={config_path.name}\n{'=' * 72}", flush=True)
    subprocess.run([sys.executable, "train_cifar10.py", "--config", str(generated_path)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="456,789")
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    configs = (
        ROOT / "configs" / "cifar10_author_source_a.json",
        ROOT / "configs" / "cifar10_v215local_authorfusion.json",
    )
    for seed in seeds:
        for config_path in configs:
            run(config_path, seed)


if __name__ == "__main__":
    main()
