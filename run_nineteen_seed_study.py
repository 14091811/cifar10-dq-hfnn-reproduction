"""Run one optimized author seed-42 validation plus nine matched seed pairs."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REMAINING_SEEDS = (456, 789, 123, 1024, 2024, 3030, 4040, 5050, 6060)
AUTHOR_CONFIG = ROOT / "configs" / "cifar10_author_source_a_vectorized.json"
LOCAL_CONFIG = ROOT / "configs" / "cifar10_v215local_authorfusion.json"


def run(config_path, seed):
    config = json.loads(config_path.read_text())
    config["seed"] = seed
    config["run_name"] = f"{config['run_name']}_seed{seed}"
    generated = ROOT / "configs" / "generated"
    generated.mkdir(exist_ok=True)
    path = generated / f"{config_path.stem}_seed{seed}.json"
    path.write_text(json.dumps(config, indent=2))
    print(f"\n{'=' * 72}\nseed={seed} config={config_path.name}\n{'=' * 72}", flush=True)
    subprocess.run([sys.executable, "train_cifar10.py", "--config", str(path)], cwd=ROOT, check=True)


run(AUTHOR_CONFIG, 42)
for seed in REMAINING_SEEDS:
    run(AUTHOR_CONFIG, seed)
    run(LOCAL_CONFIG, seed)
