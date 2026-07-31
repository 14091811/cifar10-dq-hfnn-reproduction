"""Complete the nine remaining matched seeds after the seed-42 pilot runs."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEEDS = (456, 789, 123, 1024, 2024, 3030, 4040, 5050, 6060)
CONFIGS = (
    ROOT / "configs" / "cifar10_author_source_a_vectorized.json",
    ROOT / "configs" / "cifar10_v215local_authorfusion.json",
)


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


for seed in SEEDS:
    for config_path in CONFIGS:
        run(config_path, seed)
