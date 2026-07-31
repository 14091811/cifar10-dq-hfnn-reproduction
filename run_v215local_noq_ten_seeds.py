"""Run the matched no-quantum control for all reported V215-local seeds."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEEDS = (42, 456, 789, 123, 1024, 2024, 3030, 4040, 5050, 6060)
CONFIG = ROOT / "configs" / "cifar10_v215local_noq_authorfusion.json"


for seed in SEEDS:
    config = json.loads(CONFIG.read_text())
    config["seed"] = seed
    config["run_name"] = f"{config['run_name']}_seed{seed}"
    generated = ROOT / "configs" / "generated"
    generated.mkdir(exist_ok=True)
    path = generated / f"{CONFIG.stem}_seed{seed}.json"
    path.write_text(json.dumps(config, indent=2))
    print(f"\n{'=' * 72}\nseed={seed} config={CONFIG.name}\n{'=' * 72}", flush=True)
    subprocess.run([sys.executable, "train_cifar10.py", "--config", str(path)], cwd=ROOT, check=True)
