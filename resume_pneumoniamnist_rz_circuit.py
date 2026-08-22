"""Resume the missing PneumoniaMNIST RZ circuit seeds."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = "pneumoniamnist"
MODEL = "two_block_dwt_directional_qpa_learned_d16_circuit_rz_cnn"
SEEDS = (123, 1024, 2024, 3030, 4040)
BASE_CONFIG = ROOT / "configs" / "pneumoniamnist_2x2_fixed50.json"
CONFIG_DIR = ROOT / "configs" / "generated_pneumoniamnist_circuit_ablation"


def make_config(seed):
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config.update(
        {
            "seed": seed,
            "model_type": MODEL,
            "run_name": f"{DATASET}_{MODEL}",
            "early_stopping_patience": 999,
        }
    )
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{MODEL}_seed{seed}.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def main():
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(BASE_CONFIG)
    log_dir = ROOT / "study_logs" / "circuit_ablation"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pneumoniamnist_rz_resume.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        print(f"Resume log: {log_path}", flush=True)
        for seed in SEEDS:
            config_path = make_config(seed)
            message = f"===== seed={seed} config={config_path} ====="
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()
            subprocess.run(
                [sys.executable, "-u", "train_medmnist.py", "--config", str(config_path)],
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )
    print("PneumoniaMNIST RZ circuit resume complete", flush=True)


if __name__ == "__main__":
    main()
