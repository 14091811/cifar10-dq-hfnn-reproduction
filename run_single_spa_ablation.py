"""Run a single Yang-style SP&A block in the two-block MedMNIST CNN."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "breastmnist": ROOT / "configs" / "breastmnist_paper_protocol.json",
    "pneumoniamnist": ROOT / "configs" / "pneumoniamnist_2x2_fixed50.json",
}
MODELS = (
    "two_block_classical_cnn",
    "two_block_single_spa_classical_cnn",
    "two_block_single_spa_dwt_qpa_classical_cnn",
    "two_block_single_spa_dwt_qpa_qpa_cnn",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default="42,456,789")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())

    for dataset in (item.strip() for item in args.datasets.split(",")):
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset: {dataset}")
        base = json.loads(DATASETS[dataset].read_text(encoding="utf-8"))
        config_dir = ROOT / "configs" / f"generated_{dataset}_single_spa"
        log_dir = ROOT / "study_logs" / "single_spa"
        config_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        for model in MODELS:
            log_path = log_dir / f"{dataset}__{model}.log"
            print(f"START {dataset} {model} log={log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                for seed in seeds:
                    config = dict(
                        base,
                        seed=seed,
                        model_type=model,
                        run_name=f"{dataset}_{model}",
                        early_stopping_patience=999,
                    )
                    config_path = config_dir / f"{model}_seed{seed}.json"
                    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
                    subprocess.run(
                        [sys.executable, "-u", "train_medmnist.py", "--config", str(config_path)],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )


if __name__ == "__main__":
    main()
