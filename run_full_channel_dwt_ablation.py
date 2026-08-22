"""Run the full-64-channel non-residual DWT-QPA diagnostic."""

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
    "two_block_full_channel_dwt_classical_cnn",
    "two_block_full_channel_dwt_qpa_cnn",
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
        config_dir = ROOT / "configs" / f"generated_{dataset}_full_channel_dwt"
        log_dir = ROOT / "study_logs" / "full_channel_dwt"
        config_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        for model in MODELS:
            log_path = log_dir / f"{dataset}__{model}.log"
            print(f"START {dataset} {model} log={log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                for seed in seeds:
                    cfg = dict(base, seed=seed, model_type=model,
                               run_name=f"{dataset}_{model}",
                               early_stopping_patience=999)
                    config_path = config_dir / f"{model}_seed{seed}.json"
                    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                    subprocess.run(
                        [sys.executable, "-u", "train_medmnist.py", "--config", str(config_path)],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                    )


if __name__ == "__main__":
    main()
