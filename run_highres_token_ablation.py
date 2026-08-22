"""Compare 4x4 and 8x8 token QPA on the fixed binary MedMNIST protocol."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEEDS = (42, 456, 789)
DATASETS = {
    "breastmnist": ROOT / "configs" / "breastmnist_paper_protocol.json",
    "pneumoniamnist": ROOT / "configs" / "pneumoniamnist_2x2_fixed50.json",
}
MODELS = (
    "two_block_dwt_directional_classical_learned_partial_d16_highres_token_cnn",
    "two_block_dwt_directional_qpa_learned_partial_d16_highres_token_cnn",
)


def parse_csv(value):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("value must not be empty")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()
    datasets = parse_csv(args.datasets)
    seeds = tuple(int(value) for value in parse_csv(args.seeds))
    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset: {dataset}")
        base = json.loads(DATASETS[dataset].read_text(encoding="utf-8"))
        out_dir = ROOT / "configs" / f"generated_{dataset}_highres_token"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir = ROOT / "study_logs" / "highres_token"
        log_dir.mkdir(parents=True, exist_ok=True)
        for model in MODELS:
            log_path = log_dir / f"{dataset}__{model}.log"
            print(f"START {dataset} {model} log={log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                for seed in seeds:
                    cfg = dict(base)
                    cfg.update(seed=seed, model_type=model,
                               run_name=f"{dataset}_{model}",
                               early_stopping_patience=999)
                    config_path = out_dir / f"{model}_seed{seed}.json"
                    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                    subprocess.run(
                        [sys.executable, "-u", "train_medmnist.py", "--config", str(config_path)],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                    )
        print(f"DONE {dataset}", flush=True)


if __name__ == "__main__":
    main()
