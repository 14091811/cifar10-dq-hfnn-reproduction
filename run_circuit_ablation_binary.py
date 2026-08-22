"""Run the fixed learned-D16 QPA circuit ablation on two binary datasets."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEEDS = (42, 456, 789, 5050, 6060, 123, 1024, 2024, 3030, 4040)
DATASETS = {
    "breastmnist": ROOT / "configs" / "breastmnist_paper_protocol.json",
    "pneumoniamnist": ROOT / "configs" / "pneumoniamnist_2x2_fixed50.json",
}
MODELS = (
    "two_block_dwt_directional_qpa_learned_d16_circuit_baseline_cnn",
    "two_block_dwt_directional_qpa_learned_d16_circuit_no_entanglement_cnn",
    "two_block_dwt_directional_qpa_learned_d16_circuit_rz_cnn",
    "two_block_dwt_directional_qpa_learned_d16_circuit_single_cnot_cnn",
    "two_block_dwt_directional_qpa_learned_d16_circuit_symmetric_ry_cnn",
    "two_block_dwt_directional_qpa_learned_d16_circuit_rz_single_cnot_cnn",
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
        base_config = json.loads(DATASETS[dataset].read_text(encoding="utf-8"))
        output_dir = ROOT / "configs" / f"generated_{dataset}_circuit_ablation"
        output_dir.mkdir(parents=True, exist_ok=True)
        for model_type in MODELS:
            log_path = ROOT / "study_logs" / "circuit_ablation" / f"{dataset}__{model_type}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"START {dataset} {model_type} log={log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                for seed in seeds:
                    cfg = dict(base_config)
                    cfg["seed"] = seed
                    cfg["model_type"] = model_type
                    cfg["run_name"] = f"{dataset}_{model_type}"
                    cfg["early_stopping_patience"] = 999
                    generated = output_dir / f"{model_type}_seed{seed}.json"
                    generated.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                    log_file.write(f"starting model={model_type} seed={seed}\n")
                    log_file.flush()
                    subprocess.run(
                        [sys.executable, "-u", "train_medmnist.py", "--config", str(generated)],
                        cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT, check=True,
                    )
            print(f"DONE {dataset} {model_type} log={log_path}", flush=True)
    print("Circuit ablation study complete", flush=True)


if __name__ == "__main__":
    main()
