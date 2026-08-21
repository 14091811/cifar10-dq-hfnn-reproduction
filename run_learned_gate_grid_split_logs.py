"""Run the learned gate grid with one ten-seed log per configuration."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEEDS = (42, 456, 789, 5050, 6060, 123, 1024, 2024, 3030, 4040)
DATASETS = {
    "breastmnist": ROOT / "configs" / "breastmnist_paper_protocol.json",
    "pneumoniamnist": ROOT / "configs" / "pneumoniamnist_2x2_fixed50.json",
}
MODELS = tuple(
    model
    for gate in ("", "standard_")
    for dimension in (8, 16, 32)
    for model in (
        f"two_block_dwt_directional_classical_{gate}learned_partial_d{dimension}_cnn",
        f"two_block_dwt_directional_qpa_{gate}learned_partial_d{dimension}_cnn",
    )
)


def parse_csv(value):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("value must not be empty")
    return values


def summarize(result_root, datasets, seeds):
    print("=" * 100, flush=True)
    print("LEARNED GATE GRID SUMMARY", flush=True)
    print("dataset | model | n | test_acc mean+-sd | test_f1 mean+-sd | test_auc mean+-sd", flush=True)
    for dataset in datasets:
        for model_type in MODELS:
            values = []
            experiment = f"{dataset}_{model_type}"
            for seed in seeds:
                seed_root = result_root / experiment / f"seed_{seed}"
                latest_file = seed_root / "latest.txt"
                if not latest_file.exists():
                    continue
                attempt = latest_file.read_text(encoding="utf-8").strip()
                result_file = seed_root / attempt / "result.json"
                if result_file.exists():
                    values.append(json.loads(result_file.read_text(encoding="utf-8")))
            if not values:
                print(f"{dataset} | {model_type} | 0 | incomplete", flush=True)
                continue
            metrics = {}
            for key in ("test_accuracy", "test_macro_f1", "test_auc"):
                numbers = [float(item[key]) * 100 for item in values if item.get(key) is not None]
                deviation = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
                metrics[key] = f"{statistics.mean(numbers):.2f}+-{deviation:.2f}"
            print(
                f"{dataset} | {model_type} | {len(values)} | "
                f"{metrics['test_accuracy']} | {metrics['test_macro_f1']} | {metrics['test_auc']}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--log-dir", type=Path, default=ROOT / "study_logs" / "learned_gate_grid")
    args = parser.parse_args()

    datasets = parse_csv(args.datasets)
    seeds = tuple(int(value) for value in parse_csv(args.seeds))
    args.log_dir.mkdir(parents=True, exist_ok=True)
    print(f"study=dwt_qpa_learned_gate_grid datasets={datasets} seeds={seeds}", flush=True)
    print(f"per_configuration_logs={args.log_dir}", flush=True)

    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset: {dataset}")
        base_config = json.loads(DATASETS[dataset].read_text(encoding="utf-8"))
        output_dir = ROOT / "configs" / f"generated_{dataset}_learned_gate_grid"
        output_dir.mkdir(parents=True, exist_ok=True)
        for model_type in MODELS:
            log_path = args.log_dir / f"{dataset}__{model_type}.log"
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
                        cwd=ROOT,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            print(f"DONE {dataset} {model_type} log={log_path}", flush=True)

    summarize(ROOT / "runs", datasets, seeds)
    print("Learned gate grid study complete", flush=True)


if __name__ == "__main__":
    main()
