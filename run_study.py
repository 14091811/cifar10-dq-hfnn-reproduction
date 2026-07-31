"""Run a reproducible config-by-seed study with logs and incremental summaries."""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from dq_hfnn.run_io import normalize_experiment_name, resolve_latest_run, write_json


def write_summary(study_dir, rows):
    write_json(study_dir / "summary.json", rows)
    fields = (
        "experiment",
        "seed",
        "status",
        "best_epoch",
        "test_accuracy",
        "test_macro_f1",
        "run_dir",
    )
    with (study_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    study_root = ROOT / "studies" / args.name
    study_dir = study_root / timestamp
    generated_dir = study_dir / "configs"
    generated_dir.mkdir(parents=True)
    (study_root / "latest.txt").write_text(timestamp + "\n", encoding="utf-8")
    study_log = (study_dir / "study.log").open("w", encoding="utf-8", buffering=1)

    def log(message):
        print(message, flush=True)
        study_log.write(message + "\n")

    manifest = {
        "name": args.name,
        "started_at": datetime.now().astimezone().isoformat(),
        "configs": [str(path) for path in args.config],
        "seeds": seeds,
    }
    write_json(study_dir / "manifest.json", manifest)
    rows = []

    for config_path in args.config:
        base_config = json.loads(config_path.read_text(encoding="utf-8"))
        experiment = normalize_experiment_name(base_config["run_name"])
        for seed in seeds:
            config = dict(base_config)
            config["seed"] = seed
            generated_path = generated_dir / f"{experiment}_seed{seed}.json"
            write_json(generated_path, config)
            log("=" * 80)
            log(f"Experiment={experiment} | Seed={seed} | Config={config_path}")
            process = subprocess.Popen(
                [sys.executable, "-u", "train_cifar10.py", "--config", str(generated_path)],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                log(line.rstrip("\n"))
            return_code = process.wait()
            row = {
                "experiment": experiment,
                "seed": seed,
                "status": "failed" if return_code else "completed",
                "best_epoch": "",
                "test_accuracy": "",
                "test_macro_f1": "",
                "run_dir": "",
            }
            try:
                run_dir = resolve_latest_run(ROOT, experiment, seed)
                row["run_dir"] = str(run_dir)
                if return_code == 0:
                    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
                    row["best_epoch"] = result["best_epoch"]
                    row["test_accuracy"] = result["test_accuracy"]
                    row["test_macro_f1"] = result["test_macro_f1"]
            finally:
                rows.append(row)
                write_summary(study_dir, rows)
            if return_code and not args.continue_on_error:
                study_log.close()
                raise SystemExit(return_code)

    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    write_json(study_dir / "manifest.json", manifest)
    log(f"StudyDir={study_dir}")
    study_log.close()


if __name__ == "__main__":
    main()
