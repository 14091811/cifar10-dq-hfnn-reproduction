"""List the latest structured training run for every experiment and seed."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="")
    args = parser.parse_args()
    rows = []
    runs_root = ROOT / "runs"
    if not runs_root.exists():
        print("No structured runs found.")
        return

    for experiment_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        if args.experiment and args.experiment not in experiment_dir.name:
            continue
        for seed_dir in sorted(path for path in experiment_dir.glob("seed_*") if path.is_dir()):
            try:
                attempt = (seed_dir / "latest.txt").read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                continue
            run_dir = seed_dir / attempt
            status = load_json(run_dir / "status.json")
            result = load_json(run_dir / "result.json")
            rows.append(
                (
                    experiment_dir.name,
                    seed_dir.name.removeprefix("seed_"),
                    status.get("status", "unknown"),
                    str(status.get("last_epoch", "-")),
                    str(status.get("best_epoch", "-")),
                    f"{result['test_accuracy'] * 100:.3f}" if "test_accuracy" in result else "-",
                    f"{result['test_macro_f1'] * 100:.3f}" if "test_macro_f1" in result else "-",
                    attempt,
                )
            )

    if not rows:
        print("No matching structured runs found.")
        return
    headers = ("experiment", "seed", "status", "epoch", "best", "acc%", "f1%", "attempt")
    widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]
    format_row = lambda row: "  ".join(value.ljust(widths[i]) for i, value in enumerate(row))
    print(format_row(headers))
    print(format_row(tuple("-" * width for width in widths)))
    for row in rows:
        print(format_row(row))


if __name__ == "__main__":
    main()
