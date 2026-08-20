"""Summarize fixed-vs-learned Partial-D16 Classical/QPA runs."""

import argparse
import csv
import json
import math
from pathlib import Path


MODEL_NAMES = {
    "fixed_classical": "two_block_dwt_directional_classical_partial_d16_cnn",
    "fixed_qpa": "two_block_dwt_directional_qpa_partial_d16_cnn",
    "learned_classical": "two_block_dwt_directional_classical_learned_partial_d16_cnn",
    "learned_qpa": "two_block_dwt_directional_qpa_learned_partial_d16_cnn",
}
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "auc")


def latest_result(root, experiment, seed):
    seed_dir = root / "runs" / experiment / f"seed_{seed}"
    pointer = seed_dir / "latest.txt"
    if not pointer.exists():
        return None
    result = seed_dir / pointer.read_text(encoding="utf-8").strip() / "result.json"
    return json.loads(result.read_text(encoding="utf-8")) if result.exists() else None


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def std(values):
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def fmt(value):
    return "NA" if math.isnan(value) else f"{value * 100:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", default="42,456,789,5050,6060,123,1024,2024,3030,4040")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows = []
    for condition, model in MODEL_NAMES.items():
        experiment = f"{args.dataset}_{model}"
        for seed in seeds:
            result = latest_result(root, experiment, seed)
            if result is None:
                continue
            row = {"dataset": args.dataset, "condition": condition, "model": model, "seed": seed}
            row.update({metric: result.get(f"test_{metric}") for metric in METRICS})
            row["best_epoch"] = result.get("best_epoch")
            rows.append(row)

    out_dir = root / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.dataset}_partial_d16_2x2_per_seed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for condition in MODEL_NAMES:
        selected = [row for row in rows if row["condition"] == condition]
        summary[condition] = {
            metric: (mean([row[metric] for row in selected if row[metric] is not None]),
                     std([row[metric] for row in selected if row[metric] is not None]))
            for metric in METRICS
        }

    lines = [
        f"# {args.dataset} Partial-D16 2x2 Ablation",
        "",
        "Selection metric: validation macro-F1; test evaluated once after checkpoint selection.",
        "",
        "| Condition | N | Acc | Precision | Recall | F1 | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in MODEL_NAMES:
        selected = [row for row in rows if row["condition"] == condition]
        values = summary[condition]
        lines.append(
            f"| {condition} | {len(selected)} | "
            + " | ".join(f"{fmt(values[m][0])} +/- {fmt(values[m][1])}" for m in METRICS)
            + " |"
        )

    lines += ["", "## Paired QPA Gain", "", "| Subspace | Acc gain | F1 gain | AUC gain |", "|---|---:|---:|---:|"]
    by_key = {(row["condition"], row["seed"]): row for row in rows}
    for prefix, label in (("fixed", "Fixed-D16"), ("learned", "Learned-D16")):
        pairs = []
        for seed in seeds:
            classical = by_key.get((f"{prefix}_classical", seed))
            qpa = by_key.get((f"{prefix}_qpa", seed))
            if classical and qpa:
                pairs.append({metric: qpa[metric] - classical[metric] for metric in METRICS})
        lines.append(
            f"| {label} | {fmt(mean([p['accuracy'] for p in pairs]))} | "
            f"{fmt(mean([p['macro_f1'] for p in pairs]))} | {fmt(mean([p['auc'] for p in pairs]))} |"
        )

    md_path = out_dir / f"{args.dataset}_partial_d16_2x2_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Per-seed CSV: {csv_path}")
    print(f"Summary Markdown: {md_path}")
    print("Conditions:", {condition: sum(row["condition"] == condition for row in rows) for condition in MODEL_NAMES})


if __name__ == "__main__":
    main()
