"""Structured, non-overwriting run directories and JSON persistence."""

import json
import re
from datetime import datetime
from pathlib import Path


SEED_SUFFIX = re.compile(r"_seed\d+$")


def normalize_experiment_name(run_name):
    return SEED_SUFFIX.sub("", run_name)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def create_run_directory(root, run_name, seed):
    experiment = normalize_experiment_name(run_name)
    seed_dir = Path(root) / "runs" / experiment / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone()
    attempt_name = started_at.strftime("%Y%m%d_%H%M%S_%f")
    run_dir = seed_dir / attempt_name
    run_dir.mkdir()
    (seed_dir / "latest.txt").write_text(attempt_name + "\n", encoding="utf-8")
    return run_dir, experiment, started_at


def resolve_latest_run(root, run_name, seed):
    experiment = normalize_experiment_name(run_name)
    seed_dir = Path(root) / "runs" / experiment / f"seed_{seed}"
    attempt_name = (seed_dir / "latest.txt").read_text(encoding="utf-8").strip()
    return seed_dir / attempt_name
