"""Run the one-head channel-token QPA diagnostic."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = "breastmnist"
SEEDS = (42, 456, 789)
MODELS = (
    "two_block_channelwise_dwt_classical_onehead_cnn",
    "two_block_channelwise_dwt_qpa_onehead_cnn",
)


def main():
    base = json.loads(
        (ROOT / "configs" / "breastmnist_paper_protocol.json").read_text(encoding="utf-8")
    )
    config_dir = ROOT / "configs" / "generated_breastmnist_channelwise_onehead"
    log_dir = ROOT / "study_logs" / "channelwise_onehead"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        log_path = log_dir / f"{DATASET}__{model}.log"
        print(f"START {DATASET} {model} log={log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            for seed in SEEDS:
                cfg = dict(base, seed=seed, model_type=model,
                           run_name=f"{DATASET}_{model}",
                           early_stopping_patience=999)
                config_path = config_dir / f"{model}_seed{seed}.json"
                config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                subprocess.run(
                    [sys.executable, "-u", "train_medmnist.py", "--config", str(config_path)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                )


if __name__ == "__main__":
    main()
