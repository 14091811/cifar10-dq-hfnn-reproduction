# Run And Log Management

## Directory Layout

Every training attempt receives a unique timestamped directory and never
overwrites an earlier attempt:

```text
runs/
  cifar10_v216grouped_authorfusion/
    seed_42/
      latest.txt
      20260729_120000_123456/
        train.log
        config.json
        status.json
        metrics.jsonl
        history.json
        result.json
        best.pt
```

`metrics.jsonl` is appended after every epoch. `history.json` and `result.json`
are finalized after training. `status.json` records the latest completed epoch,
best epoch, and whether the run completed. `latest.txt` identifies the newest
attempt for a seed without deleting older attempts.

## One Run

```bash
python train_cifar10.py \
  --config configs/cifar10_v216grouped_authorfusion.json
```

The terminal and `train.log` receive the same epoch lines. External redirection
is optional. For a background process:

```bash
nohup python -u train_cifar10.py \
  --config configs/cifar10_v216grouped_authorfusion.json \
  > launcher.log 2>&1 &
```

`launcher.log` is only a launcher copy; the canonical log is the `train.log`
inside the timestamped run directory.

## Multi-Seed Study

Use the common study runner instead of adding another one-off Python script:

```bash
python run_study.py \
  --name v216_pilot \
  --config configs/cifar10_v216grouped_authorfusion.json \
  --seeds 42,456,789
```

Multiple configurations can be compared in one matched study:

```bash
python run_study.py \
  --name v216_vs_v215_noq \
  --config \
    configs/cifar10_v216grouped_authorfusion.json \
    configs/cifar10_v215local_noq_authorfusion.json \
  --seeds 42,456,789
```

Study outputs are stored under:

```text
studies/<study-name>/<timestamp>/
  study.log
  manifest.json
  summary.csv
  summary.json
  configs/
```

The CSV and JSON summaries are rewritten after every completed run, so partial
results remain usable after an interruption.

## Inspect Runs

List the latest attempt for every experiment and seed:

```bash
python list_runs.py
```

Filter by part of an experiment name:

```bash
python list_runs.py --experiment v216
```

The table reports run status, latest epoch, best epoch, final accuracy and F1,
and the timestamp identifying the exact attempt directory.

## Quantum-Branch Diagnostics

Evaluate one trained local-frequency checkpoint without retraining:

```bash
python diagnose_quantum_branch.py \
  --experiment cifar10_v216grouped_authorfusion \
  --seed 42
```

The command compares normal inference with `Q=0`, batch-shuffled quantum
features, and CNOT-disabled inference. It also records classical/quantum feature
norms and cosine similarity in `quantum_diagnostics.json` inside the run
directory. A specific attempt can instead be selected with `--run-dir`.
