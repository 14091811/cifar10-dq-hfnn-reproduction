# CIFAR-10 DQ-HFNN Task Boundary

## Identity

- Task: CIFAR-10 image classification.
- Research focus: controlled classical-versus-quantum pooling, local frequency
  descriptors, and dual-qubit circuit ablations.
- Primary metric: validation-selected test accuracy and macro-F1.
- Server repository path: `/workspace/dq_hfnn_reproduction_clean`.
- Server Conda environment: `QIGN`.

Do not mix brain-tumor MRI data, splits, augmentation policies, model numbers,
or conclusions into this repository. Brain-tumor work is maintained separately
in `brain_tumor_task_pack`.

## Experiment Contract

- Use `run_study.py` for matched multi-model and multi-seed comparisons.
- Keep the experiment config, seed, status, metrics, and checkpoint under the
  generated run directory.
- Select checkpoints by validation metrics and evaluate the test set only as
  defined by the study protocol.
- Quantum variants require parameter-matched classical controls.
- Diagnose `normal`, `base` when available, `zero_q`, `shuffle_q`, and
  `no_entanglement` before claiming that a quantum branch is causally useful.
- Do not expand a single-seed result to many seeds unless it beats all required
  controls under the same protocol.

## Current Interpretation

- V240 showed that Born-style pooling made the branch causally active, but the
  parameter-free SoftPool control remained stronger.
- V241 showed that quantum and classical refinements contributed roughly zero
  beyond the SoftPool base.
- Do not repeat weak post-hoc SoftPool refinements without a new falsifiable
  mechanism and matched controls.

## Commands

Run focused tests:

```bash
python -m pytest tests
```

Run a configured study:

```bash
python run_study.py --name <study-name> --config <config...> --seeds <seeds>
```

Run branch diagnostics:

```bash
python diagnose_quantum_branch.py --experiment <experiment> --seed <seed>
```

## Repository Hygiene

- Never commit CIFAR downloads, checkpoints, run directories, study outputs,
  logs, credentials, or machine-specific caches.
- Keep architecture and protocol changes scoped to this repository.
- Run automated research on a dedicated experiment branch, never directly on
  `main`.
