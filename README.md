# Clean DQ-HFNN Reproduction

Independent, maintainable reproduction of the author DQ-HFNN pipeline.
The archived author code remains in `dq_hfnn_reproduction` untouched.

The first target is CIFAR-10 with an author-style residual CNN branch, a 3 x 3
pair sampler, class-specific dual-qubit membership circuits, log-average
aggregation, and additive fusion. `pair_source: pixels` is the author baseline;
`pair_source: haar_high` is our matched local-frequency variant. Circuit
ablations are explicit: `no_ent`, `weak_ent`, and `strong_ent`.

The default strong circuit is `Ry(x0), Ry(x1), Rz(q0), CNOT, Ry(q0), CNOT`.

Cross-region pairs are stochastic during training and fixed from the experiment
seed during evaluation. This preserves the sampling regularization while making
test metrics repeatable.

The supplied 0.1 SGD protocol uses global gradient clipping at 5.0. The
two-qubit Log-Avg branch can otherwise produce occasional large updates that
destabilize the classical fusion path.

Run: `python train_cifar10.py --config configs/cifar10_strong_ent.json`.
The test split is evaluated once after validation selects the checkpoint.

Training artifacts are organized by experiment, seed, and timestamp under
`runs/`. Each attempt automatically records its console log, exact config,
incremental epoch metrics, status, checkpoint, and final result. Use
`run_study.py` for future multi-seed or multi-model comparisons instead of
creating another one-off runner. See `docs/run_management.md`.
