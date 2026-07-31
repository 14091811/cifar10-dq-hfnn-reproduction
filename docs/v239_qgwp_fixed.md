# V239 fixed-strength QGWP

V238 showed that its learned detail strength stayed at approximately 0.05.
V239 removes that scalar optimization bottleneck and tests the same shallow
Haar-detail placement at a fixed strength of 0.10:

```text
Block1 feature x [B, 64, 32, 32]
  |-- MaxPool2d(x) ------------------------------------------|
  |-- Haar(x) -> LL, LH, HL, HH                              |
                 |-> 8 group descriptors                     |
                 |-> local two-qubit or classical gate       |
                 |-> gLH * LH + gHL * HL + gHH * HH          |
  |                                                          |
  `-> MaxPool2d(x) + 0.10 * selected_detail / sqrt(3) -------'
                         [B, 64, 16, 16]
```

The seed-42 pilot has three controls:

- `qgwp_fixed_quantum_local_rx`: 64 trainable circuit angles, no CNOT.
- `qgwp_fixed_classical_matched`: 64 trainable classical gate weights.
- `qgwp_fixed_uniform`: no gate parameters; all three detail gates equal one.

`zero_q` exactly recovers the original MaxPool path. `shuffle_q` tests whether
sample-conditioned selection matters. Uniform Haar separates useful gate
selection from a generic benefit caused by injecting more high frequency.

Pilot success requires at least 94.45% accuracy and at least 0.10 percentage
point margins over zero, shuffle, classical, and uniform controls.
