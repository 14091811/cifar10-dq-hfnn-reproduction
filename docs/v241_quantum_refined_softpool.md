# V241 quantum-refined SoftPool

V241 keeps the strongest V240 control as its initialization and asks whether a
small quantum circuit can improve it. It does not learn a residual scale. At
initialization, all trainable parameters are zero and the pooling result equals
the grouped V240 SoftPool result up to floating-point roundoff.

```text
Block1 feature [B, 64, 32, 32]
  -> grouped 2x2 patch scores
  -> normalized SoftPool probabilities s00, s01, s10, s11
  -> amplitude state [sqrt(s00), sqrt(s01), sqrt(s10), sqrt(s11)]
  -> Haar left-right, top-bottom, and diagonal phase encoding
  -> identity-initialized trainable refinement
  -> refined Born probabilities p00, p01, p10, p11
  -> sum(pij * xij)
  -> [B, 64, 16, 16]
```

There are eight channel groups. Each group owns ten trainable parameters, so
each V241 variant adds 80 parameters. Parameters are shared by all spatial
patches within a group.

The local quantum variant uses two local `RZ-RY` blocks and two middle `RY`
rotations. The entangled variant replaces those two middle rotations with
symmetric `RZZ` and `RXX` interactions. This avoids assigning a permanent
control and target direction as CNOT does. For the entangled model,
`no_entanglement` removes the data `RZZ` and trainable `RZZ/RXX` operations;
the two trained interaction parameters are therefore unused in that diagnostic.

The matched classical control also has 80 zero-initialized parameters. It
learns two descriptor-conditioned row and column corrections to the logarithms
of the SoftPool probabilities. The V240 parameter-free SoftPool result is reused
as the common baseline rather than rerun as a new architecture.

Diagnostics retain the V240 meanings:

- `base_softpool` disables only the learned refinement while preserving the
  original SoftPool probabilities;
- `zero_q` restores MaxPool, measuring dependence on the entire pooling rule;
- `shuffle_q` applies another sample's refined probabilities;
- `no_entanglement` removes entangling operations at inference;
- `pool_refinement` measures probability and pooled-feature movement away from
  the original SoftPool distribution.

The main comparison is accuracy against V240 SoftPool (`94.13%`, seed 42), then
the refined-probability distance and shuffle sensitivity. A useful refinement
must improve or preserve SoftPool while producing a nontrivial, sample-dependent
change; a large `zero_q` drop alone is not evidence of quantum benefit.
