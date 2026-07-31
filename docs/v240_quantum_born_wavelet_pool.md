# V240 quantum Born wavelet pooling

V240 replaces the first MaxPool operation with an input-conditioned 2x2
pooling kernel. Haar coefficients describe each shallow feature patch, and the
four Born probabilities of a two-qubit state select its four spatial values.

```text
Block1 feature [B, 64, 32, 32]
  -> 2x2 patches: x00, x01, x10, x11
  -> Haar descriptors: LL, LH, HL, HH
  -> eight channel-group circuits
  -> P00, P01, P10, P11
  -> sum(Pij * xij)
  -> [B, 64, 16, 16]
```

Each group contains eight channels and owns eight trainable angles. Circuit
parameters are shared across all 16x16 patches, for 64 trainable parameters in
total. The four probabilities are nonnegative and sum to one, so no learned or
fixed residual amplitude is needed.

Without CNOT, the two-qubit state is separable and its pooling distribution
satisfies `P00 * P11 = P01 * P10`. The entangled circuit uses `q0 -> q1` after
the first local block and `q1 -> q0` after the second block. This bidirectional
coupling allows nonseparable row-column spatial selection without assigning a
fixed control role to either spatial axis.

The seed-42 pilot compares:

- local-RX quantum pooling, 64 parameters;
- entangled quantum pooling, 64 parameters;
- separable trigonometric classical pooling with matched encoding, 64 parameters;
- parameter-free grouped SoftPool.

Diagnostic mode `zero_q` restores the original MaxPool exactly. `shuffle_q`
applies another sample's pooling probabilities. Reported probability entropy
and maximum probability reveal whether selection is uniform or concentrated.
