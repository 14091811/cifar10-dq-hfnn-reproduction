# V236 two-level Haar quantum logit residual

V236 replaces intermediate channel modulation with directly supervised class evidence.
The original author DQ branch is disabled.

```text
image
  |-- CNN trunk ---------------------------------> base_logits [B, 10]
  |
  `-- tap-3 features [B, 128, 16, 16]
       -> learned 1x1 projection [B, 4, 16, 16]
       -> two-level Haar
          [LH1, HL1, HH1, LL2, LH2, HL2, HH2]
       -> local descriptors [B, 4, 4, 4, 7]
       -> learned 7-to-4 angle projection
       -> four 4-qubit circuit banks at each local position
       -> local Z expectations [B, 4, 4, 4, 4]
       -> spatial mean + 16-to-10 class projection
       -> layer-normalized q_logits [B, 10]

final_logits = base_logits + alpha * q_logits
alpha = 0.10 * sigmoid(alpha_logit), initialized to 0.05
loss = CE(final_logits, y)
```

The pilot does not use auxiliary CE. The fused classification loss already
supervises the quantum head through the bounded residual, while allowing it to
learn complementary evidence instead of requiring it to be a standalone
classifier. Auxiliary CE remains available as a later ablation only.

The no-aux seed-42 diagnostic showed no accuracy change under `zero_q`, so the
pre-registered follow-up uses `0.05 * CE(q_logits, y)`. It is stored in separate
`*_aux005.json` configs and changes no other training or model setting.

The entangled candidate uses the HQNet AC+ directed CRX pattern. The matched
Local-RX control replaces every CRX by a target-qubit RX using the same angle
parameter. Both heads add exactly 733 parameters.

The pilot advances beyond seed 42 only if the entangled model reaches 94.35%,
beats Local-RX, and loses at least 0.10 percentage points under `zero_q`.
