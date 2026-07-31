# V238 quantum-gated Haar detail pooling

V238 keeps the original first max-pooling path and adds a bounded Haar-detail
bypass. Turning the gate off therefore recovers the original CNN downsampling
exactly.

```text
Block1: [B, 64, 32, 32]
  -> MaxPool2d ----------------------------------------+
  -> one-level Haar -> LL/LH/HL/HH                    |
       -> 8 channel-group descriptors                 |
       -> 8 shared two-qubit gates                    |
       -> Z0/Z1/ZZ select LH/HL/HH                    |
  -> maxpool + bounded selected-detail correction ----+
  -> [B, 64, 16, 16]
```

Each group uses LL signed response, total high-frequency energy, LH-HL
directional contrast, and HH diagonal response. Circuit parameters are shared
across all 16 x 16 positions. The entangled and local-Rx variants differ only
by two fixed CNOT operations. The classical control uses the same descriptors,
eight trainable values per group, three outputs, and the same 24 bounded-alpha
parameters. Every gate variant adds exactly 88 trainable parameters.

The three matched controls are:

- `qgwp_quantum_entangled`
- `qgwp_quantum_local_rx`
- `qgwp_classical_matched`

The diagnostic modes have direct interpretations: `zero_q` is the original
MaxPool CNN, `shuffle_q` applies another sample's detail gates, and
`no_entanglement` removes only the CNOT operations.
