# CIFAR-10 Model Flows

The two experiments share the CNN branch, fusion head, optimizer, pair budget,
and active-source A circuit. The only experimental change is the input supplied
to the quantum-fuzzy branch.

## Author Active CIFAR Source A

```mermaid
flowchart TD
    I["CIFAR-10 image\nB x 3 x 32 x 32"]

    I --> C1["Residual CNN\n64 -> pool -> 128 -> 128 -> pool\n-> 256 -> 256 -> pool\n-> 512 -> 512 -> pool"]
    C1 --> C2["Flatten 2048\nDropout 0.4 + Linear"]
    C2 --> CF["Classical feature\nB x 256"]

    I --> Q1["Flatten RGB\n3 x 32 x 32 = 3072 values"]
    Q1 --> Q2["3 x 3 grid per RGB channel\nboundaries: 0, 10, 21, 32"]
    Q2 --> Q3["60 cross-block candidates\nSample 460 random pairs per image"]
    Q3 --> Q4["Pair tensor\nB x 460 x 2"]
    Q4 --> QC

    subgraph QC["Ten class-specific A circuits"]
      direction TB
      Q5["For each pair: Ry(x0) on q0\nRy(x1) on q1"]
      Q5 --> Q6["Trainable Rz(theta0), Ry(theta1) on q0\nNo CNOT in active source"]
      Q6 --> Q7["Measure Z on q0 and q1\nNormalize each measurement to [0, 1]"]
    end

    QC --> Q8["Memberships\nB x 920 x 10"]
    Q8 --> Q9["Clamp + mean(log membership)\n10 class scores"]
    Q9 --> Q10["Linear 10 -> 256"]
    Q10 --> QF["Quantum-fuzzy feature\nB x 256"]

    CF --> F["Elementwise addition"]
    QF --> F
    F --> O["Linear 256 -> 10 logits"]
```

## Current Haar Input Experiment (Not V215)

```mermaid
flowchart TD
    I["Same CIFAR-10 image\nB x 3 x 32 x 32"]

    I --> C1["Same residual CNN branch"]
    C1 --> C2["Same 2048 -> 256 projection"]
    C2 --> CF["Classical feature\nB x 256"]

    I --> H1["One-level Haar transform per RGB channel"]
    H1 --> H2["LH, HL, HH detail bands\n3 RGB x 3 bands = 9 maps\nB x 9 x 16 x 16"]
    H2 --> H3["3 x 3 grid per detail map"]
    H3 --> H4["Same budget: sample 460\nrandom cross-block pairs per image"]
    H4 --> H5["Pair tensor\nB x 460 x 2"]
    H5 --> QC

    subgraph QC["Same ten active-source A circuits"]
      direction TB
      Q1["Ry(x0), Ry(x1)"]
      Q1 --> Q2["Trainable Rz, Ry on q0\nNo CNOT"]
      Q2 --> Q3["Two Z measurements in [0, 1]"]
    end

    QC --> Q4["Memberships\nB x 920 x 10"]
    Q4 --> Q5["Same log-average fuzzy rule"]
    Q5 --> Q6["Same Linear 10 -> 256"]
    Q6 --> QF["Quantum-fuzzy feature\nB x 256"]

    CF --> F["Same elementwise addition"]
    QF --> F
    F --> O["Same Linear 256 -> 10 logits"]
```

## Controlled Difference

| Component | Author source A | Our variant |
| --- | --- | --- |
| Quantum input | Raw normalized RGB pixel pairs | Haar LH/HL/HH detail-coefficient pairs |
| Quantum feature maps | 3 maps at 32 x 32 | 9 maps at 16 x 16 |
| Pair count | 460 | 460 |
| Circuit | Active-source A, no CNOT | Identical |
| CNN, fusion, classifier, optimizer | Identical | Identical |

The Haar comparison therefore tests whether replacing raw local values with
local directional high-frequency detail improves this exact source model. It
is a simple input substitution, not the V215 strategy.

## Actual V215 Strategy Used in the Brain-Tumor Project

```mermaid
flowchart TD
    I["MRI image"] --> B1["V91-lite learned spatial backbone\nHR stages + low/mid candidate fusion\nResidual + Laplacian enhancement"]
    B1 --> E["Expand learned feature map\nto 128 channels"]

    E --> F1["FER semantic branch"]
    F1 --> F2["Global average pool\nLinear 128 -> 256, ReLU, Dropout"]
    F2 --> BL["Base classifier logits"]

    E --> D1["DWT on learned 128-channel features"]
    D1 --> D2["LL, LH, HL, HH feature bands"]
    D2 --> D3["For every band: channel mean and RMS energy\nAdaptive pool to 4 x 4 local positions\nLayer normalization"]
    D3 --> D4["Two fixed band relations at every position\nLL-HH and LH-HL"]
    D4 --> D5["Four angles per relation\nsigned(A), signed(B), energy(A), energy(B)"]

    subgraph VQ["V215 class-specific DQ head"]
      direction TB
      Q1["Encode signed values with Ry on two qubits"]
      Q1 --> Q2["Trainable Rz/Ry on both qubits + CNOT"]
      Q2 --> Q3["Re-upload energy values with Rz"]
      Q3 --> Q4["Second trainable Rz/Ry block + CNOT\nMeasure both qubits"]
    end

    D5 --> VQ
    VQ --> Q5["Log-membership average\nover 16 positions and two relations"]
    Q5 --> Q6["LayerNorm class scores, zero mean\nLearned residual scale alpha <= 0.10"]
    Q6 --> QR["Small quantum logit residual"]
    BL --> ADD["Add logits"]
    QR --> ADD
    ADD --> O["Final class logits"]
```

V215 differs materially from the current CIFAR Haar experiment: it applies DWT
after a learned feature extractor, uses two fixed frequency relations across a
4 x 4 grid, re-uploads signed and energy descriptors, has two entangling
two-qubit blocks, and adds a bounded residual directly to the base logits.

## V215 Local Strategy With Author Feature Fusion

```mermaid
flowchart LR
    I["CIFAR image"] --> CNN["Author residual CNN"]
    CNN --> T["Tap: 256 x 8 x 8\nthen continue author CNN"]
    CNN --> C["Author classical feature\nB x 256"]
    T --> A["1 x 1 adapter: 256 -> 128"]
    A --> D["DWT -> LL/LH/HL/HH\n128 x 4 x 4"]
    D --> L["V215 local descriptors\n4 x 4 positions; LL-HH and LH-HL\nsigned + energy"]
    L --> Q["V215 re-uploading CNOT circuits\nclass scores B x classes"]
    Q --> P["Author fusion projection\nLinear classes -> 256"]
    C --> F["Elementwise addition"]
    P --> F
    F --> O["Author classifier\nLinear 256 -> classes"]
```

This is the implemented `v215_local` source. It intentionally omits V215's
`alpha * quantum_logits` classifier-logit residual. The fusion operation is
the author model's feature-level addition instead.

## Proposed V216 Group-Preserving Local Quantum Head

```mermaid
flowchart TD
    I["CIFAR-10 image\nB x 3 x 32 x 32"] --> CNN["Author residual CNN"]
    CNN --> TAP["Middle-layer tap\nB x 256 x 8 x 8"]
    CNN --> CF["Author classical feature\nB x 256"]

    TAP --> ADAPT["Learned 1 x 1 adapter\n256 -> 128 channels"]
    ADAPT --> DWT["Haar DWT per channel"]
    DWT --> BANDS["LL, LH, HL, HH\neach B x 128 x 4 x 4"]

    BANDS --> GROUPS["Split each band into 4 groups\n32 channels per group"]
    GROUPS --> DESC["Per group and band\nchannel signed mean + RMS energy\nB x 4 groups x 16 positions"]
    DESC --> REL["Two fixed band relations\nLL-HH and LH-HL"]
    REL --> ANG["Four angles per relation\nsigned(A), signed(B), energy(A), energy(B)"]

    subgraph PQC["One shared V215 PQC bank"]
      direction TB
      ENC["Ry signed-value encoding\non two qubits"]
      ENC --> TB1["Trainable Rz/Ry + CNOT"]
      TB1 --> REUP["Rz energy re-uploading"]
      REUP --> TB2["Trainable Rz/Ry + CNOT"]
      TB2 --> MEAS["Measure both qubits"]
    end

    ANG --> PQC
    PQC --> GS["Log-membership scores\nB x 10 classes x 4 groups"]
    GS --> GW["Learned class-group softmax weights\nweighted aggregation over 4 groups"]
    GW --> QS["Quantum class evidence\nB x 10"]
    QS --> FP["Unchanged author projection\nLinear 10 -> 256"]

    CF --> ADD["Unchanged elementwise addition"]
    FP --> ADD
    ADD --> OUT["Unchanged author classifier\nLinear 256 -> 10 logits"]
```

V216 changes only the descriptor path from one all-channel statistic to four
group-preserving statistics. The learned adapter can organize related channels
before the fixed split. All groups share the existing class-specific V215 PQC
parameters; the circuit depth, measurements, feature-level fusion, classifier,
and training protocol remain unchanged. This isolates whether preserving
channel-group information improves the quantum branch.

## V217 Cross-Band Relative Energy

V217 keeps the complete V216 architecture and changes only the energy
descriptor. V216 independently applies spatial `LayerNorm` to each band's 16
energy values. V217 instead computes, for every sample, channel group, and
local position:

```text
r_b = log(E_b + eps) - mean_k(log(E_k + eps)), k in {LL, LH, HL, HH}
```

The four relative-energy values sum to zero and are invariant to a common
feature-amplitude scaling. Signed descriptors retain V216's spatial
normalization. The shared PQC bank, LL-HH/LH-HL relations, class-group pooling,
author projection, feature addition, classifier, and training protocol are
unchanged. This controlled experiment tests whether explicit cross-band energy
contrast is useful to the quantum branch.

## V218 High-Resolution Grouped Local Head

V218 tests whether CIFAR spatial information was compressed too aggressively.
It takes the learned `256 x 16 x 16` feature map before the second pooling
operation, applies the same one-level Haar transform, and retains an `8 x 8`
descriptor grid instead of V216's `4 x 4` grid:

```text
V216: 256 x 8 x 8 -> DWT -> 4 x 4 -> 16 positions
V218: 256 x 16 x 16 -> DWT -> 8 x 8 -> 64 positions
```

The channel groups, two LL-HH/LH-HL relations, shared class-specific
re-uploading circuit, group aggregation, author feature fusion, and optimizer
are unchanged. V218 increases shared circuit evaluations by four times without
assigning independent parameters to each spatial position.

## V219 Shallow Entangled Frequency Fuzzy Head

V219 keeps V218's `16 x 16` tap, `8 x 8` DWT grid, and four channel groups.
It replaces the deep signed/energy re-uploading circuit with four shallow
class-specific circuit banks:

```text
signed LL-HH, signed LH-HL, energy LL-HH, energy LH-HL
    -> Ry(input0), Ry(input1), Rz(theta0), CNOT, Ry(theta1)
    -> two membership measurements
```

Parameters are shared over all 64 positions and four channel groups. The four
circuit-bank outputs, positions, relations, and measurements are combined with
the DQFNN log-domain fuzzy product (geometric mean). A matched `simple_noent`
source removes only the CNOT for a direct entanglement ablation.

## V220 Joint ZZ Membership

V220 changes only V219's measurement set. In addition to the two local
memberships from `Z0` and `Z1`, it measures the joint frequency correlation:

```text
<ZZ> = p00 - p01 - p10 + p11
mu_ZZ = (<ZZ> + 1) / 2
```

The resulting three memberships are combined by the same DQFNN log-domain
fuzzy product. V220 adds no trainable parameters. Matched entangled and no-ent
sources isolate whether CNOT-generated joint evidence is useful.

## V221 Quantum No-Decay Training

V221 restores V216 unchanged and modifies only optimizer regularization. The
CNN and classifier retain `weight_decay=0.012`; the local-frequency head and
`Linear(classes, 256)` fuzzy projection use `weight_decay=0`. This tests whether
uniform regularization caused the observed quantum feature-norm collapse.
