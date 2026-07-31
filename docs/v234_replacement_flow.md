# V234 Projected-Wavelet Replacement

This model removes the author's raw-pixel DQ branch. The only quantum module is
the projected-wavelet Ring attention inside the CNN trunk.

```mermaid
flowchart TD
    I["CIFAR-10 image<br/>B x 3 x 32 x 32"]
    I --> E["Author CNN front<br/>B x 128 x 16 x 16"]
    E --> P["Learned 1 x 1 projection<br/>128 to 4"]
    P --> W["Haar LL/LH/HL/HH<br/>then 2 x 2 average pooling"]
    W --> D["Descriptors<br/>4 projections x 4 x 4 regions x 4 bands"]
    D --> Q

    subgraph Q["Four independent 4-qubit circuits"]
      direction TB
      Q1["Ry descriptor encoding"] --> Q2["Trainable Ry/Rz"]
      Q2 --> Q3["CNOT Ring: 0-1-2-3-0"]
      Q3 --> Q4["Trainable Ry and Z measurements"]
    end

    Q --> G["16 local gate maps<br/>4 circuits x 4 measurements"]
    G --> U["Nearest upsample to 16 x 16<br/>repeat each gate over 8 channels"]
    E --> M["Residual modulation<br/>F' = F * (1 + alpha * gate)"]
    U --> M
    M --> B["Author CNN back<br/>Flatten 2048 to feature 256"]
    B --> C["Linear 256 to 10 logits"]
```

Removed components:

- 460 raw RGB pixel pairs
- ten author class-specific two-qubit circuits
- 920 fuzzy memberships and log-domain aggregation
- author `Linear(10, 256)` fuzzy projection
- feature addition between the CNN and author DQ branches

The Ring, no-entanglement, and classical matched variants each add exactly 576
parameters over the pure CNN control: 512 projection weights, 48 interaction
parameters, and 16 bounded modulation parameters.
