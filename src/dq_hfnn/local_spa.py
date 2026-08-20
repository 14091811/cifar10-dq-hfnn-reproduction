"""Single local self-proliferation-and-attention blocks for CIFAR features."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frequency_qpa import DirectionalFrequencyQPAResidual, FrequencyQPAResidual
from .model import ClassicalBranch


class GCStyleAttention(nn.Module):
    """Lightweight global-context attention used as the Yang-style control."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 8)
        self.score = nn.Conv2d(channels, 1, 1, bias=False)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.LayerNorm([hidden, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        weights = self.score(x).flatten(2).softmax(dim=-1)
        values = x.flatten(2).transpose(1, 2)
        context = torch.bmm(weights, values).transpose(1, 2)
        context = context.reshape(batch, channels, 1, 1)
        return x + self.transform(context)


class YangSPABlock(nn.Module):
    """One Yang-style SP&A block with a pluggable attention operator."""

    def __init__(self, channels=128, attention="gc", qpa_mode="classical"):
        super().__init__()
        if attention not in {"gc", "dwt_qpa"}:
            raise ValueError(f"Unsupported local attention: {attention}")

        expanded = channels * 2
        self.mapping = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.expansion = nn.Sequential(
            nn.Conv2d(channels, expanded, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(expanded),
            nn.ReLU(inplace=True),
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded, bias=False),
            nn.BatchNorm2d(expanded),
        )
        if attention == "gc":
            self.attention = GCStyleAttention(expanded)
        else:
            self.attention = FrequencyQPAResidual(
                channels=expanded,
                reduced_channels=8,
                mode=qpa_mode,
                entangled=qpa_mode == "torchquantum",
                alpha_max=0.10,
                alpha_init=0.02,
            )
        self.compression = nn.Sequential(
            nn.Conv2d(expanded, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        y = self.mapping(x)
        y = self.expansion(y)
        y = self.dwconv(y)
        y = self.attention(y)
        y = self.compression(y)
        return F.relu(x + y, inplace=True)


class LocalSPACNN(nn.Module):
    """Classical CNN with one local Yang-style or DWT-QPA block."""

    has_quantum_auxiliary = False

    def __init__(self, num_classes=10, hidden_dim=256, attention="none", qpa_mode="classical"):
        super().__init__()
        self.classical = ClassicalBranch(hidden_dim)
        if attention != "none":
            self.classical.features[3] = YangSPABlock(
                channels=128,
                attention=attention,
                qpa_mode=qpa_mode,
            )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.classical(x))


class ClassicalCNN(LocalSPACNN):
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, attention="none")


class YangSPACNN(LocalSPACNN):
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, attention="gc")


class DWTClassicalSPACNN(LocalSPACNN):
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, attention="dwt_qpa", qpa_mode="classical")


class DWTQPASPACNN(LocalSPACNN):
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, attention="dwt_qpa", qpa_mode="torchquantum")


class DirectionalDWTQPACNN(nn.Module):
    """CNN with a standalone DWT-QPA residual at the 128-channel tap."""

    has_quantum_auxiliary = False

    def __init__(self, num_classes=10, hidden_dim=256, mode="torchquantum"):
        super().__init__()
        self.classical = ClassicalBranch(hidden_dim)
        self.classical.frequency_tap_index = 3
        self.classical.frequency_modulator = DirectionalFrequencyQPAResidual(
            channels=128,
            reduced_channels=64,
            relation_dim=16,
            mode=mode,
            entangled=mode == "torchquantum",
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.classical(x))


class DirectionalDWTClassicalCNN(DirectionalDWTQPACNN):
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="classical")


class DirectionalDWTQuantumCNN(DirectionalDWTQPACNN):
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum")
