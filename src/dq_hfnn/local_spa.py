"""Single local self-proliferation-and-attention blocks for CIFAR features."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frequency_qpa import DirectionalFrequencyQPAResidual, FrequencyQPAResidual
from .model import Block, ClassicalBranch


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

    def __init__(self, num_classes=10, hidden_dim=256, mode="torchquantum",
                 relation_dim=16):
        super().__init__()
        self.classical = ClassicalBranch(hidden_dim)
        self.classical.frequency_tap_index = 3
        self.classical.frequency_modulator = DirectionalFrequencyQPAResidual(
            channels=128,
            reduced_channels=64,
            relation_dim=relation_dim,
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


class DirectionalDWTClassicalD8CNN(DirectionalDWTQPACNN):
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="classical", relation_dim=8)


class DirectionalDWTQuantumD8CNN(DirectionalDWTQPACNN):
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=8)


class CompactClassicalBranch(nn.Module):
    """Capacity-controlled CIFAR backbone with a 128x16x16 frequency tap."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.features = nn.Sequential(
            Block(3, 32),
            nn.MaxPool2d(2),
            Block(32, 64),
            Block(64, 128),
            nn.MaxPool2d(2),
            Block(128, 128),
            nn.MaxPool2d(2),
        )
        self.project = nn.Sequential(nn.Dropout(0.25), nn.Linear(128 * 4 * 4, hidden_dim))
        self.frequency_modulator = None
        self.frequency_tap_index = 3

    def forward(self, x):
        for index, layer in enumerate(self.features):
            x = layer(x)
            if self.frequency_modulator is not None and index == self.frequency_tap_index:
                x = self.frequency_modulator(x)
        return self.project(x.flatten(1))


class CompactDirectionalDWTQPACNN(nn.Module):
    """Compact paired control for binary small-sample DWT-QPA experiments."""

    has_quantum_auxiliary = False

    def __init__(self, num_classes=2, hidden_dim=128, mode=None, relation_dim=8):
        super().__init__()
        self.classical = CompactClassicalBranch(hidden_dim)
        if mode is not None:
            self.classical.frequency_modulator = DirectionalFrequencyQPAResidual(
                channels=128,
                reduced_channels=64,
                relation_dim=relation_dim,
                mode=mode,
                entangled=mode == "torchquantum",
            )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.classical(x))


class CompactClassicalCNN(CompactDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)


class CompactDirectionalDWTClassicalD8CNN(CompactDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="classical", relation_dim=8)


class CompactDirectionalDWTQuantumD8CNN(CompactDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="torchquantum", relation_dim=8)


class TwoBlockBinaryBranch(nn.Module):
    """Two residual blocks only, with a 128x16x16 DWT-QPA insertion point."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block1 = Block(64, 128)
        self.block2 = Block(128, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.frequency_modulator = None

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        if self.frequency_modulator is not None:
            x = self.frequency_modulator(x)
        x = self.block2(x)
        return self.pool(x).flatten(1)


class TwoBlockDirectionalDWTQPACNN(nn.Module):
    """Paired two-block controls for the CIFAR binary small-sample study."""

    has_quantum_auxiliary = False

    def __init__(self, num_classes=2, hidden_dim=128, mode=None, relation_dim=8,
                 partial_value=False, rope_2d=False):
        super().__init__()
        if hidden_dim != 128:
            raise ValueError("TwoBlockDirectionalDWTQPACNN uses a fixed 128D GAP feature")
        self.classical = TwoBlockBinaryBranch()
        if mode is not None:
            self.classical.frequency_modulator = DirectionalFrequencyQPAResidual(
                channels=128,
                reduced_channels=64,
                relation_dim=relation_dim,
                mode=mode,
                entangled=mode == "torchquantum",
                partial_value=partial_value,
                rope_2d=rope_2d,
            )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.classical(x))


class TwoBlockClassicalCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)


class TwoBlockDirectionalDWTClassicalD8CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="classical", relation_dim=8)


class TwoBlockDirectionalDWTQuantumD8CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="torchquantum", relation_dim=8)


class TwoBlockDirectionalDWTClassicalD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="classical", relation_dim=16)


class TwoBlockDirectionalDWTQuantumD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="torchquantum", relation_dim=16)


class TwoBlockDirectionalDWTClassicalPartialD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="classical",
                         relation_dim=16, partial_value=True)


class TwoBlockDirectionalDWTQuantumPartialD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="torchquantum",
                         relation_dim=16, partial_value=True)


class TwoBlockDirectionalDWTClassicalRoPED16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="classical",
                         relation_dim=16, rope_2d=True)


class TwoBlockDirectionalDWTQuantumRoPED16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="torchquantum",
                         relation_dim=16, rope_2d=True)
