"""Clean CNN + frequency-conditioned QPSAN model for CIFAR-10."""

import torch.nn as nn

from .frequency_qpa import FrequencyQPAResidual
from .model import ClassicalBranch


class FrequencyQPANet(nn.Module):
    """CNN classifier with a single frequency QPA residual at 128x16x16."""

    has_quantum_auxiliary = False

    def __init__(self, num_classes=10, hidden_dim=256, mode="torchquantum", entangled=True):
        super().__init__()
        self.classical = ClassicalBranch(hidden_dim)
        self.classical.frequency_tap_index = 3
        self.classical.frequency_modulator = FrequencyQPAResidual(
            channels=128, reduced_channels=8, mode=mode, entangled=entangled
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    @property
    def qpa_residual(self):
        return self.classical.frequency_modulator

    def forward(self, x):
        return self.classifier(self.classical(x))
