"""Two-level Haar class-evidence heads for the V236 logit residual."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fca_attention import V232QuantumFcaAttention, V235HQNetProjectedWaveletAttention


class V236TwoLevelHaarQuantumLogitHead(nn.Module):
    """Produce bounded class evidence from local two-level Haar descriptors."""

    CONNECTIONS = V235HQNetProjectedWaveletAttention.CONNECTIONS

    def __init__(
        self,
        channels=128,
        feature_size=16,
        num_classes=10,
        num_circuits=4,
        entangled=True,
        alpha_max=0.1,
        alpha_init=0.05,
    ):
        super().__init__()
        if feature_size % 4:
            raise ValueError("V236 requires a feature size divisible by four")
        if num_circuits != 4:
            raise ValueError("V236 requires four projected channels/circuits")
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must be between zero and alpha_max")
        self.channels = channels
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.num_circuits = num_circuits
        self.entangled = entangled
        self.alpha_max = alpha_max
        self.projection = nn.Conv2d(channels, num_circuits, 1, bias=False)
        self.band_projection = nn.Linear(7, 4, bias=False)
        self.theta = nn.Parameter(torch.empty(num_circuits, len(self.CONNECTIONS)))
        self.class_projection = nn.Linear(num_circuits * 4, num_classes, bias=False)
        ratio = torch.tensor(alpha_init / alpha_max)
        self.alpha_logit = nn.Parameter(torch.logit(ratio))
        self._initialize_parameters()

    def _initialize_parameters(self):
        channels_per_circuit = self.channels // self.num_circuits
        with torch.no_grad():
            self.projection.weight.zero_()
            for circuit in range(self.num_circuits):
                start = circuit * channels_per_circuit
                stop = start + channels_per_circuit
                self.projection.weight[circuit, start:stop, 0, 0] = (
                    1.0 / channels_per_circuit
                )
        nn.init.orthogonal_(self.band_projection.weight)
        nn.init.normal_(self.theta, mean=0.0, std=0.1)
        nn.init.xavier_uniform_(self.class_projection.weight)

    @property
    def alpha(self):
        return self.alpha_max * self.alpha_logit.sigmoid()

    @staticmethod
    def _dwt(features):
        top_left = features[:, :, 0::2, 0::2]
        top_right = features[:, :, 0::2, 1::2]
        bottom_left = features[:, :, 1::2, 0::2]
        bottom_right = features[:, :, 1::2, 1::2]
        ll = (top_left + top_right + bottom_left + bottom_right) * 0.5
        lh = (top_left - top_right + bottom_left - bottom_right) * 0.5
        hl = (top_left + top_right - bottom_left - bottom_right) * 0.5
        hh = (top_left - top_right - bottom_left + bottom_right) * 0.5
        return ll, lh, hl, hh

    def descriptors(self, features):
        expected = (self.channels, self.feature_size, self.feature_size)
        if tuple(features.shape[1:]) != expected:
            raise ValueError(
                f"Expected [batch, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(features.shape)}"
            )
        projected = self.projection(features)
        ll1, lh1, hl1, hh1 = self._dwt(projected)
        ll2, lh2, hl2, hh2 = self._dwt(ll1)
        level1_high = [F.avg_pool2d(band, 2, 2) for band in (lh1, hl1, hh1)]
        bands = torch.stack(level1_high + [ll2, lh2, hl2, hh2], dim=-1)
        return F.layer_norm(bands, bands.shape[2:])

    def _measurements(self, descriptors):
        batch, circuits, height, width, _ = descriptors.shape
        encoded = self.band_projection(descriptors)
        angles = math.pi * torch.tanh(encoded).reshape(-1, 4)
        parameters = self.theta.view(1, circuits, 1, 1, -1)
        parameters = parameters.expand(batch, -1, height, width, -1)
        parameters = parameters.reshape(-1, len(self.CONNECTIONS))

        state = torch.zeros(
            angles.shape[0], 16, device=angles.device, dtype=torch.complex64
        )
        state[:, 0] = 1.0
        for qubit in range(4):
            state = V235HQNetProjectedWaveletAttention._rx(
                state, angles[:, qubit], qubit
            )
            state = V232QuantumFcaAttention._rz(
                state, angles[:, qubit], qubit
            )
        for edge, (control, target) in enumerate(self.CONNECTIONS):
            if self.entangled:
                state = V235HQNetProjectedWaveletAttention._crx(
                    state, parameters[:, edge], control, target
                )
            else:
                state = V235HQNetProjectedWaveletAttention._rx(
                    state, parameters[:, edge], target
                )
        measured = V232QuantumFcaAttention._measure_z(state)
        return measured.reshape(batch, circuits, height, width, 4)

    def forward(self, features):
        local_measurements = self._measurements(self.descriptors(features))
        pooled = local_measurements.mean(dim=(2, 3)).flatten(1)
        logits = self.class_projection(pooled)
        return F.layer_norm(logits, (self.num_classes,))
