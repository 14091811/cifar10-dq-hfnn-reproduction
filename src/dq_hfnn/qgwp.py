"""Quantum-gated Haar detail bypass for the first CNN downsampling stage."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .circuit import DualQubitMembershipCircuit


class QGWPHaarDetailPool(nn.Module):
    """Preserve selected Haar details alongside the original max-pool path."""

    VALID_GATE_TYPES = {"quantum", "classical", "uniform"}
    diagnostic_gate_labels = ("lh", "hl", "hh")

    def __init__(
        self,
        channels=64,
        num_groups=8,
        gate_type="quantum",
        entangled=True,
        alpha_max=0.2,
        alpha_init=0.05,
        fixed_beta=None,
        eps=1e-6,
    ):
        super().__init__()
        if channels % num_groups:
            raise ValueError("channels must be divisible by num_groups")
        if gate_type not in self.VALID_GATE_TYPES:
            raise ValueError(f"Unsupported QGWP gate type: {gate_type}")
        if fixed_beta is None and not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must be between zero and alpha_max")
        if fixed_beta is not None and fixed_beta <= 0.0:
            raise ValueError("fixed_beta must be positive")

        self.channels = channels
        self.num_groups = num_groups
        self.gate_type = gate_type
        self.entangled = entangled
        self.alpha_max = alpha_max
        self.fixed_beta = fixed_beta
        self.eps = eps
        self.mode = "normal"

        if gate_type == "uniform":
            self.register_parameter("theta", None)
        else:
            # Two local rotation blocks per channel group, four angles per block.
            self.theta = nn.Parameter(torch.empty(num_groups, 2, 4))
            nn.init.normal_(self.theta, mean=0.0, std=0.1)
        if fixed_beta is None:
            ratio = torch.tensor(alpha_init / alpha_max)
            self.alpha_logits = nn.Parameter(
                torch.full((num_groups, 3), torch.logit(ratio).item())
            )
        else:
            self.register_parameter("alpha_logits", None)

    @property
    def alpha(self):
        if self.fixed_beta is not None:
            return torch.full(
                (self.num_groups, 3),
                self.fixed_beta,
                device=self._device(),
            )
        return self.alpha_max * self.alpha_logits.sigmoid()

    def _device(self):
        parameter = next(self.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    @staticmethod
    def _dwt(features):
        top_left = features[:, :, 0::2, 0::2]
        top_right = features[:, :, 0::2, 1::2]
        bottom_left = features[:, :, 1::2, 0::2]
        bottom_right = features[:, :, 1::2, 1::2]
        return (
            (top_left + top_right + bottom_left + bottom_right) * 0.5,
            (top_left - top_right + bottom_left - bottom_right) * 0.5,
            (top_left + top_right - bottom_left - bottom_right) * 0.5,
            (top_left - top_right - bottom_left + bottom_right) * 0.5,
        )

    def descriptors(self, bands):
        ll, lh, hl, hh = bands
        grouped = []
        channels_per_group = self.channels // self.num_groups
        for band in bands:
            grouped.append(
                band.reshape(
                    band.shape[0],
                    self.num_groups,
                    channels_per_group,
                    band.shape[2],
                    band.shape[3],
                )
            )

        ll_signed = grouped[0].mean(dim=2)
        high_energy = torch.stack(
            [value.square().mean(dim=2) for value in grouped[1:]], dim=0
        ).mean(dim=0).add(self.eps).sqrt()
        directional = (grouped[1] - grouped[2]).mean(dim=2)
        diagonal = grouped[3].mean(dim=2)
        normalized = []
        for descriptor in (ll_signed, high_energy, directional, diagonal):
            height, width = descriptor.shape[-2:]
            normalized.append(
                F.layer_norm(
                    descriptor.flatten(2), (height * width,)
                ).reshape_as(descriptor)
            )
        return torch.stack(normalized, dim=-1)

    @staticmethod
    def _local_rotation_block(state, parameters):
        state = DualQubitMembershipCircuit._rz_q0(state, parameters[:, 0])
        state = DualQubitMembershipCircuit._ry_q0(state, parameters[:, 1])
        state = DualQubitMembershipCircuit._rz_q1(state, parameters[:, 2])
        return DualQubitMembershipCircuit._ry_q1(state, parameters[:, 3])

    def _quantum_gates(self, descriptors):
        batch, groups, height, width, _ = descriptors.shape
        angles = math.pi * torch.tanh(descriptors).reshape(-1, 4)
        parameters = self.theta.view(1, groups, 1, 1, 2, 4)
        parameters = parameters.expand(batch, -1, height, width, -1, -1)
        parameters = parameters.reshape(-1, 2, 4)

        state = torch.zeros(
            angles.shape[0], 4, device=angles.device, dtype=torch.complex64
        )
        state[:, 0] = 1.0
        state = DualQubitMembershipCircuit._ry_q0(state, angles[:, 0])
        state = DualQubitMembershipCircuit._ry_q1(state, angles[:, 1])
        state = self._local_rotation_block(state, parameters[:, 0])
        if self.entangled:
            state = DualQubitMembershipCircuit._cnot(state)
        state = DualQubitMembershipCircuit._rz_q0(state, angles[:, 2])
        state = DualQubitMembershipCircuit._rz_q1(state, angles[:, 3])
        state = self._local_rotation_block(state, parameters[:, 1])
        if self.entangled:
            state = DualQubitMembershipCircuit._cnot(state)

        probabilities = state.real.square() + state.imag.square()
        z0 = (
            probabilities[:, 0]
            + probabilities[:, 1]
            - probabilities[:, 2]
            - probabilities[:, 3]
        )
        z1 = (
            probabilities[:, 0]
            - probabilities[:, 1]
            + probabilities[:, 2]
            - probabilities[:, 3]
        )
        zz = (
            probabilities[:, 0]
            - probabilities[:, 1]
            - probabilities[:, 2]
            + probabilities[:, 3]
        )
        return torch.stack((z0, z1, zz), dim=-1).reshape(
            batch, groups, height, width, 3
        )

    def _classical_gates(self, descriptors):
        first = (
            descriptors * self.theta[:, 0].view(1, self.num_groups, 1, 1, 4)
        ).sum(dim=-1)
        second = (
            descriptors * self.theta[:, 1].view(1, self.num_groups, 1, 1, 4)
        ).sum(dim=-1)
        first, second = torch.tanh(first), torch.tanh(second)
        return torch.stack((first, second, first * second), dim=-1)

    def gates(self, bands):
        if self.gate_type == "uniform":
            ll = bands[0]
            gates = ll.new_ones(
                ll.shape[0], self.num_groups, ll.shape[2], ll.shape[3], 3
            )
            return gates
        descriptors = self.descriptors(bands)
        gates = (
            self._quantum_gates(descriptors)
            if self.gate_type == "quantum"
            else self._classical_gates(descriptors)
        )
        if self.mode == "shuffle":
            gates = gates.roll(1, dims=0)
        return gates

    def components(self, features):
        """Return the baseline, gates, and applied detail correction."""
        baseline = F.max_pool2d(features, 2, 2)
        bands = self._dwt(features)
        gates = self.gates(bands)
        channels_per_group = self.channels // self.num_groups
        channel_gates = gates.repeat_interleave(channels_per_group, dim=1)
        details = torch.stack(bands[1:], dim=-1)
        if self.fixed_beta is None:
            strength = self.alpha.repeat_interleave(channels_per_group, dim=0)
            strength = strength.view(1, self.channels, 1, 1, 3)
            correction = (strength * channel_gates * details).sum(dim=-1)
        else:
            correction = self.fixed_beta * (channel_gates * details).sum(dim=-1)
        return baseline, gates, correction / math.sqrt(3.0)

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                f"Expected [batch, {self.channels}, height, width], got {tuple(features.shape)}"
            )
        if features.shape[-2] % 2 or features.shape[-1] % 2:
            raise ValueError("QGWP requires even spatial dimensions")

        if self.mode == "zero":
            return F.max_pool2d(features, 2, 2)
        baseline, _, correction = self.components(features)
        return baseline + correction
