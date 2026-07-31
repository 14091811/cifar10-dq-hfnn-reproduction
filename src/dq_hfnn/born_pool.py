"""Two-qubit Born-probability pooling for shallow Haar features."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .circuit import DualQubitMembershipCircuit


class QuantumBornHaarPool(nn.Module):
    """Use four normalized selection probabilities as a dynamic 2x2 pool."""

    VALID_GATE_TYPES = {"quantum", "classical", "softpool"}
    is_probability_pool = True
    diagnostic_gate_labels = ("top_left", "top_right", "bottom_left", "bottom_right")

    def __init__(
        self,
        channels=64,
        num_groups=8,
        gate_type="quantum",
        entangled=False,
        eps=1e-6,
    ):
        super().__init__()
        if channels % num_groups:
            raise ValueError("channels must be divisible by num_groups")
        if gate_type not in self.VALID_GATE_TYPES:
            raise ValueError(f"Unsupported Born-pool gate type: {gate_type}")

        self.channels = channels
        self.num_groups = num_groups
        self.gate_type = gate_type
        self.entangled = entangled
        self.eps = eps
        self.mode = "normal"

        if gate_type == "softpool":
            self.register_parameter("theta", None)
        else:
            # Two four-angle local blocks per channel group.
            self.theta = nn.Parameter(torch.empty(num_groups, 2, 4))
            nn.init.normal_(self.theta, mean=0.0, std=0.1)

    @staticmethod
    def patches(features):
        return torch.stack(
            (
                features[:, :, 0::2, 0::2],
                features[:, :, 0::2, 1::2],
                features[:, :, 1::2, 0::2],
                features[:, :, 1::2, 1::2],
            ),
            dim=-1,
        )

    def descriptors(self, patches):
        top_left, top_right, bottom_left, bottom_right = patches.unbind(dim=-1)
        bands = (
            (top_left + top_right + bottom_left + bottom_right) * 0.5,
            (top_left - top_right + bottom_left - bottom_right) * 0.5,
            (top_left + top_right - bottom_left - bottom_right) * 0.5,
            (top_left - top_right - bottom_left + bottom_right) * 0.5,
        )
        channels_per_group = self.channels // self.num_groups
        grouped = [
            band.reshape(
                band.shape[0],
                self.num_groups,
                channels_per_group,
                band.shape[2],
                band.shape[3],
            ).mean(dim=2)
            for band in bands
        ]
        normalized = []
        for descriptor in grouped:
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

    def _quantum_probabilities(self, descriptors):
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
            state = DualQubitMembershipCircuit._cnot_reverse(state)

        probabilities = state.real.square() + state.imag.square()
        return probabilities.reshape(batch, groups, height, width, 4)

    def _classical_probabilities(self, descriptors):
        row_correction = (
            descriptors * self.theta[:, 0].view(1, self.num_groups, 1, 1, 4)
        ).sum(dim=-1)
        column_correction = (
            descriptors * self.theta[:, 1].view(1, self.num_groups, 1, 1, 4)
        ).sum(dim=-1)
        row_angle = math.pi * torch.tanh(descriptors[..., 0]) + row_correction
        column_angle = (
            math.pi * torch.tanh(descriptors[..., 1]) + column_correction
        )
        row_zero = torch.cos(0.5 * row_angle).square()
        row_one = torch.sin(0.5 * row_angle).square()
        column_zero = torch.cos(0.5 * column_angle).square()
        column_one = torch.sin(0.5 * column_angle).square()
        return torch.stack(
            (
                row_zero * column_zero,
                row_zero * column_one,
                row_one * column_zero,
                row_one * column_one,
            ),
            dim=-1,
        )

    def _softpool_probabilities(self, patches):
        channels_per_group = self.channels // self.num_groups
        scores = patches.reshape(
            patches.shape[0],
            self.num_groups,
            channels_per_group,
            patches.shape[2],
            patches.shape[3],
            4,
        ).mean(dim=2)
        scores = scores - scores.mean(dim=-1, keepdim=True)
        scores = scores / scores.square().mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return scores.softmax(dim=-1)

    def probabilities(self, patches):
        if self.gate_type == "softpool":
            probabilities = self._softpool_probabilities(patches)
        else:
            descriptors = self.descriptors(patches)
            probabilities = (
                self._quantum_probabilities(descriptors)
                if self.gate_type == "quantum"
                else self._classical_probabilities(descriptors)
            )
        if self.mode == "shuffle":
            probabilities = probabilities.roll(1, dims=0)
        return probabilities

    def components(self, features):
        baseline = F.max_pool2d(features, 2, 2)
        patches = self.patches(features)
        probabilities = self.probabilities(patches)
        channels_per_group = self.channels // self.num_groups
        channel_probabilities = probabilities.repeat_interleave(
            channels_per_group, dim=1
        )
        pooled = (patches * channel_probabilities).sum(dim=-1)
        return baseline, probabilities, pooled - baseline

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                f"Expected [batch, {self.channels}, height, width], got {tuple(features.shape)}"
            )
        if features.shape[-2] % 2 or features.shape[-1] % 2:
            raise ValueError("Born pooling requires even spatial dimensions")
        if self.mode == "zero":
            return F.max_pool2d(features, 2, 2)
        baseline, _, delta = self.components(features)
        return baseline + delta
