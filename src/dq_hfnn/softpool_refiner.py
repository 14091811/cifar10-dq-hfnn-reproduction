"""Identity-initialized quantum refinement of grouped SoftPool weights."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .circuit import DualQubitMembershipCircuit


class QuantumRefinedSoftPool(nn.Module):
    """Prepare SoftPool amplitudes and learn a small unitary probability update."""

    VALID_GATE_TYPES = {"quantum", "classical"}
    diagnostic_gate_labels = ("top_left", "top_right", "bottom_left", "bottom_right")
    is_probability_pool = True

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
            raise ValueError(f"Unsupported SoftPool refiner: {gate_type}")

        self.channels = channels
        self.num_groups = num_groups
        self.gate_type = gate_type
        self.entangled = entangled
        self.configured_entangled = entangled
        self.eps = eps
        self.mode = "normal"

        if gate_type == "quantum":
            self.theta = nn.Parameter(torch.zeros(num_groups, 10))
        else:
            # Two four-input corrections and their biases: 10 parameters/group.
            self.theta = nn.Parameter(torch.zeros(num_groups, 2, 5))

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

    def _group_scores(self, patches):
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
        return scores / scores.square().mean(dim=-1, keepdim=True).add(self.eps).sqrt()

    def base_probabilities(self, patches):
        return self._group_scores(patches).softmax(dim=-1)

    @staticmethod
    def _local_rotation_block(state, parameters):
        state = DualQubitMembershipCircuit._rz_q0(state, parameters[:, 0])
        state = DualQubitMembershipCircuit._ry_q0(state, parameters[:, 1])
        state = DualQubitMembershipCircuit._rz_q1(state, parameters[:, 2])
        return DualQubitMembershipCircuit._ry_q1(state, parameters[:, 3])

    @staticmethod
    def _rzz(state, angle):
        zero = torch.zeros_like(angle)
        even = torch.exp(torch.complex(zero, -0.5 * angle))
        odd = torch.exp(torch.complex(zero, 0.5 * angle))
        return state * torch.stack((even, odd, odd, even), dim=1)

    @staticmethod
    def _rxx(state, angle):
        cosine = torch.cos(0.5 * angle).unsqueeze(1)
        sine = torch.sin(0.5 * angle).unsqueeze(1)
        flipped = state[:, (3, 2, 1, 0)]
        return cosine * state - 1j * sine * flipped

    def _expanded_quantum_parameters(self, descriptors):
        batch, groups, height, width, _ = descriptors.shape
        return self.theta.view(1, groups, 1, 1, 10).expand(
            batch, -1, height, width, -1
        ).reshape(-1, 10)

    def _quantum_probabilities(self, base, descriptors):
        parameters = self._expanded_quantum_parameters(descriptors)
        state = base.reshape(-1, 4).clamp_min(0.0).sqrt().to(torch.complex64)

        # Directional Haar contrasts enter as phase while preserving SoftPool
        # probabilities at the identity initialization.
        left_right = descriptors[..., 0] - descriptors[..., 1] + descriptors[..., 2] - descriptors[..., 3]
        top_bottom = descriptors[..., 0] + descriptors[..., 1] - descriptors[..., 2] - descriptors[..., 3]
        diagonal = descriptors[..., 0] - descriptors[..., 1] - descriptors[..., 2] + descriptors[..., 3]
        state = DualQubitMembershipCircuit._rz_q0(
            state, math.pi * torch.tanh(top_bottom.reshape(-1))
        )
        state = DualQubitMembershipCircuit._rz_q1(
            state, math.pi * torch.tanh(left_right.reshape(-1))
        )
        if self.entangled:
            state = self._rzz(state, math.pi * torch.tanh(diagonal.reshape(-1)))

        state = self._local_rotation_block(state, parameters[:, 0:4])
        if self.entangled:
            state = self._rzz(state, parameters[:, 8])
            state = self._rxx(state, parameters[:, 9])
        elif not self.configured_entangled:
            state = DualQubitMembershipCircuit._ry_q0(state, parameters[:, 8])
            state = DualQubitMembershipCircuit._ry_q1(state, parameters[:, 9])
        state = self._local_rotation_block(state, parameters[:, 4:8])

        probabilities = state.real.square() + state.imag.square()
        return probabilities.reshape_as(base)

    def _classical_probabilities(self, base, descriptors):
        weights = self.theta[..., :4]
        biases = self.theta[..., 4]
        row = (
            descriptors * weights[:, 0].view(1, self.num_groups, 1, 1, 4)
        ).sum(dim=-1) + biases[:, 0].view(1, self.num_groups, 1, 1)
        column = (
            descriptors * weights[:, 1].view(1, self.num_groups, 1, 1, 4)
        ).sum(dim=-1) + biases[:, 1].view(1, self.num_groups, 1, 1)
        correction = torch.stack(
            (row + column, row - column, -row + column, -row - column),
            dim=-1,
        )
        return (base.clamp_min(self.eps).log() + correction).softmax(dim=-1)

    def probabilities(self, patches):
        descriptors = self._group_scores(patches)
        base = descriptors.softmax(dim=-1)
        if self.mode == "base":
            return base
        probabilities = (
            self._quantum_probabilities(base, descriptors)
            if self.gate_type == "quantum"
            else self._classical_probabilities(base, descriptors)
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
            raise ValueError("SoftPool refinement requires even spatial dimensions")
        if self.mode == "zero":
            return F.max_pool2d(features, 2, 2)
        baseline, _, delta = self.components(features)
        return baseline + delta
