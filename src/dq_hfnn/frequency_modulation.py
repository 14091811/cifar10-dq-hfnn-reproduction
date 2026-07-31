"""Dense wavelet modulation driven by a shared two-qubit fuzzy circuit."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .circuit import DualQubitMembershipCircuit


class V223QuantumFrequencyModulator(nn.Module):
    """Modulate dense CNN wavelet maps without class-specific compression."""

    def __init__(
        self,
        channels,
        num_groups=4,
        entangled=True,
        alpha_max=0.5,
        alpha_init=0.1,
        eps=1e-6,
    ):
        super().__init__()
        if channels % num_groups != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_groups ({num_groups})"
            )
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must be between zero and alpha_max")
        self.channels = channels
        self.num_groups = num_groups
        self.entangled = entangled
        self.alpha_max = alpha_max
        self.eps = eps
        self.mode = "normal"
        # [group, LL-HH/LH-HL relation, re-upload block, Rz0/Ry0/Rz1/Ry1]
        self.theta = nn.Parameter(torch.empty(num_groups, 2, 2, 4))
        nn.init.uniform_(self.theta, 0.0, 2.0 * math.pi)
        initial_probability = alpha_init / alpha_max
        initial_logit = math.log(initial_probability / (1.0 - initial_probability))
        self.alpha_logits = nn.Parameter(torch.full((num_groups, 3), initial_logit))

    @property
    def alpha(self):
        return self.alpha_max * self.alpha_logits.sigmoid()

    @staticmethod
    def _dwt(features):
        a = features[:, :, 0::2, 0::2]
        b = features[:, :, 0::2, 1::2]
        c = features[:, :, 1::2, 0::2]
        d = features[:, :, 1::2, 1::2]
        return (
            (a + b + c + d) * 0.5,
            (a - b + c - d) * 0.5,
            (a + b - c - d) * 0.5,
            (a - b - c + d) * 0.5,
        )

    @staticmethod
    def _idwt(bands):
        ll, lh, hl, hh = bands
        batch, channels, height, width = ll.shape
        output = ll.new_empty(batch, channels, height * 2, width * 2)
        output[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
        output[:, :, 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
        output[:, :, 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
        output[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
        return output

    def _group_descriptors(self, bands):
        signed, energy = [], []
        for band in bands:
            batch, channels, height, width = band.shape
            grouped = band.reshape(
                batch,
                self.num_groups,
                channels // self.num_groups,
                height,
                width,
            )
            signed_map = grouped.mean(dim=2)
            energy_map = grouped.square().mean(dim=2).add(self.eps).sqrt()
            signed.append(F.layer_norm(signed_map.flatten(2), (height * width,)).reshape_as(signed_map))
            log_energy = energy_map.add(self.eps).log()
            energy.append(F.layer_norm(log_energy.flatten(2), (height * width,)).reshape_as(log_energy))
        return signed, energy

    @staticmethod
    def _to_angle(values):
        return (torch.tanh(values) + 1.0) * (0.5 * math.pi)

    def _trainable_block(self, state, parameters):
        state = DualQubitMembershipCircuit._rz_q0(state, parameters[:, 0])
        state = DualQubitMembershipCircuit._ry_q0(state, parameters[:, 1])
        state = DualQubitMembershipCircuit._rz_q1(state, parameters[:, 2])
        state = DualQubitMembershipCircuit._ry_q1(state, parameters[:, 3])
        if self.entangled:
            state = DualQubitMembershipCircuit._cnot(state)
        return state

    def _relation_gates(self, bands):
        signed, energy = self._group_descriptors(bands)
        relations = []
        for left, right in ((0, 3), (1, 2)):
            relations.append(
                torch.stack(
                    (
                        self._to_angle(signed[left]),
                        self._to_angle(signed[right]),
                        self._to_angle(energy[left]),
                        self._to_angle(energy[right]),
                    ),
                    dim=-1,
                )
            )
        angles = torch.stack(relations, dim=4)
        batch, groups, height, width, relation_count, _ = angles.shape
        parameters = self.theta.view(1, groups, 1, 1, relation_count, 2, 4)
        parameters = parameters.expand(batch, -1, height, width, -1, -1, -1)
        angles = angles.reshape(-1, 4)
        parameters = parameters.reshape(-1, 2, 4)

        state = torch.zeros(angles.shape[0], 4, device=angles.device, dtype=torch.complex64)
        state[:, 0] = 1
        state = DualQubitMembershipCircuit._ry_q0(state, angles[:, 0])
        state = DualQubitMembershipCircuit._ry_q1(state, angles[:, 1])
        state = self._trainable_block(state, parameters[:, 0])
        state = DualQubitMembershipCircuit._rz_q0(state, angles[:, 2])
        state = DualQubitMembershipCircuit._rz_q1(state, angles[:, 3])
        state = self._trainable_block(state, parameters[:, 1])
        probabilities = state.real.square() + state.imag.square()
        z0 = probabilities[:, 0] + probabilities[:, 1] - probabilities[:, 2] - probabilities[:, 3]
        z1 = probabilities[:, 0] - probabilities[:, 1] + probabilities[:, 2] - probabilities[:, 3]
        memberships = torch.stack(((z0 + 1.0) * 0.5, (z1 + 1.0) * 0.5), dim=-1)
        memberships = memberships.clamp_min(self.eps)
        # DQFNN product relation, centered so modulation can suppress or amplify.
        gates = 2.0 * memberships.log().mean(dim=-1).exp() - 1.0
        gates = gates.reshape(batch, groups, height, width, relation_count)
        if self.mode == "shuffle":
            gates = gates.roll(1, dims=0)
        return gates

    def forward(self, features):
        if self.mode == "zero":
            return features
        original_height, original_width = features.shape[-2:]
        pad_height = original_height % 2
        pad_width = original_width % 2
        if pad_height or pad_width:
            features = F.pad(features, (0, pad_width, 0, pad_height), mode="replicate")

        bands = self._dwt(features)
        gates = self._relation_gates(bands)
        channels_per_group = self.channels // self.num_groups
        gates = gates.repeat_interleave(channels_per_group, dim=1)
        alpha = self.alpha.repeat_interleave(channels_per_group, dim=0)
        alpha = alpha.view(1, self.channels, 1, 1, 3)

        ll, lh, hl, hh = bands
        relation_hh = gates[..., 0]
        relation_direction = gates[..., 1]
        lh = lh * (1.0 + alpha[..., 0] * relation_direction)
        hl = hl * (1.0 + alpha[..., 1] * relation_direction)
        hh = hh * (1.0 + alpha[..., 2] * relation_hh)
        reconstructed = self._idwt((ll, lh, hl, hh))
        return reconstructed[:, :, :original_height, :original_width]
