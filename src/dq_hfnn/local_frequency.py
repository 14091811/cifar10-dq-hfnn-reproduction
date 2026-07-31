"""V215 local-frequency quantum head with the author feature-level fusion API."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .circuit import DualQubitMembershipCircuit


class V215LocalFrequencyHead(nn.Module):
    """Return class scores, not a residual: fusion remains the author design."""

    PAIR_TYPES = ((0, 3), (1, 2))

    def __init__(self, num_classes, feature_channels=256, grid_size=4, eps=1e-4):
        super().__init__()
        self.grid_size, self.eps = grid_size, eps
        self.entangled = True
        self.adapter = nn.Conv2d(feature_channels, 128, kernel_size=1)
        self.num_classes = num_classes
        # [class, relation, re-upload block, Rz0/Ry0/Rz1/Ry1]
        self.theta = nn.Parameter(torch.empty(num_classes, len(self.PAIR_TYPES), 2, 4))
        nn.init.uniform_(self.theta, 0.0, 2.0 * math.pi)

    @staticmethod
    def _dwt(features):
        a, b = features[:, :, 0::2, 0::2], features[:, :, 0::2, 1::2]
        c, d = features[:, :, 1::2, 0::2], features[:, :, 1::2, 1::2]
        return ((a + b + c + d) * 0.5, (a - b + c - d) * 0.5, (a + b - c - d) * 0.5, (a - b - c + d) * 0.5)

    def _descriptors(self, bands):
        signed, energy = [], []
        for band in bands:
            signed_map = band.mean(dim=1, keepdim=True)
            energy_map = band.square().mean(dim=1, keepdim=True).add(1e-8).sqrt()
            signed.append(F.layer_norm(F.adaptive_avg_pool2d(signed_map, (self.grid_size, self.grid_size)).flatten(1), (self.grid_size ** 2,)))
            energy.append(F.layer_norm(F.adaptive_avg_pool2d(energy_map, (self.grid_size, self.grid_size)).flatten(1), (self.grid_size ** 2,)))
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

    def _memberships(self, relation_angles):
        """Evaluate all classes and relations together to avoid tiny GPU calls."""
        batch, positions, relations, _ = relation_angles.shape
        angles = relation_angles.unsqueeze(2).expand(-1, -1, self.num_classes, -1, -1)
        parameters = self.theta.view(1, 1, self.num_classes, relations, 2, 4).expand(batch, positions, -1, -1, -1, -1)
        angles, parameters = angles.reshape(-1, 4), parameters.reshape(-1, 2, 4)
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
        return torch.stack(((z0 + 1) / 2, (z1 + 1) / 2), dim=1).reshape(batch, positions, self.num_classes, relations, 2)

    def forward(self, local_features):
        signed, energy = self._descriptors(self._dwt(self.adapter(local_features)))
        batch = local_features.shape[0]
        relation_angles = torch.stack(
            [torch.stack((self._to_angle(signed[left]), self._to_angle(signed[right]), self._to_angle(energy[left]), self._to_angle(energy[right])), dim=-1) for left, right in self.PAIR_TYPES],
            dim=2,
        )
        memberships = self._memberships(relation_angles).clamp(self.eps, 1.0)
        raw_scores = (0.5 * memberships.log().sum(dim=-1)).mean(dim=1).mean(dim=-1)
        return F.layer_norm(raw_scores, (raw_scores.shape[1],))


class V216GroupedLocalFrequencyHead(V215LocalFrequencyHead):
    """Preserve channel-group statistics while sharing the V215 PQC bank."""

    def __init__(self, num_classes, feature_channels=256, grid_size=4, num_groups=4, eps=1e-4):
        super().__init__(num_classes, feature_channels, grid_size, eps)
        if self.adapter.out_channels % num_groups != 0:
            raise ValueError(
                f"Adapter channels ({self.adapter.out_channels}) must be divisible "
                f"by num_groups ({num_groups})"
            )
        self.num_groups = num_groups
        # Uniform at initialization; each class can later select useful groups.
        self.class_group_logits = nn.Parameter(torch.zeros(num_classes, num_groups))

    def _descriptors(self, bands):
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
            energy_map = grouped.square().mean(dim=2).add(1e-8).sqrt()

            def pool_and_normalize(values):
                values = values.reshape(batch * self.num_groups, 1, height, width)
                values = F.adaptive_avg_pool2d(values, (self.grid_size, self.grid_size))
                values = values.flatten(1).reshape(batch, self.num_groups, -1)
                return F.layer_norm(values, (self.grid_size ** 2,))

            signed.append(pool_and_normalize(signed_map))
            energy.append(pool_and_normalize(energy_map))
        return signed, energy

    def _grouped_memberships(self, relation_angles):
        """Evaluate positions, classes, groups, and relations in one batch."""
        batch, positions, groups, relations, _ = relation_angles.shape
        angles = relation_angles.unsqueeze(2).expand(
            -1, -1, self.num_classes, -1, -1, -1
        )
        parameters = self.theta.view(
            1, 1, self.num_classes, 1, relations, 2, 4
        ).expand(batch, positions, -1, groups, -1, -1, -1)
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
        memberships = torch.stack(((z0 + 1) / 2, (z1 + 1) / 2), dim=1)
        return memberships.reshape(batch, positions, self.num_classes, groups, relations, 2)

    def forward(self, local_features):
        signed, energy = self._descriptors(self._dwt(self.adapter(local_features)))
        relations = []
        for left, right in self.PAIR_TYPES:
            relation = torch.stack(
                (
                    self._to_angle(signed[left]),
                    self._to_angle(signed[right]),
                    self._to_angle(energy[left]),
                    self._to_angle(energy[right]),
                ),
                dim=-1,
            )
            relations.append(relation.permute(0, 2, 1, 3))
        relation_angles = torch.stack(relations, dim=3)

        memberships = self._grouped_memberships(relation_angles).clamp(self.eps, 1.0)
        # [batch, class, group], retaining groups until the final learned pooling.
        group_scores = memberships.log().mean(dim=-1).mean(dim=1).mean(dim=-1)
        group_weights = self.class_group_logits.softmax(dim=-1).unsqueeze(0)
        raw_scores = (group_scores * group_weights).sum(dim=-1)
        return F.layer_norm(raw_scores, (raw_scores.shape[1],))


class V217RelativeEnergyLocalFrequencyHead(V216GroupedLocalFrequencyHead):
    """Retain cross-band energy ratios instead of normalizing each band alone."""

    def _descriptors(self, bands):
        signed, log_energy = [], []
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
            energy_map = grouped.square().mean(dim=2).add(1e-8).sqrt()

            def pool(values):
                values = values.reshape(batch * self.num_groups, 1, height, width)
                values = F.adaptive_avg_pool2d(
                    values, (self.grid_size, self.grid_size)
                )
                return values.flatten(1).reshape(batch, self.num_groups, -1)

            pooled_signed = pool(signed_map)
            signed.append(
                F.layer_norm(pooled_signed, (self.grid_size ** 2,))
            )
            log_energy.append(pool(energy_map).add(1e-8).log())

        # Center over LL/LH/HL/HH at every sample, group, and local position.
        relative_energy = torch.stack(log_energy, dim=2)
        relative_energy = relative_energy - relative_energy.mean(dim=2, keepdim=True)
        return signed, list(relative_energy.unbind(dim=2))


class V219SimpleFrequencyHead(V216GroupedLocalFrequencyHead):
    """Many shallow frequency circuits followed by DQFNN log-domain pooling."""

    def __init__(
        self,
        num_classes,
        feature_channels=256,
        grid_size=8,
        num_groups=4,
        eps=1e-4,
        entangled=True,
        measure_zz=False,
    ):
        super().__init__(num_classes, feature_channels, grid_size, num_groups, eps)
        self.entangled = entangled
        self.measure_zz = measure_zz
        # Independent shallow circuits for signed/energy and both band relations.
        # [class, descriptor, relation, Rz0/Ry0]
        self.theta = nn.Parameter(
            torch.empty(num_classes, 2, len(self.PAIR_TYPES), 2)
        )
        nn.init.uniform_(self.theta, 0.0, 2.0 * math.pi)

    def _simple_memberships(self, relation_angles, descriptor_index):
        """Evaluate one descriptor stream over positions, classes, and groups."""
        batch, positions, groups, relations, _ = relation_angles.shape
        angles = relation_angles.unsqueeze(2).expand(
            -1, -1, self.num_classes, -1, -1, -1
        )
        parameters = self.theta[:, descriptor_index].view(
            1, 1, self.num_classes, 1, relations, 2
        ).expand(batch, positions, -1, groups, -1, -1)
        angles = angles.reshape(-1, 2)
        parameters = parameters.reshape(-1, 2)

        state = torch.zeros(
            angles.shape[0], 4, device=angles.device, dtype=torch.complex64
        )
        state[:, 0] = 1
        state = DualQubitMembershipCircuit._ry_q0(state, angles[:, 0])
        state = DualQubitMembershipCircuit._ry_q1(state, angles[:, 1])
        state = DualQubitMembershipCircuit._rz_q0(state, parameters[:, 0])
        if self.entangled:
            state = DualQubitMembershipCircuit._cnot(state)
        state = DualQubitMembershipCircuit._ry_q0(state, parameters[:, 1])

        probabilities = state.real.square() + state.imag.square()
        z0 = probabilities[:, 0] + probabilities[:, 1] - probabilities[:, 2] - probabilities[:, 3]
        z1 = probabilities[:, 0] - probabilities[:, 1] + probabilities[:, 2] - probabilities[:, 3]
        observables = [z0, z1]
        if self.measure_zz:
            zz = probabilities[:, 0] - probabilities[:, 1] - probabilities[:, 2] + probabilities[:, 3]
            observables.append(zz)
        memberships = torch.stack(tuple((value + 1) / 2 for value in observables), dim=1)
        return memberships.reshape(
            batch, positions, self.num_classes, groups, relations, len(observables)
        )

    def forward(self, local_features):
        descriptors = self._descriptors(self._dwt(self.adapter(local_features)))
        descriptor_scores = []
        for descriptor_index, values in enumerate(descriptors):
            relations = []
            for left, right in self.PAIR_TYPES:
                pair = torch.stack(
                    (self._to_angle(values[left]), self._to_angle(values[right])),
                    dim=-1,
                )
                relations.append(pair.permute(0, 2, 1, 3))
            relation_angles = torch.stack(relations, dim=3)
            memberships = self._simple_memberships(
                relation_angles, descriptor_index
            ).clamp(self.eps, 1.0)
            # Fuzzy product over positions, relations, and both measurements.
            descriptor_scores.append(
                memberships.log().mean(dim=-1).mean(dim=1).mean(dim=-1)
            )

        group_scores = torch.stack(descriptor_scores, dim=-1).mean(dim=-1)
        group_weights = self.class_group_logits.softmax(dim=-1).unsqueeze(0)
        raw_scores = (group_scores * group_weights).sum(dim=-1)
        return F.layer_norm(raw_scores, (raw_scores.shape[1],))
