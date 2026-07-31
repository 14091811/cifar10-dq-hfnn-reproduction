"""FcaNet-style DCT channel attention driven by compact four-qubit circuits."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# First 16 entries of FcaNet's commonly used top-frequency selection.
FCA_TOP16_FREQUENCIES = (
    (0, 0),
    (0, 1),
    (6, 0),
    (0, 5),
    (0, 2),
    (1, 0),
    (1, 2),
    (4, 0),
    (5, 0),
    (1, 6),
    (3, 0),
    (0, 4),
    (0, 6),
    (0, 3),
    (3, 5),
    (2, 2),
)


class GroupedDCTDescriptors(nn.Module):
    """Compress channel groups with fixed two-dimensional DCT bases."""

    def __init__(self, channels=128, height=16, width=16, num_groups=16):
        super().__init__()
        if channels % num_groups:
            raise ValueError("channels must be divisible by num_groups")
        if num_groups != len(FCA_TOP16_FREQUENCIES):
            raise ValueError("V232 currently requires 16 DCT channel groups")
        self.channels = channels
        self.height = height
        self.width = width
        self.num_groups = num_groups
        self.register_buffer("filters", self._build_filters(height, width))

    @staticmethod
    def _basis(length, frequency):
        positions = torch.arange(length, dtype=torch.float32) + 0.5
        basis = torch.cos(math.pi * frequency * positions / length) / math.sqrt(length)
        if frequency:
            basis = basis * math.sqrt(2.0)
        return basis

    @classmethod
    def _build_filters(cls, height, width):
        filters = []
        for frequency_y, frequency_x in FCA_TOP16_FREQUENCIES:
            vertical = cls._basis(height, frequency_y)
            horizontal = cls._basis(width, frequency_x)
            filters.append(vertical[:, None] * horizontal[None, :])
        return torch.stack(filters)

    def forward(self, features):
        expected = (self.channels, self.height, self.width)
        if tuple(features.shape[1:]) != expected:
            raise ValueError(
                f"Expected [batch, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(features.shape)}"
            )
        grouped = features.reshape(
            features.shape[0],
            self.num_groups,
            self.channels // self.num_groups,
            self.height,
            self.width,
        ).mean(dim=2)
        descriptors = (grouped * self.filters.unsqueeze(0)).sum(dim=(-2, -1))
        return torch.nn.functional.layer_norm(descriptors, (self.num_groups,))


class FcaNetChannelAttention(nn.Module):
    """Standard multi-spectral channel attention used as the classical control."""

    def __init__(
        self,
        channels=128,
        feature_size=16,
        num_groups=16,
        reduction=16,
    ):
        super().__init__()
        if channels % num_groups:
            raise ValueError("channels must be divisible by num_groups")
        grouped_filters = GroupedDCTDescriptors._build_filters(
            feature_size, feature_size
        )
        filters = grouped_filters.repeat_interleave(channels // num_groups, dim=0)
        self.channels = channels
        self.feature_size = feature_size
        self.register_buffer("filters", filters)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, features):
        expected = (self.channels, self.feature_size, self.feature_size)
        if tuple(features.shape[1:]) != expected:
            raise ValueError(
                f"Expected [batch, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(features.shape)}"
            )
        descriptors = (features * self.filters.unsqueeze(0)).sum(dim=(-2, -1))
        gates = self.mlp(descriptors)
        return features * gates.view(features.shape[0], self.channels, 1, 1)


class V232ClassicalMatchedFcaAttention(nn.Module):
    """A 64-parameter classical control with the same DCT inputs and gate count."""

    def __init__(
        self,
        channels=128,
        feature_size=16,
        num_groups=16,
        num_circuits=4,
        alpha_max=0.25,
        alpha_init=0.1,
    ):
        super().__init__()
        if num_groups != 16 or num_circuits != 4:
            raise ValueError("The matched V232 control requires 16 groups and four banks")
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must be between zero and alpha_max")
        self.channels = channels
        self.num_groups = num_groups
        self.num_circuits = num_circuits
        self.alpha_max = alpha_max
        self.descriptors = GroupedDCTDescriptors(
            channels, feature_size, feature_size, num_groups
        )
        # Each output keeps its own descriptor and learns mixing from the other three.
        self.cross_weights = nn.Parameter(torch.empty(num_circuits, 4, 3))
        nn.init.normal_(self.cross_weights, mean=0.0, std=0.1)
        initial_probability = alpha_init / alpha_max
        initial_logit = math.log(initial_probability / (1.0 - initial_probability))
        self.alpha_logits = nn.Parameter(torch.full((num_groups,), initial_logit))

    @property
    def alpha(self):
        return self.alpha_max * self.alpha_logits.sigmoid()

    def _gates(self, descriptors):
        grouped = descriptors.reshape(descriptors.shape[0], self.num_circuits, 4)
        other_inputs = []
        for output_index in range(4):
            other_inputs.append(
                torch.cat(
                    (
                        grouped[:, :, :output_index],
                        grouped[:, :, output_index + 1 :],
                    ),
                    dim=2,
                )
            )
        others = torch.stack(other_inputs, dim=2)
        mixed = grouped + (others * self.cross_weights.unsqueeze(0)).sum(dim=3)
        return torch.tanh(mixed).reshape(descriptors.shape[0], self.num_groups)

    def forward(self, features):
        gates = self._gates(self.descriptors(features))
        channels_per_group = self.channels // self.num_groups
        gates = gates.repeat_interleave(channels_per_group, dim=1)
        alpha = self.alpha.repeat_interleave(channels_per_group)
        modulation = alpha.view(1, -1, 1, 1) * gates.view(
            features.shape[0], -1, 1, 1
        )
        return features * (1.0 + modulation)


class V232QuantumFcaAttention(nn.Module):
    """Generate 16 FcaNet-style channel-group gates with four 4-qubit VQCs."""

    def __init__(
        self,
        channels=128,
        feature_size=16,
        num_groups=16,
        num_circuits=4,
        shared_circuit=False,
        entangled=True,
        alpha_max=0.25,
        alpha_init=0.1,
    ):
        super().__init__()
        if num_groups != 16 or num_circuits != 4:
            raise ValueError("V232 requires 16 groups and four 4-qubit circuit executions")
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must be between zero and alpha_max")
        self.channels = channels
        self.num_groups = num_groups
        self.num_circuits = num_circuits
        self.shared_circuit = shared_circuit
        self.entangled = entangled
        self.alpha_max = alpha_max
        self.mode = "normal"
        self.descriptors = GroupedDCTDescriptors(
            channels, feature_size, feature_size, num_groups
        )
        template_count = 1 if shared_circuit else num_circuits
        # Per qubit: pre-entanglement Ry/Rz and post-entanglement Ry.
        self.theta = nn.Parameter(torch.empty(template_count, 4, 3))
        nn.init.uniform_(self.theta, 0.0, 2.0 * math.pi)
        initial_probability = alpha_init / alpha_max
        initial_logit = math.log(initial_probability / (1.0 - initial_probability))
        self.alpha_logits = nn.Parameter(torch.full((num_groups,), initial_logit))

    @property
    def alpha(self):
        return self.alpha_max * self.alpha_logits.sigmoid()

    @staticmethod
    def _ry(state, angle, qubit):
        batch = state.shape[0]
        shaped = state.reshape(batch, 2 ** qubit, 2, 2 ** (3 - qubit))
        zero, one = shaped[:, :, 0, :], shaped[:, :, 1, :]
        cosine = torch.cos(angle / 2).reshape(batch, 1, 1)
        sine = torch.sin(angle / 2).reshape(batch, 1, 1)
        return torch.stack(
            (cosine * zero - sine * one, sine * zero + cosine * one), dim=2
        ).reshape(batch, 16)

    @staticmethod
    def _rz(state, angle, qubit):
        batch = state.shape[0]
        shaped = state.reshape(batch, 2 ** qubit, 2, 2 ** (3 - qubit))
        phase_zero = torch.exp(torch.complex(torch.zeros_like(angle), -angle / 2))
        phase_one = torch.exp(torch.complex(torch.zeros_like(angle), angle / 2))
        phases = torch.stack((phase_zero, phase_one), dim=1).reshape(batch, 1, 2, 1)
        return (shaped * phases).reshape(batch, 16)

    @staticmethod
    def _cnot(state, control, target):
        indices = torch.arange(16, device=state.device)
        control_mask = 1 << (3 - control)
        target_mask = 1 << (3 - target)
        permutation = indices ^ ((indices & control_mask).ne(0).long() * target_mask)
        return state[:, permutation]

    @staticmethod
    def _measure_z(state):
        probabilities = state.real.square() + state.imag.square()
        indices = torch.arange(16, device=state.device)
        expectations = []
        for qubit in range(4):
            signs = 1.0 - 2.0 * ((indices >> (3 - qubit)) & 1).to(probabilities.dtype)
            expectations.append((probabilities * signs.unsqueeze(0)).sum(dim=1))
        return torch.stack(expectations, dim=1)

    def _quantum_gates(self, descriptors):
        batch = descriptors.shape[0]
        angles = (torch.tanh(descriptors) + 1.0) * (0.5 * math.pi)
        angles = angles.reshape(batch, self.num_circuits, 4).reshape(-1, 4)
        parameters = self.theta
        if self.shared_circuit:
            parameters = parameters.expand(self.num_circuits, -1, -1)
        parameters = parameters.unsqueeze(0).expand(batch, -1, -1, -1).reshape(-1, 4, 3)

        state = torch.zeros(angles.shape[0], 16, device=angles.device, dtype=torch.complex64)
        state[:, 0] = 1.0
        for qubit in range(4):
            state = self._ry(state, angles[:, qubit], qubit)
            state = self._ry(state, parameters[:, qubit, 0], qubit)
            state = self._rz(state, parameters[:, qubit, 1], qubit)
        if self.entangled:
            for control, target in ((0, 1), (1, 2), (2, 3), (3, 0)):
                state = self._cnot(state, control, target)
        for qubit in range(4):
            state = self._ry(state, parameters[:, qubit, 2], qubit)
        return self._measure_z(state).reshape(batch, self.num_groups)

    def forward(self, features):
        if self.mode == "zero":
            return features
        gates = self._quantum_gates(self.descriptors(features))
        if self.mode == "shuffle":
            gates = gates.roll(1, dims=0)
        channels_per_group = self.channels // self.num_groups
        gates = gates.repeat_interleave(channels_per_group, dim=1)
        alpha = self.alpha.repeat_interleave(channels_per_group)
        modulation = alpha.view(1, -1, 1, 1) * gates.view(
            features.shape[0], -1, 1, 1
        )
        return features * (1.0 + modulation)


class LocalWaveletDescriptors(nn.Module):
    """Retain LL/LH/HL/HH descriptors on a four-by-four spatial grid."""

    def __init__(self, channels=128, feature_size=16, num_circuits=4):
        super().__init__()
        if channels % num_circuits:
            raise ValueError("channels must be divisible by num_circuits")
        if feature_size % 4:
            raise ValueError("feature_size must be divisible by four")
        self.channels = channels
        self.feature_size = feature_size
        self.num_circuits = num_circuits
        self.region_grid = feature_size // 4

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

    def forward(self, features):
        expected = (self.channels, self.feature_size, self.feature_size)
        if tuple(features.shape[1:]) != expected:
            raise ValueError(
                f"Expected [batch, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(features.shape)}"
            )
        pooled_bands = []
        for band in self._dwt(features):
            grouped = band.reshape(
                features.shape[0],
                self.num_circuits,
                self.channels // self.num_circuits,
                self.feature_size // 2,
                self.feature_size // 2,
            ).mean(dim=2)
            pooled_bands.append(F.avg_pool2d(grouped, kernel_size=2, stride=2))
        descriptors = torch.stack(pooled_bands, dim=-1)
        return F.layer_norm(
            descriptors,
            (self.region_grid, self.region_grid, 4),
        )


class V233ClassicalLocalWaveletAttention(nn.Module):
    """Parameter-matched classical local LL/LH/HL/HH interaction control."""

    def __init__(
        self,
        channels=128,
        feature_size=16,
        num_groups=16,
        num_circuits=4,
        alpha_max=0.25,
        alpha_init=0.1,
    ):
        super().__init__()
        if num_groups != 16 or num_circuits != 4:
            raise ValueError("V233 requires 16 output groups and four interaction banks")
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must be between zero and alpha_max")
        self.channels = channels
        self.feature_size = feature_size
        self.num_groups = num_groups
        self.num_circuits = num_circuits
        self.alpha_max = alpha_max
        self.descriptors = LocalWaveletDescriptors(
            channels, feature_size, num_circuits
        )
        self.cross_weights = nn.Parameter(torch.empty(num_circuits, 4, 3))
        nn.init.normal_(self.cross_weights, mean=0.0, std=0.1)
        initial_probability = alpha_init / alpha_max
        initial_logit = math.log(initial_probability / (1.0 - initial_probability))
        self.alpha_logits = nn.Parameter(torch.full((num_groups,), initial_logit))

    @property
    def alpha(self):
        return self.alpha_max * self.alpha_logits.sigmoid()

    def _local_gates(self, descriptors):
        other_inputs = []
        for output_index in range(4):
            other_inputs.append(
                torch.cat(
                    (
                        descriptors[..., :output_index],
                        descriptors[..., output_index + 1 :],
                    ),
                    dim=-1,
                )
            )
        others = torch.stack(other_inputs, dim=-2)
        weights = self.cross_weights.view(1, self.num_circuits, 1, 1, 4, 3)
        return torch.tanh(descriptors + (others * weights).sum(dim=-1))

    def _apply_gates(self, features, gates):
        gates = gates.permute(0, 1, 4, 2, 3).reshape(
            features.shape[0], self.num_groups, self.descriptors.region_grid, -1
        )
        gates = F.interpolate(
            gates, size=(self.feature_size, self.feature_size), mode="nearest"
        )
        channels_per_group = self.channels // self.num_groups
        gates = gates.repeat_interleave(channels_per_group, dim=1)
        alpha = self.alpha.repeat_interleave(channels_per_group)
        return features * (1.0 + alpha.view(1, -1, 1, 1) * gates)

    def forward(self, features):
        descriptors = self.descriptors(features)
        return self._apply_gates(features, self._local_gates(descriptors))


class V233QuantumLocalWaveletAttention(V233ClassicalLocalWaveletAttention):
    """Use four independent 4-qubit Ring circuits at 16 spatial regions."""

    def __init__(
        self,
        channels=128,
        feature_size=16,
        num_groups=16,
        num_circuits=4,
        entangled=True,
        alpha_max=0.25,
        alpha_init=0.1,
    ):
        nn.Module.__init__(self)
        if num_groups != 16 or num_circuits != 4:
            raise ValueError("V233 requires 16 output groups and four circuits")
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must be between zero and alpha_max")
        self.channels = channels
        self.feature_size = feature_size
        self.num_groups = num_groups
        self.num_circuits = num_circuits
        self.entangled = entangled
        self.alpha_max = alpha_max
        self.mode = "normal"
        self.descriptors = LocalWaveletDescriptors(
            channels, feature_size, num_circuits
        )
        self.theta = nn.Parameter(torch.empty(num_circuits, 4, 3))
        nn.init.uniform_(self.theta, 0.0, 2.0 * math.pi)
        initial_probability = alpha_init / alpha_max
        initial_logit = math.log(initial_probability / (1.0 - initial_probability))
        self.alpha_logits = nn.Parameter(torch.full((num_groups,), initial_logit))

    def _local_gates(self, descriptors):
        batch, circuits, height, width, _ = descriptors.shape
        angles = (torch.tanh(descriptors) + 1.0) * (0.5 * math.pi)
        angles = angles.reshape(-1, 4)
        parameters = self.theta.view(1, circuits, 1, 1, 4, 3)
        parameters = parameters.expand(batch, -1, height, width, -1, -1)
        parameters = parameters.reshape(-1, 4, 3)

        state = torch.zeros(
            angles.shape[0], 16, device=angles.device, dtype=torch.complex64
        )
        state[:, 0] = 1.0
        for qubit in range(4):
            state = V232QuantumFcaAttention._ry(state, angles[:, qubit], qubit)
            state = V232QuantumFcaAttention._ry(
                state, parameters[:, qubit, 0], qubit
            )
            state = V232QuantumFcaAttention._rz(
                state, parameters[:, qubit, 1], qubit
            )
        if self.entangled:
            for control, target in ((0, 1), (1, 2), (2, 3), (3, 0)):
                state = V232QuantumFcaAttention._cnot(state, control, target)
        for qubit in range(4):
            state = V232QuantumFcaAttention._ry(
                state, parameters[:, qubit, 2], qubit
            )
        gates = V232QuantumFcaAttention._measure_z(state)
        return gates.reshape(batch, circuits, height, width, 4)

    def forward(self, features):
        if self.mode == "zero":
            return features
        gates = self._local_gates(self.descriptors(features))
        if self.mode == "shuffle":
            gates = gates.roll(1, dims=0)
        return self._apply_gates(features, gates)


class LearnedProjectedWaveletDescriptors(LocalWaveletDescriptors):
    """Learn four full-channel projections before local wavelet decomposition."""

    def __init__(self, channels=128, feature_size=16, num_circuits=4):
        super().__init__(channels, feature_size, num_circuits)
        self.projection = nn.Conv2d(
            channels, num_circuits, kernel_size=1, bias=False
        )
        channels_per_circuit = channels // num_circuits
        with torch.no_grad():
            self.projection.weight.zero_()
            for circuit in range(num_circuits):
                start = circuit * channels_per_circuit
                stop = start + channels_per_circuit
                self.projection.weight[circuit, start:stop, 0, 0] = (
                    1.0 / channels_per_circuit
                )

    def forward(self, features):
        expected = (self.channels, self.feature_size, self.feature_size)
        if tuple(features.shape[1:]) != expected:
            raise ValueError(
                f"Expected [batch, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(features.shape)}"
            )
        projected = self.projection(features)
        pooled_bands = [
            F.avg_pool2d(band, kernel_size=2, stride=2)
            for band in self._dwt(projected)
        ]
        descriptors = torch.stack(pooled_bands, dim=-1)
        return F.layer_norm(
            descriptors,
            (self.region_grid, self.region_grid, 4),
        )


class V234ClassicalProjectedWaveletAttention(V233ClassicalLocalWaveletAttention):
    """Classical 64-parameter interaction after a learned 128-to-4 projection."""

    def __init__(
        self,
        channels=128,
        feature_size=16,
        num_groups=16,
        num_circuits=4,
        alpha_max=0.25,
        alpha_init=0.1,
    ):
        super().__init__(
            channels,
            feature_size,
            num_groups,
            num_circuits,
            alpha_max,
            alpha_init,
        )
        self.descriptors = LearnedProjectedWaveletDescriptors(
            channels, feature_size, num_circuits
        )


class V234QuantumProjectedWaveletAttention(V233QuantumLocalWaveletAttention):
    """Four local Ring circuits driven by learned full-channel projections."""

    def __init__(
        self,
        channels=128,
        feature_size=16,
        num_groups=16,
        num_circuits=4,
        entangled=True,
        alpha_max=0.25,
        alpha_init=0.1,
    ):
        super().__init__(
            channels,
            feature_size,
            num_groups,
            num_circuits,
            entangled,
            alpha_max,
            alpha_init,
        )
        self.descriptors = LearnedProjectedWaveletDescriptors(
            channels, feature_size, num_circuits
        )


class V235ClassicalHQNetProjectedWaveletAttention(
    V234ClassicalProjectedWaveletAttention
):
    """Parameter-matched directed cross-band control for the HQNet RQC."""

    CONNECTIONS = (
        (0, 1),
        (0, 3),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 0),
    )

    def __init__(
        self,
        channels=128,
        feature_size=16,
        num_groups=16,
        num_circuits=4,
        alpha_max=0.25,
        alpha_init=0.1,
    ):
        super().__init__(
            channels,
            feature_size,
            num_groups,
            num_circuits,
            alpha_max,
            alpha_init,
        )
        self.cross_weights = nn.Parameter(
            torch.empty(num_circuits, len(self.CONNECTIONS))
        )
        nn.init.normal_(self.cross_weights, mean=0.0, std=0.1)

    def _local_gates(self, descriptors):
        weights = self.cross_weights.view(
            1, self.num_circuits, 1, 1, len(self.CONNECTIONS)
        )
        outputs = []
        for target in range(4):
            incoming = [
                descriptors[..., control] * weights[..., edge]
                for edge, (control, edge_target) in enumerate(self.CONNECTIONS)
                if edge_target == target
            ]
            outputs.append(
                torch.tanh(descriptors[..., target] + torch.stack(incoming).sum(0))
            )
        return torch.stack(outputs, dim=-1)


class V235HQNetProjectedWaveletAttention(V234QuantumProjectedWaveletAttention):
    """HQNet Fig. 7 Rx/Rz encoding with trainable AC+ CRX connections."""

    CONNECTIONS = V235ClassicalHQNetProjectedWaveletAttention.CONNECTIONS

    def __init__(
        self,
        channels=128,
        feature_size=16,
        num_groups=16,
        num_circuits=4,
        entangled=True,
        alpha_max=0.25,
        alpha_init=0.1,
    ):
        super().__init__(
            channels,
            feature_size,
            num_groups,
            num_circuits,
            entangled,
            alpha_max,
            alpha_init,
        )
        self.theta = nn.Parameter(
            torch.empty(num_circuits, len(self.CONNECTIONS))
        )
        nn.init.normal_(self.theta, mean=0.0, std=0.1)

    @staticmethod
    def _rx(state, angle, qubit):
        batch = state.shape[0]
        shaped = state.reshape(batch, 2 ** qubit, 2, 2 ** (3 - qubit))
        zero, one = shaped[:, :, 0, :], shaped[:, :, 1, :]
        cosine = torch.cos(angle / 2).reshape(batch, 1, 1)
        sine = torch.sin(angle / 2).reshape(batch, 1, 1)
        imaginary_sine = torch.complex(torch.zeros_like(sine), sine)
        return torch.stack(
            (
                cosine * zero - imaginary_sine * one,
                -imaginary_sine * zero + cosine * one,
            ),
            dim=2,
        ).reshape(batch, 16)

    @staticmethod
    def _crx(state, angle, control, target):
        indices = torch.arange(16, device=state.device)
        control_mask = 1 << (3 - control)
        target_mask = 1 << (3 - target)
        zero_indices = indices[
            indices.bitwise_and(control_mask).ne(0)
            & indices.bitwise_and(target_mask).eq(0)
        ]
        one_indices = zero_indices.bitwise_xor(target_mask)
        zero, one = state[:, zero_indices], state[:, one_indices]
        cosine = torch.cos(angle / 2).unsqueeze(1)
        sine = torch.sin(angle / 2).unsqueeze(1)
        imaginary_sine = torch.complex(torch.zeros_like(sine), sine)
        output = state.clone()
        output[:, zero_indices] = cosine * zero - imaginary_sine * one
        output[:, one_indices] = -imaginary_sine * zero + cosine * one
        return output

    def _local_gates(self, descriptors):
        batch, circuits, height, width, _ = descriptors.shape
        angles = (torch.tanh(descriptors) + 1.0) * (0.5 * math.pi)
        angles = angles.reshape(-1, 4)
        parameters = self.theta.view(1, circuits, 1, 1, -1)
        parameters = parameters.expand(batch, -1, height, width, -1)
        parameters = parameters.reshape(-1, len(self.CONNECTIONS))

        state = torch.zeros(
            angles.shape[0], 16, device=angles.device, dtype=torch.complex64
        )
        state[:, 0] = 1.0
        for qubit in range(4):
            state = self._rx(state, angles[:, qubit], qubit)
            state = V232QuantumFcaAttention._rz(
                state, angles[:, qubit], qubit
            )
        for edge, (control, target) in enumerate(self.CONNECTIONS):
            if self.entangled:
                state = self._crx(
                    state, parameters[:, edge], control, target
                )
            else:
                state = self._rx(state, parameters[:, edge], target)
        gates = V232QuantumFcaAttention._measure_z(state)
        return gates.reshape(batch, circuits, height, width, 4)
