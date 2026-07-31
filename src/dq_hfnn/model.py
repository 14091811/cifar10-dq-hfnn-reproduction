import torch
import torch.nn as nn

from .circuit import DualQubitMembershipCircuit
from .features import (
    CNNSpatialFrequencyPairs,
    DirectionAwareSpatialPatchPairs,
    FuzzyPooledPatchPairs,
    FrequencyGuidedPixelPairs,
    HaarHighFrequencyPairs,
    SpatialFuzzyPatchPairs,
)
from .frequency_modulation import V223QuantumFrequencyModulator
from .fca_attention import (
    FcaNetChannelAttention,
    V232ClassicalMatchedFcaAttention,
    V232QuantumFcaAttention,
    V233ClassicalLocalWaveletAttention,
    V233QuantumLocalWaveletAttention,
    V234ClassicalProjectedWaveletAttention,
    V234QuantumProjectedWaveletAttention,
    V235ClassicalHQNetProjectedWaveletAttention,
    V235HQNetProjectedWaveletAttention,
)
from .local_frequency import (
    V215LocalFrequencyHead,
    V216GroupedLocalFrequencyHead,
    V217RelativeEnergyLocalFrequencyHead,
    V219SimpleFrequencyHead,
)
from .pairing import GridPairSampler
from .qgwp import QGWPHaarDetailPool
from .born_pool import QuantumBornHaarPool
from .softpool_refiner import QuantumRefinedSoftPool
from .v236_logit_residual import V236TwoLevelHaarQuantumLogitHead


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels))
        self.skip = nn.Identity() if in_channels == out_channels else nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.body(x) + self.skip(x))


class ClassicalBranch(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.features = nn.Sequential(Block(3, 64), nn.MaxPool2d(2), Block(64, 128), Block(128, 128), nn.MaxPool2d(2), Block(128, 256), Block(256, 256), nn.MaxPool2d(2), Block(256, 512), Block(512, 512), nn.MaxPool2d(2))
        self.project = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, hidden_dim))
        self.input_frequency_modulator = None
        self.frequency_modulator = None
        self.frequency_tap_index = 3

    def _forward_features(self, x, local_tap_index=None):
        local_features = None
        if self.input_frequency_modulator is not None:
            x = self.input_frequency_modulator(x)
        for index, layer in enumerate(self.features):
            x = layer(x)
            if self.frequency_modulator is not None and index == self.frequency_tap_index:
                x = self.frequency_modulator(x)
            if index == local_tap_index:
                local_features = x
        return x, local_features

    def forward(self, x):
        x, _ = self._forward_features(x)
        return self.project(x.flatten(1))

    def forward_with_local_features(self, x, tap_index=6):
        x, local_features = self._forward_features(x, tap_index)
        if local_features is None:
            raise ValueError(f"tap_index {tap_index} did not produce local features")
        return self.project(x.flatten(1)), local_features


class DQHFNN(nn.Module):
    """CIFAR-10 DQ-HFNN: CNN + class-specific DQ fuzzy memberships."""

    def __init__(self, num_classes=10, hidden_dim=256, total_pairs=153, random_pair_ratio=0.3, circuit_variant="strong_ent", sampler_seed=42, pair_source="pixels", pairing_layout="compact", evaluation_pairing="fixed", vectorize_class_circuits=False, quantum_enabled=True, local_tap_index=6, local_grid_size=4, frequency_tap_index=3, frequency_num_groups=4, frequency_alpha_max=0.5, frequency_alpha_init=0.1, measurement_basis="z", quantum_auxiliary_weight=0.0, fusion_alpha_max=0.5, fusion_alpha_init=0.1, channel_attention="none", fca_num_groups=16, fca_num_circuits=4, author_dq_branch_enabled=True, first_pool="maxpool", frequency_beta=None):
        super().__init__()
        self.feature_quantum_adapter = pair_source == "v231_cnn_spatial_frequency_quantum"
        self.class_logit_residual = pair_source in {
            "v236_twolevel_haar_quantum",
            "v236_twolevel_haar_local_rx",
        }
        self.standalone_quantum = pair_source in {
            "v226_fuzzy_pooled_quantum",
            "v227_spatial_fuzzy_quantum",
            "v228_superpixel9_uniform_quantum",
            "v230_directional_spatial_quantum",
        }
        self.spatial_fuzzy_aggregation = pair_source in {
            "v227_spatial_fuzzy_quantum",
            "v230_directional_spatial_quantum",
            "v231_cnn_spatial_frequency_quantum",
        }
        self.classical = nn.Identity() if self.standalone_quantum else ClassicalBranch(hidden_dim)
        self.vectorize_class_circuits = vectorize_class_circuits
        self.quantum_enabled = quantum_enabled
        self.quantum_auxiliary_weight = quantum_auxiliary_weight
        self.author_dq_branch_enabled = author_dq_branch_enabled
        self.local_tap_index = local_tap_index
        self.local_frequency_head = None
        self.channel_attention = channel_attention
        self.first_pool = first_pool
        if first_pool != "maxpool":
            qgwp_variants = {
                "qgwp_quantum_entangled": ("quantum", True, None),
                "qgwp_quantum_local_rx": ("quantum", False, None),
                "qgwp_classical_matched": ("classical", False, None),
                "qgwp_fixed_quantum_local_rx": ("quantum", False, frequency_beta),
                "qgwp_fixed_classical_matched": ("classical", False, frequency_beta),
                "qgwp_fixed_uniform": ("uniform", False, frequency_beta),
            }
            born_pool_variants = {
                "qbwp_quantum_local_rx": ("quantum", False),
                "qbwp_quantum_entangled": ("quantum", True),
                "qbwp_classical_matched": ("classical", False),
                "qbwp_softpool": ("softpool", False),
            }
            softpool_refiner_variants = {
                "qsrp_quantum_local_rx": ("quantum", False),
                "qsrp_quantum_entangled": ("quantum", True),
                "qsrp_classical_matched": ("classical", False),
            }
            if (
                first_pool not in qgwp_variants
                and first_pool not in born_pool_variants
                and first_pool not in softpool_refiner_variants
            ):
                raise ValueError(f"Unsupported first pool: {first_pool}")
            if self.standalone_quantum:
                raise ValueError("Custom first pooling requires the CNN branch")
            if first_pool in qgwp_variants:
                gate_type, entangled, fixed_beta = qgwp_variants[first_pool]
                if first_pool.startswith("qgwp_fixed") and fixed_beta is None:
                    raise ValueError("Fixed QGWP requires frequency_beta")
                self.classical.features[1] = QGWPHaarDetailPool(
                    channels=64,
                    num_groups=frequency_num_groups,
                    gate_type=gate_type,
                    entangled=entangled,
                    alpha_max=frequency_alpha_max,
                    alpha_init=frequency_alpha_init,
                    fixed_beta=fixed_beta,
                )
            elif first_pool in born_pool_variants:
                gate_type, entangled = born_pool_variants[first_pool]
                self.classical.features[1] = QuantumBornHaarPool(
                    channels=64,
                    num_groups=frequency_num_groups,
                    gate_type=gate_type,
                    entangled=entangled,
                )
            else:
                gate_type, entangled = softpool_refiner_variants[first_pool]
                self.classical.features[1] = QuantumRefinedSoftPool(
                    channels=64,
                    num_groups=frequency_num_groups,
                    gate_type=gate_type,
                    entangled=entangled,
                )
        if pair_source == "none":
            self.pair_source = None
        elif pair_source == "pixels":
            self.pair_source = GridPairSampler(total_pairs, random_pair_ratio, sampler_seed, layout=pairing_layout, evaluation_mode=evaluation_pairing)
        elif pair_source == "haar_high":
            self.pair_source = HaarHighFrequencyPairs(total_pairs, random_pair_ratio, sampler_seed, layout=pairing_layout, evaluation_mode=evaluation_pairing)
        elif pair_source == "frequency_guided_pixels":
            self.pair_source = FrequencyGuidedPixelPairs(total_pairs)
        elif pair_source == "v226_fuzzy_pooled_quantum":
            self.pair_source = FuzzyPooledPatchPairs()
        elif pair_source == "v227_spatial_fuzzy_quantum":
            self.pair_source = SpatialFuzzyPatchPairs()
        elif pair_source == "v228_superpixel9_uniform_quantum":
            self.pair_source = SpatialFuzzyPatchPairs()
        elif pair_source == "v230_directional_spatial_quantum":
            self.pair_source = DirectionAwareSpatialPatchPairs()
        elif pair_source == "v231_cnn_spatial_frequency_quantum":
            self.pair_source = CNNSpatialFrequencyPairs()
        elif self.class_logit_residual:
            self.pair_source = None
        elif pair_source == "v215_local":
            self.pair_source = None
            self.local_frequency_head = V215LocalFrequencyHead(num_classes, grid_size=local_grid_size)
        elif pair_source == "v216_grouped_local":
            self.pair_source = None
            self.local_frequency_head = V216GroupedLocalFrequencyHead(num_classes, grid_size=local_grid_size)
        elif pair_source == "v217_relative_energy":
            self.pair_source = None
            self.local_frequency_head = V217RelativeEnergyLocalFrequencyHead(num_classes, grid_size=local_grid_size)
        elif pair_source in {"v219_simple_entangled", "v219_simple_noent"}:
            self.pair_source = None
            self.local_frequency_head = V219SimpleFrequencyHead(
                num_classes,
                grid_size=local_grid_size,
                entangled=pair_source == "v219_simple_entangled",
            )
        elif pair_source in {"v220_zz_entangled", "v220_zz_noent"}:
            self.pair_source = None
            self.local_frequency_head = V219SimpleFrequencyHead(
                num_classes,
                grid_size=local_grid_size,
                entangled=pair_source == "v220_zz_entangled",
                measure_zz=True,
            )
        elif pair_source in {"v223_trunk_entangled", "v223_trunk_noent"}:
            self.pair_source = None
            self.classical.frequency_modulator = V223QuantumFrequencyModulator(
                channels=128,
                num_groups=frequency_num_groups,
                entangled=pair_source == "v223_trunk_entangled",
                alpha_max=frequency_alpha_max,
                alpha_init=frequency_alpha_init,
            )
            self.classical.frequency_tap_index = frequency_tap_index
        elif pair_source in {"v224_rawfreq_entangled", "v224_rawfreq_noent"}:
            self.pair_source = None
            self.classical.input_frequency_modulator = V223QuantumFrequencyModulator(
                channels=3,
                num_groups=frequency_num_groups,
                entangled=pair_source == "v224_rawfreq_entangled",
                alpha_max=frequency_alpha_max,
                alpha_init=frequency_alpha_init,
            )
        else:
            raise ValueError(f"Unsupported pair source: {pair_source}")
        if channel_attention in {
            "v232_quantum_fca_independent",
            "v232_quantum_fca_shared",
            "v232_quantum_fca_noent",
        }:
            self.classical.frequency_modulator = V232QuantumFcaAttention(
                channels=128,
                feature_size=16,
                num_groups=fca_num_groups,
                num_circuits=fca_num_circuits,
                shared_circuit=channel_attention == "v232_quantum_fca_shared",
                entangled=channel_attention != "v232_quantum_fca_noent",
                alpha_max=frequency_alpha_max,
                alpha_init=frequency_alpha_init,
            )
            self.classical.frequency_tap_index = frequency_tap_index
        elif channel_attention == "fcanet":
            self.classical.frequency_modulator = FcaNetChannelAttention(
                channels=128,
                feature_size=16,
                num_groups=fca_num_groups,
            )
            self.classical.frequency_tap_index = frequency_tap_index
        elif channel_attention == "v232_classical_fca_matched":
            self.classical.frequency_modulator = V232ClassicalMatchedFcaAttention(
                channels=128,
                feature_size=16,
                num_groups=fca_num_groups,
                num_circuits=fca_num_circuits,
                alpha_max=frequency_alpha_max,
                alpha_init=frequency_alpha_init,
            )
            self.classical.frequency_tap_index = frequency_tap_index
        elif channel_attention in {
            "v233_local_wavelet_quantum",
            "v233_local_wavelet_quantum_noent",
        }:
            self.classical.frequency_modulator = V233QuantumLocalWaveletAttention(
                channels=128,
                feature_size=16,
                num_groups=fca_num_groups,
                num_circuits=fca_num_circuits,
                entangled=channel_attention == "v233_local_wavelet_quantum",
                alpha_max=frequency_alpha_max,
                alpha_init=frequency_alpha_init,
            )
            self.classical.frequency_tap_index = frequency_tap_index
        elif channel_attention == "v233_local_wavelet_classical_matched":
            self.classical.frequency_modulator = V233ClassicalLocalWaveletAttention(
                channels=128,
                feature_size=16,
                num_groups=fca_num_groups,
                num_circuits=fca_num_circuits,
                alpha_max=frequency_alpha_max,
                alpha_init=frequency_alpha_init,
            )
            self.classical.frequency_tap_index = frequency_tap_index
        elif channel_attention in {
            "v234_projected_wavelet_quantum",
            "v234_projected_wavelet_quantum_noent",
        }:
            self.classical.frequency_modulator = V234QuantumProjectedWaveletAttention(
                channels=128,
                feature_size=16,
                num_groups=fca_num_groups,
                num_circuits=fca_num_circuits,
                entangled=channel_attention == "v234_projected_wavelet_quantum",
                alpha_max=frequency_alpha_max,
                alpha_init=frequency_alpha_init,
            )
            self.classical.frequency_tap_index = frequency_tap_index
        elif channel_attention == "v234_projected_wavelet_classical_matched":
            self.classical.frequency_modulator = V234ClassicalProjectedWaveletAttention(
                channels=128,
                feature_size=16,
                num_groups=fca_num_groups,
                num_circuits=fca_num_circuits,
                alpha_max=frequency_alpha_max,
                alpha_init=frequency_alpha_init,
            )
            self.classical.frequency_tap_index = frequency_tap_index
        elif channel_attention in {
            "v235_hqnet_projected_wavelet_quantum",
            "v235_hqnet_projected_wavelet_quantum_local_rx",
        }:
            self.classical.frequency_modulator = V235HQNetProjectedWaveletAttention(
                channels=128,
                feature_size=16,
                num_groups=fca_num_groups,
                num_circuits=fca_num_circuits,
                entangled=channel_attention == "v235_hqnet_projected_wavelet_quantum",
                alpha_max=frequency_alpha_max,
                alpha_init=frequency_alpha_init,
            )
            self.classical.frequency_tap_index = frequency_tap_index
        elif channel_attention == "v235_hqnet_projected_wavelet_classical_matched":
            self.classical.frequency_modulator = V235ClassicalHQNetProjectedWaveletAttention(
                channels=128,
                feature_size=16,
                num_groups=fca_num_groups,
                num_circuits=fca_num_circuits,
                alpha_max=frequency_alpha_max,
                alpha_init=frequency_alpha_init,
            )
            self.classical.frequency_tap_index = frequency_tap_index
        elif channel_attention != "none":
            raise ValueError(f"Unsupported channel attention: {channel_attention}")
        self.quantum_logit_head = (
            V236TwoLevelHaarQuantumLogitHead(
                channels=128,
                feature_size=16,
                num_classes=num_classes,
                num_circuits=fca_num_circuits,
                entangled=pair_source == "v236_twolevel_haar_quantum",
                alpha_max=fusion_alpha_max,
                alpha_init=fusion_alpha_init,
            )
            if self.class_logit_residual
            else None
        )
        if self.standalone_quantum or self.feature_quantum_adapter:
            self.circuits = nn.ModuleList(DualQubitMembershipCircuit(circuit_variant, measurement_basis) for _ in range(num_classes))
            self.fuzzy_projection = nn.Identity()
        elif self.class_logit_residual or not author_dq_branch_enabled:
            self.circuits = nn.ModuleList()
            self.fuzzy_projection = nn.Identity()
        elif self.trunk_frequency_modulator is None or channel_attention != "none":
            self.circuits = nn.ModuleList(DualQubitMembershipCircuit(circuit_variant, measurement_basis) for _ in range(num_classes))
            self.fuzzy_projection = nn.Linear(num_classes, hidden_dim)
        else:
            self.circuits = nn.ModuleList()
            self.fuzzy_projection = nn.Identity()
        self.class_position_logits = (
            nn.Parameter(torch.zeros(num_classes, self.pair_source.output_pairs))
            if self.spatial_fuzzy_aggregation
            else None
        )
        self.classifier = nn.Identity() if self.standalone_quantum else nn.Linear(hidden_dim, num_classes)
        if self.feature_quantum_adapter:
            if not 0.0 < fusion_alpha_init < fusion_alpha_max:
                raise ValueError("fusion_alpha_init must be between zero and fusion_alpha_max")
            self.fusion_alpha_max = fusion_alpha_max
            ratio = torch.tensor(fusion_alpha_init / fusion_alpha_max)
            self.fusion_alpha_logit = nn.Parameter(torch.logit(ratio))
        elif self.class_logit_residual:
            self.fusion_alpha_max = fusion_alpha_max
            self.register_parameter("fusion_alpha_logit", None)
        else:
            self.fusion_alpha_max = None
            self.register_parameter("fusion_alpha_logit", None)

    @property
    def has_quantum_auxiliary(self):
        return (
            self.feature_quantum_adapter or self.class_logit_residual
        ) and self.quantum_auxiliary_weight > 0.0

    @property
    def fusion_alpha(self):
        if self.quantum_logit_head is not None:
            return self.quantum_logit_head.alpha
        if self.fusion_alpha_logit is None:
            return None
        return self.fusion_alpha_max * self.fusion_alpha_logit.sigmoid()

    @property
    def trunk_frequency_modulator(self):
        frequency_modulator = getattr(self.classical, "frequency_modulator", None)
        if frequency_modulator is not None:
            return frequency_modulator
        input_frequency_modulator = getattr(
            self.classical, "input_frequency_modulator", None
        )
        if input_frequency_modulator is not None:
            return input_frequency_modulator
        if not self.standalone_quantum and isinstance(
            self.classical.features[1],
            (QGWPHaarDetailPool, QuantumBornHaarPool, QuantumRefinedSoftPool),
        ):
            return self.classical.features[1]
        return None

    def memberships(self, source):
        if self.local_frequency_head is not None:
            raise RuntimeError("V215 local scores require CNN local features")
        pairs = self.pair_source(source)
        batch, pair_count, _ = pairs.shape
        if self.vectorize_class_circuits:
            variants = {circuit.variant for circuit in self.circuits}
            measurement_bases = {circuit.measurement_basis for circuit in self.circuits}
            if len(variants) != 1:
                raise RuntimeError("Class-vectorized execution requires one shared circuit variant")
            if len(measurement_bases) != 1:
                raise RuntimeError("Class-vectorized execution requires one shared measurement basis")
            num_classes = len(self.circuits)
            inputs = pairs.unsqueeze(2).expand(-1, -1, num_classes, -1).reshape(-1, 2)
            theta = torch.stack([circuit.theta for circuit in self.circuits])
            theta = theta.view(1, 1, num_classes, 2).expand(batch, pair_count, -1, -1).reshape(-1, 2)
            state = torch.zeros(inputs.shape[0], 4, device=inputs.device, dtype=torch.complex64)
            state[:, 0] = 1
            state = DualQubitMembershipCircuit._ry_q0(state, inputs[:, 0])
            state = DualQubitMembershipCircuit._ry_q1(state, inputs[:, 1])
            state = DualQubitMembershipCircuit._rz_q0(state, theta[:, 0])
            variant = next(iter(variants))
            if variant != "no_ent":
                state = DualQubitMembershipCircuit._cnot(state)
            state = DualQubitMembershipCircuit._ry_q0(state, theta[:, 1])
            if variant == "strong_ent":
                state = DualQubitMembershipCircuit._cnot(state)
            memberships = DualQubitMembershipCircuit._measure_memberships(
                state, next(iter(measurement_bases))
            )
            return memberships.reshape(batch, pair_count, num_classes, 2).permute(0, 1, 3, 2).reshape(batch, pair_count * 2, num_classes)
        flat = pairs.reshape(-1, 2)
        return torch.stack([circuit(flat).reshape(batch, pair_count * 2) for circuit in self.circuits], dim=-1)

    def quantum_features(self, images, local_features=None):
        if self.standalone_quantum:
            raise RuntimeError("V226 returns quantum class scores directly")
        if self.trunk_frequency_modulator is not None and self.channel_attention == "none":
            raise RuntimeError("Quantum features are integrated inside the CNN trunk")
        if self.local_frequency_head is not None:
            return self.fuzzy_projection(self.local_frequency_head(local_features))
        return self.fuzzy_projection(self.memberships(images).clamp_min(1e-7).log().mean(dim=1))

    def quantum_class_scores(self, source):
        if not (self.standalone_quantum or self.feature_quantum_adapter):
            raise RuntimeError("Direct quantum class scores require a class-scoring quantum model")
        memberships = self.memberships(source).clamp_min(1e-7)
        if self.spatial_fuzzy_aggregation:
            pair_count = memberships.shape[1] // 2
            position_scores = memberships.reshape(
                memberships.shape[0], pair_count, 2, memberships.shape[2]
            ).log().mean(dim=2)
            position_weights = self.class_position_logits.softmax(dim=1).transpose(0, 1)
            scores = (position_scores * position_weights.unsqueeze(0)).sum(dim=1)
        else:
            scores = memberships.log().mean(dim=1)
        return torch.nn.functional.layer_norm(scores, (scores.shape[1],))

    def hybrid_components(self, images):
        if not (self.feature_quantum_adapter or self.class_logit_residual):
            raise RuntimeError("Hybrid components require a quantum logit adapter")
        classical_features, local_features = self.classical.forward_with_local_features(
            images, self.local_tap_index
        )
        classical_logits = self.classifier(classical_features)
        if self.quantum_enabled:
            quantum_logits = (
                self.quantum_logit_head(local_features)
                if self.class_logit_residual
                else self.quantum_class_scores(local_features)
            )
            fused_logits = classical_logits + self.fusion_alpha * quantum_logits
        else:
            quantum_logits = torch.zeros_like(classical_logits)
            fused_logits = classical_logits
        return fused_logits, classical_logits, quantum_logits

    def forward_with_auxiliary(self, images):
        if not (self.feature_quantum_adapter or self.class_logit_residual):
            raise RuntimeError("Auxiliary output requires a quantum logit adapter")
        fused_logits, _, quantum_logits = self.hybrid_components(images)
        return fused_logits, quantum_logits

    def forward(self, images):
        if self.standalone_quantum:
            if not self.quantum_enabled:
                return images.new_zeros(images.shape[0], len(self.circuits))
            return self.quantum_class_scores(images)
        if self.feature_quantum_adapter or self.class_logit_residual:
            return self.hybrid_components(images)[0]
        if self.trunk_frequency_modulator is not None:
            if hasattr(self.trunk_frequency_modulator, "mode"):
                previous_mode = self.trunk_frequency_modulator.mode
                try:
                    if not self.quantum_enabled:
                        self.trunk_frequency_modulator.mode = "zero"
                    classical_features = self.classical(images)
                finally:
                    self.trunk_frequency_modulator.mode = previous_mode
            else:
                classical_features = self.classical(images)
            if (
                self.channel_attention != "none"
                and self.quantum_enabled
                and self.author_dq_branch_enabled
            ):
                return self.classifier(classical_features + self.quantum_features(images))
            return self.classifier(classical_features)
        if self.local_frequency_head is not None:
            classical_features, local_features = self.classical.forward_with_local_features(
                images, self.local_tap_index
            )
            if not self.quantum_enabled:
                return self.classifier(classical_features)
            return self.classifier(classical_features + self.quantum_features(images, local_features))
        classical_features = self.classical(images)
        if self.author_dq_branch_enabled:
            classical_features = classical_features + self.quantum_features(images)
        return self.classifier(classical_features)
