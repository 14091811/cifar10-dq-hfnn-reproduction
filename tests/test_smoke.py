import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dq_hfnn.circuit import DualQubitMembershipCircuit
from dq_hfnn.model import DQHFNN
from dq_hfnn.frequency_modulation import V223QuantumFrequencyModulator
from dq_hfnn.fca_attention import (
    FcaNetChannelAttention,
    GroupedDCTDescriptors,
    V232ClassicalMatchedFcaAttention,
    V232QuantumFcaAttention,
    LocalWaveletDescriptors,
    V233ClassicalLocalWaveletAttention,
    V233QuantumLocalWaveletAttention,
    LearnedProjectedWaveletDescriptors,
    V234ClassicalProjectedWaveletAttention,
    V234QuantumProjectedWaveletAttention,
    V235ClassicalHQNetProjectedWaveletAttention,
    V235HQNetProjectedWaveletAttention,
)
from dq_hfnn.features import (
    CNNSpatialFrequencyPairs,
    DirectionAwareSpatialPatchPairs,
    FuzzyPooledPatchPairs,
    FrequencyGuidedPixelPairs,
    SpatialFuzzyPatchPairs,
)
from dq_hfnn.v236_logit_residual import V236TwoLevelHaarQuantumLogitHead
from dq_hfnn.qgwp import QGWPHaarDetailPool
from dq_hfnn.born_pool import QuantumBornHaarPool
from dq_hfnn.softpool_refiner import QuantumRefinedSoftPool

model = DQHFNN(total_pairs=9, random_pair_ratio=0.33)
output = model(torch.randn(2, 3, 32, 32))
assert output.shape == (2, 10)
output.sum().backward()
assert model.circuits[0].theta.grad is not None

grouped_model = DQHFNN(pair_source="v216_grouped_local", total_pairs=0)
grouped_output = grouped_model(torch.randn(2, 3, 32, 32))
assert grouped_output.shape == (2, 10)
assert torch.isfinite(grouped_output).all()
grouped_output.square().mean().backward()
assert grouped_model.local_frequency_head.theta.grad is not None
assert grouped_model.local_frequency_head.class_group_logits.grad is not None
assert torch.isfinite(grouped_model.local_frequency_head.theta.grad).all()
assert torch.isfinite(grouped_model.local_frequency_head.class_group_logits.grad).all()

relative_model = DQHFNN(pair_source="v217_relative_energy", total_pairs=0)
relative_input = torch.randn(2, 3, 32, 32)
relative_output = relative_model(relative_input)
assert relative_output.shape == (2, 10)
assert torch.isfinite(relative_output).all()
relative_output.square().mean().backward()
assert relative_model.local_frequency_head.theta.grad is not None
assert relative_model.local_frequency_head.class_group_logits.grad is not None
assert torch.isfinite(relative_model.local_frequency_head.theta.grad).all()
assert torch.isfinite(relative_model.local_frequency_head.class_group_logits.grad).all()

head = relative_model.local_frequency_head
bands = tuple(torch.rand(2, 128, 4, 4) + 0.1 for _ in range(4))
_, relative_energy = head._descriptors(bands)
stacked_energy = torch.stack(relative_energy, dim=2)
assert torch.allclose(stacked_energy.sum(dim=2), torch.zeros_like(stacked_energy[:, :, 0]), atol=1e-6)
_, scaled_relative_energy = head._descriptors(tuple(band * 3.0 for band in bands))
assert torch.allclose(
    stacked_energy,
    torch.stack(scaled_relative_energy, dim=2),
    atol=1e-5,
    rtol=1e-5,
)

fca_descriptors = GroupedDCTDescriptors(channels=128, height=16, width=16)
descriptor_output = fca_descriptors(torch.randn(2, 128, 16, 16))
assert descriptor_output.shape == (2, 16)
assert torch.isfinite(descriptor_output).all()

fcanet_attention = FcaNetChannelAttention()
fcanet_output = fcanet_attention(torch.randn(2, 128, 16, 16))
assert fcanet_output.shape == (2, 128, 16, 16)
assert torch.isfinite(fcanet_output).all()

matched_attention = V232ClassicalMatchedFcaAttention()
assert sum(parameter.numel() for parameter in matched_attention.parameters()) == 64
matched_output = matched_attention(torch.randn(2, 128, 16, 16))
assert matched_output.shape == (2, 128, 16, 16)
matched_output.mean().backward()
assert matched_attention.cross_weights.grad is not None
assert matched_attention.alpha_logits.grad is not None

v232_attention = V232QuantumFcaAttention()
v232_input = torch.randn(2, 128, 16, 16)
v232_output = v232_attention(v232_input)
assert v232_output.shape == v232_input.shape
assert torch.isfinite(v232_output).all()
v232_output.square().mean().backward()
assert v232_attention.theta.grad is not None
assert v232_attention.alpha_logits.grad is not None
assert torch.isfinite(v232_attention.theta.grad).all()
assert torch.isfinite(v232_attention.alpha_logits.grad).all()
v232_attention.mode = "zero"
assert torch.equal(v232_attention(v232_input), v232_input)

entangled_attention = V232QuantumFcaAttention(entangled=True)
noent_attention = V232QuantumFcaAttention(entangled=False)
noent_attention.load_state_dict(entangled_attention.state_dict())
test_descriptors = torch.linspace(-1.0, 1.0, 32).reshape(2, 16)
entangled_gates = entangled_attention._quantum_gates(test_descriptors)
noent_gates = noent_attention._quantum_gates(test_descriptors)
assert entangled_gates.shape == (2, 16)
assert torch.isfinite(entangled_gates).all()
assert entangled_gates.abs().max() <= 1.0 + 1e-6
assert not torch.allclose(entangled_gates, noent_gates)
assert entangled_attention.theta.shape == (4, 4, 3)

v232_model = DQHFNN(
    total_pairs=460,
    random_pair_ratio=1.0,
    circuit_variant="no_ent",
    pair_source="pixels",
    pairing_layout="author_cifar",
    evaluation_pairing="stochastic",
    vectorize_class_circuits=True,
    frequency_tap_index=3,
    frequency_alpha_max=0.25,
    frequency_alpha_init=0.1,
    channel_attention="v232_quantum_fca_independent",
)
v232_model_output = v232_model(torch.randn(2, 3, 32, 32))
assert v232_model_output.shape == (2, 10)
assert torch.isfinite(v232_model_output).all()
v232_model_output.mean().backward()
assert v232_model.trunk_frequency_modulator.theta.grad is not None
assert v232_model.circuits[0].theta.grad is not None

wavelet_descriptors = LocalWaveletDescriptors()
wavelet_descriptor_output = wavelet_descriptors(torch.randn(2, 128, 16, 16))
assert wavelet_descriptor_output.shape == (2, 4, 4, 4, 4)
assert torch.isfinite(wavelet_descriptor_output).all()

v233_classical = V233ClassicalLocalWaveletAttention()
assert sum(parameter.numel() for parameter in v233_classical.parameters()) == 64
v233_classical_output = v233_classical(torch.randn(2, 128, 16, 16))
assert v233_classical_output.shape == (2, 128, 16, 16)
v233_classical_output.mean().backward()
assert v233_classical.cross_weights.grad is not None

v233_attention = V233QuantumLocalWaveletAttention()
assert sum(parameter.numel() for parameter in v233_attention.parameters()) == 64
v233_input = torch.randn(2, 128, 16, 16)
v233_output = v233_attention(v233_input)
assert v233_output.shape == v233_input.shape
assert torch.isfinite(v233_output).all()
v233_output.square().mean().backward()
assert v233_attention.theta.grad is not None
assert v233_attention.alpha_logits.grad is not None
assert torch.isfinite(v233_attention.theta.grad).all()
v233_attention.mode = "zero"
assert torch.equal(v233_attention(v233_input), v233_input)

v233_entangled = V233QuantumLocalWaveletAttention(entangled=True)
v233_noent = V233QuantumLocalWaveletAttention(entangled=False)
v233_noent.load_state_dict(v233_entangled.state_dict())
local_descriptors = wavelet_descriptors(torch.randn(2, 128, 16, 16))
assert not torch.allclose(
    v233_entangled._local_gates(local_descriptors),
    v233_noent._local_gates(local_descriptors),
)

v233_model = DQHFNN(
    total_pairs=460,
    random_pair_ratio=1.0,
    circuit_variant="no_ent",
    pair_source="pixels",
    pairing_layout="author_cifar",
    evaluation_pairing="stochastic",
    vectorize_class_circuits=True,
    frequency_tap_index=3,
    frequency_alpha_max=0.25,
    frequency_alpha_init=0.1,
    channel_attention="v233_local_wavelet_quantum",
)
v233_model_output = v233_model(torch.randn(2, 3, 32, 32))
assert v233_model_output.shape == (2, 10)
assert torch.isfinite(v233_model_output).all()
v233_model_output.mean().backward()
assert v233_model.trunk_frequency_modulator.theta.grad is not None
assert v233_model.circuits[0].theta.grad is not None

projected_descriptors = LearnedProjectedWaveletDescriptors()
projected_input = torch.randn(2, 128, 16, 16)
initial_projection = projected_descriptors.projection(projected_input)
expected_projection = projected_input.reshape(2, 4, 32, 16, 16).mean(dim=2)
assert torch.allclose(initial_projection, expected_projection, atol=1e-6, rtol=1e-6)
projected_descriptor_output = projected_descriptors(projected_input)
assert projected_descriptor_output.shape == (2, 4, 4, 4, 4)

v234_classical = V234ClassicalProjectedWaveletAttention()
assert sum(parameter.numel() for parameter in v234_classical.parameters()) == 576
v234_classical_output = v234_classical(torch.randn(2, 128, 16, 16))
assert v234_classical_output.shape == (2, 128, 16, 16)
v234_classical_output.mean().backward()
assert v234_classical.descriptors.projection.weight.grad is not None
assert v234_classical.cross_weights.grad is not None

v234_attention = V234QuantumProjectedWaveletAttention()
assert sum(parameter.numel() for parameter in v234_attention.parameters()) == 576
v234_input = torch.randn(2, 128, 16, 16)
v234_output = v234_attention(v234_input)
assert v234_output.shape == v234_input.shape
assert torch.isfinite(v234_output).all()
v234_output.square().mean().backward()
assert v234_attention.descriptors.projection.weight.grad is not None
assert v234_attention.theta.grad is not None
assert v234_attention.alpha_logits.grad is not None
assert torch.isfinite(v234_attention.descriptors.projection.weight.grad).all()

v234_model = DQHFNN(
    total_pairs=460,
    random_pair_ratio=1.0,
    circuit_variant="no_ent",
    pair_source="pixels",
    pairing_layout="author_cifar",
    evaluation_pairing="stochastic",
    vectorize_class_circuits=True,
    frequency_tap_index=3,
    frequency_alpha_max=0.25,
    frequency_alpha_init=0.1,
    channel_attention="v234_projected_wavelet_quantum",
)
v234_model_output = v234_model(torch.randn(2, 3, 32, 32))
assert v234_model_output.shape == (2, 10)
assert torch.isfinite(v234_model_output).all()
v234_model_output.mean().backward()
assert v234_model.trunk_frequency_modulator.theta.grad is not None
assert v234_model.trunk_frequency_modulator.descriptors.projection.weight.grad is not None
assert v234_model.circuits[0].theta.grad is not None

highres_model = DQHFNN(
    pair_source="v216_grouped_local",
    total_pairs=0,
    local_tap_index=5,
    local_grid_size=8,
)
highres_output = highres_model(torch.randn(2, 3, 32, 32))
assert highres_output.shape == (2, 10)
assert torch.isfinite(highres_output).all()

for pair_source, expected_entanglement in (
    ("v219_simple_entangled", True),
    ("v219_simple_noent", False),
):
    simple_model = DQHFNN(
        pair_source=pair_source,
        total_pairs=0,
        local_tap_index=5,
        local_grid_size=8,
    )
    simple_output = simple_model(torch.randn(2, 3, 32, 32))
    assert simple_output.shape == (2, 10)
    assert torch.isfinite(simple_output).all()
    assert simple_model.local_frequency_head.entangled is expected_entanglement
    simple_output.square().mean().backward()
    assert simple_model.local_frequency_head.theta.grad is not None
    assert torch.isfinite(simple_model.local_frequency_head.theta.grad).all()

for pair_source, expected_entanglement in (
    ("v220_zz_entangled", True),
    ("v220_zz_noent", False),
):
    zz_model = DQHFNN(
        pair_source=pair_source,
        total_pairs=0,
        local_tap_index=5,
        local_grid_size=8,
    )
    zz_output = zz_model(torch.randn(2, 3, 32, 32))
    assert zz_output.shape == (2, 10)
    assert torch.isfinite(zz_output).all()
    assert zz_model.local_frequency_head.entangled is expected_entanglement
    assert zz_model.local_frequency_head.measure_zz
    zz_output.square().mean().backward()
    assert zz_model.local_frequency_head.theta.grad is not None
    assert torch.isfinite(zz_model.local_frequency_head.theta.grad).all()

modulator = V223QuantumFrequencyModulator(channels=8, num_groups=2)
wavelet_input = torch.randn(2, 8, 10, 12)
reconstructed = modulator._idwt(modulator._dwt(wavelet_input))
assert torch.allclose(reconstructed, wavelet_input, atol=1e-6, rtol=1e-6)
modulator.mode = "zero"
odd_input = torch.randn(2, 8, 9, 11)
assert torch.equal(modulator(odd_input), odd_input)

trunk_model = DQHFNN(
    pair_source="v223_trunk_entangled",
    total_pairs=0,
    frequency_tap_index=3,
    frequency_num_groups=4,
)
trunk_output = trunk_model(torch.randn(2, 3, 32, 32))
assert trunk_output.shape == (2, 10)
assert torch.isfinite(trunk_output).all()
trunk_output.square().mean().backward()
trunk_modulator = trunk_model.trunk_frequency_modulator
assert trunk_modulator.theta.grad is not None
assert trunk_modulator.alpha_logits.grad is not None
assert torch.isfinite(trunk_modulator.theta.grad).all()
assert torch.isfinite(trunk_modulator.alpha_logits.grad).all()

odd_features = torch.randn(2, 128, 15, 17)
odd_output = trunk_modulator(odd_features)
assert odd_output.shape == odd_features.shape
assert torch.isfinite(odd_output).all()

raw_model = DQHFNN(
    pair_source="v224_rawfreq_entangled",
    total_pairs=0,
    frequency_num_groups=3,
    frequency_alpha_init=0.05,
    frequency_alpha_max=0.25,
)
raw_input = torch.randn(2, 3, 32, 32)
raw_output = raw_model(raw_input)
assert raw_output.shape == (2, 10)
assert torch.isfinite(raw_output).all()
raw_output.square().mean().backward()
raw_modulator = raw_model.trunk_frequency_modulator
assert raw_modulator is raw_model.classical.input_frequency_modulator
assert raw_model.classical.frequency_modulator is None
assert raw_modulator.theta.grad is not None
assert raw_modulator.alpha_logits.grad is not None
assert torch.isfinite(raw_modulator.theta.grad).all()
assert torch.isfinite(raw_modulator.alpha_logits.grad).all()
assert torch.allclose(
    raw_modulator.alpha,
    torch.full_like(raw_modulator.alpha, 0.05),
    atol=1e-7,
)

frequency_selector = FrequencyGuidedPixelPairs(total_pairs=144)
selected_pairs = frequency_selector(torch.randn(2, 3, 32, 32))
assert selected_pairs.shape == (2, 144, 2)
assert torch.isfinite(selected_pairs).all()

single_position_selector = FrequencyGuidedPixelPairs(total_pairs=6)
synthetic_image = torch.zeros(1, 3, 32, 32)
synthetic_image[:, :, 4:6, 6:8] = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
synthetic_pairs = single_position_selector(synthetic_image)
expected_pairs = torch.tensor([[[1.0, 1.0], [-1.0, -1.0]] * 3])
assert torch.equal(synthetic_pairs, expected_pairs)

frequency_selected_model = DQHFNN(
    pair_source="frequency_guided_pixels",
    total_pairs=144,
    circuit_variant="weak_ent",
    vectorize_class_circuits=True,
)
frequency_selected_output = frequency_selected_model(torch.randn(2, 3, 32, 32))
assert frequency_selected_output.shape == (2, 10)
assert torch.isfinite(frequency_selected_output).all()
frequency_selected_output.square().mean().backward()
for circuit in frequency_selected_model.circuits:
    assert circuit.variant == "weak_ent"
    assert circuit.theta.grad is not None
    assert torch.isfinite(circuit.theta.grad).all()

fuzzy_patch_pairs = FuzzyPooledPatchPairs()
pooled_pairs = fuzzy_patch_pairs(torch.randn(2, 3, 32, 32))
assert pooled_pairs.shape == (2, 16, 2)
assert torch.isfinite(pooled_pairs).all()
assert pooled_pairs.min() >= 0.0
assert pooled_pairs.max() <= torch.pi

quantum_only_model = DQHFNN(
    pair_source="v226_fuzzy_pooled_quantum",
    total_pairs=16,
    circuit_variant="weak_ent",
    vectorize_class_circuits=True,
)
assert sum(parameter.numel() for parameter in quantum_only_model.parameters()) == 118
quantum_only_output = quantum_only_model(torch.randn(2, 3, 32, 32))
assert quantum_only_output.shape == (2, 10)
assert torch.isfinite(quantum_only_output).all()
quantum_only_output.square().mean().backward()
assert quantum_only_model.pair_source.patch_projection.weight.grad is not None
assert torch.isfinite(quantum_only_model.pair_source.patch_projection.weight.grad).all()
for circuit in quantum_only_model.circuits:
    assert circuit.theta.grad is not None
    assert torch.isfinite(circuit.theta.grad).all()

spatial_patch_pairs = SpatialFuzzyPatchPairs()
spatial_pairs = spatial_patch_pairs(torch.randn(2, 3, 32, 32))
assert spatial_pairs.shape == (2, 16, 2)
assert torch.isfinite(spatial_pairs).all()

spatial_quantum_model = DQHFNN(
    pair_source="v227_spatial_fuzzy_quantum",
    total_pairs=16,
    circuit_variant="weak_ent",
    vectorize_class_circuits=True,
)
assert sum(parameter.numel() for parameter in spatial_quantum_model.parameters()) == 641
spatial_quantum_output = spatial_quantum_model(torch.randn(2, 3, 32, 32))
assert spatial_quantum_output.shape == (2, 10)
assert torch.isfinite(spatial_quantum_output).all()
spatial_quantum_output.square().mean().backward()
assert spatial_quantum_model.class_position_logits.grad is not None
assert torch.isfinite(spatial_quantum_model.class_position_logits.grad).all()
assert spatial_quantum_model.pair_source.patch_projection[0].weight.grad is not None
assert spatial_quantum_model.pair_source.patch_projection[2].weight.grad is not None

directional_features = torch.tensor(
    [[[[1.0], [3.0]], [[5.0], [7.0]]]]
)
directional_mean, horizontal_change, vertical_change = (
    DirectionAwareSpatialPatchPairs._directional_descriptors(directional_features)
)
assert torch.equal(directional_mean, torch.tensor([[[[4.0]]]]))
assert torch.equal(horizontal_change, torch.tensor([[[[2.0]]]]))
assert torch.equal(vertical_change, torch.tensor([[[[4.0]]]]))

directional_patch_pairs = DirectionAwareSpatialPatchPairs()
directional_pairs = directional_patch_pairs(torch.randn(2, 3, 32, 32))
assert directional_pairs.shape == (2, 16, 2)
assert torch.isfinite(directional_pairs).all()
assert directional_pairs.min() >= 0.0
assert directional_pairs.max() <= torch.pi

directional_quantum_model = DQHFNN(
    pair_source="v230_directional_spatial_quantum",
    total_pairs=16,
    circuit_variant="weak_ent",
    vectorize_class_circuits=True,
    measurement_basis="z",
)
assert sum(parameter.numel() for parameter in directional_quantum_model.parameters()) == 641
directional_output = directional_quantum_model(torch.randn(2, 3, 32, 32))
assert directional_output.shape == (2, 10)
assert torch.isfinite(directional_output).all()
directional_output.square().mean().backward()
assert directional_quantum_model.class_position_logits.grad is not None
assert directional_quantum_model.pair_source.patch_projection[0].weight.grad is not None
assert directional_quantum_model.pair_source.direction_projection.weight.grad is not None
assert directional_quantum_model.pair_source.direction_bias.grad is not None
for circuit in directional_quantum_model.circuits:
    assert circuit.theta.grad is not None
    assert torch.isfinite(circuit.theta.grad).all()

constant_features = torch.ones(1, 2, 4, 4)
constant_high_energy = CNNSpatialFrequencyPairs._haar_high_energy(constant_features)
assert torch.allclose(constant_high_energy, torch.full_like(constant_high_energy, 1e-4))
checkerboard_features = torch.tensor(
    [[[[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]] * 2]]
)
checkerboard_high_energy = CNNSpatialFrequencyPairs._haar_high_energy(
    checkerboard_features
)
assert checkerboard_high_energy.mean() > constant_high_energy.mean()

cnn_spatial_frequency_pairs = CNNSpatialFrequencyPairs()
cnn_feature_pairs = cnn_spatial_frequency_pairs(torch.randn(2, 128, 16, 16))
assert cnn_feature_pairs.shape == (2, 16, 2)
assert torch.isfinite(cnn_feature_pairs).all()
assert cnn_feature_pairs.min() >= 0.0
assert cnn_feature_pairs.max() <= torch.pi
assert sum(parameter.numel() for parameter in cnn_spatial_frequency_pairs.parameters()) == 1181

hybrid_adapter_model = DQHFNN(
    pair_source="v231_cnn_spatial_frequency_quantum",
    total_pairs=16,
    circuit_variant="weak_ent",
    vectorize_class_circuits=True,
    measurement_basis="z",
    local_tap_index=3,
    quantum_auxiliary_weight=0.2,
    fusion_alpha_init=0.1,
    fusion_alpha_max=0.5,
)
assert hybrid_adapter_model.has_quantum_auxiliary
assert torch.allclose(hybrid_adapter_model.fusion_alpha, torch.tensor(0.1))
hybrid_input = torch.randn(2, 3, 32, 32)
hybrid_output, hybrid_quantum_output = hybrid_adapter_model.forward_with_auxiliary(
    hybrid_input
)
assert hybrid_output.shape == (2, 10)
assert hybrid_quantum_output.shape == (2, 10)
assert torch.isfinite(hybrid_output).all()
assert torch.isfinite(hybrid_quantum_output).all()
(hybrid_output.square().mean() + 0.2 * hybrid_quantum_output.square().mean()).backward()
assert hybrid_adapter_model.fusion_alpha_logit.grad is not None
assert hybrid_adapter_model.class_position_logits.grad is not None
assert hybrid_adapter_model.pair_source.feature_projection[0].weight.grad is not None
assert hybrid_adapter_model.pair_source.spatial_projection.weight.grad is not None
assert hybrid_adapter_model.pair_source.frequency_projection.weight.grad is not None
assert hybrid_adapter_model.classical.features[0].body[0].weight.grad is not None
for circuit in hybrid_adapter_model.circuits:
    assert circuit.theta.grad is not None
    assert torch.isfinite(circuit.theta.grad).all()

x_circuit = DualQubitMembershipCircuit("no_ent", measurement_basis="x")
with torch.no_grad():
    x_circuit.theta.zero_()
x_memberships = x_circuit(torch.tensor([[torch.pi / 2, 0.0]]))
assert torch.allclose(x_memberships, torch.tensor([[1.0, 0.5]]), atol=1e-6)

x_spatial_quantum_model = DQHFNN(
    pair_source="v227_spatial_fuzzy_quantum",
    total_pairs=16,
    circuit_variant="weak_ent",
    vectorize_class_circuits=True,
    measurement_basis="x",
)
assert sum(parameter.numel() for parameter in x_spatial_quantum_model.parameters()) == 641
x_reference_model = DQHFNN(
    pair_source="v227_spatial_fuzzy_quantum",
    total_pairs=16,
    circuit_variant="weak_ent",
    vectorize_class_circuits=False,
    measurement_basis="x",
)
x_reference_model.load_state_dict(x_spatial_quantum_model.state_dict())
x_comparison_input = torch.randn(2, 3, 32, 32)
assert torch.allclose(
    x_spatial_quantum_model.memberships(x_comparison_input),
    x_reference_model.memberships(x_comparison_input),
    atol=1e-6,
    rtol=1e-5,
)
x_spatial_output = x_spatial_quantum_model(torch.randn(2, 3, 32, 32))
assert x_spatial_output.shape == (2, 10)
assert torch.isfinite(x_spatial_output).all()
x_spatial_output.square().mean().backward()
assert x_spatial_quantum_model.class_position_logits.grad is not None
for circuit in x_spatial_quantum_model.circuits:
    assert circuit.measurement_basis == "x"
    assert circuit.theta.grad is not None
    assert torch.isfinite(circuit.theta.grad).all()

v234_replacement = DQHFNN(
    total_pairs=0,
    random_pair_ratio=0.0,
    circuit_variant="no_ent",
    pair_source="none",
    channel_attention="v234_projected_wavelet_quantum",
    frequency_tap_index=3,
    fca_num_groups=16,
    fca_num_circuits=4,
    frequency_alpha_init=0.1,
    frequency_alpha_max=0.25,
    author_dq_branch_enabled=False,
)
assert len(v234_replacement.circuits) == 0
assert v234_replacement.pair_source is None
assert isinstance(v234_replacement.fuzzy_projection, nn.Identity)
replacement_output = v234_replacement(torch.randn(2, 3, 32, 32))
assert replacement_output.shape == (2, 10)
assert torch.isfinite(replacement_output).all()
replacement_output.square().mean().backward()
assert v234_replacement.trunk_frequency_modulator.theta.grad is not None
assert v234_replacement.trunk_frequency_modulator.descriptors.projection.weight.grad is not None

crx_state = torch.zeros(1, 16, dtype=torch.complex64)
crx_state[:, 8] = 1.0
zero_crx = V235HQNetProjectedWaveletAttention._crx(
    crx_state, torch.zeros(1), 0, 1
)
assert torch.equal(zero_crx, crx_state)
pi_crx = V235HQNetProjectedWaveletAttention._crx(
    crx_state, torch.full((1,), torch.pi), 0, 1
)
assert torch.allclose(pi_crx[:, 12], torch.tensor([0.0 - 1.0j]), atol=1e-6)

v235_common = dict(
    total_pairs=0,
    random_pair_ratio=0.0,
    circuit_variant="no_ent",
    pair_source="none",
    frequency_tap_index=3,
    fca_num_groups=16,
    fca_num_circuits=4,
    frequency_alpha_init=0.1,
    frequency_alpha_max=0.25,
    author_dq_branch_enabled=False,
)
v235_hqnet = DQHFNN(
    **v235_common,
    channel_attention="v235_hqnet_projected_wavelet_quantum",
)
v235_local_rx = DQHFNN(
    **v235_common,
    channel_attention="v235_hqnet_projected_wavelet_quantum_local_rx",
)
v235_classical = DQHFNN(
    **v235_common,
    channel_attention="v235_hqnet_projected_wavelet_classical_matched",
)
v235_cnn = DQHFNN(**v235_common, channel_attention="none")
v235_added_parameters = {
    sum(parameter.numel() for parameter in model.parameters())
    - sum(parameter.numel() for parameter in v235_cnn.parameters())
    for model in (v235_hqnet, v235_local_rx, v235_classical)
}
assert v235_added_parameters == {560}
assert v235_hqnet.trunk_frequency_modulator.theta.shape == (4, 8)
assert isinstance(
    v235_classical.trunk_frequency_modulator,
    V235ClassicalHQNetProjectedWaveletAttention,
)
v235_local_rx.load_state_dict(v235_hqnet.state_dict())
v235_input = torch.randn(2, 3, 32, 32)
v235_output = v235_hqnet(v235_input)
v235_local_rx_output = v235_local_rx(v235_input)
assert v235_output.shape == (2, 10)
assert torch.isfinite(v235_output).all()
assert torch.isfinite(v235_local_rx_output).all()
assert not torch.allclose(v235_output, v235_local_rx_output)
v235_output.square().mean().backward()
assert v235_hqnet.trunk_frequency_modulator.theta.grad is not None
assert torch.isfinite(v235_hqnet.trunk_frequency_modulator.theta.grad).all()
assert v235_hqnet.trunk_frequency_modulator.theta.grad.abs().sum() > 0
assert v235_hqnet.trunk_frequency_modulator.descriptors.projection.weight.grad is not None
assert v235_hqnet.trunk_frequency_modulator.descriptors.projection.weight.grad.abs().sum() > 0

v236_entangled = V236TwoLevelHaarQuantumLogitHead(entangled=True)
v236_local_rx = V236TwoLevelHaarQuantumLogitHead(entangled=False)
assert sum(parameter.numel() for parameter in v236_entangled.parameters()) == 733
assert sum(parameter.numel() for parameter in v236_local_rx.parameters()) == 733
v236_features = torch.randn(2, 128, 16, 16)
v236_descriptors = v236_entangled.descriptors(v236_features)
assert v236_descriptors.shape == (2, 4, 4, 4, 7)
v236_local_rx.load_state_dict(v236_entangled.state_dict())
assert not torch.allclose(
    v236_entangled(v236_features),
    v236_local_rx(v236_features),
)

v236_model = DQHFNN(
    pair_source="v236_twolevel_haar_quantum",
    total_pairs=0,
    local_tap_index=3,
    quantum_auxiliary_weight=0.05,
    fusion_alpha_init=0.05,
    fusion_alpha_max=0.1,
    author_dq_branch_enabled=False,
)
assert v236_model.has_quantum_auxiliary
assert len(v236_model.circuits) == 0
assert torch.allclose(v236_model.fusion_alpha, torch.tensor(0.05))
v236_logits, v236_quantum_logits = v236_model.forward_with_auxiliary(
    torch.randn(2, 3, 32, 32)
)
assert v236_logits.shape == (2, 10)
assert v236_quantum_logits.shape == (2, 10)
assert torch.isfinite(v236_logits).all()
assert torch.isfinite(v236_quantum_logits).all()
(v236_logits.square().mean() + 0.05 * v236_quantum_logits.square().mean()).backward()
assert v236_model.quantum_logit_head.theta.grad is not None
assert v236_model.quantum_logit_head.theta.grad.abs().sum() > 0
assert v236_model.quantum_logit_head.projection.weight.grad is not None
assert v236_model.quantum_logit_head.band_projection.weight.grad is not None
assert v236_model.quantum_logit_head.class_projection.weight.grad is not None
assert v236_model.quantum_logit_head.alpha_logit.grad is not None

qgwp_entangled = QGWPHaarDetailPool(
    channels=64,
    num_groups=8,
    gate_type="quantum",
    entangled=True,
    alpha_init=0.05,
    alpha_max=0.2,
)
qgwp_local = QGWPHaarDetailPool(
    channels=64,
    num_groups=8,
    gate_type="quantum",
    entangled=False,
    alpha_init=0.05,
    alpha_max=0.2,
)
qgwp_classical = QGWPHaarDetailPool(
    channels=64,
    num_groups=8,
    gate_type="classical",
    entangled=False,
    alpha_init=0.05,
    alpha_max=0.2,
)
assert {
    sum(parameter.numel() for parameter in module.parameters())
    for module in (qgwp_entangled, qgwp_local, qgwp_classical)
} == {88}
qgwp_local.load_state_dict(qgwp_entangled.state_dict())
qgwp_input = torch.randn(2, 64, 32, 32)
qgwp_entangled_output = qgwp_entangled(qgwp_input)
qgwp_local_output = qgwp_local(qgwp_input)
assert qgwp_entangled_output.shape == (2, 64, 16, 16)
assert torch.isfinite(qgwp_entangled_output).all()
assert torch.isfinite(qgwp_local_output).all()
assert not torch.allclose(qgwp_entangled_output, qgwp_local_output)
qgwp_entangled.mode = "zero"
assert torch.equal(
    qgwp_entangled(qgwp_input), nn.functional.max_pool2d(qgwp_input, 2)
)
qgwp_entangled.mode = "normal"
qgwp_entangled_output.square().mean().backward()
assert qgwp_entangled.theta.grad is not None
assert qgwp_entangled.theta.grad.abs().sum() > 0
assert qgwp_entangled.alpha_logits.grad is not None
assert qgwp_entangled.alpha_logits.grad.abs().sum() > 0

qgwp_common = dict(
    pair_source="none",
    total_pairs=0,
    author_dq_branch_enabled=False,
    frequency_num_groups=8,
    frequency_alpha_init=0.05,
    frequency_alpha_max=0.2,
)
qgwp_quantum_model = DQHFNN(
    **qgwp_common, first_pool="qgwp_quantum_entangled"
)
qgwp_local_model = DQHFNN(
    **qgwp_common, first_pool="qgwp_quantum_local_rx"
)
qgwp_classical_model = DQHFNN(
    **qgwp_common, first_pool="qgwp_classical_matched"
)
qgwp_cnn_model = DQHFNN(**qgwp_common, first_pool="maxpool")
assert {
    sum(parameter.numel() for parameter in model.parameters())
    - sum(parameter.numel() for parameter in qgwp_cnn_model.parameters())
    for model in (qgwp_quantum_model, qgwp_local_model, qgwp_classical_model)
} == {88}
qgwp_model_output = qgwp_quantum_model(torch.randn(2, 3, 32, 32))
assert qgwp_model_output.shape == (2, 10)
assert torch.isfinite(qgwp_model_output).all()
qgwp_model_output.square().mean().backward()
assert qgwp_quantum_model.trunk_frequency_modulator.theta.grad is not None
assert qgwp_quantum_model.trunk_frequency_modulator.theta.grad.abs().sum() > 0

qgwp_fixed_local = QGWPHaarDetailPool(
    channels=64,
    num_groups=8,
    gate_type="quantum",
    entangled=False,
    fixed_beta=0.1,
)
qgwp_fixed_classical = QGWPHaarDetailPool(
    channels=64,
    num_groups=8,
    gate_type="classical",
    entangled=False,
    fixed_beta=0.1,
)
qgwp_fixed_uniform = QGWPHaarDetailPool(
    channels=64,
    num_groups=8,
    gate_type="uniform",
    entangled=False,
    fixed_beta=0.1,
)
assert sum(parameter.numel() for parameter in qgwp_fixed_local.parameters()) == 64
assert sum(parameter.numel() for parameter in qgwp_fixed_classical.parameters()) == 64
assert sum(parameter.numel() for parameter in qgwp_fixed_uniform.parameters()) == 0
assert qgwp_fixed_local.alpha_logits is None
fixed_baseline, fixed_gates, fixed_detail = qgwp_fixed_local.components(qgwp_input)
assert fixed_baseline.shape == (2, 64, 16, 16)
assert fixed_gates.shape == (2, 8, 16, 16, 3)
assert fixed_detail.shape == fixed_baseline.shape
qgwp_fixed_local.mode = "zero"
assert torch.equal(qgwp_fixed_local(qgwp_input), fixed_baseline)
qgwp_fixed_local.mode = "normal"
qgwp_fixed_local(qgwp_input).square().mean().backward()
assert qgwp_fixed_local.theta.grad is not None
assert qgwp_fixed_local.theta.grad.abs().sum() > 0
uniform_gates = qgwp_fixed_uniform.gates(qgwp_fixed_uniform._dwt(qgwp_input))
assert torch.equal(uniform_gates, torch.ones_like(uniform_gates))

qgwp_fixed_common = {
    **qgwp_common,
    "frequency_beta": 0.1,
}
qgwp_fixed_local_model = DQHFNN(
    **qgwp_fixed_common, first_pool="qgwp_fixed_quantum_local_rx"
)
qgwp_fixed_classical_model = DQHFNN(
    **qgwp_fixed_common, first_pool="qgwp_fixed_classical_matched"
)
qgwp_fixed_uniform_model = DQHFNN(
    **qgwp_fixed_common, first_pool="qgwp_fixed_uniform"
)
assert {
    sum(parameter.numel() for parameter in model.parameters())
    - sum(parameter.numel() for parameter in qgwp_cnn_model.parameters())
    for model in (qgwp_fixed_local_model, qgwp_fixed_classical_model)
} == {64}
assert (
    sum(parameter.numel() for parameter in qgwp_fixed_uniform_model.parameters())
    == sum(parameter.numel() for parameter in qgwp_cnn_model.parameters())
)
assert qgwp_fixed_local_model(torch.randn(2, 3, 32, 32)).shape == (2, 10)

born_local = QuantumBornHaarPool(
    channels=64, num_groups=8, gate_type="quantum", entangled=False
)
born_entangled = QuantumBornHaarPool(
    channels=64, num_groups=8, gate_type="quantum", entangled=True
)
born_classical = QuantumBornHaarPool(
    channels=64, num_groups=8, gate_type="classical", entangled=False
)
born_softpool = QuantumBornHaarPool(
    channels=64, num_groups=8, gate_type="softpool", entangled=False
)
assert {
    sum(parameter.numel() for parameter in module.parameters())
    for module in (born_local, born_entangled, born_classical)
} == {64}
assert sum(parameter.numel() for parameter in born_softpool.parameters()) == 0
born_entangled.load_state_dict(born_local.state_dict())
test_state = torch.randn(3, 4, dtype=torch.complex64)
assert torch.equal(
    DualQubitMembershipCircuit._cnot_reverse(test_state),
    test_state[:, (0, 3, 2, 1)],
)
born_patches = born_local.patches(qgwp_input)
born_local_probabilities = born_local.probabilities(born_patches)
born_entangled_probabilities = born_entangled.probabilities(born_patches)
assert born_local_probabilities.shape == (2, 8, 16, 16, 4)
assert torch.allclose(
    born_local_probabilities.sum(dim=-1),
    torch.ones_like(born_local_probabilities[..., 0]),
    atol=1e-5,
)
assert torch.allclose(
    born_local_probabilities[..., 0] * born_local_probabilities[..., 3],
    born_local_probabilities[..., 1] * born_local_probabilities[..., 2],
    atol=1e-5,
)
assert not torch.allclose(born_local_probabilities, born_entangled_probabilities)
born_classical_probabilities = born_classical.probabilities(born_patches)
assert torch.allclose(
    born_classical_probabilities.sum(dim=-1),
    torch.ones_like(born_classical_probabilities[..., 0]),
    atol=1e-5,
)
assert torch.allclose(
    born_classical_probabilities[..., 0] * born_classical_probabilities[..., 3],
    born_classical_probabilities[..., 1] * born_classical_probabilities[..., 2],
    atol=1e-5,
)
born_output = born_local(qgwp_input)
assert born_output.shape == (2, 64, 16, 16)
assert torch.isfinite(born_output).all()
born_local.mode = "zero"
assert torch.equal(born_local(qgwp_input), nn.functional.max_pool2d(qgwp_input, 2))
born_local.mode = "normal"
born_output.square().mean().backward()
assert born_local.theta.grad is not None
assert born_local.theta.grad.abs().sum() > 0

born_common = dict(
    pair_source="none",
    total_pairs=0,
    author_dq_branch_enabled=False,
    frequency_num_groups=8,
)
born_local_model = DQHFNN(
    **born_common, first_pool="qbwp_quantum_local_rx"
)
born_entangled_model = DQHFNN(
    **born_common, first_pool="qbwp_quantum_entangled"
)
born_classical_model = DQHFNN(
    **born_common, first_pool="qbwp_classical_matched"
)
born_softpool_model = DQHFNN(**born_common, first_pool="qbwp_softpool")
assert {
    sum(parameter.numel() for parameter in model.parameters())
    - sum(parameter.numel() for parameter in qgwp_cnn_model.parameters())
    for model in (born_local_model, born_entangled_model, born_classical_model)
} == {64}
assert (
    sum(parameter.numel() for parameter in born_softpool_model.parameters())
    == sum(parameter.numel() for parameter in qgwp_cnn_model.parameters())
)
born_model_output = born_entangled_model(torch.randn(2, 3, 32, 32))
assert born_model_output.shape == (2, 10)
born_model_output.square().mean().backward()
assert born_entangled_model.trunk_frequency_modulator.theta.grad is not None
assert born_entangled_model.trunk_frequency_modulator.theta.grad.abs().sum() > 0

refined_local = QuantumRefinedSoftPool(
    channels=64, num_groups=8, gate_type="quantum", entangled=False
)
refined_entangled = QuantumRefinedSoftPool(
    channels=64, num_groups=8, gate_type="quantum", entangled=True
)
refined_classical = QuantumRefinedSoftPool(
    channels=64, num_groups=8, gate_type="classical", entangled=False
)
assert {
    sum(parameter.numel() for parameter in module.parameters())
    for module in (refined_local, refined_entangled, refined_classical)
} == {80}
refined_patches = refined_local.patches(qgwp_input)
softpool_probabilities = born_softpool.probabilities(refined_patches)
for module in (refined_local, refined_entangled, refined_classical):
    initial_probabilities = module.probabilities(refined_patches)
    assert torch.allclose(
        initial_probabilities, softpool_probabilities, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        initial_probabilities.sum(dim=-1),
        torch.ones_like(initial_probabilities[..., 0]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        module(qgwp_input), born_softpool(qgwp_input), atol=1e-6, rtol=1e-6
    )
    module(qgwp_input).square().mean().backward()
    assert module.theta.grad is not None
    assert torch.isfinite(module.theta.grad).all()
    assert module.theta.grad.abs().sum() > 0

refined_local.mode = "zero"
assert torch.equal(
    refined_local(qgwp_input), nn.functional.max_pool2d(qgwp_input, 2)
)
refined_local.mode = "normal"
with torch.no_grad():
    refined_entangled.theta.normal_(mean=0.0, std=0.2)
entangled_probabilities = refined_entangled.probabilities(refined_patches)
refined_entangled.mode = "base"
assert torch.equal(
    refined_entangled.probabilities(refined_patches), softpool_probabilities
)
assert torch.equal(refined_entangled(qgwp_input), born_softpool(qgwp_input))
refined_entangled.mode = "normal"
refined_entangled.entangled = False
no_entanglement_probabilities = refined_entangled.probabilities(refined_patches)
refined_entangled.entangled = True
assert not torch.allclose(entangled_probabilities, no_entanglement_probabilities)

refined_common = dict(
    pair_source="none",
    total_pairs=0,
    author_dq_branch_enabled=False,
    frequency_num_groups=8,
)
refined_local_model = DQHFNN(
    **refined_common, first_pool="qsrp_quantum_local_rx"
)
refined_entangled_model = DQHFNN(
    **refined_common, first_pool="qsrp_quantum_entangled"
)
refined_classical_model = DQHFNN(
    **refined_common, first_pool="qsrp_classical_matched"
)
assert {
    sum(parameter.numel() for parameter in model.parameters())
    - sum(parameter.numel() for parameter in qgwp_cnn_model.parameters())
    for model in (
        refined_local_model,
        refined_entangled_model,
        refined_classical_model,
    )
} == {80}
refined_model_output = refined_entangled_model(torch.randn(2, 3, 32, 32))
assert refined_model_output.shape == (2, 10)
assert torch.isfinite(refined_model_output).all()
refined_model_output.square().mean().backward()
assert refined_entangled_model.trunk_frequency_modulator.theta.grad is not None
assert refined_entangled_model.trunk_frequency_modulator.theta.grad.abs().sum() > 0

uniform_superpixel_model = DQHFNN(
    pair_source="v228_superpixel9_uniform_quantum",
    total_pairs=16,
    circuit_variant="weak_ent",
    vectorize_class_circuits=True,
)
assert uniform_superpixel_model.class_position_logits is None
assert sum(parameter.numel() for parameter in uniform_superpixel_model.parameters()) == 481
uniform_superpixel_output = uniform_superpixel_model(torch.randn(2, 3, 32, 32))
assert uniform_superpixel_output.shape == (2, 10)
assert torch.isfinite(uniform_superpixel_output).all()
uniform_superpixel_output.square().mean().backward()
assert uniform_superpixel_model.pair_source.patch_projection[0].weight.grad is not None
assert uniform_superpixel_model.pair_source.patch_projection[2].weight.grad is not None
for circuit in uniform_superpixel_model.circuits:
    assert circuit.theta.grad is not None
    assert torch.isfinite(circuit.theta.grad).all()
