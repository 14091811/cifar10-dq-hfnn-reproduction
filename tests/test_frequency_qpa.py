import importlib.util

import pytest
import torch

from dq_hfnn.frequency_qpa import (
    FrequencyQPAResidual,
    haar_dwt2,
    haar_idwt2,
)
from dq_hfnn.frequency_qpa_model import FrequencyQPANet


def test_haar_round_trip():
    features = torch.randn(2, 8, 16, 16)
    assert torch.allclose(haar_idwt2(*haar_dwt2(features)), features, atol=1e-6)


def test_classical_frequency_qpa_is_finite_and_trainable():
    block = FrequencyQPAResidual(mode="classical")
    features = torch.randn(2, 128, 16, 16, requires_grad=True)
    output = block(features)
    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert block.attention.scorer.weight.grad is not None
    assert torch.isfinite(block.attention.scorer.weight.grad).all()


def test_disabled_frequency_qpa_is_exact_identity():
    block = FrequencyQPAResidual(mode="none")
    features = torch.randn(2, 128, 16, 16)
    assert torch.equal(block(features), features)


def test_frequency_qpa_network_output_shape():
    model = FrequencyQPANet(mode="classical")
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(
    importlib.util.find_spec("torchquantum") is None,
    reason="TorchQuantum is not installed in this local environment",
)
def test_torchquantum_qpa_is_finite_and_entanglement_changes_scores(monkeypatch):
    monkeypatch.setenv("TORCHQUANTUM_PAIR_CHUNK", "128")
    torch.manual_seed(7)
    entangled = FrequencyQPAResidual(mode="torchquantum", entangled=True)
    no_entanglement = FrequencyQPAResidual(mode="torchquantum", entangled=False)
    no_entanglement.load_state_dict(entangled.state_dict())
    features = torch.randn(1, 128, 16, 16, requires_grad=True)
    entangled_output = entangled(features)
    no_entanglement_output = no_entanglement(features)
    assert torch.isfinite(entangled_output).all()
    assert (entangled_output - no_entanglement_output).abs().max() > 0
    entangled_output.square().mean().backward()
    assert entangled.attention.scorer.input_scale.grad is not None
