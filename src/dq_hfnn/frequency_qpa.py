"""Frequency-conditioned QPSAN residual for small spatial feature maps.

This module is intentionally independent of the legacy DQ-HFNN fuzzy-pair
pipeline.  It operates on a CNN feature map, not sampled input pixel pairs.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_dwt2(x):
    """One-level orthonormal Haar DWT for an even-sized feature map."""
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("Haar DWT requires even spatial dimensions")
    top_left = x[..., 0::2, 0::2]
    top_right = x[..., 0::2, 1::2]
    bottom_left = x[..., 1::2, 0::2]
    bottom_right = x[..., 1::2, 1::2]
    return (
        (top_left + top_right + bottom_left + bottom_right) * 0.5,
        (top_left - top_right + bottom_left - bottom_right) * 0.5,
        (top_left + top_right - bottom_left - bottom_right) * 0.5,
        (top_left - top_right - bottom_left + bottom_right) * 0.5,
    )


def haar_idwt2(ll, lh, hl, hh):
    """Inverse of :func:`haar_dwt2`."""
    batch, channels, height, width = ll.shape
    output = ll.new_empty(batch, channels, height * 2, width * 2)
    output[..., 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
    output[..., 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
    output[..., 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
    output[..., 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
    return output


class ClassicalQPAScorer(nn.Module):
    """Five-parameter classical control with the QPSAN input variables."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(5))
        with torch.no_grad():
            self.weight[0] = 0.5
            self.weight[1:].normal_(mean=0.0, std=0.12)

    def forward(self, q, k):
        q, k = torch.broadcast_tensors(q, k)
        features = torch.stack((q, k, q - k, q + k, q * k), dim=-1)
        return torch.sigmoid((features * self.weight).sum(dim=-1))


class TorchQuantumQPAScorer(nn.Module):
    """Two-qubit QPSAN similarity: probability of the even-parity states."""

    def __init__(self, entangled=True, pair_chunk=None):
        super().__init__()
        self.entangled = entangled
        self.pair_chunk = pair_chunk
        self.input_scale = nn.Parameter(torch.tensor(0.5))
        self.diff_scale = nn.Parameter(torch.empty(()))
        self.sum_scale = nn.Parameter(torch.empty(()))
        self.ent_scale = nn.Parameter(torch.empty(()))
        self.mixer_angle = nn.Parameter(torch.empty(()))
        nn.init.normal_(self.diff_scale, mean=0.0, std=0.12)
        nn.init.normal_(self.sum_scale, mean=0.0, std=0.12)
        nn.init.normal_(self.ent_scale, mean=0.0, std=0.12)
        nn.init.normal_(self.mixer_angle, mean=0.0, std=0.12)
        try:
            import torchquantum as tq
        except ImportError as error:
            raise ImportError(
                "TorchQuantumQPAScorer requires torchquantum. Install the "
                "project requirements in the QIGN environment."
            ) from error
        self.tq = tq

    def _chunk_size(self):
        configured = self.pair_chunk or os.environ.get("TORCHQUANTUM_PAIR_CHUNK", "65536")
        chunk = int(configured)
        if chunk <= 0:
            raise ValueError("TORCHQUANTUM_PAIR_CHUNK must be positive")
        return chunk

    def _run_chunk(self, q, k):
        qdev = self.tq.QuantumDevice(n_wires=2, bsz=q.numel(), device=q.device, record_op=False)
        qdev.reset_states(bsz=q.numel())
        first = math.pi / 4 + self.input_scale * q + self.diff_scale * (q - k) + self.sum_scale * (q + k)
        second = math.pi / 4 + self.input_scale * k - self.diff_scale * (q - k) + self.sum_scale * (q + k)
        self.tq.functional.ry(qdev, wires=0, params=first)
        self.tq.functional.ry(qdev, wires=1, params=second)
        if self.entangled:
            self.tq.functional.cnot(qdev, wires=[0, 1])
        self.tq.functional.ry(qdev, wires=1, params=self.ent_scale * (q + k))
        if self.entangled:
            self.tq.functional.cnot(qdev, wires=[1, 0])
        self.tq.functional.rx(qdev, wires=0, params=2.0 * self.mixer_angle)
        self.tq.functional.rx(qdev, wires=1, params=2.0 * self.mixer_angle)
        probabilities = qdev.get_states_1d().abs().square()
        return probabilities[:, 0] + probabilities[:, 3]

    def forward(self, q, k):
        q, k = torch.broadcast_tensors(q, k)
        shape = q.shape
        q, k = q.reshape(-1), k.reshape(-1)
        scores = [self._run_chunk(q[start : start + self._chunk_size()], k[start : start + self._chunk_size()]) for start in range(0, q.numel(), self._chunk_size())]
        return torch.cat(scores).reshape(shape)


class FrequencyQPAAttention(nn.Module):
    """LL attention whose Q/K are conditioned on the three Haar detail bands."""

    def __init__(self, channels=8, heads=2, mode="torchquantum", entangled=True):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must divide evenly across attention heads")
        if mode not in {"none", "classical", "torchquantum"}:
            raise ValueError(f"Unsupported FrequencyQPA mode: {mode}")
        self.channels = channels
        self.heads = heads
        self.mode = mode
        self.q_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.frequency_q = nn.Conv2d(channels * 3, channels, 1, bias=False)
        self.frequency_k = nn.Conv2d(channels * 3, channels, 1, bias=False)
        self.output_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.frequency_scale = nn.Parameter(torch.tensor(0.0))
        self.scorer = (
            ClassicalQPAScorer() if mode == "classical" else
            TorchQuantumQPAScorer(entangled=entangled) if mode == "torchquantum" else
            None
        )
        self._initialize_parameters()

    def _initialize_parameters(self):
        for layer in (
            self.q_projection, self.k_projection, self.v_projection,
            self.frequency_q, self.frequency_k, self.output_projection,
        ):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        # The branch begins as an LL-only attention path.  Its candidate
        # projections are nonzero so the zero scalar can receive a gradient.

    def forward(self, ll, high):
        if self.mode == "none":
            return ll
        batch, _, height, width = ll.shape
        if (height, width) != (8, 8):
            raise ValueError("FrequencyQPA expects an 8x8 LL feature map")
        q = F.avg_pool2d(self.q_projection(ll), 2)
        k = F.avg_pool2d(self.k_projection(ll), 2)
        v = F.avg_pool2d(self.v_projection(ll), 2)
        q = q + self.frequency_scale * F.avg_pool2d(self.frequency_q(high), 2)
        k = k + self.frequency_scale * F.avg_pool2d(self.frequency_k(high), 2)
        dimension = self.channels // self.heads
        q = q.reshape(batch, self.heads, dimension, 16).transpose(2, 3)
        k = k.reshape(batch, self.heads, dimension, 16).transpose(2, 3)
        v = v.reshape(batch, self.heads, dimension, 16).transpose(2, 3)
        pair_scores = self.scorer(q.unsqueeze(3), k.unsqueeze(2)).mean(dim=-1)
        weights = pair_scores.softmax(dim=-1)
        context = weights @ v
        context = context.transpose(2, 3).reshape(batch, self.channels, 4, 4)
        context = F.interpolate(context, size=(height, width), mode="nearest")
        return ll + self.output_projection(context)


class DirectionalHighBandRefinement(nn.Module):
    """Learn a bounded directional correction for LH, HL, and HH details."""

    def __init__(self, channels=8):
        super().__init__()
        self.refine = nn.Conv2d(channels * 3, channels * 3, 1, groups=3, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.0))
        nn.init.kaiming_uniform_(self.refine.weight, a=math.sqrt(5))

    def forward(self, lh, hl, hh):
        details = torch.cat((lh, hl, hh), dim=1)
        correction = self.refine(details)
        lh_c, hl_c, hh_c = correction.chunk(3, dim=1)
        return lh + self.scale * lh_c, hl + self.scale * hl_c, hh + self.scale * hh_c


class FrequencyQPAResidual(nn.Module):
    """DWT -> frequency-conditioned QPA -> IDWT residual for a 128x16x16 tap."""

    def __init__(self, channels=128, reduced_channels=8, mode="torchquantum", entangled=True, alpha_max=0.10, alpha_init=0.02):
        super().__init__()
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must be between zero and alpha_max")
        self.channels = channels
        self.mode = mode
        self.reduce = nn.Conv2d(channels, reduced_channels, 1, bias=False)
        self.attention = FrequencyQPAAttention(reduced_channels, mode=mode, entangled=entangled)
        self.detail_refinement = DirectionalHighBandRefinement(reduced_channels)
        self.expand = nn.Conv2d(reduced_channels, channels, 1, bias=False)
        self.alpha_max = alpha_max
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(alpha_init / alpha_max)))

    @property
    def alpha(self):
        return self.alpha_max * self.alpha_logit.sigmoid()

    def forward(self, x):
        if self.mode == "none":
            return x
        reduced = self.reduce(x)
        ll, lh, hl, hh = haar_dwt2(reduced)
        high = torch.cat((lh, hl, hh), dim=1)
        ll = self.attention(ll, high)
        lh, hl, hh = self.detail_refinement(lh, hl, hh)
        reconstruction = haar_idwt2(ll, lh, hl, hh)
        return x + self.alpha * self.expand(reconstruction)


class DirectionalFrequencyQPAResidual(nn.Module):
    """QPA residual that uses only LH/HL; HH is an IDWT-only bypass.

    The 16-dimensional Q/K relation space controls aggregation of the
    directional detail values. LL remains the low-frequency base and HH is
    deliberately excluded from the attention path.
    """

    def __init__(self, channels=128, reduced_channels=64, relation_dim=16,
                 mode="torchquantum", entangled=True, alpha_max=0.10,
                 alpha_init=0.02):
        super().__init__()
        if relation_dim <= 0 or relation_dim > reduced_channels:
            raise ValueError("relation_dim must be in [1, reduced_channels]")
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must be between zero and alpha_max")
        if mode not in {"classical", "torchquantum"}:
            raise ValueError(f"Unsupported directional QPA mode: {mode}")
        self.channels = channels
        self.reduced_channels = reduced_channels
        self.relation_dim = relation_dim
        self.mode = mode
        self.reduce = nn.Conv2d(channels, reduced_channels, 1, bias=False)
        directional_channels = reduced_channels * 2
        self.q_projection = nn.Conv2d(directional_channels, relation_dim, 1, bias=False)
        self.k_projection = nn.Conv2d(directional_channels, relation_dim, 1, bias=False)
        self.v_projection = nn.Conv2d(directional_channels, directional_channels, 1, bias=False)
        self.output_projection = nn.Conv2d(directional_channels, directional_channels, 1, bias=False)
        self.scorer = (
            ClassicalQPAScorer() if mode == "classical"
            else TorchQuantumQPAScorer(entangled=entangled)
        )
        self.expand = nn.Conv2d(reduced_channels, channels, 1, bias=False)
        self.alpha_max = alpha_max
        self.alpha_logit = nn.Parameter(
            torch.logit(torch.tensor(alpha_init / alpha_max))
        )
        for layer in (self.q_projection, self.k_projection, self.v_projection,
                      self.output_projection, self.expand):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        # Start as a nearly identity path while retaining gradients in QPA.
        nn.init.zeros_(self.output_projection.weight)

    @property
    def alpha(self):
        return self.alpha_max * self.alpha_logit.sigmoid()

    def forward(self, x):
        reduced = self.reduce(x)
        ll, lh, hl, hh = haar_dwt2(reduced)
        batch, _, height, width = lh.shape
        directional = torch.cat((lh, hl), dim=1)
        q = F.avg_pool2d(self.q_projection(directional), 2)
        k = F.avg_pool2d(self.k_projection(directional), 2)
        v = F.avg_pool2d(self.v_projection(directional), 2)
        tokens = q.shape[-2] * q.shape[-1]
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        pair_scores = self.scorer(q.unsqueeze(2), k.unsqueeze(1)).mean(dim=-1)
        weights = pair_scores.softmax(dim=-1)
        context = weights @ v
        context = context.transpose(1, 2).reshape(
            batch, self.reduced_channels * 2, int(tokens ** 0.5), int(tokens ** 0.5)
        )
        context = F.interpolate(context, size=(height, width), mode="nearest")
        detail_update = self.output_projection(context)
        lh_update, hl_update = detail_update.chunk(2, dim=1)
        reconstruction = haar_idwt2(ll, lh + self.alpha * lh_update,
                                     hl + self.alpha * hl_update, hh)
        return x + self.alpha * self.expand(reconstruction)


class WideTwoLevelFrequencyQPAAttention(nn.Module):
    """Use narrow frequency-conditioned Q/K scores to aggregate wide value tokens."""

    def __init__(self, channels=128, qk_channels=8, heads=2, mode="torchquantum", entangled=True):
        super().__init__()
        if qk_channels % heads or channels % heads:
            raise ValueError("Q/K and value channels must divide evenly across attention heads")
        if mode not in {"none", "classical", "torchquantum"}:
            raise ValueError(f"Unsupported wide FrequencyQPA mode: {mode}")
        self.channels = channels
        self.qk_channels = qk_channels
        self.heads = heads
        self.mode = mode
        self.q_projection = nn.Conv2d(channels, qk_channels, 1, bias=False)
        self.k_projection = nn.Conv2d(channels, qk_channels, 1, bias=False)
        self.v_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.frequency_q = nn.Conv2d(channels * 3, qk_channels, 1, bias=False)
        self.frequency_k = nn.Conv2d(channels * 3, qk_channels, 1, bias=False)
        self.frequency_scale = nn.Parameter(torch.zeros(()))
        self.output_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.scorer = (
            ClassicalQPAScorer() if mode == "classical" else
            TorchQuantumQPAScorer(entangled=entangled) if mode == "torchquantum" else
            None
        )
        for layer in (
            self.q_projection, self.k_projection, self.v_projection,
            self.frequency_q, self.frequency_k,
        ):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        # Identity start without an outer residual branch.  Once this projection opens,
        # gradients reach the Q/K, V, frequency, and scorer paths on the next update.
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, ll, high):
        if self.mode == "none":
            return ll
        batch, _, height, width = ll.shape
        if (height, width) != (4, 4):
            raise ValueError("Wide two-level FrequencyQPA expects a 4x4 LL2 feature map")
        q = self.q_projection(ll) + self.frequency_scale * self.frequency_q(high)
        k = self.k_projection(ll) + self.frequency_scale * self.frequency_k(high)
        v = self.v_projection(ll)
        qk_dimension = self.qk_channels // self.heads
        value_dimension = self.channels // self.heads
        q = q.reshape(batch, self.heads, qk_dimension, 16).transpose(2, 3)
        k = k.reshape(batch, self.heads, qk_dimension, 16).transpose(2, 3)
        v = v.reshape(batch, self.heads, value_dimension, 16).transpose(2, 3)
        pair_scores = self.scorer(q.unsqueeze(3), k.unsqueeze(2)).mean(dim=-1)
        weights = pair_scores.softmax(dim=-1)
        context = weights @ v
        context = context.transpose(2, 3).reshape(batch, self.channels, height, width)
        return ll + self.output_projection(context)


class WideTwoLevelFrequencyQPALayer(nn.Module):
    """Full-width two-level Haar reconstruction with narrow quantum Q/K and wide V."""

    def __init__(self, channels=128, qk_channels=8, mode="torchquantum", entangled=True):
        super().__init__()
        self.channels = channels
        self.mode = mode
        self.attention = WideTwoLevelFrequencyQPAAttention(
            channels=channels,
            qk_channels=qk_channels,
            mode=mode,
            entangled=entangled,
        )
        self.detail_refinement_1 = DirectionalHighBandRefinement(channels)
        self.detail_refinement_2 = DirectionalHighBandRefinement(channels)

    def forward(self, x):
        if self.mode == "none":
            return x
        ll1, lh1, hl1, hh1 = haar_dwt2(x)
        ll2, lh2, hl2, hh2 = haar_dwt2(ll1)
        ll2 = self.attention(ll2, torch.cat((lh2, hl2, hh2), dim=1))
        lh2, hl2, hh2 = self.detail_refinement_2(lh2, hl2, hh2)
        ll1 = haar_idwt2(ll2, lh2, hl2, hh2)
        lh1, hl1, hh1 = self.detail_refinement_1(lh1, hl1, hh1)
        return haar_idwt2(ll1, lh1, hl1, hh1)


class FullWidthTwoLevelFrequencyQPAAttention(nn.Module):
    """Compact LL2 QPA with directional full-width detail conditioning."""

    def __init__(self, channels=8, heads=2, mode="torchquantum", entangled=True):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must divide evenly across attention heads")
        if mode not in {"none", "classical", "torchquantum"}:
            raise ValueError(f"Unsupported full-width FrequencyQPA mode: {mode}")
        self.channels = channels
        self.heads = heads
        self.mode = mode
        self.q_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.frequency_q = nn.Conv2d(channels * 3, channels, 1, bias=False)
        self.frequency_k = nn.Conv2d(channels * 3, channels, 1, bias=False)
        self.frequency_scale = nn.Parameter(torch.zeros(()))
        self.output_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.scorer = (
            ClassicalQPAScorer() if mode == "classical" else
            TorchQuantumQPAScorer(entangled=entangled) if mode == "torchquantum" else
            None
        )
        for layer in (
            self.q_projection, self.k_projection, self.v_projection,
            self.frequency_q, self.frequency_k,
        ):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        # The full-width reconstruction starts exactly as identity.
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, ll, high):
        if self.mode == "none":
            return ll
        batch, _, height, width = ll.shape
        if (height, width) != (4, 4):
            raise ValueError("Full-width two-level FrequencyQPA expects a 4x4 LL2 feature map")
        q = self.q_projection(ll) + self.frequency_scale * self.frequency_q(high)
        k = self.k_projection(ll) + self.frequency_scale * self.frequency_k(high)
        v = self.v_projection(ll)
        dimension = self.channels // self.heads
        q = q.reshape(batch, self.heads, dimension, 16).transpose(2, 3)
        k = k.reshape(batch, self.heads, dimension, 16).transpose(2, 3)
        v = v.reshape(batch, self.heads, dimension, 16).transpose(2, 3)
        pair_scores = self.scorer(q.unsqueeze(3), k.unsqueeze(2)).mean(dim=-1)
        weights = pair_scores.softmax(dim=-1)
        context = weights @ v
        context = context.transpose(2, 3).reshape(batch, self.channels, height, width)
        return ll + self.output_projection(context)


class FullWidthTwoLevelFrequencyQPALayer(nn.Module):
    """DWA-style residual wrapper around full-width DWT and compact LL2 QPA."""

    def __init__(
        self, channels=128, reduced_channels=8, mode="torchquantum", entangled=True,
        alpha_max=0.10, alpha_init=0.02,
    ):
        super().__init__()
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must be between zero and alpha_max")
        self.channels = channels
        self.mode = mode
        self.reduce_ll2 = nn.Conv2d(channels, reduced_channels, 1, bias=False)
        self.reduce_frequency = nn.Conv2d(channels * 3, reduced_channels * 3, 1, bias=False)
        self.attention = FullWidthTwoLevelFrequencyQPAAttention(
            reduced_channels, mode=mode, entangled=entangled
        )
        self.expand_delta = nn.Conv2d(reduced_channels, channels, 1, bias=False)
        self.alpha_max = alpha_max
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(alpha_init / alpha_max)))

    @property
    def alpha(self):
        return self.alpha_max * self.alpha_logit.sigmoid()

    def forward(self, x):
        if self.mode == "none":
            return x
        ll1, lh1, hl1, hh1 = haar_dwt2(x)
        ll2, lh2, hl2, hh2 = haar_dwt2(ll1)
        compact_ll2 = self.reduce_ll2(ll2)
        frequency_context = self.reduce_frequency(torch.cat((lh2, hl2, hh2), dim=1))
        updated_compact_ll2 = self.attention(compact_ll2, frequency_context)
        ll2 = ll2 + self.expand_delta(updated_compact_ll2 - compact_ll2)
        ll1 = haar_idwt2(ll2, lh2, hl2, hh2)
        candidate = haar_idwt2(ll1, lh1, hl1, hh1)
        return x + self.alpha * (candidate - x)
