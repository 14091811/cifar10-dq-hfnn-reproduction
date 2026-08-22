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


def apply_2d_rope(tokens, height, width):
    """Apply fixed 2D RoPE: first half encodes rows, second half columns."""
    batch, token_count, dimension = tokens.shape
    if token_count != height * width or dimension % 4:
        raise ValueError("2D RoPE requires a token grid and a dimension divisible by four")
    half = dimension // 2
    pair_count = half // 2
    inv_frequency = 1.0 / (10000 ** (
        torch.arange(pair_count, device=tokens.device, dtype=tokens.dtype) / pair_count
    ))
    rows = torch.arange(height, device=tokens.device, dtype=tokens.dtype).repeat_interleave(width)
    columns = torch.arange(width, device=tokens.device, dtype=tokens.dtype).repeat(height)

    def rotate(values, positions):
        values = values.reshape(batch, token_count, pair_count, 2)
        angles = positions[:, None] * inv_frequency[None, :]
        cosine, sine = angles.cos()[None], angles.sin()[None]
        first, second = values.unbind(dim=-1)
        return torch.stack((first * cosine - second * sine, first * sine + second * cosine), dim=-1).flatten(2)

    return torch.cat((rotate(tokens[..., :half], rows), rotate(tokens[..., half:], columns)), dim=-1)


def relative_position_bucket(offsets):
    """Map signed offsets to 0, +/-1, +/-2, and +/-3-or-farther buckets."""
    magnitude = offsets.abs()
    compact = torch.where(magnitude <= 2, magnitude, torch.full_like(magnitude, 3))
    return torch.where(offsets == 0, torch.zeros_like(offsets), torch.where(offsets > 0, compact, 3 + compact)).long()


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

    def __init__(self, entangled=True, pair_chunk=None, circuit_variant="baseline"):
        super().__init__()
        if circuit_variant not in {
            "baseline", "no_entanglement", "single_cnot", "symmetric_ry",
            "rz_entangled", "rz_single_cnot",
        }:
            raise ValueError(f"Unsupported circuit variant: {circuit_variant}")
        self.entangled = entangled
        self.circuit_variant = circuit_variant
        self.pair_chunk = pair_chunk
        self.input_scale = nn.Parameter(torch.tensor(0.5))
        self.diff_scale = nn.Parameter(torch.empty(()))
        self.sum_scale = nn.Parameter(torch.empty(()))
        self.ent_scale = nn.Parameter(torch.empty(()))
        self.mixer_angle = nn.Parameter(torch.empty(()))
        if circuit_variant in {"rz_entangled", "rz_single_cnot"}:
            self.rz_scale = nn.Parameter(torch.empty(()))
        nn.init.normal_(self.diff_scale, mean=0.0, std=0.12)
        nn.init.normal_(self.sum_scale, mean=0.0, std=0.12)
        nn.init.normal_(self.ent_scale, mean=0.0, std=0.12)
        nn.init.normal_(self.mixer_angle, mean=0.0, std=0.12)
        if circuit_variant in {"rz_entangled", "rz_single_cnot"}:
            nn.init.normal_(self.rz_scale, mean=0.0, std=0.12)
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
        use_entanglement = self.entangled and self.circuit_variant != "no_entanglement"
        if self.circuit_variant in {"rz_entangled", "rz_single_cnot"}:
            self.tq.functional.rz(qdev, wires=0, params=self.rz_scale * q)
            self.tq.functional.rz(qdev, wires=1, params=self.rz_scale * k)
        if use_entanglement:
            self.tq.functional.cnot(qdev, wires=[0, 1])
        middle_angle = self.ent_scale * (q + k)
        self.tq.functional.ry(qdev, wires=1, params=middle_angle)
        if self.circuit_variant == "symmetric_ry":
            self.tq.functional.ry(qdev, wires=0, params=middle_angle)
        if use_entanglement and self.circuit_variant not in {"single_cnot", "rz_single_cnot"}:
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


class ChannelWiseFrequencyQPA(nn.Module):
    """Channel-token QPA: Q/K are channel descriptors and V keeps HxW maps."""

    def __init__(self, channels=64, heads=4, mode="torchquantum", entangled=True):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must divide evenly across channel-QPA heads")
        if mode not in {"classical", "torchquantum"}:
            raise ValueError(f"Unsupported channel-QPA mode: {mode}")
        self.channels = channels
        self.heads = heads
        self.head_channels = channels // heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.high_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1,
                      groups=channels * 2, bias=False),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.output_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.scorer = (
            ClassicalQPAScorer() if mode == "classical"
            else TorchQuantumQPAScorer(entangled=entangled)
        )

    def forward(self, ll, lh, hl):
        batch, _, height, width = ll.shape
        q, k, v = self.qkv(ll).chunk(3, dim=1)
        v = v * self.high_gate(torch.cat((lh, hl), dim=1))

        # Each channel is a token. Q/K use global descriptors, while V keeps
        # the full spatial response map for channel-to-channel aggregation.
        q = q.mean(dim=(-2, -1)).reshape(batch, self.heads, self.head_channels)
        k = k.mean(dim=(-2, -1)).reshape(batch, self.heads, self.head_channels)
        v = v.reshape(batch, self.heads, self.head_channels, height * width)
        scores = self.scorer(q.unsqueeze(-1), k.unsqueeze(-2))
        weights = scores.softmax(dim=-1)
        context = weights @ v
        context = context.reshape(batch, self.channels, height, width)
        return self.output_projection(context)


class StarHighFrequencyGate(nn.Module):
    """Star-style LH/HL gate with multiplicative feature interaction."""

    def __init__(self, channels):
        super().__init__()
        input_channels = channels * 2
        self.local = nn.Conv2d(
            input_channels, input_channels, 3, padding=1,
            groups=input_channels, bias=False,
        )
        self.branch_a = nn.Conv2d(input_channels, channels, 1, bias=False)
        self.branch_b = nn.Conv2d(input_channels, channels, 1, bias=False)
        self.output = nn.Conv2d(channels, channels, 1, bias=False)
        self.activation = nn.ReLU6(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, high):
        local = self.local(high)
        interaction = self.branch_a(local) * self.activation(self.branch_b(local))
        return self.gate(self.output(interaction))


class StandardHighFrequencyGate(nn.Module):
    """LH/HL gate using a standard convolution for cross-channel mixing."""

    def __init__(self, channels):
        super().__init__()
        input_channels = channels * 2
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(input_channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, high):
        return self.features(high)


class SeparateBandDepthwiseGate(nn.Module):
    """Apply an independent depthwise filter to LH, HL, and HH before fusion."""

    def __init__(self, channels):
        super().__init__()
        self.lh = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.hl = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.hh = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, lh, hl, hh):
        high = torch.cat((self.lh(lh), self.hl(hl), self.hh(hh)), dim=1)
        return self.fuse(high)


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
    """DWA-faithful QPA residual: LL carries Q/K/V, LH/HL gate V, HH bypasses."""

    def __init__(self, channels=128, reduced_channels=64, relation_dim=16,
                 mode="torchquantum", entangled=True, alpha_max=0.10,
                 alpha_init=0.02, partial_value=False, rope_2d=False,
                 bucketed_relative_position_bias=False, grouped_relation=False,
                 group_size=4, learned_partial_selection=False,
                 include_hh_in_gate=False, gate_value_before_attention=False,
                 star_value_gate=False, standard_value_gate=False,
                 circuit_variant="baseline", token_pool_factor=2,
                 residual_output=True, separate_band_depthwise_gate=False):
        super().__init__()
        if relation_dim <= 0 or relation_dim > reduced_channels:
            raise ValueError("relation_dim must be in [1, reduced_channels]")
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must be between zero and alpha_max")
        if mode not in {"classical", "torchquantum"}:
            raise ValueError(f"Unsupported directional QPA mode: {mode}")
        if token_pool_factor not in {1, 2}:
            raise ValueError("token_pool_factor must be 1 or 2")
        if rope_2d and relation_dim % 4:
            raise ValueError("2D RoPE requires relation_dim divisible by four")
        if grouped_relation and (relation_dim % group_size):
            raise ValueError("grouped relation_dim must be divisible by group_size")
        self.channels = channels
        self.reduced_channels = reduced_channels
        self.relation_dim = relation_dim
        self.partial_value = partial_value
        self.learned_partial_selection = learned_partial_selection
        self.include_hh_in_gate = include_hh_in_gate
        self.gate_value_before_attention = gate_value_before_attention
        self.star_value_gate = star_value_gate
        self.standard_value_gate = standard_value_gate
        self.rope_2d = rope_2d
        self.bucketed_relative_position_bias = bucketed_relative_position_bias
        self.grouped_relation = grouped_relation
        self.group_size = group_size
        self.mode = mode
        self.circuit_variant = circuit_variant
        self.token_pool_factor = token_pool_factor
        self.residual_output = residual_output
        self.separate_band_depthwise_gate = separate_band_depthwise_gate
        self.reduce = nn.Conv2d(channels, reduced_channels, 1, bias=False)
        high_gate_input_channels = reduced_channels * (3 if include_hh_in_gate else 2)
        if star_value_gate and standard_value_gate:
            raise ValueError("star_value_gate and standard_value_gate are mutually exclusive")
        if separate_band_depthwise_gate:
            if not include_hh_in_gate:
                raise ValueError("Separate band depthwise gate requires HH in the gate")
            self.high_gate = SeparateBandDepthwiseGate(reduced_channels)
        elif star_value_gate:
            if include_hh_in_gate:
                raise ValueError("Star value gate screening currently uses LH/HL only")
            self.high_gate = StarHighFrequencyGate(reduced_channels)
        elif standard_value_gate:
            if include_hh_in_gate:
                raise ValueError("Standard value gate screening currently uses LH/HL only")
            self.high_gate = StandardHighFrequencyGate(reduced_channels)
        else:
            self.high_gate = nn.Sequential(
                nn.Conv2d(high_gate_input_channels, high_gate_input_channels, 3,
                          padding=1, groups=high_gate_input_channels, bias=False),
                nn.GELU(),
                nn.Conv2d(high_gate_input_channels, reduced_channels, 1, bias=False),
                nn.Sigmoid(),
            )
        self.qkv = nn.Conv2d(reduced_channels, reduced_channels * 3, 1, bias=False)
        self.output_projection = nn.Conv2d(reduced_channels, reduced_channels, 1, bias=False)
        self.scorer = (
            ClassicalQPAScorer() if mode == "classical"
            else TorchQuantumQPAScorer(
                entangled=entangled, circuit_variant=circuit_variant
            )
        )
        if learned_partial_selection:
            if not partial_value or reduced_channels != 64:
                raise ValueError(
                    "learned partial selection currently requires partial_value=True "
                    "and reduced_channels=64"
                )
            # Each row softly selects one active relation dimension from the 64D Q/K/V.
            self.partial_selector_logits = nn.Parameter(torch.full((relation_dim, reduced_channels), -4.0))
            with torch.no_grad():
                self.partial_selector_logits.diagonal().fill_(4.0)
        if grouped_relation:
            group_count = relation_dim // group_size
            self.group_q_projection = nn.Linear(group_size, 1, bias=False)
            self.group_k_projection = nn.Linear(group_size, 1, bias=False)
            nn.init.normal_(self.group_q_projection.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.group_k_projection.weight, mean=0.0, std=0.02)
        self.expand = nn.Conv2d(reduced_channels, channels, 1, bias=False)
        self.alpha_max = alpha_max
        self.alpha_logit = nn.Parameter(
            torch.logit(torch.tensor(alpha_init / alpha_max))
        )
        if bucketed_relative_position_bias:
            coordinates = torch.stack(torch.meshgrid(
                torch.arange(4), torch.arange(4), indexing="ij"
            ), dim=-1).flatten(0, 1)
            relative = coordinates[:, None] - coordinates[None, :]
            self.register_buffer("relative_row_bucket", relative_position_bucket(relative[..., 0]))
            self.register_buffer("relative_column_bucket", relative_position_bucket(relative[..., 1]))
            # Starts exactly equivalent to position-free QPA.
            self.relative_position_bias = nn.Parameter(torch.zeros(7, 7))
        for layer in self.high_gate.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        for layer in (self.qkv, self.output_projection, self.expand):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        # Start as a nearly identity path while retaining gradients in QPA.
        nn.init.zeros_(self.output_projection.weight)

    @property
    def alpha(self):
        return self.alpha_max * self.alpha_logit.sigmoid()

    def update_ll(self, ll, lh, hl, hh=None, value_gate=None):
        batch, _, height, width = lh.shape
        if value_gate is not None:
            block_gate = value_gate
        else:
            high_inputs = (lh, hl, hh) if self.include_hh_in_gate and hh is not None else (lh, hl)
            if self.separate_band_depthwise_gate:
                block_gate = self.high_gate(lh, hl, hh)
            else:
                block_gate = self.high_gate(torch.cat(high_inputs, dim=1))
        q, k, v = self.qkv(ll).chunk(3, dim=1)
        if self.token_pool_factor == 2:
            block_gate = F.avg_pool2d(block_gate, 2)
            q, k, v = (F.avg_pool2d(tensor, 2) for tensor in (q, k, v))
        if self.gate_value_before_attention:
            # DWA-style placement: gate V before token relation aggregation.
            v = v * block_gate
        token_height, token_width = q.shape[-2:]
        tokens = token_height * token_width
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        if self.learned_partial_selection:
            selector = self.partial_selector_logits.softmax(dim=-1)
            q_active = torch.einsum("bnd,rd->bnr", q, selector)
            k_active = torch.einsum("bnd,rd->bnr", k, selector)
            v_active = torch.einsum("bnd,rd->bnr", v, selector)
        else:
            q_active = q[..., :self.relation_dim]
            k_active = k[..., :self.relation_dim]
        if self.grouped_relation:
            batch_size, token_count, _ = q_active.shape
            group_count = self.relation_dim // self.group_size
            q_groups = q_active.reshape(batch_size, token_count, group_count, self.group_size)
            k_groups = k_active.reshape(batch_size, token_count, group_count, self.group_size)
            q_active = self.group_q_projection(q_groups).squeeze(-1)
            k_active = self.group_k_projection(k_groups).squeeze(-1)
        if self.rope_2d:
            q_active = apply_2d_rope(q_active, token_height, token_width)
            k_active = apply_2d_rope(k_active, token_height, token_width)
        pair_scores = self.scorer(
            q_active.unsqueeze(2), k_active.unsqueeze(1)
        ).mean(dim=-1)
        if self.bucketed_relative_position_bias:
            if tokens != 16:
                raise ValueError("Bucketed position bias is configured for the 4x4 token grid")
            pair_scores = pair_scores + self.relative_position_bias[
                self.relative_row_bucket, self.relative_column_bucket
            ].unsqueeze(0)
        weights = pair_scores.softmax(dim=-1)
        if self.partial_value:
            # Partial attention: only active relation channels exchange
            # information across tokens; the remaining value channels bypass.
            attended_active = weights @ (v_active if self.learned_partial_selection else v[..., :self.relation_dim])
            bypass = v[..., self.relation_dim:]
            context = torch.cat((attended_active, bypass), dim=-1)
        else:
            context = weights @ v
        side = int(tokens ** 0.5)
        context = context.transpose(1, 2).reshape(batch, self.reduced_channels, side, side)
        base = F.avg_pool2d(ll, self.token_pool_factor) if self.token_pool_factor == 2 else ll
        delta = F.interpolate(self.output_projection(context - base),
                              size=(height, width), mode="nearest")
        if self.gate_value_before_attention:
            updated_ll = ll + delta
        else:
            block_gate = F.interpolate(block_gate, size=(height, width), mode="nearest")
            updated_ll = ll + block_gate * delta
        return updated_ll

    def forward(self, x):
        reduced = self.reduce(x)
        ll, lh, hl, hh = haar_dwt2(reduced)
        updated_ll = self.update_ll(ll, lh, hl, hh)
        reconstruction = haar_idwt2(updated_ll, lh, hl, hh)
        transformed = self.expand(reconstruction)
        if not self.residual_output:
            return transformed
        return x + self.alpha * transformed


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
