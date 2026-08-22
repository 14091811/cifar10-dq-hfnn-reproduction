"""Single local self-proliferation-and-attention blocks for CIFAR features."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frequency_qpa import (
    ChannelWiseFrequencyQPA,
    DirectionalFrequencyQPAResidual,
    FrequencyQPAResidual,
    haar_dwt2,
    haar_idwt2,
)
from .model import Block, ClassicalBranch


class GCStyleAttention(nn.Module):
    """Lightweight global-context attention used as the Yang-style control."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 8)
        self.score = nn.Conv2d(channels, 1, 1, bias=False)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.LayerNorm([hidden, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        weights = self.score(x).flatten(2).softmax(dim=-1)
        values = x.flatten(2).transpose(1, 2)
        context = torch.bmm(weights, values).transpose(1, 2)
        context = context.reshape(batch, channels, 1, 1)
        return x + self.transform(context)


class YangSPABlock(nn.Module):
    """One Yang-style SP&A block with a pluggable attention operator."""

    def __init__(self, channels=128, attention="gc", qpa_mode="classical"):
        super().__init__()
        if attention not in {"gc", "dwt_qpa"}:
            raise ValueError(f"Unsupported local attention: {attention}")

        expanded = channels * 2
        self.mapping = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.expansion = nn.Sequential(
            nn.Conv2d(channels, expanded, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(expanded),
            nn.ReLU(inplace=True),
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded, bias=False),
            nn.BatchNorm2d(expanded),
        )
        if attention == "gc":
            self.attention = GCStyleAttention(expanded)
        else:
            self.attention = FrequencyQPAResidual(
                channels=expanded,
                reduced_channels=8,
                mode=qpa_mode,
                entangled=qpa_mode == "torchquantum",
                alpha_max=0.10,
                alpha_init=0.02,
            )
        self.compression = nn.Sequential(
            nn.Conv2d(expanded, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        y = self.mapping(x)
        y = self.expansion(y)
        y = self.dwconv(y)
        y = self.attention(y)
        y = self.compression(y)
        return F.relu(x + y, inplace=True)


class LocalSPACNN(nn.Module):
    """Classical CNN with one local Yang-style or DWT-QPA block."""

    has_quantum_auxiliary = False

    def __init__(self, num_classes=10, hidden_dim=256, attention="none", qpa_mode="classical"):
        super().__init__()
        self.classical = ClassicalBranch(hidden_dim)
        if attention != "none":
            self.classical.features[3] = YangSPABlock(
                channels=128,
                attention=attention,
                qpa_mode=qpa_mode,
            )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.classical(x))


class ClassicalCNN(LocalSPACNN):
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, attention="none")


class YangSPACNN(LocalSPACNN):
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, attention="gc")


class DWTClassicalSPACNN(LocalSPACNN):
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, attention="dwt_qpa", qpa_mode="classical")


class DWTQPASPACNN(LocalSPACNN):
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, attention="dwt_qpa", qpa_mode="torchquantum")


class DirectionalDWTQPACNN(nn.Module):
    """CNN with a standalone DWT-QPA residual at the 128-channel tap."""

    has_quantum_auxiliary = False

    def __init__(self, num_classes=10, hidden_dim=256, mode="torchquantum",
                 relation_dim=16):
        super().__init__()
        self.classical = ClassicalBranch(hidden_dim)
        self.classical.frequency_tap_index = 3
        self.classical.frequency_modulator = DirectionalFrequencyQPAResidual(
            channels=128,
            reduced_channels=64,
            relation_dim=relation_dim,
            mode=mode,
            entangled=mode == "torchquantum",
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.classical(x))


class DirectionalDWTClassicalCNN(DirectionalDWTQPACNN):
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="classical")


class DirectionalDWTQuantumCNN(DirectionalDWTQPACNN):
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum")


class DirectionalDWTClassicalD8CNN(DirectionalDWTQPACNN):
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="classical", relation_dim=8)


class DirectionalDWTQuantumD8CNN(DirectionalDWTQPACNN):
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=8)


class CompactClassicalBranch(nn.Module):
    """Capacity-controlled CIFAR backbone with a 128x16x16 frequency tap."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.features = nn.Sequential(
            Block(3, 32),
            nn.MaxPool2d(2),
            Block(32, 64),
            Block(64, 128),
            nn.MaxPool2d(2),
            Block(128, 128),
            nn.MaxPool2d(2),
        )
        self.project = nn.Sequential(nn.Dropout(0.25), nn.Linear(128 * 4 * 4, hidden_dim))
        self.frequency_modulator = None
        self.frequency_tap_index = 3

    def forward(self, x):
        for index, layer in enumerate(self.features):
            x = layer(x)
            if self.frequency_modulator is not None and index == self.frequency_tap_index:
                x = self.frequency_modulator(x)
        return self.project(x.flatten(1))


class CompactDirectionalDWTQPACNN(nn.Module):
    """Compact paired control for binary small-sample DWT-QPA experiments."""

    has_quantum_auxiliary = False

    def __init__(self, num_classes=2, hidden_dim=128, mode=None, relation_dim=8):
        super().__init__()
        self.classical = CompactClassicalBranch(hidden_dim)
        if mode is not None:
            self.classical.frequency_modulator = DirectionalFrequencyQPAResidual(
                channels=128,
                reduced_channels=64,
                relation_dim=relation_dim,
                mode=mode,
                entangled=mode == "torchquantum",
            )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.classical(x))


class CompactClassicalCNN(CompactDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)


class CompactDirectionalDWTClassicalD8CNN(CompactDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="classical", relation_dim=8)


class CompactDirectionalDWTQuantumD8CNN(CompactDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="torchquantum", relation_dim=8)


class TwoBlockBinaryBranch(nn.Module):
    """Two residual blocks only, with a 128x16x16 DWT-QPA insertion point."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block1 = Block(64, 128)
        self.block2 = Block(128, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.frequency_modulator = None

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        if self.frequency_modulator is not None:
            x = self.frequency_modulator(x)
        x = self.block2(x)
        return self.pool(x).flatten(1)


class PartialChannelDWTQPA(nn.Module):
    """Non-residual partial-channel DWT-QPA layer between two ResBlocks."""

    def __init__(self, mode="torchquantum"):
        super().__init__()
        self.reduce = nn.Conv2d(128, 64, 1, bias=False)
        self.active_qpa = DirectionalFrequencyQPAResidual(
            channels=16,
            reduced_channels=16,
            relation_dim=16,
            mode=mode,
            entangled=mode == "torchquantum",
            gate_value_before_attention=True,
            residual_output=False,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        reduced = self.reduce(x)
        active, bypass = reduced[:, :16], reduced[:, 16:]
        active = self.active_qpa(active)
        return self.fuse(torch.cat((active, bypass), dim=1))


class ChannelWiseDWTQPA(nn.Module):
    """Non-residual DWT layer whose QPA tokens are feature channels."""

    def __init__(self, mode="torchquantum", heads=4):
        super().__init__()
        self.reduce = nn.Conv2d(128, 64, 1, bias=False)
        self.attention = ChannelWiseFrequencyQPA(
            channels=64, heads=heads, mode=mode, entangled=mode == "torchquantum"
        )
        self.expand = nn.Conv2d(64, 128, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        reduced = self.reduce(x)
        ll, lh, hl, hh = haar_dwt2(reduced)
        ll = self.attention(ll, lh, hl)
        return self.fuse(self.expand(haar_idwt2(ll, lh, hl, hh)))


class FullChannelDWTQPA(nn.Module):
    """Non-residual layer that sends all 64 reduced channels through DWT-QPA."""

    def __init__(self, mode="torchquantum"):
        super().__init__()
        self.reduce = nn.Conv2d(128, 64, 1, bias=False)
        self.dwt_qpa = DirectionalFrequencyQPAResidual(
            channels=64,
            reduced_channels=64,
            relation_dim=16,
            mode=mode,
            entangled=mode == "torchquantum",
            partial_value=True,
            learned_partial_selection=True,
            gate_value_before_attention=True,
            residual_output=False,
        )
        self.expand = nn.Conv2d(64, 128, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fuse(self.expand(self.dwt_qpa(self.reduce(x))))


class MWHLFusedFullChannelDWTQPA(nn.Module):
    """Non-residual full-channel DWT-QPA with an LL/original-spatial fusion."""

    def __init__(self, mode="torchquantum"):
        super().__init__()
        self.reduce = nn.Conv2d(128, 64, 1, bias=False)
        self.ll_fusion = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.dwt_qpa = DirectionalFrequencyQPAResidual(
            channels=64,
            reduced_channels=64,
            relation_dim=16,
            mode=mode,
            entangled=mode == "torchquantum",
            partial_value=True,
            learned_partial_selection=True,
            gate_value_before_attention=True,
            residual_output=False,
        )
        self.expand = nn.Conv2d(64, 128, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        reduced = self.reduce(x)
        ll, lh, hl, hh = haar_dwt2(reduced)
        spatial = F.max_pool2d(reduced, 2)
        fused_ll = self.ll_fusion(torch.cat((ll, spatial), dim=1))
        updated_ll = self.dwt_qpa.update_ll(fused_ll, lh, hl, hh)
        reconstructed = haar_idwt2(updated_ll, lh, hl, hh)
        return self.fuse(self.expand(reconstructed))


class DynamicBlockDownsample(nn.Module):
    """Learned four-block spatial downsampling from the DBDM design."""

    def __init__(self, in_channels=128, out_channels=128):
        super().__init__()
        self.offset_conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.offset_conv2 = nn.Conv2d(in_channels, 8, 3, stride=2, padding=1)
        self.block_conv = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.final_conv = nn.Conv2d(out_channels, out_channels, 1)

    @staticmethod
    def _base_grid(height, width, device, dtype):
        out_h, out_w = height // 2, width // 2
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, out_h, device=device, dtype=dtype),
            torch.linspace(-1, 1, out_w, device=device, dtype=dtype),
            indexing="ij",
        )
        base = torch.stack((x, y), dim=-1).unsqueeze(0)
        offsets = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]],
            device=device, dtype=dtype,
        ).view(4, 1, 1, 2)
        return base * 0.5 + offsets

    def forward(self, x):
        batch, _, height, width = x.shape
        offsets = F.relu(self.offset_conv1(x))
        offsets = self.offset_conv2(offsets)
        offsets = offsets.view(batch, 4, 2, height // 2, width // 2).permute(0, 1, 3, 4, 2)
        grid = self._base_grid(height, width, x.device, x.dtype) + offsets
        grid = grid.clamp(-1, 1)
        blocks = [
            F.grid_sample(x, grid[:, index], align_corners=True)
            for index in range(4)
        ]
        dynamic = torch.cat([self.block_conv(block) for block in blocks], dim=1)
        return self.final_conv(dynamic + self.residual_conv(x))


class DWTQPADBDMDownsample(nn.Module):
    """DWT-QPA feature processing followed by learned spatial downsampling."""

    def __init__(self, mode="torchquantum"):
        super().__init__()
        self.qpa = DirectionalFrequencyQPAResidual(
            channels=128,
            reduced_channels=64,
            relation_dim=16,
            mode=mode,
            entangled=mode == "torchquantum",
            partial_value=True,
            learned_partial_selection=True,
            gate_value_before_attention=True,
            residual_output=False,
        )
        self.downsample = DynamicBlockDownsample(128, 128)

    def forward(self, x):
        return self.downsample(self.qpa(x))


class FrequencyPoMGate(nn.Module):
    """Use PoM-style polynomial context mixing to produce a spatial V gate."""

    def __init__(self, channels=64, degree=2):
        super().__init__()
        self.degree = degree
        self.context_proj = nn.Linear(channels, channels, bias=False)
        self.query_proj = nn.Linear(channels, channels, bias=False)
        self.coeff = nn.Parameter(torch.zeros(channels, degree))
        self.gate_proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, ll, lh, hl, hh):
        batch, channels, height, width = ll.shape
        query = ll.flatten(2).transpose(1, 2)
        context = torch.cat((lh, hl, hh), dim=2).flatten(2).transpose(1, 2)
        context = self.context_proj(context)
        activated = torch.clamp(F.leaky_relu(context, 0.01), -0.1, 6.0)
        powers = []
        power = activated
        for _ in range(self.degree):
            powers.append(power)
            power = power * activated
        polynomial = torch.stack(powers, dim=-1)
        aggregated = (polynomial * self.coeff.view(1, 1, channels, self.degree)).sum(dim=-1)
        aggregated = aggregated.mean(dim=1, keepdim=True)
        selection = F.hardsigmoid(self.query_proj(query))
        gated = selection * aggregated
        gated = gated.transpose(1, 2).reshape(batch, channels, height, width)
        return torch.sigmoid(self.gate_proj(gated))


class DWTQPAFrequencyPoMDownsample(nn.Module):
    """DWT downsampling with LL-QPA and PoM-generated high-frequency V gates."""

    def __init__(self, mode="torchquantum"):
        super().__init__()
        self.reduce = nn.Conv2d(128, 64, 1, bias=False)
        self.high_gate = FrequencyPoMGate(channels=64, degree=2)
        self.qpa = DirectionalFrequencyQPAResidual(
            channels=64,
            reduced_channels=64,
            relation_dim=16,
            mode=mode,
            entangled=mode == "torchquantum",
            partial_value=True,
            learned_partial_selection=True,
            gate_value_before_attention=True,
            residual_output=False,
        )
        self.expand = nn.Conv2d(64, 128, 1, bias=False)
        self.output = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(inplace=True))

    def forward(self, x):
        reduced = self.reduce(x)
        ll, lh, hl, hh = haar_dwt2(reduced)
        value_gate = self.high_gate(ll, lh, hl, hh)
        updated_ll = self.qpa.update_ll(ll, lh, hl, hh, value_gate=value_gate)
        updated_ll = self.qpa.expand(updated_ll)
        return self.output(self.expand(updated_ll))


class MWHLQPADownsample(nn.Module):
    """MWHL-style DWT downsampler with QPA and no IDWT or outer residual."""

    def __init__(self, mode="torchquantum"):
        super().__init__()
        self.reduce = nn.Conv2d(128, 64, 1, bias=False)
        self.ll_fusion = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.qpa = DirectionalFrequencyQPAResidual(
            channels=64,
            reduced_channels=64,
            relation_dim=16,
            mode=mode,
            entangled=mode == "torchquantum",
            partial_value=True,
            learned_partial_selection=True,
            gate_value_before_attention=True,
            residual_output=False,
        )
        self.expand = nn.Conv2d(64, 128, 1, bias=False)
        self.output = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        reduced = self.reduce(x)
        ll, lh, hl, hh = haar_dwt2(reduced)
        spatial = F.max_pool2d(reduced, 2)
        fused_ll = self.ll_fusion(torch.cat((ll, spatial), dim=1))
        updated_ll = self.qpa.update_ll(fused_ll, lh, hl, hh)
        updated_ll = self.qpa.expand(updated_ll)
        return self.output(self.expand(updated_ll))


class SeparateBandDepthwiseMWHLDownsample(nn.Module):
    """MWHL downsampling with separate LH/HL/HH depthwise value gating."""

    def __init__(self, mode="torchquantum"):
        super().__init__()
        self.reduce = nn.Conv2d(128, 64, 1, bias=False)
        self.ll_fusion = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.qpa = DirectionalFrequencyQPAResidual(
            channels=64,
            reduced_channels=64,
            relation_dim=16,
            mode=mode,
            entangled=mode == "torchquantum",
            partial_value=True,
            learned_partial_selection=True,
            gate_value_before_attention=True,
            include_hh_in_gate=True,
            separate_band_depthwise_gate=True,
            residual_output=False,
        )
        self.expand = nn.Conv2d(64, 128, 1, bias=False)
        self.output = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(inplace=True))

    def forward(self, x):
        reduced = self.reduce(x)
        ll, lh, hl, hh = haar_dwt2(reduced)
        spatial = F.max_pool2d(reduced, 2)
        fused_ll = self.ll_fusion(torch.cat((ll, spatial), dim=1))
        updated_ll = self.qpa.update_ll(fused_ll, lh, hl, hh)
        updated_ll = self.qpa.expand(updated_ll)
        return self.output(self.expand(updated_ll))


class HighFrequencyQPADownsample(nn.Module):
    """DWT downsampler that explicitly fuses LL with learned high-frequency features.

    Unlike MWHL, the second input to the fusion layer is not max-pooled spatial
    input.  LH, HL, and HH are projected at the native DWT resolution and are
    fused with LL before QPA.  The directional bands are still passed to QPA
    so its value gate can use the same frequency information.
    """

    def __init__(self, mode="torchquantum"):
        super().__init__()
        self.reduce = nn.Conv2d(128, 64, 1, bias=False)
        self.high_fusion = nn.Sequential(
            nn.Conv2d(192, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.ll_fusion = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.qpa = DirectionalFrequencyQPAResidual(
            channels=64,
            reduced_channels=64,
            relation_dim=16,
            mode=mode,
            entangled=mode == "torchquantum",
            partial_value=True,
            learned_partial_selection=True,
            gate_value_before_attention=True,
            residual_output=False,
        )
        self.expand = nn.Conv2d(64, 128, 1, bias=False)
        self.output = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(inplace=True))

    def forward(self, x):
        reduced = self.reduce(x)
        ll, lh, hl, hh = haar_dwt2(reduced)
        high = self.high_fusion(torch.cat((lh, hl, hh), dim=1))
        fused_ll = self.ll_fusion(torch.cat((ll, high), dim=1))
        updated_ll = self.qpa.update_ll(fused_ll, lh, hl, hh)
        updated_ll = self.qpa.expand(updated_ll)
        return self.output(self.expand(updated_ll))


class TwoBlockDirectionalDWTQPACNN(nn.Module):
    """Paired two-block controls for the CIFAR binary small-sample study."""

    has_quantum_auxiliary = False

    def __init__(self, num_classes=2, hidden_dim=128, mode=None, relation_dim=8,
                 partial_value=False, rope_2d=False, bucketed_relative_position_bias=False,
                 grouped_relation=False, group_size=4, learned_partial_selection=False,
                 include_hh_in_gate=False, gate_value_before_attention=False,
                 star_value_gate=False, standard_value_gate=False,
                 circuit_variant="baseline", token_pool_factor=2):
        super().__init__()
        if hidden_dim != 128:
            raise ValueError("TwoBlockDirectionalDWTQPACNN uses a fixed 128D GAP feature")
        self.classical = TwoBlockBinaryBranch()
        if mode is not None:
            self.classical.frequency_modulator = DirectionalFrequencyQPAResidual(
                channels=128,
                reduced_channels=64,
                relation_dim=relation_dim,
                mode=mode,
                entangled=mode == "torchquantum",
                partial_value=partial_value,
                rope_2d=rope_2d,
                bucketed_relative_position_bias=bucketed_relative_position_bias,
                grouped_relation=grouped_relation,
                group_size=group_size,
                learned_partial_selection=learned_partial_selection,
                include_hh_in_gate=include_hh_in_gate,
                gate_value_before_attention=gate_value_before_attention,
                star_value_gate=star_value_gate,
                standard_value_gate=standard_value_gate,
                circuit_variant=circuit_variant,
                token_pool_factor=token_pool_factor,
            )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.classical(x))


class TwoBlockClassicalCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)


class TwoBlockPartialChannelDWTClassicalCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = PartialChannelDWTQPA(mode="classical")


class TwoBlockPartialChannelDWTQuantumCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = PartialChannelDWTQPA(mode="torchquantum")


class TwoBlockChannelWiseDWTClassicalCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = ChannelWiseDWTQPA(mode="classical")


class TwoBlockChannelWiseDWTQuantumCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = ChannelWiseDWTQPA(mode="torchquantum")


class TwoBlockChannelWiseDWTClassicalOneHeadCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = ChannelWiseDWTQPA(mode="classical", heads=1)


class TwoBlockChannelWiseDWTQuantumOneHeadCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = ChannelWiseDWTQPA(mode="torchquantum", heads=1)


class TwoBlockFullChannelDWTClassicalCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = FullChannelDWTQPA(mode="classical")


class TwoBlockFullChannelDWTQuantumCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = FullChannelDWTQPA(mode="torchquantum")


class TwoBlockMWHLFusedFullChannelDWTClassicalCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = MWHLFusedFullChannelDWTQPA(mode="classical")


class TwoBlockMWHLFusedFullChannelDWTQuantumCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = MWHLFusedFullChannelDWTQPA(mode="torchquantum")


class TwoBlockMWHLDownsampleClassicalCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = MWHLQPADownsample(mode="classical")


class TwoBlockMWHLDownsampleQuantumCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = MWHLQPADownsample(mode="torchquantum")


class TwoBlockSeparateBandDepthwiseMWHLClassicalCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = SeparateBandDepthwiseMWHLDownsample(mode="classical")


class TwoBlockSeparateBandDepthwiseMWHLQuantumCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = SeparateBandDepthwiseMWHLDownsample(mode="torchquantum")


class TwoBlockHighFrequencyDownsampleClassicalCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = HighFrequencyQPADownsample(mode="classical")


class TwoBlockHighFrequencyDownsampleQuantumCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = HighFrequencyQPADownsample(mode="torchquantum")


class TwoBlockDBDMClassicalCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = DynamicBlockDownsample(128, 128)


class TwoBlockDWTQPADBDMClassicalCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = DWTQPADBDMDownsample(mode="classical")


class TwoBlockDWTQPADBDMQuantumCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = DWTQPADBDMDownsample(mode="torchquantum")


class TwoBlockDWTQPAFrequencyPoMClassicalCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = DWTQPAFrequencyPoMDownsample(mode="classical")


class TwoBlockDWTQPAFrequencyPoMQuantumCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode=None)
        self.classical.frequency_modulator = DWTQPAFrequencyPoMDownsample(mode="torchquantum")


class TwoBlockDirectionalDWTClassicalD8CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="classical", relation_dim=8)


class TwoBlockDirectionalDWTQuantumD8CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="torchquantum", relation_dim=8)


class TwoBlockDirectionalDWTClassicalD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="classical", relation_dim=16)


class TwoBlockDirectionalDWTQuantumD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="torchquantum", relation_dim=16)


class TwoBlockDirectionalDWTClassicalHHGateD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="classical", relation_dim=16, include_hh_in_gate=True)


class TwoBlockDirectionalDWTQuantumHHGateD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=16, include_hh_in_gate=True)


class TwoBlockDirectionalDWTClassicalValueGateD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="classical", relation_dim=16,
                         gate_value_before_attention=True)


class TwoBlockDirectionalDWTQuantumValueGateD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=16,
                         gate_value_before_attention=True)


class TwoBlockDirectionalDWTClassicalStarValueGateD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="classical", relation_dim=16,
                         gate_value_before_attention=True, star_value_gate=True)


class TwoBlockDirectionalDWTQuantumStarValueGateD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=16,
                         gate_value_before_attention=True, star_value_gate=True)


class TwoBlockDirectionalDWTClassicalStandardValueGateD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="classical", relation_dim=16,
                         gate_value_before_attention=True, standard_value_gate=True)


class TwoBlockDirectionalDWTQuantumStandardValueGateD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=16,
                         gate_value_before_attention=True, standard_value_gate=True)


class TwoBlockDirectionalDWTClassicalLearnedPartialD8CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="classical", relation_dim=8, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True)


class TwoBlockDirectionalDWTQuantumLearnedPartialD8CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=8, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True)


class TwoBlockDirectionalDWTClassicalLearnedPartialD32CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="classical", relation_dim=32, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True)


class TwoBlockDirectionalDWTQuantumLearnedPartialD32CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=32, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True)


class TwoBlockDirectionalDWTClassicalStandardLearnedPartialD8CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="classical", relation_dim=8, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True,
                         standard_value_gate=True)


class TwoBlockDirectionalDWTQuantumStandardLearnedPartialD8CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=8, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True,
                         standard_value_gate=True)


class TwoBlockDirectionalDWTClassicalStandardLearnedPartialD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="classical", relation_dim=16, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True,
                         standard_value_gate=True)


class TwoBlockDirectionalDWTQuantumStandardLearnedPartialD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=16, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True,
                         standard_value_gate=True)


class TwoBlockDirectionalDWTClassicalStandardLearnedPartialD32CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="classical", relation_dim=32, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True,
                         standard_value_gate=True)


class TwoBlockDirectionalDWTQuantumStandardLearnedPartialD32CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=32, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True,
                         standard_value_gate=True)


class TwoBlockDirectionalDWTClassicalD24CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="classical", relation_dim=24)


class TwoBlockDirectionalDWTQuantumD24CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="torchquantum", relation_dim=24)


class TwoBlockDirectionalDWTClassicalD64CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="classical", relation_dim=64)


class TwoBlockDirectionalDWTQuantumD64CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="torchquantum", relation_dim=64)


class TwoBlockDirectionalDWTClassicalGroupedD64CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="classical",
                         relation_dim=64, grouped_relation=True, group_size=4)


class TwoBlockDirectionalDWTQuantumGroupedD64CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="torchquantum",
                         relation_dim=64, grouped_relation=True, group_size=4)


class TwoBlockDirectionalDWTClassicalPartialD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="classical",
                 relation_dim=16, partial_value=True)


class TwoBlockDirectionalDWTQuantumPartialD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="torchquantum",
                         relation_dim=16, partial_value=True)


class TwoBlockDirectionalDWTClassicalLearnedPartialD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="classical",
                         relation_dim=16, partial_value=True,
                         learned_partial_selection=True)


class TwoBlockDirectionalDWTQuantumLearnedPartialD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="torchquantum",
                         relation_dim=16, partial_value=True,
                         learned_partial_selection=True)


class TwoBlockDirectionalDWTQuantumLearnedPartialD16BaselineCircuitCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=16, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True,
                         circuit_variant="baseline")


class TwoBlockDirectionalDWTQuantumLearnedPartialD16NoEntanglementCircuitCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=16, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True,
                         circuit_variant="no_entanglement")


class TwoBlockDirectionalDWTQuantumLearnedPartialD16RZCircuitCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=16, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True,
                         circuit_variant="rz_entangled")


class TwoBlockDirectionalDWTQuantumLearnedPartialD16SingleCNOTCircuitCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=16, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True,
                         circuit_variant="single_cnot")


class TwoBlockDirectionalDWTQuantumLearnedPartialD16SymmetricRYCircuitCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=16, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True,
                         circuit_variant="symmetric_ry")


class TwoBlockDirectionalDWTQuantumLearnedPartialD16RZSingleCNOTCircuitCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim,
                         mode="torchquantum", relation_dim=16, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True,
                         circuit_variant="rz_single_cnot")


class TwoBlockDirectionalDWTClassicalLearnedPartialD16HighResTokenCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="classical",
                         relation_dim=16, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True,
                         token_pool_factor=1)


class TwoBlockDirectionalDWTQuantumLearnedPartialD16HighResTokenCNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="torchquantum",
                         relation_dim=16, partial_value=True,
                         learned_partial_selection=True, gate_value_before_attention=True,
                         token_pool_factor=1)


class TwoBlockDirectionalDWTClassicalRoPED16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="classical",
                         relation_dim=16, rope_2d=True)


class TwoBlockDirectionalDWTQuantumRoPED16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="torchquantum",
                         relation_dim=16, rope_2d=True)


class TwoBlockDirectionalDWTClassicalBucketD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="classical",
                         relation_dim=16, bucketed_relative_position_bias=True)


class TwoBlockDirectionalDWTQuantumBucketD16CNN(TwoBlockDirectionalDWTQPACNN):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__(num_classes=num_classes, hidden_dim=hidden_dim, mode="torchquantum",
                         relation_dim=16, bucketed_relative_position_bias=True)
