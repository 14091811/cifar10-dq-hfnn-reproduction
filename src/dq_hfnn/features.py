"""Feature-pair sources for the author baseline and our frequency variant."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pairing import GridPairSampler


class HaarHighFrequencyPairs(nn.Module):
    """Sample DQ inputs from local Haar detail bands instead of raw pixels.

    A 32 x 32 RGB image becomes nine 16 x 16 detail maps: LH, HL, and HH for
    each RGB channel. The pair budget and sampler policy remain unchanged.
    """

    def __init__(self, total_pairs=153, random_ratio=0.3, seed=42, layout="compact", evaluation_mode="fixed"):
        super().__init__()
        self.sampler = GridPairSampler(total_pairs, random_ratio, seed, image_size=16, channels=9, layout=layout, evaluation_mode=evaluation_mode)

    @staticmethod
    def _haar_details(images):
        top_left, top_right = images[:, :, 0::2, 0::2], images[:, :, 0::2, 1::2]
        bottom_left, bottom_right = images[:, :, 1::2, 0::2], images[:, :, 1::2, 1::2]
        lh = (top_left - top_right + bottom_left - bottom_right) * 0.5
        hl = (top_left + top_right - bottom_left - bottom_right) * 0.5
        hh = (top_left - top_right - bottom_left + bottom_right) * 0.5
        return torch.cat((lh, hl, hh), dim=1)

    def forward(self, images):
        return self.sampler(self._haar_details(images))


class FrequencyGuidedPixelPairs(nn.Module):
    """Use Haar detail energy to select original-image pixel pairs."""

    def __init__(self, total_pairs=144, channels=3):
        super().__init__()
        pairs_per_position = channels * 2
        if total_pairs % pairs_per_position != 0:
            raise ValueError(
                f"total_pairs must be divisible by {pairs_per_position}, got {total_pairs}"
            )
        self.total_pairs = total_pairs
        self.channels = channels
        self.selected_positions = total_pairs // pairs_per_position

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != self.channels:
            raise ValueError(
                f"Expected [batch, {self.channels}, height, width], got {tuple(images.shape)}"
            )
        batch, _, height, width = images.shape
        if height % 2 or width % 2:
            raise ValueError("Frequency-guided pixel selection requires even spatial dimensions")

        top_left = images[:, :, 0::2, 0::2]
        top_right = images[:, :, 0::2, 1::2]
        bottom_left = images[:, :, 1::2, 0::2]
        bottom_right = images[:, :, 1::2, 1::2]
        lh = (top_left - top_right + bottom_left - bottom_right) * 0.5
        hl = (top_left + top_right - bottom_left - bottom_right) * 0.5
        hh = (top_left - top_right - bottom_left + bottom_right) * 0.5
        detail_energy = (lh.square() + hl.square() + hh.square()).mean(dim=1)
        available_positions = detail_energy.shape[-2] * detail_energy.shape[-1]
        if self.selected_positions > available_positions:
            raise ValueError(
                f"Requested {self.selected_positions} positions, only {available_positions} available"
            )
        selected = detail_energy.flatten(1).topk(
            self.selected_positions, dim=1, sorted=False
        ).indices

        block_width = width // 2
        rows, columns = selected // block_width, selected % block_width
        upper_left = (2 * rows) * width + 2 * columns
        upper_right = upper_left + 1
        lower_left = upper_left + width
        lower_right = lower_left + 1
        flat = images.flatten(2)

        def gather(indices):
            return torch.gather(
                flat,
                2,
                indices.unsqueeze(1).expand(batch, self.channels, -1),
            )

        main_diagonal = torch.stack(
            (gather(upper_left), gather(lower_right)), dim=-1
        )
        anti_diagonal = torch.stack(
            (gather(upper_right), gather(lower_left)), dim=-1
        )
        pairs = torch.stack((main_diagonal, anti_diagonal), dim=3)
        return pairs.permute(0, 2, 1, 3, 4).reshape(batch, self.total_pairs, 2)


class FuzzyPooledPatchPairs(nn.Module):
    """Compress all image patches into a small grid using fuzzy products."""

    def __init__(self, image_size=32, patch_size=4, channels=3, eps=1e-6):
        super().__init__()
        if image_size % (patch_size * 2) != 0:
            raise ValueError("image_size must be divisible by twice the patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.eps = eps
        self.patch_projection = nn.Linear(channels * patch_size * patch_size, 2)

    @property
    def output_pairs(self):
        pooled_side = self.image_size // (self.patch_size * 2)
        return pooled_side * pooled_side

    def forward(self, images):
        expected = (self.channels, self.image_size, self.image_size)
        if tuple(images.shape[1:]) != expected:
            raise ValueError(f"Expected [batch, {expected[0]}, {expected[1]}, {expected[2]}], got {tuple(images.shape)}")
        patches = F.unfold(
            images,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        memberships = self.patch_projection(patches).sigmoid().clamp_min(self.eps)
        batch = images.shape[0]
        patch_side = self.image_size // self.patch_size
        memberships = memberships.reshape(batch, patch_side, patch_side, 2)
        # Pool each neighboring 2x2 patch group with the DQFNN fuzzy product.
        memberships = memberships.reshape(
            batch,
            patch_side // 2,
            2,
            patch_side // 2,
            2,
            2,
        )
        pooled = memberships.log().mean(dim=(2, 4)).exp()
        return pooled.reshape(batch, self.output_pairs, 2) * torch.pi


class SpatialFuzzyPatchPairs(FuzzyPooledPatchPairs):
    """Use a paper-style nine-element superpixel before quantum compression."""

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        channels=3,
        superpixel_elements=9,
        eps=1e-6,
    ):
        super().__init__(image_size, patch_size, channels, eps)
        patch_features = channels * patch_size * patch_size
        self.patch_projection = nn.Sequential(
            nn.Linear(patch_features, superpixel_elements),
            nn.ReLU(),
            nn.Linear(superpixel_elements, 2),
        )


class DirectionAwareSpatialPatchPairs(FuzzyPooledPatchPairs):
    """Encode regional appearance with horizontal and vertical patch changes."""

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        channels=3,
        superpixel_elements=9,
        eps=1e-6,
    ):
        super().__init__(image_size, patch_size, channels, eps)
        patch_features = channels * patch_size * patch_size
        self.patch_projection = nn.Sequential(
            nn.Linear(patch_features, superpixel_elements),
            nn.ReLU(),
        )
        self.direction_projection = nn.Linear(
            2 * superpixel_elements, 1, bias=False
        )
        self.direction_bias = nn.Parameter(torch.zeros(2))

    @staticmethod
    def _directional_descriptors(features):
        if features.ndim != 4:
            raise ValueError(
                f"Expected [batch, height, width, features], got {tuple(features.shape)}"
            )
        batch, height, width, feature_count = features.shape
        if height % 2 or width % 2:
            raise ValueError("Directional pooling requires even spatial dimensions")
        grouped = features.reshape(
            batch,
            height // 2,
            2,
            width // 2,
            2,
            feature_count,
        )
        mean = grouped.mean(dim=(2, 4))
        horizontal = (
            grouped[:, :, :, :, 1, :].mean(dim=2)
            - grouped[:, :, :, :, 0, :].mean(dim=2)
        )
        vertical = (
            grouped[:, :, 1, :, :, :].mean(dim=3)
            - grouped[:, :, 0, :, :, :].mean(dim=3)
        )
        return mean, horizontal, vertical

    def forward(self, images):
        expected = (self.channels, self.image_size, self.image_size)
        if tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"Expected [batch, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(images.shape)}"
            )
        patches = F.unfold(
            images,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        patch_features = self.patch_projection(patches)
        patch_side = self.image_size // self.patch_size
        patch_features = patch_features.reshape(
            images.shape[0], patch_side, patch_side, -1
        )
        mean, horizontal, vertical = self._directional_descriptors(patch_features)
        directional_inputs = torch.stack(
            (
                torch.cat((mean, horizontal), dim=-1),
                torch.cat((mean, vertical), dim=-1),
            ),
            dim=-2,
        )
        angles = self.direction_projection(directional_inputs).squeeze(-1)
        angles = (angles + self.direction_bias).sigmoid() * torch.pi
        return angles.reshape(images.shape[0], self.output_pairs, 2)


class CNNSpatialFrequencyPairs(nn.Module):
    """Build spatial/texture angle pairs from an intermediate CNN feature map."""

    def __init__(self, in_channels=128, feature_size=16, embedding_dim=9):
        super().__init__()
        if feature_size % 4:
            raise ValueError("feature_size must be divisible by 4")
        self.in_channels = in_channels
        self.feature_size = feature_size
        self.embedding_dim = embedding_dim
        self.feature_projection = nn.Sequential(
            nn.Conv2d(in_channels, embedding_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.spatial_projection = nn.Conv2d(embedding_dim, 1, kernel_size=1)
        self.frequency_projection = nn.Conv2d(embedding_dim, 1, kernel_size=1)

    @property
    def output_pairs(self):
        regions_per_side = self.feature_size // 4
        return regions_per_side * regions_per_side

    @staticmethod
    def _haar_high_energy(features):
        top_left = features[:, :, 0::2, 0::2]
        top_right = features[:, :, 0::2, 1::2]
        bottom_left = features[:, :, 1::2, 0::2]
        bottom_right = features[:, :, 1::2, 1::2]
        lh = (top_left - top_right + bottom_left - bottom_right) * 0.5
        hl = (top_left + top_right - bottom_left - bottom_right) * 0.5
        hh = (top_left - top_right - bottom_left + bottom_right) * 0.5
        return torch.sqrt((lh.square() + hl.square() + hh.square()) / 3.0 + 1e-8)

    def forward(self, features):
        expected = (self.in_channels, self.feature_size, self.feature_size)
        if tuple(features.shape[1:]) != expected:
            raise ValueError(
                f"Expected [batch, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(features.shape)}"
            )
        embedded = self.feature_projection(features)
        spatial = F.avg_pool2d(embedded, kernel_size=4, stride=4)
        high_energy = self._haar_high_energy(embedded)
        high_energy = F.avg_pool2d(high_energy, kernel_size=2, stride=2)
        spatial_angles = self.spatial_projection(spatial)
        frequency_angles = self.frequency_projection(torch.log1p(high_energy))
        angles = torch.cat((spatial_angles, frequency_angles), dim=1)
        angles = angles.sigmoid() * torch.pi
        return angles.permute(0, 2, 3, 1).reshape(
            features.shape[0], self.output_pairs, 2
        )
