import math

import torch
import torch.nn as nn


class GridPairSampler(nn.Module):
    """Fixed local pairs plus per-batch cross-region pairs from 32 x 32 RGB."""

    def __init__(self, total_pairs=153, random_ratio=0.3, seed=42, image_size=32, channels=3, layout="compact", evaluation_mode="fixed"):
        super().__init__()
        if layout not in {"compact", "author_cifar"}:
            raise ValueError(f"Unsupported pairing layout: {layout}")
        if evaluation_mode not in {"fixed", "stochastic"}:
            raise ValueError(f"Unsupported evaluation mode: {evaluation_mode}")
        self.total_pairs = total_pairs
        self.random_pairs = round(total_pairs * random_ratio)
        self.fixed_pairs = total_pairs - self.random_pairs
        self.image_size, self.channels = image_size, channels
        self.evaluation_mode = evaluation_mode
        boundaries = (0, 10, 21, 32) if layout == "author_cifar" and image_size == 32 else (0, image_size // 3, math.ceil(2 * image_size / 3), image_size)
        blocks, candidates = [], []
        for channel in range(channels):
            offset = channel * image_size * image_size
            for row in range(3):
                for col in range(3):
                    block = [offset + y * image_size + x for y in range(boundaries[row], boundaries[row + 1]) for x in range(boundaries[col], boundaries[col + 1])]
                    blocks.append(torch.tensor(block, dtype=torch.long))
                    for y in range(boundaries[row], boundaries[row + 1]):
                        for x in range(boundaries[col], boundaries[col + 1]):
                            index = offset + y * image_size + x
                            if x + 1 < boundaries[col + 1]: candidates.append((index, index + 1))
                            if y + 1 < boundaries[row + 1]: candidates.append((index, index + image_size))
        generator = torch.Generator().manual_seed(seed)
        choice = torch.randperm(len(candidates), generator=generator)[:self.fixed_pairs]
        fixed_indices = torch.tensor([candidates[i] for i in choice.tolist()], dtype=torch.long)
        self.register_buffer("fixed_indices", fixed_indices.reshape(-1, 2))
        self.blocks = blocks
        if layout == "author_cifar":
            within_channel = []
            for row in range(3):
                for col in range(2):
                    within_channel.append((row * 3 + col, row * 3 + col + 1))
            for row in range(2):
                for col in range(3):
                    within_channel.append((row * 3 + col, (row + 1) * 3 + col))
            within_channel.extend(((0, 4), (1, 4), (2, 4), (3, 4), (5, 4), (4, 6), (4, 7), (4, 8)))
        else:
            within_channel = ((0, 4), (1, 4), (2, 4), (3, 4), (4, 5), (4, 6), (4, 7), (4, 8))
        self.region_pairs = [(base + source, base + target) for base in range(0, channels * 9, 9) for source, target in within_channel]
        self.register_buffer("evaluation_cross_indices", self._build_evaluation_indices(seed + 1))

    def _build_evaluation_indices(self, seed):
        if not self.random_pairs:
            return torch.empty(0, 2, dtype=torch.long)
        generator = torch.Generator().manual_seed(seed)
        base, extra = divmod(self.random_pairs, len(self.region_pairs))
        chunks = []
        for index, (left, right) in enumerate(self.region_pairs):
            count = base + int(index < extra)
            if count:
                lhs, rhs = self.blocks[left], self.blocks[right]
                chunks.append(torch.stack((lhs[torch.randint(len(lhs), (count,), generator=generator)], rhs[torch.randint(len(rhs), (count,), generator=generator)]), dim=-1))
        return torch.cat(chunks, dim=0)

    def _cross_indices(self, batch, device):
        if not self.random_pairs:
            return torch.empty(batch, 0, 2, dtype=torch.long, device=device)
        base, extra = divmod(self.random_pairs, len(self.region_pairs))
        chunks = []
        for index, (left, right) in enumerate(self.region_pairs):
            count = base + int(index < extra)
            if count:
                lhs, rhs = self.blocks[left].to(device), self.blocks[right].to(device)
                chunks.append(torch.stack((lhs[torch.randint(len(lhs), (batch, count), device=device)], rhs[torch.randint(len(rhs), (batch, count), device=device)]), dim=-1))
        return torch.cat(chunks, dim=1)

    def forward(self, images):
        if tuple(images.shape[1:]) != (self.channels, self.image_size, self.image_size):
            raise ValueError(f"Expected [batch, {self.channels}, {self.image_size}, {self.image_size}], got {tuple(images.shape)}")
        batch, flat = images.shape[0], images.flatten(1)
        fixed = self.fixed_indices.unsqueeze(0).expand(batch, -1, -1)
        cross = self._cross_indices(batch, images.device) if self.training or self.evaluation_mode == "stochastic" else self.evaluation_cross_indices.to(images.device).unsqueeze(0).expand(batch, -1, -1)
        indices = torch.cat((fixed, cross), dim=1)
        return torch.gather(flat.unsqueeze(1).expand(-1, self.total_pairs, -1), 2, indices)
