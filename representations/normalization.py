"""Training-only latent statistics and reversible channel normalization."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite, sqrt
from typing import Iterable, Tuple

import torch


@dataclass(frozen=True)
class NormalizationStats:
    """Frozen per-channel statistics fitted from one training data version."""

    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    training_data_version: str
    sample_count: int
    value_count_per_channel: int
    training_split: str = "train"

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", tuple(self.mean))
        object.__setattr__(self, "std", tuple(self.std))
        if self.training_split != "train":
            raise ValueError("normalization statistics must come from the train split")
        if not self.training_data_version:
            raise ValueError("training_data_version must be explicit")
        if not self.mean or len(self.mean) != len(self.std):
            raise ValueError("normalization mean/std must be non-empty and aligned")
        if not all(isfinite(value) for value in (*self.mean, *self.std)):
            raise ValueError("normalization statistics must be finite")
        if any(value <= 0 for value in self.std):
            raise ValueError("normalization standard deviations must be positive")
        if self.sample_count <= 0 or self.value_count_per_channel <= 0:
            raise ValueError("normalization counts must be positive")

    @property
    def identity(self) -> str:
        payload = {
            "mean": self.mean,
            "sample_count": self.sample_count,
            "std": self.std,
            "training_data_version": self.training_data_version,
            "training_split": self.training_split,
            "value_count_per_channel": self.value_count_per_channel,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()


class ChannelNormalizer:
    """Apply a frozen affine transform to ``[B, C, T, H, W]`` latents."""

    def __init__(self, stats: NormalizationStats) -> None:
        self.stats = stats

    @classmethod
    def fit_from_training(
        cls,
        batches: Iterable[torch.Tensor],
        *,
        split: str,
        data_version: str,
        minimum_std: float = 1e-6,
    ) -> "ChannelNormalizer":
        if split != "train":
            raise ValueError("normalization may only be fitted from the train split")
        if not data_version:
            raise ValueError("data_version must be explicit")
        if minimum_std <= 0:
            raise ValueError("minimum_std must be positive")

        channel_sum: torch.Tensor | None = None
        channel_square_sum: torch.Tensor | None = None
        sample_count = 0
        value_count = 0
        for batch in batches:
            _validate_latents(batch)
            values = batch.detach().to(device="cpu", dtype=torch.float64)
            if channel_sum is not None and values.shape[1] != channel_sum.numel():
                raise ValueError("all normalization batches must have the same channels")
            reduced_dims = (0, 2, 3, 4)
            current_sum = values.sum(dim=reduced_dims)
            current_square_sum = values.square().sum(dim=reduced_dims)
            channel_sum = current_sum if channel_sum is None else channel_sum + current_sum
            channel_square_sum = (
                current_square_sum
                if channel_square_sum is None
                else channel_square_sum + current_square_sum
            )
            sample_count += values.shape[0]
            value_count += values.shape[0] * values.shape[2] * values.shape[3] * values.shape[4]

        if channel_sum is None or channel_square_sum is None:
            raise ValueError("at least one training latent batch is required")
        mean = channel_sum / value_count
        variance = torch.clamp(channel_square_sum / value_count - mean.square(), min=0.0)
        std = torch.clamp(variance.sqrt(), min=minimum_std)
        stats = NormalizationStats(
            mean=tuple(float(value) for value in mean),
            std=tuple(float(value) for value in std),
            training_data_version=data_version,
            sample_count=sample_count,
            value_count_per_channel=value_count,
        )
        return cls(stats)

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        mean, std = self._parameters_for(latents)
        return (latents - mean) / std

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        mean, std = self._parameters_for(latents)
        return latents * std + mean

    def _parameters_for(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_latents(latents)
        if latents.shape[1] != len(self.stats.mean):
            raise ValueError("latent channels do not match normalization statistics")
        shape = (1, -1, 1, 1, 1)
        mean = latents.new_tensor(self.stats.mean).view(shape)
        std = latents.new_tensor(self.stats.std).view(shape)
        return mean, std


def _validate_latents(value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 5:
        raise ValueError("latents must be a [B, C, T, H, W] tensor")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError("latent batch and channel dimensions must be non-empty")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError("latents must contain finite floating-point values")
