"""Adapter for the selected official SANA causal video VAE implementation."""

from __future__ import annotations

import torch
from torch import nn

from representations.base import CodecSpec, Representation
from representations.normalization import ChannelNormalizer


class SanaCausalVideoVAEAdapter(Representation):
    """Freeze an injected official SANA VAE and normalize its latent boundary.

    Loading the upstream implementation and verified checkpoint remains outside
    this adapter. The injected module must expose tensor-to-tensor ``encode`` and
    ``decode`` methods using ``[B, C, T, H, W]``.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        spec: CodecSpec,
        normalizer: ChannelNormalizer,
    ) -> None:
        if normalizer.stats != spec.normalization:
            raise ValueError("normalizer statistics must match the codec specification")
        self._model = model.eval()
        self._model.requires_grad_(False)
        self._spec = spec
        self._normalizer = normalizer

    @property
    def spec(self) -> CodecSpec:
        return self._spec

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        _validate_video_tensor(video, "video")
        with torch.no_grad():
            latents = self._model.encode(video)
        _validate_video_tensor(latents, "codec latents")
        if latents.shape[1] != self.spec.latent_channels:
            raise ValueError("codec output channels do not match its specification")
        return self._normalizer.normalize(latents)

    def decode(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        _validate_video_tensor(normalized_latents, "normalized latents")
        latents = self._normalizer.denormalize(normalized_latents)
        with torch.no_grad():
            video = self._model.decode(latents)
        _validate_video_tensor(video, "decoded video")
        return video


def _validate_video_tensor(value: object, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 5:
        raise ValueError(f"{name} must be a [B, C, T, H, W] tensor")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")
