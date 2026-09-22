"""Adapter for the selected official SANA causal video VAE implementation."""

from __future__ import annotations

from copy import deepcopy

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
        self._model = deepcopy(model).eval()
        self._model.requires_grad_(False)
        self._codec_dtype = _floating_model_dtype(self._model)
        if self._codec_dtype is None:
            raise ValueError("codec must have a floating-point parameter or buffer")
        if _dtype_name(self._codec_dtype) != spec.execution_dtype:
            raise ValueError("codec execution dtype must match its specification")
        self._spec = spec
        self._normalizer = ChannelNormalizer(spec.normalization)

    @property
    def spec(self) -> CodecSpec:
        return self._spec

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        _validate_video_tensor(video, "video")
        self._model.eval()
        with torch.no_grad():
            latents = self._model.encode(video)
        _validate_video_tensor(latents, "codec latents")
        if latents.shape[1] != self.spec.latent_channels:
            raise ValueError("codec output channels do not match its specification")
        return self._normalizer.normalize(latents)

    def decode(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        _validate_video_tensor(normalized_latents, "normalized latents")
        latents = self._normalizer.denormalize(normalized_latents)
        if self._codec_dtype is not None:
            latents = latents.to(dtype=self._codec_dtype)
        _validate_video_tensor(latents, "codec latents")
        self._model.eval()
        with torch.no_grad():
            video = self._model.decode(latents)
        _validate_video_tensor(video, "decoded video")
        return video


def _validate_video_tensor(value: object, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 5:
        raise ValueError(f"{name} must be a [B, C, T, H, W] tensor")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")


def _floating_model_dtype(model: nn.Module) -> torch.dtype | None:
    for value in (*model.parameters(), *model.buffers()):
        if torch.is_floating_point(value):
            return value.dtype
    return None


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")
