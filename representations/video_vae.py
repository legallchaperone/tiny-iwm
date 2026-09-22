"""Adapter for the selected official SANA causal video VAE implementation."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from threading import RLock
from typing import Iterator

import torch
from torch import nn

from representations.base import CodecSpec, Representation
from representations.normalization import ChannelNormalizer


_TF32_POLICY_LOCK = RLock()


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
        self._codec_dtype = _uniform_floating_model_dtype(self._model)
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
        codec_video = video.to(dtype=self._codec_dtype)
        _validate_video_tensor(codec_video, "codec video")
        self._model.eval()
        with torch.no_grad(), _codec_math_mode(
            codec_video.device.type, self._codec_dtype
        ):
            latents = self._model.encode(codec_video)
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
        with torch.no_grad(), _codec_math_mode(latents.device.type, self._codec_dtype):
            video = self._model.decode(latents)
        _validate_video_tensor(video, "decoded video")
        return video


def _validate_video_tensor(value: object, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 5:
        raise ValueError(f"{name} must be a [B, C, T, H, W] tensor")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")


def _uniform_floating_model_dtype(model: nn.Module) -> torch.dtype | None:
    dtypes = {
        value.dtype
        for value in (*model.parameters(), *model.buffers())
        if torch.is_floating_point(value)
    }
    if len(dtypes) > 1:
        raise ValueError("codec floating-point state must use one execution dtype")
    return next(iter(dtypes), None)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@contextmanager
def _codec_math_mode(
    device_type: str, dtype: torch.dtype
) -> Iterator[None]:
    """Run with declared precision, isolated from ambient autocast and TF32."""

    with torch.autocast(device_type=device_type, enabled=False):
        if device_type != "cuda" or dtype != torch.float32:
            yield
            return
        with _TF32_POLICY_LOCK:
            cudnn_tf32 = torch.backends.cudnn.allow_tf32
            matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
                yield
            finally:
                torch.backends.cudnn.allow_tf32 = cudnn_tf32
                torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
