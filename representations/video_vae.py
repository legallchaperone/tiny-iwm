"""Adapter for the selected official SANA causal video VAE implementation."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import platform
from typing import Iterator

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
        snapshot_model: bool = True,
    ) -> None:
        if normalizer.stats != spec.normalization:
            raise ValueError("normalizer statistics must match the codec specification")
        # Small injected modules are copied by default so later caller mutation
        # cannot silently change the codec identity.  The released SANA-WM VAE
        # is almost 5 GB, so its explicit factory transfers ownership instead
        # of making a second in-memory copy.
        self._model = (deepcopy(model) if snapshot_model else model).eval()
        self._model.requires_grad_(False)
        state = _uniform_floating_model_state(self._model)
        if state is None:
            raise ValueError("codec must have a floating-point parameter or buffer")
        self._codec_dtype, self._codec_device = state
        if _dtype_name(self._codec_dtype) != spec.execution_dtype:
            raise ValueError("codec execution dtype must match its specification")
        if self._codec_device.type != spec.execution_backend:
            raise ValueError("codec execution backend must match its specification")
        if _backend_fingerprint(self._codec_device) != spec.backend_fingerprint:
            raise ValueError("codec backend fingerprint must match its specification")
        if spec.execution_backend == "cuda":
            configure_cuda_math_policy(spec.cuda_math_policy)
        self._spec = spec
        self._normalizer = ChannelNormalizer(spec.normalization)

    @property
    def spec(self) -> CodecSpec:
        return self._spec

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        _validate_video_tensor(video, "video")
        if video.device != self._codec_device:
            raise ValueError("video device must match the codec execution backend")
        codec_video = video.to(dtype=self._codec_dtype)
        _validate_video_tensor(codec_video, "codec video")
        self._model.eval()
        with torch.no_grad(), _codec_math_mode(
            codec_video.device.type, self.spec.cuda_math_policy
        ):
            latents = self._model.encode(codec_video)
        _validate_video_tensor(latents, "codec latents")
        if latents.shape[1] != self.spec.latent_channels:
            raise ValueError("codec output channels do not match its specification")
        return self._normalizer.normalize(latents)

    def decode(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        _validate_video_tensor(normalized_latents, "normalized latents")
        if normalized_latents.device != self._codec_device:
            raise ValueError(
                "normalized latent device must match the codec execution backend"
            )
        latents = self._normalizer.denormalize(normalized_latents)
        if self._codec_dtype is not None:
            latents = latents.to(dtype=self._codec_dtype)
        _validate_video_tensor(latents, "codec latents")
        self._model.eval()
        with torch.no_grad(), _codec_math_mode(
            latents.device.type, self.spec.cuda_math_policy
        ):
            video = self._model.decode(latents)
        _validate_video_tensor(video, "decoded video")
        return video


def _validate_video_tensor(value: object, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 5:
        raise ValueError(f"{name} must be a [B, C, T, H, W] tensor")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")


def _uniform_floating_model_state(
    model: nn.Module,
) -> tuple[torch.dtype, torch.device] | None:
    states = {
        (value.dtype, value.device)
        for value in (*model.parameters(), *model.buffers())
        if torch.is_floating_point(value)
    }
    if len(states) > 1:
        raise ValueError(
            "codec floating-point state must use one execution dtype and device"
        )
    return next(iter(states), None)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _backend_fingerprint(device: torch.device) -> str:
    parts = [f"torch={torch.__version__}", f"backend={device.type}"]
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        parts.extend(
            (
                f"cuda={torch.version.cuda}",
                f"cudnn={torch.backends.cudnn.version()}",
                f"device={torch.cuda.get_device_name(index)}",
                f"capability={torch.cuda.get_device_capability(index)}",
            )
        )
    else:
        parts.append(f"platform={platform.platform()}")
    return ";".join(parts)


def configure_cuda_math_policy(cuda_math_policy: str) -> None:
    """Configure the declared CUDA policy once, before worker threads start."""

    if cuda_math_policy != "strict_no_tf32_no_reduced_reduction_v1":
        raise ValueError("unsupported CUDA math policy")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False


@contextmanager
def _codec_math_mode(
    device_type: str, cuda_math_policy: str
) -> Iterator[None]:
    """Disable autocast and verify the declared process-wide CUDA policy."""

    with torch.autocast(device_type=device_type, enabled=False):
        if device_type == "cuda":
            _validate_cuda_math_policy(cuda_math_policy)
        yield


def _validate_cuda_math_policy(cuda_math_policy: str) -> None:
    if cuda_math_policy != "strict_no_tf32_no_reduced_reduction_v1":
        raise ValueError("unsupported CUDA math policy")
    enabled = {
        "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul.allow_fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "matmul.allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "matmul.allow_fp16_accumulation": (
            torch.backends.cuda.matmul.allow_fp16_accumulation
        ),
    }
    violations = [name for name, value in enabled.items() if value]
    if violations:
        raise RuntimeError(
            "CUDA backend does not match strict codec math policy: "
            + ", ".join(violations)
        )
