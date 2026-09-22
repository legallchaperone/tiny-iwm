"""End-to-end diagnostics for the M1 frozen video codec gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, log10
from typing import Any, Mapping

import numpy as np
import torch

from core.video_layout import VideoLayout
from datasets.sana_wm.reader import SANASample
from representations.base import Representation
from representations.cache import CacheIdentity


@dataclass(frozen=True)
class CodecGateSettings:
    """Fixed requirements for one complete M1 validation run."""

    fps: float = 16.0
    rgb_frame_count: int = 961
    future_start_rgb: int = 481
    causality_atol: float = 0.0

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.rgb_frame_count <= 1:
            raise ValueError("rgb_frame_count must be greater than one")
        if not 0 < self.future_start_rgb < self.rgb_frame_count:
            raise ValueError("future_start_rgb must be inside the sample")
        if not isfinite(self.causality_atol) or self.causality_atol < 0:
            raise ValueError("causality_atol must be finite and non-negative")


def validate_complete_sample(
    sample: SANASample,
    *,
    codec: Representation,
    layout: VideoLayout,
    preprocessing: Mapping[str, object],
    settings: CodecGateSettings = CodecGateSettings(),
) -> dict[str, Any]:
    """Read-independent M1 gate for a held-out sample and frozen codec.

    The caller supplies a sample produced by :class:`SANAReader`. This boundary
    keeps filesystem decoding separate while still validating the exact tensor
    that is encoded, decoded, aligned to camera data, and cache-keyed.
    """

    _validate_sample_contract(sample, codec, layout, settings)
    video = _video_tensor(sample.frames_rgb, codec)
    if layout.rgb_frame_count > settings.rgb_frame_count:
        padding = video[:, :, -1:].expand(
            -1, -1, layout.rgb_frame_count - settings.rgb_frame_count, -1, -1
        )
        video = torch.cat((video, padding), dim=2)

    latents = codec.encode(video)
    _validate_finite_tensor(latents, "codec latents")
    if latents.shape[0] != video.shape[0]:
        raise ValueError("codec latent batch size does not match its input")
    if latents.shape[1] != codec.spec.latent_channels:
        raise ValueError("codec latent channels do not match its specification")
    if latents.shape[2] != layout.latent_frame_count:
        raise ValueError(
            "codec latent length does not match VideoLayout: "
            f"{latents.shape[2]} != {layout.latent_frame_count}"
        )
    expected_latent_height = video.shape[3] // codec.spec.spatial_compression[0]
    expected_latent_width = video.shape[4] // codec.spec.spatial_compression[1]
    if latents.shape[3:] != (expected_latent_height, expected_latent_width):
        raise ValueError("codec latent spatial shape does not match its specification")
    reconstruction = codec.decode(latents)
    _validate_finite_tensor(reconstruction, "decoded video")
    if reconstruction.shape != video.shape:
        raise ValueError(
            "decoded video shape does not match codec input: "
            f"{tuple(reconstruction.shape)} != {tuple(video.shape)}"
        )

    history_latent_count = sum(
        rgb_range.stop <= settings.future_start_rgb
        for rgb_range in layout.latent_to_rgb
    )
    if history_latent_count <= 0:
        raise ValueError("causality split leaves no history latents")
    perturbed_video = video.clone()
    future = perturbed_video[:, :, settings.future_start_rgb : settings.rgb_frame_count]
    replacement = 0.0 if float(future.mean()) >= 0.5 else 1.0
    future.fill_(replacement)
    if torch.equal(
        future,
        video[:, :, settings.future_start_rgb : settings.rgb_frame_count],
    ):
        raise ValueError("future perturbation did not change the input")
    perturbed_latents = codec.encode(perturbed_video)
    _validate_finite_tensor(perturbed_latents, "perturbed codec latents")
    if perturbed_latents.shape != latents.shape:
        raise ValueError("codec latent shape changed after future perturbation")
    history_delta = (
        latents[:, :, :history_latent_count]
        - perturbed_latents[:, :, :history_latent_count]
    ).abs()
    max_history_delta = float(history_delta.max())
    if max_history_delta > settings.causality_atol:
        raise ValueError(
            "codec causality check failed: future RGB changed history latents "
            f"by {max_history_delta:.9g} (allowed {settings.causality_atol:.9g})"
        )

    valid_video = video[:, :, : settings.rgb_frame_count].float()
    valid_reconstruction = reconstruction[:, :, : settings.rgb_frame_count].float()
    error = valid_reconstruction - valid_video
    mse = float(error.square().mean())
    cache_identity = CacheIdentity(
        data_version=sample.record.data_version,
        sample_id=sample.record.sample_id,
        frame_indices=tuple(range(settings.rgb_frame_count)),
        preprocessing=preprocessing,
        codec=codec.spec,
    )
    camera_indices = tuple(
        layout.camera_rgb_index_for_latent(index)
        for index in range(layout.latent_frame_count)
    )
    timestamps = sample.camera.timestamps_seconds
    expected_timestamps = np.arange(settings.rgb_frame_count, dtype=np.float64) / settings.fps
    timestamp_error = np.abs(timestamps - expected_timestamps)

    return {
        "schema_version": 1,
        "status": "passed",
        "sample": {
            "sample_id": sample.record.sample_id,
            "source": sample.record.source,
            "scene_id": sample.record.scene_id,
            "split": sample.record.split,
            "data_version": sample.record.data_version,
            "rgb_shape_thwc": list(sample.frames_rgb.shape),
            "fps": settings.fps,
            "duration_seconds": (settings.rgb_frame_count - 1) / settings.fps,
        },
        "codec": {
            "codec_id": codec.spec.codec_id,
            "weights_sha256": codec.spec.weights_sha256,
            "causal": codec.spec.causal,
            "temporal_compression": codec.spec.temporal_compression,
            "spatial_compression": list(codec.spec.spatial_compression),
            "encoding_policy": codec.spec.encoding_policy,
            "execution_dtype": codec.spec.execution_dtype,
            "execution_backend": codec.spec.execution_backend,
            "backend_fingerprint": codec.spec.backend_fingerprint,
        },
        "layout": {
            "rgb_frame_count": layout.rgb_frame_count,
            "valid_rgb_frame_count": layout.output_rgb_frame_count,
            "latent_frame_count": layout.latent_frame_count,
            "first_latent_rgb_range": asdict(layout.rgb_range_for_latent(0)),
            "last_latent_rgb_range": asdict(
                layout.rgb_range_for_latent(layout.latent_frame_count - 1)
            ),
            "padding_rgb_indices": [
                index for index, padded in enumerate(layout.rgb_is_padding) if padded
            ],
            "last_writer_rgb_index": max(
                index
                for chunk_index in range(len(layout.chunk_to_latent))
                for index in layout.writer_rgb_indices_for_chunk(chunk_index)
            ),
        },
        "camera_alignment": {
            "camera_frame_count": int(sample.camera.c2w.shape[0]),
            "first_latent_camera_rgb_index": camera_indices[0],
            "last_latent_camera_rgb_index": camera_indices[-1],
            "latent_camera_rgb_indices": list(camera_indices),
            "latent_camera_timestamps_seconds": [
                float(timestamps[index]) for index in camera_indices
            ],
            "first_timestamp_seconds": float(timestamps[0]),
            "last_timestamp_seconds": float(timestamps[-1]),
            "max_timestamp_error_seconds": float(timestamp_error.max()),
        },
        "causality": {
            "future_start_rgb": settings.future_start_rgb,
            "history_latent_count": history_latent_count,
            "max_abs_history_latent_delta": max_history_delta,
            "atol": settings.causality_atol,
        },
        "reconstruction": {
            "mse": mse,
            "mae": float(error.abs().mean()),
            "psnr_db": None if mse == 0 else 10.0 * log10(1.0 / mse),
            "minimum": float(valid_reconstruction.min()),
            "maximum": float(valid_reconstruction.max()),
        },
        "latents": _latent_statistics(latents),
        "cache": {
            "schema_version": cache_identity.payload["schema_version"],
            "key": cache_identity.key,
            "relative_path": str(cache_identity.relative_path),
            "frame_count": len(cache_identity.frame_indices),
            "identity_payload": _json_value(cache_identity.payload),
        },
    }


def _validate_sample_contract(
    sample: SANASample,
    codec: Representation,
    layout: VideoLayout,
    settings: CodecGateSettings,
) -> None:
    if sample.record.split not in {"validation", "test"}:
        raise ValueError("codec diagnostics must run on a validation or test sample")
    frames = sample.frames_rgb
    expected_shape = (settings.rgb_frame_count,)
    if not isinstance(frames, np.ndarray) or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("sample frames must have shape [T, H, W, 3]")
    if frames.shape[:1] != expected_shape:
        raise ValueError(
            f"sample must contain exactly {settings.rgb_frame_count} RGB frames"
        )
    if frames.dtype != np.uint8:
        raise ValueError("sample RGB frames must use uint8 values")
    if not codec.spec.causal:
        raise ValueError("codec must be declared causal")
    height_stride, width_stride = codec.spec.spatial_compression
    if frames.shape[1] % height_stride or frames.shape[2] % width_stride:
        raise ValueError("RGB size must be divisible by codec spatial compression")
    if layout.fps != settings.fps:
        raise ValueError("VideoLayout FPS does not match gate settings")
    if layout.output_rgb_frame_count != settings.rgb_frame_count:
        raise ValueError("VideoLayout valid length does not match gate settings")
    covered = tuple(
        index
        for rgb_range in layout.latent_to_rgb
        for index in range(rgb_range.start, rgb_range.stop)
    )
    if covered != tuple(range(layout.rgb_frame_count)):
        raise ValueError("VideoLayout latent ranges must cover each RGB frame exactly once")
    if any(
        len(rgb_range) != codec.spec.temporal_compression
        for rgb_range in layout.latent_to_rgb[1:]
    ):
        raise ValueError("VideoLayout temporal groups do not match codec compression")
    if sample.camera.c2w.shape[0] != settings.rgb_frame_count:
        raise ValueError("camera trajectory length does not match RGB frames")
    timestamps = sample.camera.timestamps_seconds
    expected = np.arange(settings.rgb_frame_count, dtype=np.float64) / settings.fps
    if not np.allclose(timestamps, expected, rtol=0.0, atol=1e-6):
        raise ValueError("camera timestamps are not aligned to the declared FPS")


def _video_tensor(frames: np.ndarray, codec: Representation) -> torch.Tensor:
    try:
        device = torch.device(codec.spec.execution_backend)
    except RuntimeError as exc:
        raise ValueError("codec execution backend is unavailable") from exc
    return (
        torch.from_numpy(frames.copy())
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )


def _latent_statistics(latents: torch.Tensor) -> dict[str, object]:
    values = latents.detach().float().cpu()
    reduce_dims = (0, 2, 3, 4)
    return {
        "shape_bcthw": list(values.shape),
        "mean_per_channel": values.mean(dim=reduce_dims).tolist(),
        "std_per_channel": values.std(dim=reduce_dims, unbiased=False).tolist(),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _validate_finite_tensor(value: object, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 5:
        raise ValueError(f"{name} must have shape [B, C, T, H, W]")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")


def _json_value(value: object) -> object:
    """Thaw cache identity containers into JSON-native values for the report."""

    if isinstance(value, Mapping):
        return {key: _json_value(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_json_value(child) for child in value]
    return value
