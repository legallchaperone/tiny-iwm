"""Camera conventions shared by data, model, rollout, and evaluation code.

External camera poses enter this package as camera-to-world transforms (``c2w``):
a homogeneous point expressed in camera coordinates is left-multiplied by ``c2w``
to obtain world coordinates. Code that requires world-to-camera transforms must
perform one explicit matrix inverse at the adapter boundary and name the result
``w2c``. Stored conditions remain ``c2w``.

Intrinsics use the standard pixel-coordinate matrix with focal lengths in
``K[0, 0]`` and ``K[1, 1]`` and principal point in ``K[0, 2]`` and ``K[1, 2]``.
Every condition declares whether those coordinates refer to the RGB image or a
latent grid. Resize and crop operations must produce a new matrix and append a
description to ``preprocessing``; callers must never reinterpret RGB intrinsics as
latent-grid intrinsics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Tuple

import torch


class IntrinsicsSpace(str, Enum):
    """The pixel/grid coordinate system in which an intrinsic matrix is defined."""

    RGB_PIXELS = "rgb_pixels"
    LATENT_GRID = "latent_grid"


@dataclass(frozen=True)
class CameraCondition:
    """A batched camera trajectory with explicit geometry metadata.

    Tensor-like fields deliberately use ``Any`` so the data contract stays usable
    at I/O boundaries. The numerical resize, crop, conversion, and inversion
    helpers below require floating-point PyTorch tensors.

    Shapes:
        c2w: ``[B, T, 4, 4]`` camera-to-world homogeneous transforms.
        intrinsics: ``[B, T, 3, 3]`` matrices in ``intrinsics_space``.
        timestamps_seconds: ``[B, T]`` physical timestamps.
    """

    c2w: Any
    intrinsics: Any
    timestamps_seconds: Any
    intrinsics_space: IntrinsicsSpace
    length_unit: str = "meter"
    reference_frame: str = "world"
    preprocessing: Tuple[str, ...] = ()

    @property
    def extrinsics_convention(self) -> str:
        """Return the fixed storage convention for consumers and provenance."""

        return "camera_to_world"


def resize_rgb_camera(
    camera: CameraCondition,
    *,
    old_size: tuple[int, int],
    new_size: tuple[int, int],
) -> CameraCondition:
    """Resize RGB-pixel intrinsics while preserving stored c2w poses."""
    _require_rgb_intrinsics(camera)
    old_height, old_width = _positive_size(old_size, "old_size")
    new_height, new_width = _positive_size(new_size, "new_size")
    intrinsics = _camera_tensor(camera.intrinsics, "intrinsics").clone()
    scale_x, scale_y = new_width / old_width, new_height / old_height
    intrinsics[..., 0, :] *= scale_x
    intrinsics[..., 1, :] *= scale_y
    return replace(
        camera,
        intrinsics=intrinsics,
        preprocessing=camera.preprocessing
        + (f"resize_rgb:{old_height}x{old_width}->{new_height}x{new_width}",),
    )


def crop_rgb_camera(
    camera: CameraCondition,
    *,
    left: int,
    top: int,
    crop_size: tuple[int, int],
) -> CameraCondition:
    """Move the RGB principal point into a cropped image coordinate system."""
    _require_rgb_intrinsics(camera)
    crop_height, crop_width = _positive_size(crop_size, "crop_size")
    if left < 0 or top < 0:
        raise ValueError("crop offsets must be non-negative")
    intrinsics = _camera_tensor(camera.intrinsics, "intrinsics").clone()
    intrinsics[..., 0, 2] -= left
    intrinsics[..., 1, 2] -= top
    return replace(
        camera,
        intrinsics=intrinsics,
        preprocessing=camera.preprocessing
        + (f"crop_rgb:left={left},top={top},size={crop_height}x{crop_width}",),
    )


def rgb_to_latent_grid_camera(
    camera: CameraCondition, *, spatial_compression: tuple[int, int]
) -> CameraCondition:
    """Explicitly convert RGB-pixel intrinsics to latent-grid coordinates."""
    _require_rgb_intrinsics(camera)
    compression_y, compression_x = _positive_size(
        spatial_compression, "spatial_compression"
    )
    intrinsics = _camera_tensor(camera.intrinsics, "intrinsics").clone()
    intrinsics[..., 0, :] /= compression_x
    intrinsics[..., 1, :] /= compression_y
    return replace(
        camera,
        intrinsics=intrinsics,
        intrinsics_space=IntrinsicsSpace.LATENT_GRID,
        preprocessing=camera.preprocessing
        + (f"rgb_to_latent_grid:{compression_y}x{compression_x}",),
    )


def normalized_rgb_intrinsics(
    camera: CameraCondition, *, image_size: tuple[int, int]
) -> torch.Tensor:
    """Normalize RGB intrinsics to centered image coordinates for PRoPE."""
    _require_rgb_intrinsics(camera)
    height, width = _positive_size(image_size, "image_size")
    source = _camera_tensor(camera.intrinsics, "intrinsics")
    _validate_matrix_shape(source, (3, 3), "intrinsics")
    if torch.any(source[..., 0, 0] <= 0) or torch.any(source[..., 1, 1] <= 0):
        raise ValueError("RGB focal lengths must be positive")
    normalized = torch.zeros_like(source)
    normalized[..., 0, 0] = source[..., 0, 0] / width
    normalized[..., 1, 1] = source[..., 1, 1] / height
    normalized[..., 0, 2] = source[..., 0, 2] / width - 0.5
    normalized[..., 1, 2] = source[..., 1, 2] / height - 0.5
    normalized[..., 2, 2] = 1
    if not torch.isfinite(normalized).all():
        raise ValueError("normalized intrinsics must be finite")
    return normalized


def world_to_camera(camera: CameraCondition, *, translation_scale: float = 1.0) -> torch.Tensor:
    """Convert stored rigid c2w poses to w2c, with one explicit unit scale."""
    if translation_scale <= 0:
        raise ValueError("translation_scale must be positive")
    c2w = _camera_tensor(camera.c2w, "c2w").clone()
    _validate_matrix_shape(c2w, (4, 4), "c2w")
    expected_bottom = torch.zeros_like(c2w[..., 3, :])
    expected_bottom[..., 3] = 1
    if not torch.allclose(c2w[..., 3, :], expected_bottom, atol=1e-6, rtol=0):
        raise ValueError("c2w must use a homogeneous [0, 0, 0, 1] bottom row")
    c2w[..., :3, 3] *= translation_scale
    rotation = c2w[..., :3, :3]
    translation = c2w[..., :3, 3]
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    if not torch.allclose(
        rotation.transpose(-1, -2) @ rotation,
        identity.expand_as(rotation),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError("c2w rotations must be orthonormal")
    w2c = torch.zeros_like(c2w)
    inverse_rotation = rotation.transpose(-1, -2)
    w2c[..., :3, :3] = inverse_rotation
    w2c[..., :3, 3] = -(inverse_rotation @ translation.unsqueeze(-1)).squeeze(-1)
    w2c[..., 3, 3] = 1
    return w2c


def _require_rgb_intrinsics(camera: CameraCondition) -> None:
    if camera.intrinsics_space is not IntrinsicsSpace.RGB_PIXELS:
        raise ValueError("operation requires RGB-pixel intrinsics")


def _camera_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _validate_matrix_shape(value: torch.Tensor, shape: tuple[int, int], name: str) -> None:
    if value.ndim < 2 or value.shape[-2:] != shape:
        raise ValueError(f"{name} must end in shape {shape}")


def _positive_size(value: tuple[int, int], name: str) -> tuple[int, int]:
    if len(value) != 2 or any(item <= 0 for item in value):
        raise ValueError(f"{name} must contain two positive values")
    return value
