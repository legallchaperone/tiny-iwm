"""Video PRoPE adapted from Matrix-Game 3.5's tiled projection rule.

The upstream implementation fixes four sub-frame cameras in a particular head
layout. Here the temporal sub-frames come from ``VideoLayout`` and each matrix
is repeated over a configured leading slice of every attention head.
"""

from dataclasses import dataclass

import torch

from core.camera import CameraCondition, normalized_rgb_intrinsics, world_to_camera
from core.video_layout import VideoLayout


@dataclass(frozen=True)
class TokenCameraProjection:
    """Paired token projection matrices, all shaped ``[B, N, S, 4, 4]``."""

    projection: torch.Tensor
    transpose: torch.Tensor
    inverse: torch.Tensor

    def __post_init__(self) -> None:
        if self.projection.shape != self.transpose.shape or self.projection.shape != self.inverse.shape:
            raise ValueError("PRoPE projection matrices must have identical shapes")
        if self.projection.ndim != 5 or self.projection.shape[-2:] != (4, 4):
            raise ValueError("PRoPE matrices must have shape [B, N, S, 4, 4]")
        if not all(
            torch.isfinite(value).all()
            for value in (self.projection, self.transpose, self.inverse)
        ):
            raise ValueError("PRoPE projection matrices must be finite")

    @property
    def token_count(self) -> int:
        return self.projection.shape[1]

    @property
    def subframes(self) -> int:
        return self.projection.shape[2]


def build_token_camera_projection(
    camera: CameraCondition,
    layout: VideoLayout,
    *,
    grid_shape: tuple[int, int, int],
    image_size: tuple[int, int],
    translation_scale: float = 1.0,
) -> TokenCameraProjection:
    """Build per-token matrices using VideoLayout's temporal patch mapping."""
    grid_time, grid_height, grid_width = grid_shape
    if grid_time != len(layout.token_to_latent_ranges):
        raise ValueError("model temporal grid must match VideoLayout token ranges")
    c2w = camera.c2w
    intrinsics = camera.intrinsics
    if not isinstance(c2w, torch.Tensor) or not isinstance(intrinsics, torch.Tensor):
        raise TypeError("camera c2w and intrinsics must be torch tensors")
    if c2w.shape[:2] != intrinsics.shape[:2] or c2w.ndim != 4 or intrinsics.ndim != 4:
        raise ValueError("camera c2w and intrinsics must align as [B, RGB_T, ...]")

    w2c = world_to_camera(camera, translation_scale=translation_scale)
    normalized_k = normalized_rgb_intrinsics(camera, image_size=image_size)
    lifted_k = _lift_intrinsics(normalized_k)
    projection_by_rgb = lifted_k @ w2c
    inverse_by_rgb = _invert_rigid(w2c) @ _lift_intrinsics(_invert_intrinsics(normalized_k))

    subframes = max(len(item) for item in layout.token_to_latent_ranges)
    temporal_indices: list[list[int]] = []
    for latent_range in layout.token_to_latent_ranges:
        indices = [
            layout.camera_rgb_index_for_latent(index)
            for index in range(latent_range.start, latent_range.stop)
        ]
        indices.extend([indices[-1]] * (subframes - len(indices)))
        temporal_indices.append(indices)
    gather = torch.tensor(temporal_indices, device=c2w.device, dtype=torch.long)
    if int(gather.max()) >= c2w.shape[1]:
        raise ValueError("camera trajectory is shorter than VideoLayout requires")

    def gather_and_tile(value: torch.Tensor) -> torch.Tensor:
        selected = value[:, gather]  # [B, grid_T, subframes, 4, 4]
        return (
            selected[:, :, None, None]
            .expand(-1, -1, grid_height, grid_width, -1, -1, -1)
            .reshape(value.shape[0], grid_time * grid_height * grid_width, subframes, 4, 4)
            .contiguous()
        )

    projection = gather_and_tile(projection_by_rgb)
    inverse = gather_and_tile(inverse_by_rgb)
    return TokenCameraProjection(projection, projection.transpose(-1, -2), inverse)


def apply_camera_projection(
    features: torch.Tensor,
    matrices: torch.Tensor,
    *,
    camera_dims: int,
) -> torch.Tensor:
    """Tile each sub-frame's 4x4 transform through a leading head slice."""
    if features.ndim != 4:
        raise ValueError("attention features must have shape [B, H, N, D]")
    if matrices.shape[:2] != (features.shape[0], features.shape[2]):
        raise ValueError("PRoPE matrix batch/token axes must match attention features")
    subframes = matrices.shape[2]
    if camera_dims <= 0 or camera_dims > features.shape[-1]:
        raise ValueError("camera_dims must fit within attention head_dim")
    if camera_dims % (4 * subframes):
        raise ValueError("camera_dims must be divisible by four times the sub-frame count")
    camera_features, tail = features[..., :camera_dims], features[..., camera_dims:]
    repeats = camera_dims // (4 * subframes)
    camera_features = camera_features.reshape(
        *features.shape[:-1], repeats, subframes, 4
    )
    transformed = torch.einsum("bnsij,bhnrsj->bhnrsi", matrices, camera_features)
    return torch.cat((transformed.flatten(-3), tail), dim=-1)


def _lift_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    lifted = torch.zeros(
        (*intrinsics.shape[:-2], 4, 4), dtype=intrinsics.dtype, device=intrinsics.device
    )
    lifted[..., :3, :3] = intrinsics
    lifted[..., 3, 3] = 1
    return lifted


def _invert_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    inverse = torch.zeros_like(intrinsics)
    inverse[..., 0, 0] = 1 / intrinsics[..., 0, 0]
    inverse[..., 1, 1] = 1 / intrinsics[..., 1, 1]
    inverse[..., 0, 2] = -intrinsics[..., 0, 2] / intrinsics[..., 0, 0]
    inverse[..., 1, 2] = -intrinsics[..., 1, 2] / intrinsics[..., 1, 1]
    inverse[..., 2, 2] = 1
    return inverse


def _invert_rigid(transform: torch.Tensor) -> torch.Tensor:
    inverse = torch.zeros_like(transform)
    rotation = transform[..., :3, :3].transpose(-1, -2)
    inverse[..., :3, :3] = rotation
    inverse[..., :3, 3] = -(rotation @ transform[..., :3, 3, None]).squeeze(-1)
    inverse[..., 3, 3] = 1
    return inverse
