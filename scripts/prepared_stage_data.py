"""Read the already-prepared Stage A/B latent and camera sample once."""

import numpy as np
import torch

from algorithms.world_model.models.prope import (
    TokenCameraProjection,
    build_token_camera_projection,
)
from core.camera import CameraCondition, IntrinsicsSpace
from core.video_layout import VideoLayout


def validate_prepared_record(record: dict[str, object], layout: VideoLayout) -> None:
    """Validate manifest geometry without opening tensors or allocating a GPU."""
    latent_shape = record.get("latent_shape")
    camera_frames = record.get("camera_frames")
    if (
        not isinstance(latent_shape, list)
        or len(latent_shape) != 4
        or not all(isinstance(value, int) and value > 0 for value in latent_shape)
    ):
        raise ValueError("prepared record has no valid [C,T,H,W] latent_shape")
    if latent_shape[1] < layout.latent_frame_count:
        raise ValueError(
            f"prepared source has {latent_shape[1]} latent frames, but layout requires "
            f"{layout.latent_frame_count}"
        )
    if not isinstance(camera_frames, int) or camera_frames < layout.output_rgb_frame_count:
        raise ValueError(
            f"prepared source has {camera_frames!r} camera frames, but layout requires "
            f"{layout.output_rgb_frame_count}"
        )


def load_prepared_sample(
    record: dict[str, object],
    *,
    device: torch.device,
    dtype: torch.dtype,
    layout: VideoLayout,
) -> tuple[torch.Tensor, CameraCondition, TokenCameraProjection]:
    validate_prepared_record(record, layout)
    arrays = np.load(str(record["prepared_path"]), allow_pickle=False)
    latent_frames = layout.latent_frame_count
    rgb_frames = layout.output_rgb_frame_count
    if arrays["z"].shape[1] < latent_frames or arrays["pose"].shape[0] < rgb_frames:
        raise ValueError("prepared source is shorter than the resolved VideoLayout")
    clean = torch.from_numpy(arrays["z"][:, :latent_frames]).unsqueeze(0).to(device=device, dtype=dtype)
    c2w = (
        torch.from_numpy(arrays["pose"][:rgb_frames])
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )
    values = torch.from_numpy(arrays["intrinsics"][:rgb_frames]).to(
        device=device, dtype=torch.float32
    )
    k = torch.zeros((rgb_frames, 3, 3), device=device, dtype=torch.float32)
    k[:, 0, 0] = values[:, 0] * (1280 / 1920)
    k[:, 1, 1] = values[:, 1] * (720 / 1080)
    k[:, 0, 2] = values[:, 2] * (1280 / 1920) - 576
    k[:, 1, 2] = values[:, 3] * (720 / 1080) - 8 - 288
    k[:, 2, 2] = 1
    camera = CameraCondition(
        c2w=c2w,
        intrinsics=k.unsqueeze(0),
        timestamps_seconds=(torch.arange(rgb_frames, device=device)[None] / layout.fps),
        intrinsics_space=IntrinsicsSpace.RGB_PIXELS,
        preprocessing=(
            "resize_rgb:1080x1920->720x1280",
            "crop_rgb:left=0,top=8,size=704x1280",
            "crop_rgb:left=576,top=288,size=128x128",
        ),
    )
    projection = build_token_camera_projection(
        camera, layout, grid_shape=(latent_frames, 2, 2), image_size=(128, 128)
    )
    projection = TokenCameraProjection(
        projection.projection.to(dtype),
        projection.transpose.to(dtype),
        projection.inverse.to(dtype),
    )
    return clean, camera, projection
