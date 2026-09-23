"""Read the already-prepared Stage A/B latent and camera sample once."""

import numpy as np
import torch

from algorithms.world_model.models.prope import (
    TokenCameraProjection,
    build_token_camera_projection,
)
from core.camera import CameraCondition, IntrinsicsSpace
from core.video_layout import VideoLayout


def load_prepared_sample(
    record: dict[str, object],
    *,
    device: torch.device,
    dtype: torch.dtype,
    layout: VideoLayout,
) -> tuple[torch.Tensor, CameraCondition, TokenCameraProjection]:
    arrays = np.load(str(record["prepared_path"]), allow_pickle=False)
    clean = torch.from_numpy(arrays["z"]).unsqueeze(0).to(device=device, dtype=dtype)
    c2w = (
        torch.from_numpy(arrays["pose"])
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )
    values = torch.from_numpy(arrays["intrinsics"]).to(
        device=device, dtype=torch.float32
    )
    k = torch.zeros((961, 3, 3), device=device, dtype=torch.float32)
    k[:, 0, 0] = values[:, 0] * (1280 / 1920)
    k[:, 1, 1] = values[:, 1] * (720 / 1080)
    k[:, 0, 2] = values[:, 2] * (1280 / 1920) - 576
    k[:, 1, 2] = values[:, 3] * (720 / 1080) - 8 - 288
    k[:, 2, 2] = 1
    camera = CameraCondition(
        c2w=c2w,
        intrinsics=k.unsqueeze(0),
        timestamps_seconds=(torch.arange(961, device=device)[None] / 16),
        intrinsics_space=IntrinsicsSpace.RGB_PIXELS,
        preprocessing=(
            "resize_rgb:1080x1920->720x1280",
            "crop_rgb:left=0,top=8,size=704x1280",
            "crop_rgb:left=576,top=288,size=128x128",
        ),
    )
    projection = build_token_camera_projection(
        camera, layout, grid_shape=(121, 2, 2), image_size=(128, 128)
    )
    projection = TokenCameraProjection(
        projection.projection.to(dtype),
        projection.transpose.to(dtype),
        projection.inverse.to(dtype),
    )
    return clean, camera, projection
