"""Read one temporal window from the prepared latent and camera cache."""

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
    valid_latent_frames = layout.valid_latent_frame_count
    assert valid_latent_frames is not None
    _require_codec_aligned_window(layout)
    if (
        not isinstance(latent_shape, list)
        or len(latent_shape) != 4
        or not all(type(value) is int and value > 0 for value in latent_shape)
    ):
        raise ValueError("prepared record has no valid [C,T,H,W] latent_shape")
    if latent_shape[1] < valid_latent_frames:
        raise ValueError(
            f"prepared source has {latent_shape[1]} latent frames, but the requested "
            f"window needs {valid_latent_frames} real latent frames"
        )
    if type(camera_frames) is not int or camera_frames < layout.output_rgb_frame_count:
        raise ValueError(
            f"prepared source has {camera_frames!r} camera frames, but layout requires "
            f"{layout.output_rgb_frame_count}"
        )


def _require_codec_aligned_window(layout: VideoLayout) -> None:
    """Reject cached latents whose final codec group extends beyond the window."""
    valid_latent_frames = layout.valid_latent_frame_count
    assert valid_latent_frames is not None
    last_range = layout.latent_to_rgb[valid_latent_frames - 1]
    if last_range.stop != layout.output_rgb_frame_count:
        raise ValueError(
            "prepared-cache window must end on a codec boundary; requested frame "
            f"{layout.output_rgb_frame_count} splits latent group "
            f"[{last_range.start}, {last_range.stop})"
        )


def pad_prepared_latents(
    latents: np.ndarray, layout: VideoLayout
) -> tuple[np.ndarray, np.ndarray]:
    """Copy only real window latents and zero-fill codec/patch suffix padding."""
    _require_codec_aligned_window(layout)
    if latents.ndim != 4:
        raise ValueError("prepared latents must have [C,T,H,W] shape")
    valid_latent_frames = layout.valid_latent_frame_count
    assert valid_latent_frames is not None
    if latents.shape[1] < valid_latent_frames:
        raise ValueError(
            f"prepared source has {latents.shape[1]} latent frames, but the requested "
            f"window needs {valid_latent_frames} real latent frames"
        )
    padded = np.zeros(
        (latents.shape[0], layout.latent_frame_count, latents.shape[2], latents.shape[3]),
        dtype=latents.dtype,
    )
    padded[:, :valid_latent_frames] = latents[:, :valid_latent_frames]
    valid_mask = np.zeros(
        (layout.latent_frame_count, latents.shape[2], latents.shape[3]),
        dtype=np.bool_,
    )
    valid_mask[:valid_latent_frames] = True
    return padded, valid_mask


def load_prepared_sample(
    record: dict[str, object],
    *,
    device: torch.device,
    dtype: torch.dtype,
    layout: VideoLayout,
) -> tuple[torch.Tensor, CameraCondition, TokenCameraProjection, torch.Tensor]:
    """Load a valid-length window, returning its latent loss mask as well."""
    validate_prepared_record(record, layout)
    arrays = np.load(str(record["prepared_path"]), allow_pickle=False)
    latent_frames = layout.latent_frame_count
    rgb_frames = layout.output_rgb_frame_count
    if arrays["pose"].shape[0] < rgb_frames or arrays["intrinsics"].shape[0] < rgb_frames:
        raise ValueError("prepared camera source is shorter than the resolved VideoLayout")
    padded, valid_mask = pad_prepared_latents(arrays["z"], layout)
    clean = torch.from_numpy(padded).unsqueeze(0).to(device=device, dtype=dtype)
    valid_latent_mask = torch.from_numpy(valid_mask).unsqueeze(0).to(device=device)
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
        camera,
        layout,
        grid_shape=(len(layout.token_to_latent_ranges), 2, 2),
        image_size=(128, 128),
    )
    projection = TokenCameraProjection(
        projection.projection.to(dtype),
        projection.transpose.to(dtype),
        projection.inverse.to(dtype),
    )
    return clean, camera, projection, valid_latent_mask
