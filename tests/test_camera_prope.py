import torch
import pytest

from algorithms.world_model.models import JointVideoDiT, JointVideoDiTConfig
from algorithms.world_model.models.prope import (
    apply_camera_projection,
    build_token_camera_projection,
)
from core.camera import (
    CameraCondition,
    IntrinsicsSpace,
    crop_rgb_camera,
    normalized_rgb_intrinsics,
    resize_rgb_camera,
    rgb_to_latent_grid_camera,
    world_to_camera,
)
from core.video_layout import CodecTemporalSpec, VideoLayout


def _camera(frame_count: int = 9) -> CameraCondition:
    c2w = torch.eye(4).repeat(1, frame_count, 1, 1)
    c2w[0, :, 0, 3] = torch.arange(frame_count)
    intrinsics = torch.tensor(
        [[100.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]
    ).repeat(1, frame_count, 1, 1)
    return CameraCondition(
        c2w=c2w,
        intrinsics=intrinsics,
        timestamps_seconds=torch.arange(frame_count)[None] / 16,
        intrinsics_space=IntrinsicsSpace.RGB_PIXELS,
    )


def test_resize_crop_and_latent_grid_intrinsics_are_explicit():
    camera = _camera()
    resized = resize_rgb_camera(camera, old_size=(120, 160), new_size=(60, 80))
    cropped = crop_rgb_camera(resized, left=10, top=5, crop_size=(50, 60))
    latent = rgb_to_latent_grid_camera(cropped, spatial_compression=(5, 10))

    assert torch.equal(camera.intrinsics[0, 0], torch.tensor(
        [[100.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]
    ))
    assert torch.equal(cropped.intrinsics[0, 0], torch.tensor(
        [[50.0, 0.0, 30.0], [0.0, 60.0, 25.0], [0.0, 0.0, 1.0]]
    ))
    assert latent.intrinsics_space is IntrinsicsSpace.LATENT_GRID
    assert torch.equal(latent.intrinsics[0, 0], torch.tensor(
        [[5.0, 0.0, 3.0], [0.0, 12.0, 5.0], [0.0, 0.0, 1.0]]
    ))
    with pytest.raises(ValueError, match="RGB-pixel"):
        normalized_rgb_intrinsics(latent, image_size=(10, 6))


def test_normalization_and_c2w_to_w2c_use_declared_scale():
    camera = _camera()
    normalized = normalized_rgb_intrinsics(camera, image_size=(120, 160))
    w2c = world_to_camera(camera, translation_scale=0.5)

    assert torch.allclose(normalized[0, 0], torch.tensor(
        [[0.625, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ))
    assert w2c[0, 8, 0, 3] == -4


def test_projection_subframes_and_spatial_tiling_come_from_video_layout():
    layout = VideoLayout.from_codec(
        fps=16,
        rgb_frame_count=9,
        codec=CodecTemporalSpec(temporal_compression=2),
        temporal_patch_size=2,
        latent_chunk_size=2,
    )
    projection = build_token_camera_projection(
        _camera(), layout, grid_shape=(3, 2, 3), image_size=(120, 160)
    )

    assert projection.projection.shape == (1, 18, 2, 4, 4)
    # Six spatial tokens share each temporal token's camera matrices.
    assert torch.equal(projection.projection[:, 0], projection.projection[:, 5])
    assert not torch.equal(projection.projection[:, 5], projection.projection[:, 6])
    # The final short temporal patch repeats its last valid sub-frame.
    assert torch.equal(projection.projection[:, 12, 0], projection.projection[:, 12, 1])


def test_paired_prope_transforms_cancel_for_one_camera():
    layout = VideoLayout.from_codec(
        fps=16,
        rgb_frame_count=1,
        codec=CodecTemporalSpec(temporal_compression=2),
    )
    projection = build_token_camera_projection(
        _camera(1), layout, grid_shape=(1, 1, 1), image_size=(120, 160)
    )
    value = torch.randn(1, 2, 1, 8)
    transformed = apply_camera_projection(value, projection.inverse, camera_dims=8)
    restored = apply_camera_projection(transformed, projection.projection, camera_dims=8)
    assert torch.allclose(restored, value, atol=1e-5)


def test_joint_dit_composes_rope_visibility_and_prope():
    config = JointVideoDiTConfig(
        latent_channels=4,
        hidden_size=32,
        depth=1,
        num_heads=4,
        patch_size=(1, 2, 2),
        mlp_ratio=2,
        prope_camera_dims=8,
    )
    model = JointVideoDiT(config)
    latents = torch.randn(1, 4, 2, 2, 2)
    layout = VideoLayout.from_codec(
        fps=16,
        rgb_frame_count=2,
        codec=CodecTemporalSpec(temporal_compression=1, first_frame_is_independent=False),
    )
    projection = build_token_camera_projection(
        _camera(2), layout, grid_shape=(2, 1, 1), image_size=(120, 160)
    )
    output = model(
        latents,
        torch.tensor([0.5]),
        visibility_mask=torch.ones(2, 2, dtype=torch.bool).tril(),
        camera_projection=projection,
    )
    assert output.shape == latents.shape

