import numpy as np
import pytest
import torch

from core.video_layout import CodecTemporalSpec, VideoLayout
from scripts.prepared_stage_data import (
    load_prepared_sample,
    pad_prepared_latents,
    validate_prepared_record,
)


def _layout():
    return VideoLayout.from_codec(
        fps=16,
        rgb_frame_count=161,
        codec=CodecTemporalSpec(8),
        temporal_patch_size=2,
        latent_chunk_size=4,
        initial_condition_frames=9,
    )


def test_prepared_record_geometry_can_be_checked_before_gpu_loading():
    validate_prepared_record(
        {"latent_shape": [128, 121, 4, 4], "camera_frames": 961}, _layout()
    )


def test_prepared_record_needs_real_window_latents_not_patch_padding():
    layout = _layout()
    assert layout.latent_frame_count == 22
    assert layout.valid_latent_frame_count == 21
    validate_prepared_record(
        {"latent_shape": [128, 21, 4, 4], "camera_frames": 161}, layout
    )


def test_prepared_record_rejects_short_latents_and_camera_trajectory():
    with pytest.raises(ValueError, match="latent frames"):
        validate_prepared_record(
            {"latent_shape": [128, 20, 4, 4], "camera_frames": 961}, _layout()
        )
    with pytest.raises(ValueError, match="camera frames"):
        validate_prepared_record(
            {"latent_shape": [128, 121, 4, 4], "camera_frames": 160}, _layout()
        )


def test_prepared_cache_rejects_a_window_inside_a_codec_group():
    layout = VideoLayout.from_codec(
        fps=16,
        rgb_frame_count=160,
        codec=CodecTemporalSpec(8),
        temporal_patch_size=2,
        latent_chunk_size=4,
        initial_condition_frames=9,
    )
    record = {"latent_shape": [128, 121, 4, 4], "camera_frames": 160}
    with pytest.raises(ValueError, match="codec boundary"):
        validate_prepared_record(record, layout)
    with pytest.raises(ValueError, match="codec boundary"):
        pad_prepared_latents(np.ones((128, 121, 4, 4), dtype=np.float32), layout)


def test_patch_padding_does_not_copy_future_latents_and_is_masked_from_loss():
    layout = _layout()
    source = np.ones((2, 121, 2, 3), dtype=np.float32)
    source[:, layout.valid_latent_frame_count :] = 99

    padded, valid_mask = pad_prepared_latents(source, layout)

    assert padded.shape == (2, layout.latent_frame_count, 2, 3)
    assert np.all(padded[:, :21] == 1)
    assert np.all(padded[:, 21] == 0)
    assert valid_mask.shape == (layout.latent_frame_count, 2, 3)
    assert valid_mask[:21].all()
    assert not valid_mask[21].any()


def test_loader_uses_temporal_token_count_for_camera_projection(tmp_path):
    layout = _layout()
    path = tmp_path / "sample.npz"
    pose = np.repeat(np.eye(4, dtype=np.float32)[None], 161, axis=0)
    intrinsics = np.tile(np.array([1920, 1080, 960, 540], dtype=np.float32), (161, 1))
    latents = np.ones((2, 121, 2, 2), dtype=np.float32)
    latents[:, 21] = 99
    np.savez(path, z=latents, pose=pose, intrinsics=intrinsics)
    record = {
        "prepared_path": str(path),
        "latent_shape": [2, 121, 2, 2],
        "camera_frames": 161,
    }

    clean, camera, projection, valid_mask = load_prepared_sample(
        record,
        device=torch.device("cpu"),
        dtype=torch.float32,
        layout=layout,
    )

    assert clean.shape == (1, 2, 22, 2, 2)
    assert clean[:, :, 20].eq(1).all()
    assert clean[:, :, 21].eq(0).all()
    assert valid_mask[:, 20].all()
    assert not valid_mask[:, 21].any()
    assert camera.c2w.shape[1] == 161
    assert projection.projection.shape[1] == len(layout.token_to_latent_ranges) * 4
