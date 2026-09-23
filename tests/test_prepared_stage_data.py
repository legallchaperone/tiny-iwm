import pytest

from core.video_layout import CodecTemporalSpec, VideoLayout
from scripts.prepared_stage_data import validate_prepared_record


def _layout():
    return VideoLayout.from_codec(
        fps=16,
        rgb_frame_count=161,
        codec=CodecTemporalSpec(8),
        latent_chunk_size=4,
    )


def test_prepared_record_geometry_can_be_checked_before_gpu_loading():
    validate_prepared_record(
        {"latent_shape": [128, 121, 4, 4], "camera_frames": 961}, _layout()
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
