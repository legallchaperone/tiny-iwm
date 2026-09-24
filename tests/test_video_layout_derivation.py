import pytest

from core.video_layout import CodecTemporalSpec, FrameRange, VideoLayout


def test_causal_codec_derives_first_frame_and_exact_final_length():
    layout = VideoLayout.from_codec(
        fps=16.0,
        rgb_frame_count=961,
        codec=CodecTemporalSpec(temporal_compression=4),
        temporal_patch_size=2,
        latent_chunk_size=64,
    )

    assert layout.valid_latent_frame_count == 241
    assert layout.latent_frame_count == 242
    assert layout.rgb_frame_count == 965
    assert layout.valid_rgb_frame_count == 961
    assert layout.rgb_range_for_latent(0) == FrameRange(0, 1)
    assert layout.rgb_range_for_latent(240) == FrameRange(957, 961)
    assert layout.latent_range_for_token(0) == FrameRange(0, 2)
    assert layout.latent_range_for_token(120) == FrameRange(240, 242)
    assert layout.latent_range_for_chunk(3) == FrameRange(192, 242)
    assert layout.writer_rgb_indices_for_chunk(3)[-1] == 960


def test_suffix_padding_is_explicit_and_writer_crops_to_requested_length():
    layout = VideoLayout.from_codec(
        fps=10.0,
        rgb_frame_count=8,
        codec=CodecTemporalSpec(temporal_compression=4),
        latent_chunk_size=2,
    )

    assert layout.latent_frame_count == 3
    assert layout.rgb_frame_count == 9
    assert layout.valid_rgb_frame_count == 8
    assert layout.rgb_is_padding == (False,) * 8 + (True,)
    assert layout.rgb_range_for_latent(2) == FrameRange(5, 9)
    assert layout.camera_rgb_index_for_latent(2) == 7
    assert layout.camera_rgb_indices_for_chunk(1) == (7,)
    assert layout.writer_rgb_indices_for_chunk(1) == (5, 6, 7)


def test_uniform_codec_and_chunk_boundaries_share_one_mapping():
    layout = VideoLayout.from_codec(
        fps=20.0,
        rgb_frame_count=10,
        codec=CodecTemporalSpec(temporal_compression=3, first_frame_is_independent=False),
        temporal_patch_size=3,
        latent_chunk_size=3,
        initial_condition_frames=2,
    )

    assert layout.latent_frame_count == 6
    assert layout.rgb_frame_count == 18
    assert layout.latent_to_rgb == (
        FrameRange(0, 3),
        FrameRange(3, 6),
        FrameRange(6, 9),
        FrameRange(9, 12),
        FrameRange(12, 15),
        FrameRange(15, 18),
    )
    assert layout.token_to_latent_ranges == (FrameRange(0, 3), FrameRange(3, 6))
    assert layout.chunk_to_latent == (FrameRange(0, 3), FrameRange(3, 6))
    assert layout.writer_rgb_indices_for_chunk(1) == (9,)
    assert layout.camera_rgb_indices_for_chunk(1) == (9, 9, 9)
    assert layout.token_time_range_seconds(1) == pytest.approx((0.45, 0.5))


def test_factory_rejects_invalid_codec_and_group_properties():
    with pytest.raises(ValueError, match="temporal_compression"):
        CodecTemporalSpec(0)
    with pytest.raises(ValueError, match="patch and chunk"):
        VideoLayout.from_codec(
            fps=16,
            rgb_frame_count=5,
            codec=CodecTemporalSpec(4),
            temporal_patch_size=0,
        )
