import pytest

from core.camera import CameraCondition, IntrinsicsSpace
from core.types import LatentSpec, VideoBatch
from core.video_layout import FrameRange, VideoLayout


def _layout():
    return VideoLayout(
        fps=16.0,
        rgb_frame_count=5,
        latent_frame_count=3,
        latent_to_rgb=(FrameRange(0, 1), FrameRange(1, 3), FrameRange(3, 5)),
        token_to_latent=(0, 0, 1, 1, 2, 2),
        chunk_to_latent=(FrameRange(0, 2), FrameRange(2, 3)),
        rgb_is_padding=(False, False, False, False, True),
        initial_condition_rgb=FrameRange(0, 1),
    )


def test_video_layout_is_the_mapping_query_surface():
    layout = _layout()

    assert layout.rgb_range_for_latent(2) == FrameRange(3, 5)
    assert layout.latent_index_for_token(4) == 2
    assert layout.latent_range_for_chunk(1) == FrameRange(2, 3)
    assert layout.rgb_time_seconds(4) == 0.25
    assert layout.is_padded_rgb(4)
    with pytest.raises(IndexError):
        layout.rgb_range_for_latent(-1)


def test_video_layout_rejects_inconsistent_boundaries():
    with pytest.raises(ValueError, match="beyond RGB"):
        VideoLayout(
            fps=16.0,
            rgb_frame_count=2,
            latent_frame_count=1,
            latent_to_rgb=(FrameRange(0, 3),),
            token_to_latent=(0,),
            chunk_to_latent=(FrameRange(0, 1),),
            rgb_is_padding=(False, False),
            initial_condition_rgb=FrameRange(0, 1),
        )


def test_initial_contracts_make_identity_and_conventions_explicit():
    latent_spec = LatentSpec(
        codec_id="example@sha256:abc",
        channels=16,
        temporal_compression=4,
        spatial_compression=(8, 8),
        normalization={"kind": "channel_stats", "version": "v1"},
        causal=True,
        encoding_policy="prefix_causal",
    )
    camera = CameraCondition(
        c2w="[B,T,4,4]",
        intrinsics="[B,T,3,3]",
        timestamps_seconds="[B,T]",
        intrinsics_space=IntrinsicsSpace.RGB_PIXELS,
    )
    batch = VideoBatch(
        sample_ids=("scene-1",),
        sources=("sana-wm",),
        layout=_layout(),
        camera=camera,
        video="[B,C,T,H,W]",
    )

    assert latent_spec.encoding_policy == "prefix_causal"
    assert camera.extrinsics_convention == "camera_to_world"
    assert batch.sample_ids == ("scene-1",)
