import json
import pickle
from dataclasses import asdict

import pytest

from core.camera import CameraCondition, IntrinsicsSpace
from core.types import LatentSpec, ProbeEvent, RolloutResult, VideoBatch
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


def test_video_batch_rejects_misaligned_text():
    camera = CameraCondition(
        c2w="[B,T,4,4]",
        intrinsics="[B,T,3,3]",
        timestamps_seconds="[B,T]",
        intrinsics_space=IntrinsicsSpace.RGB_PIXELS,
    )
    with pytest.raises(ValueError, match="text must align"):
        VideoBatch(
            sample_ids=("a", "b"),
            sources=("a.mp4", "b.mp4"),
            layout=_layout(),
            camera=camera,
            video=object(),
            text=("only one caption",),
        )


def test_rollout_result_snapshots_and_freezes_identity_mappings():
    conditions = {"camera": {"angles": [1.0, 2.0]}}
    settings = {"steps": 8}
    result = RolloutResult(
        output_files=("sample.mp4",),
        latent_files=("sample.pt",),
        checkpoint_id="checkpoint-1",
        seed=7,
        conditions=conditions,
        inference_settings=settings,
    )

    conditions["camera"]["angles"].append(3.0)
    settings["steps"] = 16

    assert result.conditions["camera"]["angles"] == (1.0, 2.0)
    assert result.inference_settings["steps"] == 8
    with pytest.raises(TypeError):
        result.conditions["camera"] = {}

    restored = pickle.loads(pickle.dumps(result))
    serialized = asdict(result)
    assert restored.conditions == result.conditions
    assert serialized["conditions"] == result.conditions
    assert json.loads(json.dumps(result.conditions))["camera"]["angles"] == [1.0, 2.0]


def test_probe_event_carries_complete_run_and_position_identity():
    event = ProbeEvent(
        sample_id="scene-1",
        training_seed=17,
        generation_seed=23,
        checkpoint_id="stage-a-step-1000",
        config_id="sha256:config",
        layer="blocks.3",
        observation_point="post_attention",
        rollout_time=8,
        flow_time=0.5,
        token_type="target",
        physical_position={"time_seconds": 2.0, "y": 4.0, "x": 6.0},
        branch="conditional",
        forward_purpose="denoise",
    )

    assert event.checkpoint_id == "stage-a-step-1000"
    assert event.generation_seed == 23
    assert event.physical_position["time_seconds"] == 2.0
