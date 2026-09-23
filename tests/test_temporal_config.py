import pytest

from core.temporal_config import resolve_temporal_protocol
from core.video_layout import CodecTemporalSpec


def _config(*, rollout=81, chunk=4):
    return {
        "train": {"window_rgb_frames": 161},
        "temporal": {
            "chunk_latent_frames": chunk,
            "initial_condition_rgb_frames": 1,
            "fps": 16,
        },
        "rollout": {"rgb_frames": rollout},
    }


def test_training_and_debug_rollout_are_independent_and_padding_is_derived():
    protocol = resolve_temporal_protocol(_config())
    train = protocol.layout(CodecTemporalSpec(8), temporal_patch_size=2, purpose="train")
    rollout = protocol.layout(CodecTemporalSpec(8), temporal_patch_size=2, purpose="rollout")
    assert train.output_rgb_frame_count == 161
    assert rollout.output_rgb_frame_count == 81
    assert train.latent_frame_count == 22
    assert train.rgb_frame_count == 169
    assert rollout.latent_frame_count == 12
    assert rollout.rgb_frame_count == 89
    assert protocol.artifact_kind == "debug"
    with pytest.raises(ValueError, match="debug artifact"):
        protocol.require_official_minute()


def test_official_preset_alone_passes_minute_gate():
    protocol = resolve_temporal_protocol(_config(rollout=961))
    protocol.require_official_minute()
    assert protocol.rollout_duration_seconds == 60


def test_preallocation_validation_rejects_data_and_memory_shortfalls():
    protocol = resolve_temporal_protocol(_config())
    with pytest.raises(ValueError, match="source has 80"):
        protocol.validate_resources(available_rgb_frames=80)
    with pytest.raises(ValueError, match="exceeds limit"):
        protocol.validate_resources(estimated_gpu_gb=24, maximum_gpu_gb=16)


def test_incompatible_chunk_and_patch_fails_during_layout_resolution():
    protocol = resolve_temporal_protocol(_config(chunk=3))
    with pytest.raises(ValueError, match="divisible"):
        protocol.layout(CodecTemporalSpec(8), temporal_patch_size=2, purpose="train")


@pytest.mark.parametrize("legacy", [("dataset", "fps"), ("dataset", "rgb_frames"), ("rollout", "duration_seconds")])
def test_legacy_duplicate_temporal_owners_are_rejected(legacy):
    config = _config()
    config.setdefault(legacy[0], {})[legacy[1]] = 1
    with pytest.raises(ValueError, match="owner|derived"):
        resolve_temporal_protocol(config)
