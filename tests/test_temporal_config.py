import pytest
from pathlib import Path

import yaml

from core.temporal_config import resolve_temporal_protocol
from core.video_layout import CodecTemporalSpec


def _config(*, rollout=81, chunk=4, initial_condition=9, train=161):
    return {
        "train": {"window_rgb_frames": train},
        "temporal": {
            "chunk_latent_frames": chunk,
            "initial_condition_rgb_frames": initial_condition,
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
    assert train.valid_latent_frame_count == 21
    assert train.latent_frame_count == 22
    assert train.rgb_frame_count == 169
    assert train.valid_target_chunk_indices == tuple(range(6))
    assert rollout.valid_latent_frame_count == 11
    assert rollout.latent_frame_count == 12
    assert rollout.rgb_frame_count == 89
    assert rollout.valid_target_chunk_indices == (0, 1, 2)
    assert protocol.artifact_kind == "debug"
    with pytest.raises(ValueError, match="debug artifact"):
        protocol.require_official_minute()



def test_validation_target_chunks_shrink_with_the_training_window():
    protocol = resolve_temporal_protocol(_config(train=81))
    layout = protocol.layout(
        CodecTemporalSpec(8), temporal_patch_size=2, purpose="train"
    )
    assert layout.valid_latent_frame_count == 11
    assert layout.valid_target_chunk_indices == (0, 1, 2)

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
    protocol.validate_resources(available_rgb_frames=161, purpose="train")
    with pytest.raises(ValueError, match="requires 961"):
        resolve_temporal_protocol(_config(rollout=961)).validate_resources(
            available_rgb_frames=161, purpose="rollout"
        )


@pytest.mark.parametrize("name", ["stage_a_baseline", "stage_b_baseline"])
def test_published_run_preserves_original_minute_training_geometry(name):
    path = Path(__file__).parents[1] / "configurations" / "runs" / f"{name}.yaml"
    protocol = resolve_temporal_protocol(yaml.safe_load(path.read_text()))
    assert protocol.train_window_rgb_frames == 961
    assert protocol.rollout_rgb_frames == 961
    assert protocol.layout(
        CodecTemporalSpec(8), temporal_patch_size=1, purpose="train"
    ).latent_frame_count == 121


def test_incompatible_chunk_and_patch_fails_during_layout_resolution():
    protocol = resolve_temporal_protocol(_config(chunk=3))
    with pytest.raises(ValueError, match="divisible"):
        protocol.layout(CodecTemporalSpec(8), temporal_patch_size=2, purpose="train")


def test_initial_condition_must_end_on_a_temporal_patch_boundary():
    protocol = resolve_temporal_protocol(_config(initial_condition=1))
    with pytest.raises(ValueError, match="initial condition latent boundary"):
        protocol.layout(CodecTemporalSpec(8), temporal_patch_size=2, purpose="train")


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("dataset", "fps"),
        ("dataset", "rgb_frames"),
        ("rollout", "duration_seconds"),
        ("train", "chunk_latent_frames"),
        ("flow_matching", "latent_frames_per_chunk"),
        ("stage", "latent_frames_per_chunk"),
        ("stage", "chunk_duration_seconds"),
    ],
)
def test_legacy_duplicate_temporal_owners_are_rejected(section, key):
    config = _config()
    config.setdefault(section, {})[key] = 1
    with pytest.raises(ValueError, match="owner|derived|legacy"):
        resolve_temporal_protocol(config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("train", "window_rgb_frames", 161.9),
        ("train", "window_rgb_frames", True),
        ("temporal", "chunk_latent_frames", 4.0),
        ("temporal", "initial_condition_rgb_frames", 1.5),
        ("rollout", "rgb_frames", False),
    ],
)
def test_frame_counts_must_be_exact_integers(section, key, value):
    config = _config()
    config[section][key] = value
    with pytest.raises(ValueError, match="positive integer"):
        resolve_temporal_protocol(config)


@pytest.mark.parametrize("fps", [True, "16", float("inf"), 0])
def test_fps_must_be_a_finite_positive_number(fps):
    config = _config()
    config["temporal"]["fps"] = fps
    with pytest.raises(ValueError, match="fps"):
        resolve_temporal_protocol(config)
