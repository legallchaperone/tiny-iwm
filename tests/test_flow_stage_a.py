import torch
import pytest

from algorithms.world_model.flow import (
    FlowMatchSpec,
    euler_step,
    flow_matching_loss,
    interpolate,
    target_velocity,
)
from algorithms.world_model.training_batch import StageABatchBuilder
from core.camera import CameraCondition, IntrinsicsSpace
from core.types import VideoBatch
from core.video_layout import CodecTemporalSpec, FrameRange, VideoLayout


def test_native_flow_endpoints_velocity_and_reverse_sampling():
    clean = torch.tensor([[[[[2.0]]]]])
    noise = torch.tensor([[[[[5.0]]]]])

    assert torch.equal(interpolate(clean, noise, torch.tensor([0.0])), clean)
    assert torch.equal(interpolate(clean, noise, torch.tensor([1.0])), noise)
    velocity = target_velocity(clean, noise)
    assert torch.equal(velocity, torch.tensor([[[[[3.0]]]]]))
    restored = euler_step(
        noise, velocity, time=torch.tensor([1.0]), next_time=torch.tensor([0.0])
    )
    assert torch.equal(restored, clean)


def test_time_sampling_and_loss_weighting_are_explicit():
    spec = FlowMatchSpec(time_min=0.2, time_max=0.8)
    time = spec.sample_time(128, device=torch.device("cpu"), generator=torch.Generator().manual_seed(3))
    assert torch.all((time >= 0.2) & (time < 0.8))
    assert torch.equal(spec.loss_weights(time), torch.ones_like(time))
    with pytest.raises(ValueError, match="uniform"):
        FlowMatchSpec(time_distribution="logit_normal")


def _video_batch() -> VideoBatch:
    layout = VideoLayout.from_codec(
        fps=16,
        rgb_frame_count=9,
        codec=CodecTemporalSpec(temporal_compression=2),
        initial_condition_frames=1,
    )
    camera = CameraCondition(
        c2w=torch.eye(4).repeat(1, layout.rgb_frame_count, 1, 1),
        intrinsics=torch.eye(3).repeat(1, layout.rgb_frame_count, 1, 1),
        timestamps_seconds=torch.arange(layout.rgb_frame_count)[None] / 16,
        intrinsics_space=IntrinsicsSpace.RGB_PIXELS,
    )
    return VideoBatch(
        sample_ids=("scene",),
        sources=("test",),
        layout=layout,
        camera=camera,
        latents=torch.arange(5.0).view(1, 1, 5, 1, 1),
    )


def test_stage_a_keeps_condition_clean_and_masks_it_from_loss():
    batch = _video_batch()
    noise = torch.full_like(batch.latents, 10)
    built = StageABatchBuilder(FlowMatchSpec()).build(
        batch, noise=noise, flow_time=torch.tensor([0.25])
    )

    assert built.noisy_latents[0, 0, 0, 0, 0] == batch.latents[0, 0, 0, 0, 0]
    assert built.noisy_latents[0, 0, 1, 0, 0] == pytest.approx(3.25)
    assert not built.loss_mask[0, 0, 0, 0, 0]
    assert built.loss_mask[0, 0, 1:, 0, 0].all()
    assert built.target_velocity[0, 0, 0, 0, 0] == 0
    assert built.clean_condition_latents[0, 0, 1:, 0, 0].eq(0).all()
    assert built.attention_visibility.mode == "bidirectional"
    assert built.metadata["condition_latent_indices"] == (0,)


def test_stage_a_uses_one_time_per_example_and_respects_valid_mask():
    batch = _video_batch()
    valid = torch.ones(1, 5, 1, 1, dtype=torch.bool)
    valid[:, -1] = False
    built = StageABatchBuilder(FlowMatchSpec()).build(
        batch,
        noise=torch.ones_like(batch.latents),
        flow_time=torch.tensor([0.5]),
        valid_latent_mask=valid,
    )
    assert built.flow_time.shape == (1,)
    assert not built.loss_mask[0, 0, -1, 0, 0]
    assert built.target_velocity[0, 0, -1, 0, 0] == 0


def test_masked_flow_loss_ignores_conditions_and_invalid_values():
    spec = FlowMatchSpec()
    prediction = torch.tensor([[[[[100.0]], [[2.0]], [[9.0]]]]])
    target = torch.tensor([[[[[0.0]], [[0.0]], [[0.0]]]]])
    mask = torch.tensor([[[[[False]], [[True]], [[False]]]]])
    loss = flow_matching_loss(
        prediction, target, loss_mask=mask, time=torch.tensor([0.4]), spec=spec
    )
    assert loss == 4


def test_stage_a_rejects_condition_boundary_inside_a_codec_latent():
    batch = _video_batch()
    broken = VideoLayout(
        fps=batch.layout.fps,
        rgb_frame_count=batch.layout.rgb_frame_count,
        latent_frame_count=batch.layout.latent_frame_count,
        latent_to_rgb=batch.layout.latent_to_rgb,
        token_to_latent=batch.layout.token_to_latent,
        chunk_to_latent=batch.layout.chunk_to_latent,
        rgb_is_padding=batch.layout.rgb_is_padding,
        initial_condition_rgb=FrameRange(0, 2),
        valid_rgb_frame_count=batch.layout.valid_rgb_frame_count,
        token_to_latent_ranges=batch.layout.token_to_latent_ranges,
    )
    malformed = VideoBatch(
        sample_ids=batch.sample_ids,
        sources=batch.sources,
        layout=broken,
        camera=batch.camera,
        latents=batch.latents,
    )
    with pytest.raises(ValueError, match="latent boundary"):
        StageABatchBuilder(FlowMatchSpec()).build(malformed)
