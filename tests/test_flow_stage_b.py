import pytest
import torch

from algorithms.world_model.flow import FlowMatchSpec
from algorithms.world_model.models import JointVideoDiT, JointVideoDiTConfig
from algorithms.world_model.training_batch import ChunkCausalVisibility, StageBBatchBuilder
from core.camera import CameraCondition, IntrinsicsSpace
from core.types import VideoBatch
from core.video_layout import CodecTemporalSpec, VideoLayout


def _video_batch(latents: torch.Tensor | None = None) -> VideoBatch:
    layout = VideoLayout.from_codec(
        fps=1, rgb_frame_count=6, codec=CodecTemporalSpec(1),
        temporal_patch_size=1, latent_chunk_size=2, initial_condition_frames=1,
    )
    camera = CameraCondition(
        c2w=torch.eye(4).repeat(1, 6, 1, 1),
        intrinsics=torch.eye(3).repeat(1, 6, 1, 1),
        timestamps_seconds=torch.arange(6)[None].float(),
        intrinsics_space=IntrinsicsSpace.RGB_PIXELS,
    )
    if latents is None:
        latents = torch.arange(12.0).view(1, 2, 6, 1, 1)
    return VideoBatch(
        sample_ids=("scene",), sources=("test",), layout=layout,
        camera=camera, latents=latents,
    )


def test_stage_b_replaces_only_target_with_noisy_fm_state() -> None:
    batch = _video_batch()
    noise = torch.full_like(batch.latents, 10)
    built = StageBBatchBuilder(FlowMatchSpec()).build(
        batch, target_chunk=1, noise=noise, flow_time=torch.tensor([0.5])
    )
    assert torch.equal(built.noisy_latents[:, :, :2], batch.latents[:, :, :2])
    assert torch.equal(built.noisy_latents[:, :, 4:], batch.latents[:, :, 4:])
    assert torch.equal(
        built.noisy_latents[:, :, 2:4], (batch.latents[:, :, 2:4] + 10) / 2
    )
    assert built.clean_condition_latents[:, :, 2:4].eq(0).all()
    assert built.loss_mask[:, :, 2:4].all()
    assert not built.loss_mask[:, :, :2].any()
    assert not built.loss_mask[:, :, 4:].any()
    assert built.metadata["target_latent_indices"] == (2, 3)
    assert torch.equal(built.model_time, torch.tensor([[0, 0, 0.5, 0.5, 0, 0]]))


def test_chunk_mask_is_bidirectional_inside_and_causal_between_chunks() -> None:
    visibility = ChunkCausalVisibility((0, 0, 1, 1, 2, 2))
    mask = visibility.materialize(spatial_tokens_per_temporal_token=2)
    chunks = torch.tensor((0, 0, 1, 1, 2, 2)).repeat_interleave(2)
    assert torch.equal(mask, chunks[:, None] >= chunks[None, :])
    assert mask[4:8, 4:8].all()
    assert mask[4:8, :4].all()
    assert not mask[4:8, 8:].any()


def test_future_clean_values_cannot_reach_target_through_multiple_layers() -> None:
    torch.manual_seed(23)
    model = JointVideoDiT(JointVideoDiTConfig(
        latent_channels=2, hidden_size=24, depth=3, num_heads=4,
        patch_size=(1, 1, 1), mlp_ratio=2,
    )).eval()
    base = _video_batch()
    changed_latents = base.latents.clone()
    changed_latents[:, :, 4:] += 1000
    builder = StageBBatchBuilder(FlowMatchSpec())
    noise = torch.full_like(base.latents, 0.25)
    kwargs = {"target_chunk": 1, "noise": noise, "flow_time": torch.tensor([0.6])}
    first = builder.build(base, **kwargs)
    second = builder.build(_video_batch(changed_latents), **kwargs)
    mask = first.attention_visibility.materialize(spatial_tokens_per_temporal_token=1)
    with torch.no_grad():
        first_output = model(first.noisy_latents, first.flow_time, visibility_mask=mask)
        second_output = model(second.noisy_latents, second.flow_time, visibility_mask=mask)
    assert torch.equal(first_output[:, :, 2:4], second_output[:, :, 2:4])
    assert not torch.equal(first_output[:, :, 4:], second_output[:, :, 4:])


def test_clean_history_is_visible_to_target() -> None:
    torch.manual_seed(24)
    model = JointVideoDiT(JointVideoDiTConfig(
        latent_channels=2, hidden_size=16, depth=2, num_heads=4,
        patch_size=(1, 1, 1), mlp_ratio=2,
    )).eval()
    base = _video_batch()
    changed_latents = base.latents.clone()
    changed_latents[:, :, :2] += 10
    builder = StageBBatchBuilder(FlowMatchSpec())
    noise = torch.zeros_like(base.latents)
    kwargs = {"target_chunk": 1, "noise": noise, "flow_time": torch.tensor([0.5])}
    first = builder.build(base, **kwargs)
    second = builder.build(_video_batch(changed_latents), **kwargs)
    mask = first.attention_visibility.materialize(spatial_tokens_per_temporal_token=1)
    with torch.no_grad():
        first_output = model(first.noisy_latents, first.flow_time, visibility_mask=mask)
        second_output = model(second.noisy_latents, second.flow_time, visibility_mask=mask)
    assert not torch.allclose(first_output[:, :, 2:4], second_output[:, :, 2:4])


def test_temporal_patch_must_not_cross_chunk_boundary() -> None:
    layout = VideoLayout.from_codec(
        fps=1, rgb_frame_count=6, codec=CodecTemporalSpec(1),
        temporal_patch_size=3, latent_chunk_size=2, initial_condition_frames=1,
    )
    with pytest.raises(ValueError, match="must not cross"):
        ChunkCausalVisibility.from_layout(layout)
