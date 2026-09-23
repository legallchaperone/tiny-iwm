import pytest
import torch

from algorithms.world_model.models import JointVideoDiT, JointVideoDiTConfig
from algorithms.world_model.models.prope import TokenCameraProjection
from core.video_layout import CodecTemporalSpec, VideoLayout
from inference.history import HistoryIdentity, HistorySession
from inference.rollout import rollout_latents


def _setup(initial_frames=2):
    torch.manual_seed(12)
    model = JointVideoDiT(
        JointVideoDiTConfig(
            latent_channels=2,
            hidden_size=24,
            depth=2,
            num_heads=4,
            patch_size=(1, 1, 1),
            mlp_ratio=2,
            prope_camera_dims=4,
        )
    ).eval()
    layout = VideoLayout.from_codec(
        fps=1,
        rgb_frame_count=6,
        codec=CodecTemporalSpec(1),
        latent_chunk_size=2,
        initial_condition_frames=initial_frames,
    )
    identity = HistoryIdentity("episode", "checkpoint", "condition", "conditional")
    matrices = torch.eye(4).repeat(1, 24, 1, 1, 1)
    matrices[:, :, 0, 0, 3] = torch.arange(6).repeat_interleave(4) * 0.1
    inverse = torch.linalg.inv(matrices)
    camera = TokenCameraProjection(matrices, matrices.transpose(-1, -2), inverse)
    return model, layout, identity, camera


def test_multi_chunk_cache_matches_reference_and_denoising_does_not_commit():
    model, layout, identity, camera = _setup()
    session = HistorySession(model, layout, identity, camera_projection=camera)
    first = torch.randn(1, 2, 2, 2, 2)
    session.commit_clean(identity, 0, first)
    for index in (1, 2):
        target = torch.randn(1, 2, 2, 2, 2)
        old_cache = session.kv_cache
        for time in (0.9, 0.4):
            t = torch.tensor([time])
            cached = session.predict(identity, index, target, t, mode="cached")
            reference = session.predict(identity, index, target, t, mode="reference")
            torch.testing.assert_close(cached, reference, atol=2e-5, rtol=2e-5)
            assert session.kv_cache is old_cache
        session.commit_clean(identity, index, target)
        assert session.kv_cache[0].key.shape[2] == (index + 1) * 8


def test_history_rejects_cross_identity_and_out_of_order_commit():
    model, layout, identity, camera = _setup()
    session = HistorySession(model, layout, identity, camera_projection=camera)
    chunk = torch.zeros(1, 2, 2, 2, 2)
    wrong = HistoryIdentity("episode", "other-checkpoint", "condition", "conditional")
    with pytest.raises(ValueError, match="identity"):
        session.commit_clean(wrong, 0, chunk)
    with pytest.raises(ValueError, match="order"):
        session.commit_clean(identity, 1, chunk)


def test_rollout_uses_same_solver_for_cached_and_reference_modes():
    model, layout, identity, camera = _setup()
    first = (torch.randn(1, 2, 2, 2, 2),)
    cached = rollout_latents(
        model,
        layout,
        identity,
        first,
        steps=2,
        seed=7,
        mode="cached",
        camera_projection=camera,
    )
    reference = rollout_latents(
        model,
        layout,
        identity,
        first,
        steps=2,
        seed=7,
        mode="reference",
        camera_projection=camera,
    )
    assert cached.shape == (1, 2, 6, 2, 2)
    torch.testing.assert_close(cached, reference, atol=2e-5, rtol=2e-5)


def test_single_frame_initial_condition_inside_first_chunk_matches_reference():
    model, layout, identity, camera = _setup(initial_frames=1)
    first = (torch.randn(1, 2, 1, 2, 2),)
    cached = rollout_latents(
        model, layout, identity, first, steps=2, seed=7,
        mode="cached", camera_projection=camera,
    )
    reference = rollout_latents(
        model, layout, identity, first, steps=2, seed=7,
        mode="reference", camera_projection=camera,
    )
    assert cached.shape == (1, 2, 6, 2, 2)
    torch.testing.assert_close(cached, reference, atol=2e-5, rtol=2e-5)


def test_source_image_only_rollout_rejects_prefilled_history():
    model, layout, identity, camera = _setup(initial_frames=1)
    initial = torch.randn(1, 2, 1, 2, 2)
    extra = torch.randn(1, 2, 1, 2, 2)
    with pytest.raises(ValueError, match="rejects extra clean chunks"):
        rollout_latents(
            model,
            layout,
            identity,
            (initial, extra),
            steps=2,
            seed=7,
            initial_history_policy="source_image_only",
            camera_projection=camera,
        )
