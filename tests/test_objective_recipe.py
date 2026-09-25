"""E3 contract tests for FM and minimal DF through the shared recipe."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import open_dict
import pytest
import torch

from algorithms.world_model.flow import FlowMatchSpec, flow_matching_loss
from algorithms.world_model.objectives import CosineDFObjective, CosineDFSchedule, FMObjective
from algorithms.world_model.training_batch import StageBBatchBuilder
from algorithms.world_model.training_policies import build_training_policy
from core.camera import CameraCondition, IntrinsicsSpace
from core.types import VideoBatch
from core.video_layout import CodecTemporalSpec, VideoLayout
from experiments.build import build_recipe, resolve_recipe_selection
from inference.history import HistoryIdentity
from inference.rollout import rollout_latents
from inference.samplers import DFDDIMSampler, FMEulerSampler
from scripts.preflight import validate_preflight_config


CONFIG_DIR = Path(__file__).parents[1] / "configurations"


def _video(*, patch: int = 1) -> VideoBatch:
    layout = VideoLayout.from_codec(
        fps=1, rgb_frame_count=6, codec=CodecTemporalSpec(1),
        temporal_patch_size=patch, latent_chunk_size=2, initial_condition_frames=2,
    )
    camera = CameraCondition(
        c2w=torch.eye(4).repeat(1, 6, 1, 1),
        intrinsics=torch.eye(3).repeat(1, 6, 1, 1),
        timestamps_seconds=torch.arange(6)[None].float(),
        intrinsics_space=IntrinsicsSpace.RGB_PIXELS,
    )
    return VideoBatch(
        ("fixture",), ("fixture",), layout, camera,
        latents=torch.arange(12.0).reshape(1, 2, 6, 1, 1) / 10,
    )


def _small_config(objective: str):
    overrides = [
        "model.latent_channels=2", "model.hidden_size=24", "model.depth=1",
        "model.num_heads=4", "model.patch_size=[1,1,1]",
        "model.prope_camera_dims=null", "conditioning.camera.enabled=false",
        "stage=causal_direct", "training_policy=teacher_forced_causal",
        "rollout=debug_81",
    ]
    if objective == "minimal_df":
        overrides.extend(("objective=minimal_df", "sampler=df_ddim"))
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=overrides)


def test_hydra_recipes_select_both_semantics_and_direct_causal_init():
    for name, expected in (("native_fm", "velocity"), ("minimal_df", "epsilon")):
        cfg = _small_config(name)
        validate_preflight_config(cfg)
        recipe = build_recipe(cfg)
        assert recipe.objective.name == name
        assert recipe.objective.prediction_type == expected
        assert recipe.sampling_steps == 4
        assert cfg.stage.initial_checkpoint is None
        assert cfg.model.initialization == "random"


def test_direct_causal_preflight_rejects_parent_checkpoint():
    cfg = _small_config("minimal_df")
    cfg.checkpoint.init_from = "stage-a.pt"
    with pytest.raises(ValueError, match="causal_direct requires random initialization"):
        validate_preflight_config(cfg)


def test_fm_adapter_matches_existing_stage_b_math():
    video = _video()
    noise = torch.full_like(video.latents, 2)
    time = torch.tensor([0.4])
    spec = FlowMatchSpec()
    old = StageBBatchBuilder(spec).build(
        video, target_chunk=1, noise=noise, flow_time=time
    )
    new = FMObjective(spec, build_training_policy("teacher_forced_causal")).build_batch(
        video, target_chunk=1, noise=noise, flow_time=time
    )
    torch.testing.assert_close(new.model_input, old.noisy_latents)
    torch.testing.assert_close(new.noise_condition, old.model_time)
    torch.testing.assert_close(new.prediction_target, old.target_velocity)
    prediction = torch.zeros_like(noise)
    assert torch.equal(
        FMObjective(spec, build_training_policy("teacher_forced_causal")).loss(prediction, new),
        flow_matching_loss(prediction, old.target_velocity, loss_mask=old.loss_mask, time=time, spec=spec),
    )


def test_fm_stage_b_bf16_target_and_fp32_prediction_match_previous_loss():
    source = _video()
    video = VideoBatch(
        source.sample_ids, source.sources, source.layout, source.camera,
        latents=source.latents.to(torch.bfloat16),
    )
    spec = FlowMatchSpec()
    objective = FMObjective(spec, build_training_policy("teacher_forced_causal"))
    batch = objective.build_batch(
        video, target_chunk=1,
        noise=torch.full_like(video.latents, 2),
        flow_time=torch.tensor([0.5], dtype=torch.bfloat16),
    )
    prediction = torch.zeros_like(batch.prediction_target, dtype=torch.float32, requires_grad=True)
    actual = objective.loss(prediction, batch)
    expected = flow_matching_loss(
        prediction, batch.prediction_target.float(), loss_mask=batch.loss_mask,
        time=batch.loss_time.float(), spec=spec,
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_df_uses_temporal_token_noise_and_exact_clean_history():
    video = _video(patch=2)
    schedule = CosineDFSchedule(train_steps=10)
    objective = CosineDFObjective(schedule, build_training_policy("teacher_forced_causal"))
    steps = torch.tensor([[2, 8, 5]])
    noise = torch.full_like(video.latents, 3)
    batch = objective.build_batch(
        video, target_chunk=1, noise=noise, timestep_indices=steps
    )
    assert batch.prediction_type == "epsilon"
    torch.testing.assert_close(batch.noise_condition, torch.tensor([[0.0, 0.8, 0.0]]))
    torch.testing.assert_close(batch.model_input[:, :, :2], video.latents[:, :, :2])
    torch.testing.assert_close(batch.model_input[:, :, 4:], video.latents[:, :, 4:])
    assert batch.loss_mask[:, :, 2:4].all()
    assert not batch.loss_mask[:, :, :2].any()
    assert objective.loss(noise, batch) == 0
    with pytest.raises(ValueError, match="epsilon"):
        objective.loss(noise, batch.__class__(
            batch.model_input, batch.noise_condition, batch.prediction_target,
            "velocity", batch.loss_mask, batch.attention_visibility, batch.layout,
        ))


def test_df_bf16_latents_keep_adjacent_timestep_conditions_distinct():
    source = _video()
    video = VideoBatch(
        source.sample_ids, source.sources, source.layout, source.camera,
        latents=source.latents.to(torch.bfloat16),
    )
    objective = CosineDFObjective(
        CosineDFSchedule(train_steps=1000),
        build_training_policy("teacher_forced_causal"),
    )
    noise = torch.full_like(video.latents, 2)
    first = objective.build_batch(
        video, target_chunk=1, noise=noise,
        timestep_indices=torch.tensor([[1, 1, 251, 1, 1, 1]]),
    )
    second = objective.build_batch(
        video, target_chunk=1, noise=noise,
        timestep_indices=torch.tensor([[1, 1, 252, 1, 1, 1]]),
    )
    assert first.noise_condition.dtype == torch.float32
    assert second.noise_condition.dtype == torch.float32
    assert first.noise_condition[0, 2] != second.noise_condition[0, 2]
    assert first.noise_condition[0, 0] == second.noise_condition[0, 0] == 0
    assert objective.schedule.alpha_bar(first.noise_condition[0, 2]) != objective.schedule.alpha_bar(second.noise_condition[0, 2])


@pytest.mark.parametrize("objective", ["native_fm", "minimal_df"])
def test_same_recipe_training_and_rollout_path(objective):
    torch.manual_seed(7)
    recipe = build_recipe(_small_config(objective))
    video = _video()
    kwargs = {"target_chunk": 1, "noise": torch.randn_like(video.latents)}
    if objective == "native_fm":
        kwargs["flow_time"] = torch.tensor([0.5])
    else:
        kwargs["timestep_indices"] = torch.tensor([[1, 4, 6, 8, 2, 3]])
    batch = recipe.objective.build_batch(video, **kwargs)
    mask = batch.attention_visibility.materialize(spatial_tokens_per_temporal_token=1)
    optimizer = torch.optim.AdamW(recipe.model.model.parameters(), lr=1e-3)
    prediction = recipe.model.model(
        batch.model_input, batch.noise_condition, visibility_mask=mask
    )
    loss = recipe.objective.loss(prediction, batch)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    model = recipe.model.model.eval()
    identity = HistoryIdentity("episode", "checkpoint", "source", "conditional")
    initial = (video.latents[:, :, :2],)
    generated = rollout_latents(
        model, video.layout, identity, initial, steps=2, seed=5,
        mode="reference", sampler=recipe.sampler, objective_name=recipe.objective.name,
    )
    cached = rollout_latents(
        model, video.layout, identity, initial, steps=2, seed=5,
        mode="cached", sampler=recipe.sampler, objective_name=recipe.objective.name,
    )
    assert generated.shape == video.latents.shape
    assert torch.isfinite(generated).all()
    torch.testing.assert_close(generated[:, :, :2], initial[0])
    torch.testing.assert_close(cached, generated, atol=2e-5, rtol=2e-5)


def test_incompatible_recipes_fail_before_model_build():
    cfg = _small_config("minimal_df")
    cfg.sampler.name = "fm_euler"
    with pytest.raises(ValueError, match="requires sampler"):
        resolve_recipe_selection(cfg)
    cfg = _small_config("native_fm")
    cfg.sampler.prediction_type = "epsilon"
    with pytest.raises(ValueError, match="prediction_type"):
        resolve_recipe_selection(cfg)
    with pytest.raises(ValueError, match="requires objective"):
        rollout_latents(
            build_recipe(_small_config("native_fm")).model.model,
            _video().layout,
            HistoryIdentity("ep", "ckpt", "source", "conditional"),
            (_video().latents[:, :, :2],), steps=1, seed=1,
            sampler=DFDDIMSampler(), objective_name="native_fm",
        )


def test_preflight_rejects_silent_policy_and_stage_overrides():
    cfg = _small_config("minimal_df")
    cfg.training_policy.visibility = "bidirectional"
    with pytest.raises(ValueError, match="requires visibility=chunk_causal"):
        validate_preflight_config(cfg)
    cfg = _small_config("minimal_df")
    cfg.stage.attention_visibility = "bidirectional"
    with pytest.raises(ValueError, match="requires attention_visibility=chunk_causal"):
        validate_preflight_config(cfg)


def test_preflight_rejects_ignored_objective_and_sampler_fields():
    cfg = _small_config("minimal_df")
    with open_dict(cfg.sampler):
        cfg.sampler.noise_scale = 2.0
    with pytest.raises(ValueError, match="unsupported sampler fields"):
        validate_preflight_config(cfg)
    cfg = _small_config("native_fm")
    with open_dict(cfg.objective):
        cfg.objective.noise_scale = 2.0
    with pytest.raises(ValueError, match="unsupported objective fields"):
        validate_preflight_config(cfg)
    cfg = _small_config("native_fm")
    cfg.objective.flow.time_distribution = "logit_normal"
    with pytest.raises(ValueError, match="only explicit uniform"):
        validate_preflight_config(cfg)


def test_fm_preflight_rejects_training_bounds_outside_euler_domain():
    cfg = _small_config("native_fm")
    cfg.objective.flow.time_min = 0.2
    cfg.objective.flow.time_max = 0.8
    with pytest.raises(ValueError, match="fm_euler requires"):
        validate_preflight_config(cfg)
    with pytest.raises(ValueError, match="fm_euler requires"):
        build_recipe(cfg)


def test_preflight_rejects_df_step_count_without_distinct_schedule_levels():
    cfg = _small_config("minimal_df")
    cfg.sampler.steps = 1000
    with pytest.raises(ValueError, match="distinct schedule levels"):
        validate_preflight_config(cfg)
    with pytest.raises(ValueError, match="distinct schedule levels"):
        build_recipe(cfg)


def test_ddim_matches_epsilon_reconstruction_and_differs_from_fm_euler():
    sampler = DFDDIMSampler(CosineDFSchedule(train_steps=10))
    grid = sampler.grid(2, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(grid, torch.tensor([1.0, 0.5, 0.0]))
    clean = torch.full((1, 1, 1, 1, 1), 2.0)
    epsilon = torch.full_like(clean, 0.25)
    alpha = sampler.schedule.alpha_bar(torch.tensor([0.5]))
    state = alpha.sqrt().reshape(1, 1, 1, 1, 1) * clean + (1 - alpha).sqrt().reshape(1, 1, 1, 1, 1) * epsilon
    recovered = sampler.step(
        state, epsilon, time=torch.tensor([0.5]), next_time=torch.tensor([0.0])
    )
    torch.testing.assert_close(recovered, clean)
    assert not torch.allclose(
        recovered,
        FMEulerSampler().step(state, epsilon, time=torch.tensor([0.5]), next_time=torch.tensor([0.0])),
    )


def test_fm_bf16_sampler_keeps_dense_timestep_updates():
    sampler = FMEulerSampler()
    grid = sampler.grid(400, device=torch.device("cpu"), dtype=torch.bfloat16)
    assert grid.dtype == torch.float32
    assert torch.all(grid[:-1] > grid[1:])
    collapsed = torch.nonzero(
        grid[:-1].to(torch.bfloat16) == grid[1:].to(torch.bfloat16)
    ).flatten()
    assert len(collapsed) > 0
    index = int(collapsed[0])
    state = torch.zeros((1, 1, 1, 1, 1), dtype=torch.bfloat16)
    velocity = torch.ones_like(state)
    updated = sampler.step(
        state, velocity,
        time=grid[index:index + 1], next_time=grid[index + 1:index + 2],
    )
    assert updated.dtype == torch.bfloat16
    assert updated.item() < 0


def test_df_grid_stays_strictly_decreasing_with_bf16_latents():
    sampler = DFDDIMSampler(CosineDFSchedule(train_steps=1000))
    grid = sampler.grid(400, device=torch.device("cpu"), dtype=torch.bfloat16)
    assert grid.dtype == torch.float32
    assert grid[0] == 1 and grid[-1] == 0
    assert torch.all(grid[:-1] > grid[1:])
    coefficients = sampler.schedule.alpha_bar(grid)
    assert torch.all(coefficients[1:] > coefficients[:-1])
    assert grid[1] < 0.998  # skip the floored t=0.998 coefficient
    with pytest.raises(ValueError, match="distinct schedule levels"):
        sampler.grid(1000, device=torch.device("cpu"), dtype=torch.bfloat16)


def test_ddim_bf16_update_uses_distinct_fp32_schedule_coefficients():
    sampler = DFDDIMSampler(CosineDFSchedule(train_steps=1000))
    grid = sampler.grid(400, device=torch.device("cpu"), dtype=torch.bfloat16)
    time, next_time = grid[290:291], grid[291:292]
    alpha = sampler.schedule.alpha_bar(time)
    next_alpha = sampler.schedule.alpha_bar(next_time)
    assert alpha != next_alpha
    assert alpha.to(torch.bfloat16) == next_alpha.to(torch.bfloat16)
    state = torch.ones((1, 1, 1, 1, 1), dtype=torch.bfloat16)
    epsilon = torch.ones_like(state)
    updated = sampler.step(state, epsilon, time=time, next_time=next_time)
    shape = (1, 1, 1, 1, 1)
    expected_clean = (
        state.float() - (1 - alpha).sqrt().reshape(shape) * epsilon.float()
    ) / alpha.sqrt().reshape(shape)
    expected = (
        next_alpha.sqrt().reshape(shape) * expected_clean
        + (1 - next_alpha).sqrt().reshape(shape) * epsilon.float()
    )
    torch.testing.assert_close(updated, expected.to(torch.bfloat16))
    assert not torch.equal(updated, state)
