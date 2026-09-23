import json
import math

import pytest
import torch

from algorithms.world_model.models import JointVideoDiT, JointVideoDiTConfig
from algorithms.world_model.models.prope import TokenCameraProjection
from core.video_layout import CodecTemporalSpec, VideoLayout
from probing.capture import FeatureRecorder, TokenSelection
from probing.replay import _tensor_sha256, controlled_replay


def _inputs():
    torch.manual_seed(33)
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
        initial_condition_frames=1,
    )
    gt = torch.randn(1, 2, 2, 2, 2)
    generated = gt.clone()
    generated[:, :, 1] += 0.5
    target = torch.randn(1, 2, 2, 2, 2)
    noise = torch.randn_like(target)
    matrices = torch.eye(4).repeat(1, 24, 1, 1, 1)
    inverse = torch.linalg.inv(matrices)
    camera = TokenCameraProjection(matrices, matrices.transpose(-1, -2), inverse)
    arguments = dict(
        gt_history=gt,
        generated_history=generated,
        target_clean=target,
        target_noise=noise,
        flow_time=torch.tensor([0.5]),
        target_chunk=1,
        camera_projection=camera,
        checkpoint_id="checkpoint",
        config_id="config",
        sample_id="held-out-scene",
        generation_id="generation",
        selection_id="selection",
        training_seed=21,
        generation_seed=42,
    )
    return model, layout, arguments


def test_controlled_replay_changes_only_history_and_captures_matching_tokens(tmp_path):
    model, layout, arguments = _inputs()
    without_probe = controlled_replay(model, layout, **arguments)
    recorder = FeatureRecorder(
        tmp_path,
        points=("final_norm",),
        tokens=(TokenSelection(8, "target", {"rgb_frame": 2.0}),),
        max_records=2,
    )
    with_probe = controlled_replay(model, layout, recorder=recorder, **arguments)
    assert torch.equal(with_probe.gt_velocity, without_probe.gt_velocity)
    assert torch.equal(with_probe.generated_velocity, without_probe.generated_velocity)
    report = with_probe.report
    assert report["comparison"] == "controlled_gt_vs_generated_history"
    assert report["observational_rollout"] is False
    assert report["metrics"]["prediction_max_abs_delta"] > 0
    assert (
        report["fixed"]["target_noise_sha256"] != report["changed"]["gt_history_sha256"]
    )
    records = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert {record["forward_purpose"] for record in records} == {
        "controlled_gt_history",
        "controlled_generated_history",
    }
    assert {record["token_index"] for record in records} == {8}
    assert {record["flow_time"] for record in records} == {0.5}
    assert {record["rollout_time"] for record in records} == {2}


def test_controlled_replay_rejects_changed_initial_condition():
    model, layout, arguments = _inputs()
    arguments["generated_history"] = arguments["generated_history"].clone()
    arguments["generated_history"][:, :, 0] += 1
    with pytest.raises(ValueError, match="initial condition"):
        controlled_replay(model, layout, **arguments)


def test_controlled_replay_fixes_full_initial_prefix():
    model, _, arguments = _inputs()
    layout = VideoLayout.from_codec(
        fps=1,
        rgb_frame_count=6,
        codec=CodecTemporalSpec(1),
        latent_chunk_size=2,
        initial_condition_frames=2,
    )
    arguments["generated_history"] = arguments["generated_history"].clone()
    arguments["generated_history"][:, :, 1] += 1
    with pytest.raises(ValueError, match="initial condition"):
        controlled_replay(model, layout, **arguments)


def test_controlled_replay_handles_initial_prefix_across_chunks():
    model, _, arguments = _inputs()
    layout = VideoLayout.from_codec(
        fps=1,
        rgb_frame_count=7,
        codec=CodecTemporalSpec(1),
        latent_chunk_size=2,
        initial_condition_frames=3,
    )
    arguments["gt_history"] = torch.randn(1, 2, 4, 2, 2)
    arguments["generated_history"] = arguments["gt_history"].clone()
    arguments["generated_history"][:, :, 3] += 0.5
    arguments["target_chunk"] = 2
    arguments["camera_projection"] = None
    result = controlled_replay(model, layout, **arguments)
    assert result.report["target_chunk"] == 2
    assert result.report["metrics"]["prediction_max_abs_delta"] > 0


def test_controlled_replay_handles_target_inside_condition_chunk():
    model, _, arguments = _inputs()
    layout = VideoLayout.from_codec(
        fps=1,
        rgb_frame_count=7,
        codec=CodecTemporalSpec(1),
        latent_chunk_size=2,
        initial_condition_frames=3,
    )
    arguments["gt_history"] = torch.randn(1, 2, 3, 2, 2)
    arguments["generated_history"] = arguments["gt_history"].clone()
    arguments["target_clean"] = torch.randn(1, 2, 1, 2, 2)
    arguments["target_noise"] = torch.randn_like(arguments["target_clean"])
    arguments["target_chunk"] = 1
    arguments["camera_projection"] = None
    result = controlled_replay(model, layout, **arguments)
    assert result.report["rollout_time_rgb_frame"] == 3
    assert result.report["metrics"]["prediction_max_abs_delta"] == 0


def test_controlled_replay_handles_first_partially_conditioned_chunk():
    model, layout, arguments = _inputs()
    layout = VideoLayout.from_codec(
        fps=1,
        rgb_frame_count=6,
        codec=CodecTemporalSpec(1),
        latent_chunk_size=2,
        initial_condition_frames=1,
    )
    arguments["gt_history"] = torch.randn(1, 2, 1, 2, 2)
    arguments["generated_history"] = arguments["gt_history"].clone()
    arguments["target_clean"] = torch.randn(1, 2, 1, 2, 2)
    arguments["target_noise"] = torch.randn_like(arguments["target_clean"])
    arguments["target_chunk"] = 0
    arguments["camera_projection"] = None
    result = controlled_replay(model, layout, **arguments)
    assert result.report["rollout_time_rgb_frame"] == 1
    assert result.report["metrics"]["prediction_max_abs_delta"] == 0


def test_controlled_replay_rejects_nonfinite_history():
    model, layout, arguments = _inputs()
    arguments["generated_history"] = arguments["generated_history"].clone()
    arguments["generated_history"][:, :, 1] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        controlled_replay(model, layout, **arguments)


def test_controlled_replay_metrics_promote_before_subtracting(monkeypatch):
    model, layout, arguments = _inputs()
    arguments["target_clean"] = torch.full_like(
        arguments["target_clean"], 65504, dtype=torch.float16
    )
    arguments["target_noise"] = -arguments["target_clean"]
    calls = iter((40000.0, -40000.0))

    def predict(_, identity, chunk_index, noisy_chunk, flow_time, **kwargs):
        return torch.full_like(noisy_chunk, next(calls))

    monkeypatch.setattr("inference.history.HistorySession.predict", predict)
    metrics = controlled_replay(model, layout, **arguments).report["metrics"]
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["prediction_max_abs_delta"] == 80000.0


def test_tensor_hash_ignores_unrelated_backing_storage():
    values = torch.tensor([[1.0], [2.0], [3.0]])
    assert _tensor_sha256(values[1:2]) == _tensor_sha256(values[1:2].clone())
