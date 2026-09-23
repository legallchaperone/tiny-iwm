"""Held-out GT versus generated-history replay with one fixed FM target."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import math

import torch

from algorithms.world_model.flow import interpolate, target_velocity
from algorithms.world_model.models import JointVideoDiT
from algorithms.world_model.models.prope import TokenCameraProjection
from core.video_layout import VideoLayout
from inference.history import HistoryIdentity, HistorySession
from probing.capture import CaptureContext, FeatureRecorder


def _tensor_sha256(value: torch.Tensor) -> str:
    stream = BytesIO()
    torch.save(value.detach().contiguous().cpu(), stream)
    return "sha256:" + hashlib.sha256(stream.getvalue()).hexdigest()


@dataclass(frozen=True)
class ControlledReplayResult:
    gt_velocity: torch.Tensor
    generated_velocity: torch.Tensor
    report: dict


def _session_with_history(
    model: JointVideoDiT,
    layout: VideoLayout,
    identity: HistoryIdentity,
    history: torch.Tensor,
    target_chunk: int,
    camera_projection: TokenCameraProjection | None,
) -> HistorySession:
    session = HistorySession(
        model, layout, identity, camera_projection=camera_projection
    )
    condition_count = sum(
        part.stop <= layout.initial_condition_rgb.stop for part in layout.latent_to_rgb
    )
    session.commit_initial_prefix(identity, history[:, :, :condition_count])
    for index in range(target_chunk):
        if session.next_chunk > index:
            continue
        part = layout.latent_range_for_chunk(index)
        session.commit_clean(
            identity, index, history[:, :, max(part.start, condition_count) : part.stop]
        )
    if session.next_chunk != target_chunk:
        raise ValueError("history does not end at the selected target chunk")
    return session


@torch.no_grad()
def controlled_replay(
    model: JointVideoDiT,
    layout: VideoLayout,
    *,
    gt_history: torch.Tensor,
    generated_history: torch.Tensor,
    target_clean: torch.Tensor,
    target_noise: torch.Tensor,
    flow_time: torch.Tensor,
    target_chunk: int,
    camera_projection: TokenCameraProjection | None,
    checkpoint_id: str,
    config_id: str,
    sample_id: str,
    generation_id: str,
    selection_id: str,
    training_seed: int,
    generation_seed: int,
    recorder: FeatureRecorder | None = None,
) -> ControlledReplayResult:
    """Change only clean history source; preserve target, noise, time and camera."""
    if model.training or not 1 <= target_chunk < len(layout.chunk_to_latent):
        raise ValueError(
            "controlled replay requires eval mode and a history-bearing target"
        )
    target_range = layout.latent_range_for_chunk(target_chunk)
    condition_count = sum(
        part.stop <= layout.initial_condition_rgb.stop for part in layout.latent_to_rgb
    )
    target_start = max(target_range.start, condition_count)
    if target_start >= target_range.stop:
        raise ValueError("selected chunk has no unconditioned target latents")
    if (
        gt_history.shape != generated_history.shape
        or gt_history.shape[2] != target_start
    ):
        raise ValueError("both histories must cover exactly the same clean prefix")
    if (
        gt_history.shape[0] != 1
        or target_clean.shape != target_noise.shape
        or target_clean.shape[0] != 1
    ):
        raise ValueError("controlled replay requires paired batch-one target and noise")
    if (
        target_clean.shape[2] != target_range.stop - target_start
        or gt_history.shape[:2] + gt_history.shape[3:]
        != target_clean.shape[:2] + target_clean.shape[3:]
    ):
        raise ValueError("target geometry must match the selected layout and history")
    if not torch.equal(
        gt_history[:, :, :condition_count],
        generated_history[:, :, :condition_count],
    ):
        raise ValueError("the real initial condition must remain fixed")
    if not all(
        torch.isfinite(value).all()
        for value in (gt_history, generated_history, target_clean, target_noise)
    ):
        raise ValueError("replay histories, target, and noise must be finite")
    if (
        flow_time.shape != (1,)
        or not torch.isfinite(flow_time).all()
        or torch.any((flow_time < 0) | (flow_time > 1))
    ):
        raise ValueError("fixed FM time must be finite within [0, 1]")
    if not all((checkpoint_id, config_id, sample_id, generation_id, selection_id)):
        raise ValueError("replay provenance must be explicit")
    noisy_target = interpolate(target_clean, target_noise, flow_time)
    purpose_to_history = (
        ("controlled_gt_history", gt_history),
        ("controlled_generated_history", generated_history),
    )
    predictions = []
    for purpose, history in purpose_to_history:
        identity = HistoryIdentity(
            f"{generation_id}:{purpose}", checkpoint_id, sample_id, "conditional"
        )
        session = _session_with_history(
            model, layout, identity, history, target_chunk, camera_projection
        )
        context = CaptureContext(
            sample_id=sample_id,
            training_seed=training_seed,
            generation_seed=generation_seed,
            checkpoint_id=checkpoint_id,
            config_id=config_id,
            rollout_time=layout.rgb_range_for_latent(target_start).start,
            flow_time=float(flow_time[0]),
            branch="conditional",
            forward_purpose=purpose,
        )
        predictions.append(
            session.predict(
                identity,
                target_chunk,
                noisy_target,
                flow_time,
                mode="reference",
                recorder=recorder,
                context=context,
            )
        )
    gt_prediction, generated_prediction = predictions
    if (
        not torch.isfinite(gt_prediction).all()
        or not torch.isfinite(generated_prediction).all()
    ):
        raise ValueError("replay predictions must be finite")
    gt64 = gt_prediction.to(torch.float64)
    generated64 = generated_prediction.to(torch.float64)
    difference = generated64 - gt64
    target = target_velocity(
        target_clean.to(torch.float64), target_noise.to(torch.float64)
    )
    metrics = {
        "gt_target_mse": float((gt64 - target).square().mean()),
        "generated_target_mse": float((generated64 - target).square().mean()),
        "prediction_mean_abs_delta": float(difference.abs().mean()),
        "prediction_max_abs_delta": float(difference.abs().max()),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("replay metrics must be finite")
    report = {
        "schema_version": 1,
        "comparison": "controlled_gt_vs_generated_history",
        "observational_rollout": False,
        "selection_id": selection_id,
        "generation_id": generation_id,
        "sample_id": sample_id,
        "checkpoint_id": checkpoint_id,
        "config_id": config_id,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "target_chunk": target_chunk,
        "rollout_time_rgb_frame": layout.rgb_range_for_latent(target_start).start,
        "fm_time": float(flow_time[0]),
        "fixed": {
            "target_clean_sha256": _tensor_sha256(target_clean),
            "target_noise_sha256": _tensor_sha256(target_noise),
            "noisy_target_sha256": _tensor_sha256(noisy_target),
            "camera_projection_sha256": None
            if camera_projection is None
            else {
                name: _tensor_sha256(getattr(camera_projection, name))
                for name in ("projection", "transpose", "inverse")
            },
            "text_conditioning": "disabled",
            "token_selection": []
            if recorder is None
            else [asdict(token) for token in recorder.tokens],
            "capture_points": [] if recorder is None else list(recorder.points),
        },
        "changed": {
            "history_source": ["ground_truth", "generated"],
            "gt_history_sha256": _tensor_sha256(gt_history),
            "generated_history_sha256": _tensor_sha256(generated_history),
        },
        "metrics": metrics,
    }
    return ControlledReplayResult(gt_prediction, generated_prediction, report)
