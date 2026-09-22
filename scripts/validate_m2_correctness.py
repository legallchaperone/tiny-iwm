"""Run the deterministic debug-size M2 correctness gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import torch

from algorithms.world_model.flow import (
    FlowMatchSpec,
    euler_step,
    flow_matching_loss,
    interpolate,
    target_velocity,
)
from algorithms.world_model.models import JointVideoDiT, JointVideoDiTConfig
from algorithms.world_model.models.prope import build_token_camera_projection
from algorithms.world_model.training_batch import StageABatchBuilder
from core.camera import CameraCondition, IntrinsicsSpace
from core.types import VideoBatch
from core.video_layout import CodecTemporalSpec, VideoLayout


def run_gate(seed: int = 18) -> dict[str, object]:
    torch.manual_seed(seed)
    layout = VideoLayout.from_codec(
        fps=16,
        rgb_frame_count=5,
        codec=CodecTemporalSpec(temporal_compression=2),
        temporal_patch_size=1,
        initial_condition_frames=1,
    )
    latents = torch.randn(2, 4, layout.latent_frame_count, 5, 7)
    camera = _camera(batch_size=2, rgb_frames=layout.rgb_frame_count)
    video_batch = VideoBatch(
        sample_ids=("debug-a", "debug-b"),
        sources=("synthetic", "synthetic"),
        layout=layout,
        camera=camera,
        latents=latents,
    )
    noise = torch.randn(latents.shape, generator=torch.Generator().manual_seed(seed + 1))
    training_batch = StageABatchBuilder(FlowMatchSpec()).build(
        video_batch,
        noise=noise,
        flow_time=torch.tensor([0.25, 0.75]),
    )

    config = JointVideoDiTConfig(
        latent_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        patch_size=(1, 2, 2),
        mlp_ratio=2.0,
        prope_camera_dims=8,
    )
    model = JointVideoDiT(config)
    grid_shape = (
        layout.latent_frame_count,
        (latents.shape[3] + config.patch_size[1] - 1) // config.patch_size[1],
        (latents.shape[4] + config.patch_size[2] - 1) // config.patch_size[2],
    )
    camera_projection = build_token_camera_projection(
        camera,
        layout,
        grid_shape=grid_shape,
        image_size=(80, 112),
    )
    model.eval()
    prediction = model(
        training_batch.noisy_latents,
        training_batch.flow_time,
        camera_projection=camera_projection,
    )
    captured_prediction, observations = model(
        training_batch.noisy_latents,
        training_batch.flow_time,
        camera_projection=camera_projection,
        capture=("patch_tokens", "blocks.0.post_attention", "blocks.1.block_output"),
    )
    unconditioned_prediction = model(
        training_batch.noisy_latents,
        training_batch.flow_time,
    )
    loss = flow_matching_loss(
        prediction,
        training_batch.target_velocity,
        loss_mask=training_batch.loss_mask,
        time=training_batch.flow_time,
        spec=FlowMatchSpec(),
    )
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]

    clean = torch.tensor([[[[[2.0]]]]])
    endpoint_noise = torch.tensor([[[[[5.0]]]]])
    velocity = target_velocity(clean, endpoint_noise)
    endpoint_zero = interpolate(clean, endpoint_noise, torch.tensor([0.0]))
    endpoint_one = interpolate(clean, endpoint_noise, torch.tensor([1.0]))
    reverse_sample = euler_step(
        endpoint_noise,
        velocity,
        time=torch.tensor([1.0]),
        next_time=torch.tensor([0.0]),
    )

    checks = {
        "output_shape_matches_nondivisible_input": tuple(prediction.shape) == tuple(latents.shape),
        "all_outputs_finite": bool(torch.isfinite(prediction).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_present": bool(gradients),
        "all_gradients_finite": bool(gradients)
        and all(torch.isfinite(item).all() for item in gradients),
        "nonzero_gradient": any(torch.count_nonzero(item).item() > 0 for item in gradients),
        "condition_latent_stays_clean": bool(
            torch.equal(training_batch.noisy_latents[:, :, 0], latents[:, :, 0])
        ),
        "condition_excluded_from_loss": bool((~training_batch.loss_mask[:, :, 0]).all()),
        "camera_condition_changes_output": bool(
            torch.max(torch.abs(prediction.detach() - unconditioned_prediction.detach())) > 0
        ),
        "probe_is_forward_invariant": bool(torch.equal(prediction, captured_prediction)),
        "probe_names_exact": tuple(observations)
        == ("patch_tokens", "blocks.0.post_attention", "blocks.1.block_output"),
        "probe_retains_gradients": all(item.requires_grad for item in observations.values()),
        "fm_t0_is_clean": bool(torch.equal(endpoint_zero, clean)),
        "fm_t1_is_noise": bool(torch.equal(endpoint_one, endpoint_noise)),
        "solver_one_to_zero_reaches_clean": bool(torch.equal(reverse_sample, clean)),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "gate": "CWX-18 M2 debug-size correctness",
        "status": "passed" if not failures else "failed",
        "seed": seed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
        },
        "model": {
            "latent_channels": config.latent_channels,
            "hidden_size": config.hidden_size,
            "depth": config.depth,
            "num_heads": config.num_heads,
            "patch_size": list(config.patch_size),
            "prope_camera_dims": config.prope_camera_dims,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "layout": {
            "rgb_frames": layout.rgb_frame_count,
            "latent_frames": layout.latent_frame_count,
            "latent_shape": list(latents.shape),
            "token_grid_shape": list(grid_shape),
            "token_count": camera_projection.token_count,
            "camera_subframes": camera_projection.subframes,
        },
        "flow": {
            "convention": "t0_clean_t1_noise",
            "times": training_batch.flow_time.tolist(),
            "loss": float(loss.detach()),
            "condition_latent_indices": list(
                training_batch.metadata["condition_latent_indices"]
            ),
        },
        "checks": checks,
        "failures": failures,
    }


def _camera(batch_size: int, rgb_frames: int) -> CameraCondition:
    c2w = torch.eye(4).repeat(batch_size, rgb_frames, 1, 1)
    time = torch.arange(rgb_frames, dtype=torch.float32)
    c2w[:, :, 0, 3] = time[None] * 0.05
    c2w[:, :, 2, 3] = time[None] * 0.02
    intrinsics = torch.tensor(
        [[90.0, 0.0, 56.0], [0.0, 90.0, 40.0], [0.0, 0.0, 1.0]]
    ).repeat(batch_size, rgb_frames, 1, 1)
    return CameraCondition(
        c2w=c2w,
        intrinsics=intrinsics,
        timestamps_seconds=time[None].repeat(batch_size, 1) / 16,
        intrinsics_space=IntrinsicsSpace.RGB_PIXELS,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=18)
    args = parser.parse_args()
    report = run_gate(args.seed)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

