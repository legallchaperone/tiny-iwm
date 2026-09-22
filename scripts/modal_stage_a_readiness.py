"""Run the CWX-22 Stage A readiness gate against the persisted checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
VOLUME_ROOT = Path("/stage-a")
CHECKPOINT_PATH = VOLUME_ROOT / "checkpoints" / "stage-a-best.pt"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "numpy==2.2.6", "pyyaml==6.0.2")
    .add_local_dir(ROOT, remote_path="/opt/tiny-iwm")
)
app = modal.App("tiny-iwm-cwx-22-stage-a-readiness")
volume = modal.Volume.from_name("tiny-iwm-stage-a", create_if_missing=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@app.function(
    image=image,
    gpu="A10G",
    volumes={str(VOLUME_ROOT): volume},
    timeout=1800,
    memory=65536,
    cpu=8,
)
def run_gate() -> str:
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, "/opt/tiny-iwm")
    os.chdir("/opt/tiny-iwm")
    volume.reload()
    from algorithms.world_model.flow import euler_step
    from algorithms.world_model.models import JointVideoDiT, JointVideoDiTConfig
    from algorithms.world_model.models.prope import (
        TokenCameraProjection,
        build_token_camera_projection,
    )
    from core.camera import CameraCondition, IntrinsicsSpace
    from core.video_layout import CodecTemporalSpec, VideoLayout
    from runtime import (
        ExponentialMovingAverage,
        OptimizerSpec,
        SchedulerSpec,
        build_optimizer,
        build_scheduler,
        configure_cuda_math_policy,
        resume_same_run,
    )
    from scripts.stage_a_readiness import evaluate_readiness

    configure_cuda_math_policy()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    model = JointVideoDiT(
        JointVideoDiTConfig(
            latent_channels=128,
            hidden_size=832,
            depth=24,
            num_heads=16,
            patch_size=(1, 2, 2),
            mlp_ratio=4.0,
            qkv_bias=True,
            prope_camera_dims=48,
        )
    ).to(device=device, dtype=dtype)
    optimizer = build_optimizer(model, OptimizerSpec(learning_rate=1e-4, weight_decay=0.01))
    scheduler = build_scheduler(optimizer, SchedulerSpec("constant"))
    ema = ExponentialMovingAverage(model, 0.9999)
    resumed = resume_same_run(
        CHECKPOINT_PATH,
        expected_run_id="stage-a-sekai-subset-seed21-v1",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
    )
    ema.copy_to(model)
    model.eval()

    manifest = json.loads((VOLUME_ROOT / "manifest.json").read_text())
    sample_record = manifest["splits"]["validation"][0]
    arrays = np.load(sample_record["prepared_path"], allow_pickle=False)
    clean = torch.from_numpy(arrays["z"]).unsqueeze(0).to(device=device, dtype=dtype)
    c2w = torch.from_numpy(arrays["pose"]).unsqueeze(0).to(device=device, dtype=torch.float32)
    values = torch.from_numpy(arrays["intrinsics"]).to(device=device, dtype=torch.float32)
    k = torch.zeros((961, 3, 3), device=device, dtype=torch.float32)
    k[:, 0, 0] = values[:, 0] * (1280 / 1920)
    k[:, 1, 1] = values[:, 1] * (720 / 1080)
    k[:, 0, 2] = values[:, 2] * (1280 / 1920) - 576
    k[:, 1, 2] = values[:, 3] * (720 / 1080) - 8 - 288
    k[:, 2, 2] = 1

    def camera(length: int, *, frozen: bool = False) -> CameraCondition:
        poses = c2w[:, :length].clone()
        intrinsics = k[None, :length].clone()
        if frozen:
            poses[:] = poses[:, :1]
            intrinsics[:] = intrinsics[:, :1]
        return CameraCondition(
            c2w=poses,
            intrinsics=intrinsics,
            timestamps_seconds=torch.arange(length, device=device)[None] / 16,
            intrinsics_space=IntrinsicsSpace.RGB_PIXELS,
        )

    def projection(cam: CameraCondition, layout: VideoLayout) -> TokenCameraProjection:
        value = build_token_camera_projection(
            cam,
            layout,
            grid_shape=(layout.latent_frame_count, 2, 2),
            image_size=(128, 128),
        )
        return TokenCameraProjection(
            value.projection.to(dtype), value.transpose.to(dtype), value.inverse.to(dtype)
        )

    full_layout = VideoLayout.from_codec(
        fps=16, rgb_frame_count=961, codec=CodecTemporalSpec(8),
        temporal_patch_size=1, initial_condition_frames=1,
    )
    full_projection = projection(camera(961), full_layout)
    generator = torch.Generator(device=device).manual_seed(22001)
    full_input = torch.randn(clean.shape, device=device, dtype=dtype, generator=generator)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        full_output = model(
            full_input,
            torch.tensor([0.5], device=device, dtype=dtype),
            camera_projection=full_projection,
        )
    torch.cuda.synchronize()
    full_seconds = time.perf_counter() - started
    full_peak = torch.cuda.max_memory_allocated()

    short_latents = 17
    short_rgb = 1 + (short_latents - 1) * 8
    short_layout = VideoLayout.from_codec(
        fps=16, rgb_frame_count=short_rgb, codec=CodecTemporalSpec(8),
        temporal_patch_size=1, initial_condition_frames=1,
    )
    actual_projection = projection(camera(short_rgb), short_layout)
    frozen_projection = projection(camera(short_rgb, frozen=True), short_layout)
    target = clean[:, :, :short_latents]
    initial_generator = torch.Generator(device=device).manual_seed(22002)
    initial = torch.randn(target.shape, device=device, dtype=dtype, generator=initial_generator)
    initial[:, :, :1] = target[:, :, :1]

    def rollout(camera_projection: TokenCameraProjection) -> torch.Tensor:
        state = initial.clone()
        with torch.no_grad():
            for index in range(8):
                current = torch.tensor([1 - index / 8], device=device, dtype=dtype)
                following = torch.tensor([1 - (index + 1) / 8], device=device, dtype=dtype)
                velocity = model(state, current, camera_projection=camera_projection)
                state = euler_step(state, velocity, time=current, next_time=following)
                state[:, :, :1] = target[:, :, :1]
        return state

    actual = rollout(actual_projection)
    frozen = rollout(frozen_projection)
    target_region = actual[:, :, 1:]
    reference_region = target[:, :, 1:]
    delta = (actual - frozen).float()
    report = {
        "schema_version": 1,
        "issue": "CWX-22",
        "sample_id": sample_record["sample_id"],
        "traceability": {
            "profile_artifact_sha256": _sha256(Path("artifacts/m3/cwx-19-full-length-profile.json")),
            "training_artifact_sha256": _sha256(Path("artifacts/m3/cwx-21-stage-a-run.json")),
            "manifest_sha256": manifest["manifest_sha256"],
            "checkpoint_id": resumed.checkpoint_id,
        },
        "checkpoint_resume": {
            "verified": True,
            "global_step": resumed.global_step,
            "epoch": resumed.epoch,
            "restored_scope": list(resumed.restored_scope),
            "run_id": resumed.provenance["run_id"],
            "evaluation_weights": "ema",
        },
        "formal_length": {
            "completed": bool(torch.isfinite(full_output).all()),
            "rgb_frames": 961,
            "latent_shape": list(clean.shape),
            "tokens": full_projection.token_count,
            "forward_seconds": full_seconds,
            "peak_allocated_bytes": full_peak,
            "output_mean": float(full_output.float().mean()),
            "output_std": float(full_output.float().std()),
        },
        "short_term_generation": {
            "rgb_frames": short_rgb,
            "latent_frames": short_latents,
            "duration_seconds": (short_rgb - 1) / 16,
            "solver": "Euler",
            "steps": 8,
            "seed": 22002,
            "condition_max_abs_error": float((actual[:, :, :1] - target[:, :, :1]).abs().max()),
            "target_mse": float((target_region.float() - reference_region.float()).square().mean()),
            "output_mean": float(actual.float().mean()),
            "output_std": float(actual.float().std()),
        },
        "camera_conditioning": {
            "inspection": "actual trajectory versus first-camera-frozen counterfactual",
            "counterfactual_mean_abs_delta": float(delta.abs().mean()),
            "counterfactual_max_abs_delta": float(delta.abs().max()),
            "counterfactual_relative_l2": float(
                torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(actual.float())
            ),
        },
        "diagnosis": {
            "quality_warning": (
                "This is a 100-step baseline. Finite conditional generation and camera "
                "response establish integration readiness, not perceptual convergence."
            ),
            "ema_observation": (
                "CWX-21 recorded identical BF16-rounded EMA validation loss at steps 50 "
                "and 100; the deterministic tie-break selected step 50."
            ),
            "downstream_scope": (
                "Suitable as the mechanically verified Stage B initialization baseline; "
                "do not treat its generation metric as a quality target."
            ),
        },
        "hardware": {
            "platform": "Modal",
            "gpu_request": "A10G",
            "device": torch.cuda.get_device_name(device),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
        },
    }
    report["decision"] = evaluate_readiness(report)
    if report["decision"]["status"] == "failed":
        raise RuntimeError(json.dumps(report, indent=2))
    return json.dumps(report, sort_keys=True)


@app.local_entrypoint()
def main(output: str = str(ROOT / "artifacts" / "m3" / "cwx-22-stage-a-readiness.json")):
    report = json.loads(run_gate.remote())
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(destination)
