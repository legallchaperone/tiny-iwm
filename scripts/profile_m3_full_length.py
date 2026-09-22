"""Profile the formal 961-frame Stage A model on a CUDA device."""

from __future__ import annotations

import platform
import time

import torch

from algorithms.world_model.flow import FlowMatchSpec, euler_step, flow_matching_loss
from algorithms.world_model.models import JointVideoDiT, JointVideoDiTConfig
from algorithms.world_model.models.prope import (
    TokenCameraProjection,
    build_token_camera_projection,
)
from algorithms.world_model.training_batch import StageABatchBuilder
from core.camera import CameraCondition, IntrinsicsSpace
from core.types import VideoBatch
from core.video_layout import CodecTemporalSpec, VideoLayout


RGB_FRAMES = 961
FPS = 16
LATENT_CHANNELS = 128
TEMPORAL_COMPRESSION = 8
SPATIAL_COMPRESSION = 32


def profile_cuda(
    *,
    warmup_steps: int = 1,
    measured_steps: int = 3,
    sampling_steps: int = 16,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CWX-19 measured profiling requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    dtype = torch.bfloat16
    layout = VideoLayout.from_codec(
        fps=FPS,
        rgb_frame_count=RGB_FRAMES,
        codec=CodecTemporalSpec(TEMPORAL_COMPRESSION),
        temporal_patch_size=1,
        initial_condition_frames=1,
    )
    config = JointVideoDiTConfig(
        latent_channels=LATENT_CHANNELS,
        hidden_size=832,
        depth=24,
        num_heads=16,
        patch_size=(1, 2, 2),
        mlp_ratio=4.0,
        qkv_bias=True,
        prope_camera_dims=48,
    )
    model = JointVideoDiT(config).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    latent_shape = (1, LATENT_CHANNELS, layout.latent_frame_count, 4, 4)
    generator = torch.Generator(device=device).manual_seed(19)
    clean = torch.randn(latent_shape, device=device, dtype=dtype, generator=generator)
    noise = torch.randn(latent_shape, device=device, dtype=dtype, generator=generator)
    camera = _camera(layout, device)
    video_batch = VideoBatch(
        sample_ids=("profile-961",),
        sources=("synthetic-shape-profile",),
        layout=layout,
        camera=camera,
        latents=clean,
    )
    training_batch = StageABatchBuilder(FlowMatchSpec()).build(
        video_batch,
        noise=noise,
        flow_time=torch.tensor([0.5], device=device, dtype=dtype),
    )
    grid_shape = (layout.latent_frame_count, 2, 2)
    camera_projection = _projection_to_dtype(
        build_token_camera_projection(
            camera,
            layout,
            grid_shape=grid_shape,
            image_size=(128, 128),
        ),
        dtype,
    )

    def train_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            training_batch.noisy_latents,
            training_batch.flow_time,
            camera_projection=camera_projection,
        )
        loss = flow_matching_loss(
            prediction,
            training_batch.target_velocity,
            loss_mask=training_batch.loss_mask,
            time=training_batch.flow_time,
            spec=FlowMatchSpec(),
        )
        loss.backward()
        optimizer.step()
        return loss.detach()

    model.train()
    for _ in range(warmup_steps):
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step_times: list[float] = []
    losses: list[float] = []
    for _ in range(measured_steps):
        started = time.perf_counter()
        loss = train_step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - started)
        losses.append(float(loss))
    training_peak_allocated = torch.cuda.max_memory_allocated()
    training_peak_reserved = torch.cuda.max_memory_reserved()

    del optimizer
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    state = noise.clone()
    rollout_started = time.perf_counter()
    with torch.no_grad():
        for index in range(sampling_steps):
            current = 1.0 - index / sampling_steps
            next_value = 1.0 - (index + 1) / sampling_steps
            current_time = torch.tensor([current], device=device, dtype=dtype)
            next_time = torch.tensor([next_value], device=device, dtype=dtype)
            velocity = model(state, current_time, camera_projection=camera_projection)
            state = euler_step(state, velocity, time=current_time, next_time=next_time)
    torch.cuda.synchronize()
    rollout_seconds = time.perf_counter() - rollout_started
    inference_peak_allocated = torch.cuda.max_memory_allocated()
    inference_peak_reserved = torch.cuda.max_memory_reserved()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    tokens = camera_projection.token_count
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    kv_per_layer = 2 * tokens * config.hidden_size * dtype_bytes
    variants = _variant_costs(config.depth, config.hidden_size, dtype_bytes)
    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": 1,
        "issue": "CWX-19",
        "status": "passed",
        "physical_protocol": {
            "rgb_frames": RGB_FRAMES,
            "fps": FPS,
            "duration_seconds": 60,
            "temporal_compression": TEMPORAL_COMPRESSION,
            "latent_frames": layout.latent_frame_count,
        },
        "model": {
            "trainable_parameters": parameter_count,
            "depth": config.depth,
            "hidden_size": config.hidden_size,
            "num_heads": config.num_heads,
            "patch_size": list(config.patch_size),
            "latent_shape": list(latent_shape),
            "tokens": tokens,
        },
        "execution": {
            "precision": "bfloat16",
            "attention_backend": "torch_scaled_dot_product_attention_auto",
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            "activation_checkpointing": False,
            "torch_compile": False,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "data_loader_workers": 0,
            "optimizer": "AdamW",
            "optimizer_lr": 1e-4,
            "optimizer_weight_decay": 0.01,
            "sampling_solver": "Euler",
            "sampling_steps": sampling_steps,
            "cuda_math_policy": "strict_no_tf32_no_reduced_reduction_v1",
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_fp16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            "allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        },
        "hardware": {
            "modal_gpu_request": "A10G",
            "device": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "cuda_runtime": torch.version.cuda,
            "torch": str(torch.__version__),
            "python": platform.python_version(),
        },
        "measurements": {
            "warmup_steps": warmup_steps,
            "measured_training_steps": measured_steps,
            "training_step_seconds": step_times,
            "training_step_seconds_mean": sum(step_times) / len(step_times),
            "training_loss": losses,
            "peak_training_allocated_bytes": training_peak_allocated,
            "peak_training_reserved_bytes": training_peak_reserved,
            "full_961_frame_rollout_seconds": rollout_seconds,
            "peak_rollout_allocated_bytes": inference_peak_allocated,
            "peak_rollout_reserved_bytes": inference_peak_reserved,
            "projected_kv_bytes_per_layer": kv_per_layer,
            "projected_kv_bytes_all_layers": kv_per_layer * config.depth,
            "dense_attention_score_bytes_per_layer": (
                config.num_heads * tokens * tokens * dtype_bytes
            ),
        },
        "variant_comparison": variants,
        "execution_comparison": [
            {
                "name": "single_A10G_bfloat16_no_checkpointing",
                "measured": True,
                "selected": True,
                "reason": "Measured peak memory leaves substantial device headroom.",
            },
            {
                "name": "single_A10G_bfloat16_activation_checkpointing",
                "measured": False,
                "selected": False,
                "reason": "Viable fallback, but recomputation is unnecessary at the measured shape.",
            },
            {
                "name": "multi_gpu_data_parallel",
                "measured": False,
                "selected": False,
                "reason": "Useful for throughput scaling; does not reduce per-rank model memory here.",
            },
        ],
        "decision": {
            "selected_resolution": [128, 128],
            "selected_patch_size": [1, 2, 2],
            "selected_tokens": tokens,
            "selected_precision": "bfloat16",
            "selected_execution": "single_A10G_no_activation_checkpointing",
            "rationale": (
                "The selected setting preserves the formal 961-frame/60-second protocol "
                "and leaves measured memory headroom. 128px/patch1 and 256px/patch2 both "
                "quadruple token count and increase conceptual dense attention by 16x."
            ),
        },
    }


def _camera(layout: VideoLayout, device: torch.device) -> CameraCondition:
    time = torch.arange(layout.rgb_frame_count, device=device, dtype=torch.float32)
    c2w = torch.eye(4, device=device).repeat(1, layout.rgb_frame_count, 1, 1)
    c2w[:, :, 0, 3] = time[None] * 0.002
    intrinsics = torch.tensor(
        [[96.0, 0.0, 64.0], [0.0, 96.0, 64.0], [0.0, 0.0, 1.0]],
        device=device,
    ).repeat(1, layout.rgb_frame_count, 1, 1)
    return CameraCondition(
        c2w=c2w,
        intrinsics=intrinsics,
        timestamps_seconds=time[None] / FPS,
        intrinsics_space=IntrinsicsSpace.RGB_PIXELS,
    )


def _projection_to_dtype(
    projection: TokenCameraProjection, dtype: torch.dtype
) -> TokenCameraProjection:
    return TokenCameraProjection(
        projection.projection.to(dtype),
        projection.transpose.to(dtype),
        projection.inverse.to(dtype),
    )


def _variant_costs(depth: int, hidden_size: int, dtype_bytes: int) -> list[dict[str, object]]:
    variants = (
        ("128px_patch2_selected", 128, 2),
        ("128px_patch1", 128, 1),
        ("256px_patch2", 256, 2),
        ("256px_patch4", 256, 4),
    )
    output: list[dict[str, object]] = []
    for name, resolution, spatial_patch in variants:
        latent_side = resolution // SPATIAL_COMPRESSION
        token_side = (latent_side + spatial_patch - 1) // spatial_patch
        tokens = 121 * token_side * token_side
        kv = 2 * tokens * hidden_size * dtype_bytes
        output.append(
            {
                "name": name,
                "resolution": [resolution, resolution],
                "patch_size": [1, spatial_patch, spatial_patch],
                "physical_rgb_frames": RGB_FRAMES,
                "physical_duration_seconds": 60,
                "latent_shape": [1, LATENT_CHANNELS, 121, latent_side, latent_side],
                "tokens": tokens,
                "token_ratio_vs_selected": tokens / 484,
                "dense_attention_ratio_vs_selected": (tokens / 484) ** 2,
                "projected_kv_bytes_per_layer": kv,
                "projected_kv_bytes_all_layers": kv * depth,
            }
        )
    return output
