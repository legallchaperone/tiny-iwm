"""CPU preflight, then bounded Stage B training on prepared Modal data."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "numpy==2.2.6", "pyyaml==6.0.2")
    .add_local_dir(LOCAL_ROOT, remote_path="/opt/tiny-iwm")
)
VOLUME_ROOT = Path("/stage-a")
volume = modal.Volume.from_name("tiny-iwm-stage-a", create_if_missing=False)
MANIFEST_PATH = VOLUME_ROOT / "manifest.json"
CONFIG_PATH = Path("/opt/tiny-iwm/configurations/runs/stage_b_baseline.yaml")
INITIAL_CHECKPOINT = VOLUME_ROOT / "checkpoints/stage-b-init.pt"
BEST_CHECKPOINT = VOLUME_ROOT / "checkpoints/stage-b-best.pt"
app = modal.App("tiny-iwm-cwx-26-stage-b")


def _file_id(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@app.function(
    image=image, volumes={str(VOLUME_ROOT): volume}, cpu=2, memory=8192, timeout=600
)
def inspect_inputs() -> str:
    """Fail before GPU allocation if prepared artifacts or identities disagree."""
    import torch
    import yaml

    volume.reload()
    config = yaml.safe_load(CONFIG_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest["manifest_sha256"] != config["prepared_manifest_sha256"]:
        raise ValueError(
            "Stage B manifest identity differs from the pinned Stage A data"
        )
    if manifest["source"]["revision"] != config["source_revision"]:
        raise ValueError("Stage B source revision differs from prepared data")
    if (
        config["flow_matching"]["df_timestep_mixture"]
        or config["flow_matching"]["self_rollout_loss"]
    ):
        raise ValueError("excluded Stage B objectives must be disabled")
    if not INITIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(INITIAL_CHECKPOINT)
    checkpoint_id = _file_id(INITIAL_CHECKPOINT)
    sidecar = INITIAL_CHECKPOINT.with_suffix(".pt.sha256")
    if (
        not sidecar.is_file()
        or "sha256:" + sidecar.read_text().strip() != checkpoint_id
    ):
        raise ValueError(
            "Stage B initialization checkpoint failed SHA-256 verification"
        )
    payload = torch.load(INITIAL_CHECKPOINT, map_location="cpu", weights_only=False)
    provenance = payload["provenance"]
    if provenance["run_id"] != config["run_id"] or provenance["stage"] != "B":
        raise ValueError("initial checkpoint does not belong to this Stage B run")
    if provenance["parent_checkpoint_id"] != config["parent_checkpoint_id"]:
        raise ValueError("initial checkpoint parent differs from pinned Stage A")
    if payload["progress"] != {"global_step": 0, "epoch": 0}:
        raise ValueError("Stage B initialization checkpoint has nonzero progress")
    if payload["optimizer"]["state"] or payload["ema"]["num_updates"] != 0:
        raise ValueError("Stage B optimizer or EMA was not freshly initialized")
    if (
        provenance["codec"]["identity"]
        != "LTX2VAE_diffusers_704x1280_official_latent_cache"
    ):
        raise ValueError("Stage B codec identity differs from prepared latents")
    if provenance["camera"]["intrinsics_space"] != "rgb_pixels":
        raise ValueError("Stage B camera convention differs from prepared data")
    del payload
    for split in ("train", "validation"):
        for record in manifest["splits"][split]:
            prepared = Path(record["prepared_path"])
            if _file_id(prepared) != record["sha256"]:
                raise ValueError(
                    f"prepared sample failed SHA-256 verification: {prepared}"
                )
    return json.dumps(
        {
            "status": "passed",
            "device": "cpu",
            "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
            "manifest_sha256": manifest["manifest_sha256"],
            "initial_checkpoint_id": checkpoint_id,
            "train_samples": len(manifest["splits"]["train"]),
            "validation_samples": len(manifest["splits"]["validation"]),
        },
        sort_keys=True,
    )


@app.function(
    image=image,
    gpu="A10G",
    volumes={str(VOLUME_ROOT): volume},
    cpu=8,
    memory=65536,
    timeout=3600,
)
def train(preflight_json: str, verify_only: bool = False) -> str:
    """Use only prepared data and the verified Stage B initialization checkpoint."""
    import sys
    import torch
    import yaml

    sys.path.insert(0, "/opt/tiny-iwm")
    os.chdir("/opt/tiny-iwm")
    from algorithms.world_model.flow import FlowMatchSpec, flow_matching_loss
    from algorithms.world_model.models import JointVideoDiT, JointVideoDiTConfig
    from algorithms.world_model.training_batch import (
        ChunkCausalVisibility,
        StageBBatchBuilder,
    )
    from core.types import VideoBatch
    from core.video_layout import CodecTemporalSpec, VideoLayout
    from inference.history import HistoryIdentity, HistorySession
    from runtime import (
        CheckpointProvenance,
        ExponentialMovingAverage,
        OptimizerSpec,
        SchedulerSpec,
        build_optimizer,
        build_scheduler,
        configure_cuda_math_policy,
        resume_same_run,
        save_checkpoint,
    )
    from scripts.prepared_stage_data import load_prepared_sample

    preflight = json.loads(preflight_json)
    if preflight.get("status") != "passed" or preflight.get("device") != "cpu":
        raise ValueError("Stage B GPU training requires a passed CPU preflight")
    volume.reload()
    config = yaml.safe_load(
        Path("configurations/runs/stage_b_baseline.yaml").read_text()
    )
    manifest = json.loads(MANIFEST_PATH.read_text())
    if (
        hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
        != preflight["config_sha256"]
    ):
        raise ValueError("Stage B config changed after CPU preflight")
    if preflight["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("prepared manifest changed after CPU preflight")
    if _file_id(INITIAL_CHECKPOINT) != preflight["initial_checkpoint_id"]:
        raise ValueError("initial checkpoint changed after CPU preflight")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage B GPU function requires CUDA")

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    configure_cuda_math_policy()
    device, dtype = torch.device("cuda"), torch.bfloat16
    model = JointVideoDiT(
        JointVideoDiTConfig(
            **{
                **config["model"],
                "patch_size": tuple(config["model"]["patch_size"]),
            }
        )
    ).to(device=device, dtype=dtype)
    optimizer = build_optimizer(
        model,
        OptimizerSpec(
            learning_rate=config["optimization"]["learning_rate"],
            weight_decay=config["optimization"]["weight_decay"],
        ),
    )
    scheduler = build_scheduler(optimizer, SchedulerSpec("constant"))
    ema = ExponentialMovingAverage(model, config["optimization"]["ema_decay"])
    resumed = resume_same_run(
        INITIAL_CHECKPOINT,
        expected_run_id=config["run_id"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
    )
    if (
        resumed.global_step != 0
        or resumed.provenance["parent_checkpoint_id"] != config["parent_checkpoint_id"]
    ):
        raise ValueError(
            "Stage B initialization progress or parent checkpoint differs from config"
        )
    if resumed.checkpoint_id != preflight["initial_checkpoint_id"]:
        raise ValueError("Stage B checkpoint identity changed after preflight")

    layout = VideoLayout.from_codec(
        fps=16,
        rgb_frame_count=961,
        codec=CodecTemporalSpec(8),
        latent_chunk_size=config["flow_matching"]["latent_frames_per_chunk"],
        initial_condition_frames=1,
    )
    flow_spec = FlowMatchSpec()
    builder = StageBBatchBuilder(flow_spec)
    samples = {
        split: [
            load_prepared_sample(record, device=device, dtype=dtype, layout=layout)
            for record in manifest["splits"][split]
        ]
        for split in ("train", "validation")
    }
    example = samples["train"][0][0]
    spatial_tokens = (example.shape[3] // model.config.patch_size[1]) * (
        example.shape[4] // model.config.patch_size[2]
    )
    mask = (
        ChunkCausalVisibility.from_layout(layout)
        .materialize(spatial_tokens_per_temporal_token=spatial_tokens)
        .to(device)
    )

    def loss_for(
        item,
        *,
        target_chunk: int,
        generator: torch.Generator,
        fixed_time: float | None = None,
    ) -> torch.Tensor:
        clean, camera, projection = item
        video = VideoBatch(
            sample_ids=("prepared",),
            sources=("official-prepared-volume",),
            layout=layout,
            camera=camera,
            latents=clean,
        )
        flow_time = (
            None
            if fixed_time is None
            else torch.tensor([fixed_time], device=device, dtype=dtype)
        )
        batch = builder.build(
            video,
            target_chunk=target_chunk,
            generator=generator,
            flow_time=flow_time,
        )
        prediction = model(
            batch.noisy_latents,
            batch.model_time,
            visibility_mask=mask,
            camera_projection=projection,
        )
        return flow_matching_loss(
            prediction,
            batch.target_velocity,
            loss_mask=batch.loss_mask,
            time=batch.flow_time,
            spec=flow_spec,
        )

    def with_ema(task):
        raw = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }
        ema.copy_to(model)
        model.eval()
        try:
            with torch.no_grad():
                return task()
        finally:
            state = model.state_dict()
            for name, value in raw.items():
                state[name].copy_(value)
            model.train()

    def validate() -> float:
        def measure():
            losses = []
            for index, item in enumerate(samples["validation"]):
                for target_chunk in config["validation"]["target_chunks"]:
                    generator = torch.Generator(device=device).manual_seed(
                        config["validation"]["deterministic_noise_seed"]
                        + index * 100
                        + target_chunk
                    )
                    losses.append(
                        float(
                            loss_for(
                                item,
                                target_chunk=target_chunk,
                                generator=generator,
                                fixed_time=config["validation"]["flow_time"],
                            )
                        )
                    )
            return sum(losses) / len(losses)

        return with_ema(measure)

    provenance = CheckpointProvenance(
        run_id=config["run_id"],
        stage="B",
        model={
            "architecture": "joint_spatiotemporal_dit",
            "parameters": sum(p.numel() for p in model.parameters()),
            "config": config["model"],
        },
        codec={
            "identity": "LTX2VAE_diffusers_704x1280_official_latent_cache",
            "source_revision": config["source_revision"],
            "temporal_compression": 8,
            "spatial_compression": 32,
        },
        camera={
            "pose_convention": "camera_to_world",
            "intrinsics_space": "rgb_pixels",
            "preprocessing": manifest["transform"]["camera_intrinsics"],
        },
        resolved_config=config,
        parent_checkpoint_id=config["parent_checkpoint_id"],
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    trace, validations = [], []
    best_metric, best_step, best_id = float("inf"), 0, ""
    model.train()
    started = time.perf_counter()
    if not verify_only:
        for step in range(1, config["optimization"]["steps"] + 1):
            item = samples["train"][(step - 1) % len(samples["train"])]
            target_chunk = int(
                torch.randint(
                    len(layout.chunk_to_latent),
                    (1,),
                    generator=generator,
                    device=device,
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(item, target_chunk=target_chunk, generator=generator)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update(model)
            if step == 1 or step % 10 == 0:
                trace.append(
                    {
                        "step": step,
                        "target_chunk": target_chunk,
                        "loss": float(loss.detach()),
                    }
                )
            if step in config["validation"]["steps"]:
                metric = validate()
                validations.append({"step": step, "ema_flow_matching_mse": metric})
                if metric < best_metric:
                    best_metric, best_step = metric, step
                    best_id = save_checkpoint(
                        BEST_CHECKPOINT,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        ema=ema,
                        global_step=step,
                        epoch=step // len(samples["train"]),
                        provenance=provenance,
                    )
                    volume.commit()
    else:
        best_id = _file_id(BEST_CHECKPOINT)

    selected = resume_same_run(
        BEST_CHECKPOINT,
        expected_run_id=config["run_id"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
    )
    if verify_only:
        best_step = selected.global_step
        best_metric = None
    if selected.checkpoint_id != best_id or selected.global_step != best_step:
        raise AssertionError("selected Stage B checkpoint cannot be resumed exactly")

    def correctness_gate():
        clean, _, projection = samples["validation"][0]
        identity = HistoryIdentity(
            "held-out-cache-gate", best_id, "camera-and-latent", "conditional"
        )
        session = HistorySession(model, layout, identity, camera_projection=projection)
        session.commit_initial_prefix(identity, clean[:, :, :1])
        deltas = []
        for chunk_index in (0, 1, 2):
            latent_range = layout.latent_range_for_chunk(chunk_index)
            begin = sum(part.shape[2] for part in session.history)
            target = clean[:, :, begin : latent_range.stop]
            for value in (0.9, 0.5):
                t = torch.tensor([value], device=device, dtype=dtype)
                cached = session.predict(
                    identity, chunk_index, target, t, mode="cached"
                )
                reference = session.predict(
                    identity, chunk_index, target, t, mode="reference"
                )
                if (
                    not torch.isfinite(cached).all()
                    or not torch.isfinite(reference).all()
                ):
                    raise AssertionError("non-finite cached/reference prediction")
                delta = float((cached - reference).abs().max())
                deltas.append(
                    {"chunk": chunk_index, "flow_time": value, "max_abs_delta": delta}
                )
                if not math.isfinite(delta) or delta > 0.05:
                    raise AssertionError(
                        f"cached/reference delta {delta} exceeds BF16 tolerance"
                    )
            session.commit_clean(identity, chunk_index, target)
        camera = samples["validation"][0][1]
        base_video = VideoBatch(
            sample_ids=("held-out",),
            sources=("official-prepared-volume",),
            layout=layout,
            camera=camera,
            latents=clean,
        )
        noise = torch.randn(
            clean.shape,
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(26099),
        )
        fixed_time = torch.tensor([0.5], device=device, dtype=dtype)
        base = builder.build(
            base_video, target_chunk=1, noise=noise, flow_time=fixed_time
        )
        changed_future = clean.clone()
        changed_future[:, :, 8:] += 0.25
        future_video = VideoBatch(
            sample_ids=("held-out",),
            sources=("official-prepared-volume",),
            layout=layout,
            camera=camera,
            latents=changed_future,
        )
        future = builder.build(
            future_video, target_chunk=1, noise=noise, flow_time=fixed_time
        )
        changed_target = clean.clone()
        changed_target[:, :, 4:8] += 0.25
        adjusted_noise = noise.clone()
        adjusted_noise[:, :, 4:8] -= 0.25
        target_video = VideoBatch(
            sample_ids=("held-out",),
            sources=("official-prepared-volume",),
            layout=layout,
            camera=camera,
            latents=changed_target,
        )
        target = builder.build(
            target_video, target_chunk=1, noise=adjusted_noise, flow_time=fixed_time
        )
        if not target.clean_condition_latents[:, :, 4:8].eq(0).all():
            raise AssertionError("target clean GT entered the separate clean branch")
        # BF16 arithmetic makes algebraically equivalent interpolation differ by
        # rounding. Hold the target z_t tensor bit-for-bit fixed for this replay.
        fixed_target_noisy = target.noisy_latents.clone()
        fixed_target_noisy[:, :, 4:8] = base.noisy_latents[:, :, 4:8]
        inputs = (base.noisy_latents, future.noisy_latents, fixed_target_noisy)
        if not torch.equal(inputs[0], inputs[2]):
            raise AssertionError("controlled replay failed to fix the full noisy input")
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.MATH]):
            outputs = [
                model(
                    value,
                    base.model_time,
                    visibility_mask=mask,
                    camera_projection=projection,
                )[:, :, 4:8]
                for value in inputs
            ]
            repeat = model(
                inputs[0],
                base.model_time,
                visibility_mask=mask,
                camera_projection=projection,
            )[:, :, 4:8]
        if any(not torch.isfinite(value).all() for value in (*outputs, repeat)):
            raise AssertionError("non-finite leakage prediction")
        repeat_delta = float((outputs[0] - repeat).abs().max())
        future_delta = float((outputs[0] - outputs[1]).abs().max())
        target_delta = float((outputs[0] - outputs[2]).abs().max())
        if not all(
            math.isfinite(value) for value in (repeat_delta, future_delta, target_delta)
        ):
            raise AssertionError("non-finite leakage comparison")
        if max(future_delta, target_delta) > max(0.05, 2 * repeat_delta):
            raise AssertionError(
                "Stage B target or future clean branch leaked into target output"
            )
        return {
            "cache_comparison": deltas,
            "leakage": {
                "future_clean_max_abs_delta": future_delta,
                "target_clean_fixed_noisy_max_abs_delta": target_delta,
                "identical_input_repeat_max_abs_delta": repeat_delta,
                "bf16_unfixed_noisy_max_abs_delta": float(
                    (base.noisy_latents[:, :, 4:8] - target.noisy_latents[:, :, 4:8])
                    .abs()
                    .max()
                ),
                "tolerance": max(0.05, 2 * repeat_delta),
            },
        }

    gate = with_ema(correctness_gate)
    torch.cuda.synchronize()
    return json.dumps(
        {
            "schema_version": 1,
            "issue": "CWX-26",
            "status": "passed",
            "execution": "verify_only" if verify_only else "train_and_verify",
            "run_id": config["run_id"],
            "seed": seed,
            "config_sha256": preflight["config_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "parent_checkpoint_id": config["parent_checkpoint_id"],
            "initial_checkpoint_id": resumed.checkpoint_id,
            "selected_checkpoint": {
                "path": str(BEST_CHECKPOINT),
                "checkpoint_id": best_id,
                "step": best_step,
                "ema_flow_matching_mse": best_metric,
            },
            "training": {
                "steps": 0 if verify_only else config["optimization"]["steps"],
                "loss_trace": trace,
                "wall_seconds": time.perf_counter() - started,
            },
            "validation": validations,
            **gate,
            "hardware": {
                "platform": "Modal",
                "gpu_request": "A10G",
                "device": torch.cuda.get_device_name(device),
                "torch": str(torch.__version__),
            },
            "cost_control": {
                "preflight_device": "cpu",
                "downloads_inside_gpu_function": False,
                "prepared_data_volume": "tiny-iwm-stage-a",
            },
        },
        sort_keys=True,
    )


@app.local_entrypoint()
def main() -> None:
    preflight = inspect_inputs.remote()
    print("CPU preflight:", preflight, flush=True)
    result = train.remote(preflight)
    print("Stage B training:", result, flush=True)


@app.local_entrypoint()
def verify() -> None:
    preflight = inspect_inputs.remote()
    print("CPU preflight:", preflight, flush=True)
    result = train.remote(preflight, verify_only=True)
    print("Stage B verification:", result, flush=True)
