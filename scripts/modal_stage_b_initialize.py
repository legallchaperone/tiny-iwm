"""Create and verify the Stage B initialization checkpoint on Modal CPU."""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME_ROOT = Path("/stage-a")
SOURCE = VOLUME_ROOT / "checkpoints" / "stage-a-best.pt"
DESTINATION = VOLUME_ROOT / "checkpoints" / "stage-b-init.pt"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "pyyaml==6.0.2")
    .add_local_dir(ROOT, remote_path="/opt/tiny-iwm")
)
app = modal.App("tiny-iwm-cwx-24-stage-b-initialize")
volume = modal.Volume.from_name("tiny-iwm-stage-a", create_if_missing=False)


@app.function(image=image, volumes={str(VOLUME_ROOT): volume}, timeout=1800, memory=32768, cpu=8)
def initialize() -> str:
    import sys
    import torch
    import yaml

    sys.path.insert(0, "/opt/tiny-iwm")
    os.chdir("/opt/tiny-iwm")
    volume.reload()
    from algorithms.world_model.models import JointVideoDiT, JointVideoDiTConfig
    from runtime import (
        CheckpointCompatibility, CheckpointProvenance, ExponentialMovingAverage,
        OptimizerSpec, SchedulerSpec, build_optimizer, build_scheduler,
        initialize_new_stage, resume_same_run, save_checkpoint,
    )

    def new_model() -> JointVideoDiT:
        return JointVideoDiT(JointVideoDiTConfig(
            latent_channels=128, hidden_size=832, depth=24, num_heads=16,
            patch_size=(1, 2, 2), mlp_ratio=4.0, qkv_bias=True,
            prope_camera_dims=48,
        )).to(dtype=torch.bfloat16)

    optimizer_factory = lambda model: build_optimizer(
        model, OptimizerSpec(learning_rate=1e-4, weight_decay=0.01)
    )
    scheduler_factory = lambda optimizer: build_scheduler(optimizer, SchedulerSpec("constant"))
    model = new_model()
    initialized = initialize_new_stage(
        SOURCE, model=model, optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory, ema_decay=0.9999,
        source_weights="ema", ema_rule="copy_loaded_model",
        expected_compatibility=CheckpointCompatibility(
            model={"architecture": "joint_spatiotemporal_dit", "parameters": 302486592},
            codec={
                "identity": "LTX2VAE_diffusers_704x1280_official_latent_cache",
                "temporal_compression": 8, "spatial_compression": 32,
            },
            camera={"pose_convention": "camera_to_world", "intrinsics_space": "rgb_pixels"},
        ),
    )
    if (
        initialized.global_step != 0
        or initialized.epoch != 0
        or initialized.optimizer.state
        or initialized.scheduler.last_epoch != 0
        or initialized.ema.num_updates != 0
    ):
        raise RuntimeError("Stage B initialization did not reset all training state")
    stage_config = yaml.safe_load(Path("configurations/stage/causal_tf.yaml").read_text())
    if stage_config["df_timestep_mixture"] or stage_config["self_rollout_loss"]:
        raise ValueError("excluded Stage B losses must remain disabled")
    provenance = CheckpointProvenance(
        run_id="stage-b-sekai-subset-seed21-v1", stage="B",
        model={
            "architecture": "joint_spatiotemporal_dit",
            "parameters": sum(value.numel() for value in model.parameters()),
            "weights_source": "stage_a_ema",
        },
        codec={
            "identity": "LTX2VAE_diffusers_704x1280_official_latent_cache",
            "temporal_compression": 8, "spatial_compression": 32,
        },
        camera={
            "pose_convention": "camera_to_world", "intrinsics_space": "rgb_pixels",
            "method": "tiled_prope",
        },
        resolved_config={"stage": stage_config},
        parent_checkpoint_id=initialized.parent_checkpoint_id,
    )
    checkpoint_id = save_checkpoint(
        DESTINATION, model=model, optimizer=initialized.optimizer,
        scheduler=initialized.scheduler, ema=initialized.ema,
        global_step=0, epoch=0, provenance=provenance,
    )
    del model, initialized

    verify_model = new_model()
    verify_optimizer = optimizer_factory(verify_model)
    verify_scheduler = scheduler_factory(verify_optimizer)
    verify_ema = ExponentialMovingAverage(verify_model, 0.9999)
    resumed = resume_same_run(
        DESTINATION, expected_run_id="stage-b-sekai-subset-seed21-v1",
        model=verify_model, optimizer=verify_optimizer,
        scheduler=verify_scheduler, ema=verify_ema,
    )
    if verify_optimizer.state or verify_ema.num_updates != 0:
        raise RuntimeError("Stage B initialization checkpoint contains stale training state")
    volume.commit()
    return json.dumps({
        "schema_version": 1, "issue": "CWX-24", "status": "passed",
        "source_checkpoint": str(SOURCE),
        "parent_checkpoint_id": resumed.provenance["parent_checkpoint_id"],
        "source_weights": "ema",
        "compatibility_checked_before_load": ["model", "codec", "camera"],
        "initialization": {
            "optimizer": "fresh_adamw_empty_state",
            "scheduler": "fresh_constant_last_epoch_0",
            "global_step": resumed.global_step, "epoch": resumed.epoch,
            "ema": "copy_loaded_model_num_updates_0",
        },
        "excluded_objectives": {"df_timestep_mixture": False, "self_rollout_loss": False},
        "stage_b_checkpoint": str(DESTINATION),
        "stage_b_checkpoint_id": checkpoint_id,
        "resume_verification": {"passed": True, "restored_scope": list(resumed.restored_scope)},
        "execution": {"platform": "Modal", "device": "cpu", "gpu_used": False},
    }, sort_keys=True)


@app.local_entrypoint()
def main(output: str = str(ROOT / "artifacts" / "m4" / "cwx-24-stage-b-initialization.json")):
    report = json.loads(initialize.remote())
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(destination)
