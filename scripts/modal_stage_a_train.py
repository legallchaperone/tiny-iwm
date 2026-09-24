"""Prepare official latents on CPU, then train CWX-21 on an A10G.

The expensive function never accesses Hugging Face. Prepared data and the
selected resumable checkpoint live in a persistent Modal Volume.
"""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage-a-sekai-subset-seed21-v1"
REVISION = "4d965e94b9ea11b9c5ba085251ffa7a0345e006f"
REPOSITORY = "Efficient-Large-Model/SANA-WM-example-training-dataset"
BASE_URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}"
DATA_PREFIX = "data/sekai_game_train_961frames_16fps_ovl640"
LATENT_ZIP = (
    "data/vae_cache/LTX2VAE_diffusers_704x1280/"
    "sekai_game_train_961frames_16fps_ovl640/sekai_game_train_00000000.zip"
)
CAMERA_FILE = f"{DATA_PREFIX}/sekai_game_train_00000000_camera.npz"
VOLUME_ROOT = Path("/stage-a")
MANIFEST_PATH = VOLUME_ROOT / "manifest.json"
CHECKPOINT_PATH = VOLUME_ROOT / "checkpoints" / "stage-a-best.pt"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "numpy==2.2.6",
        "requests==2.32.5",
        "remotezip==0.12.3",
        "pyyaml==6.0.2",
    )
    .add_local_dir(ROOT, remote_path="/opt/tiny-iwm")
)
app = modal.App("tiny-iwm-cwx-21-stage-a")
volume = modal.Volume.from_name("tiny-iwm-stage-a", create_if_missing=True)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=3600,
    memory=8192,
    cpu=4,
)
def prepare_data(force: bool = False) -> str:
    """Range-read 20 latent members and download camera metadata on CPU."""
    import sys

    import numpy as np
    import requests
    from remotezip import RemoteZip

    sys.path.insert(0, "/opt/tiny-iwm")
    os.chdir("/opt/tiny-iwm")
    from scripts.stage_a_data import scene_id, select_scene_disjoint_samples

    volume.reload()
    if MANIFEST_PATH.is_file() and not force:
        cached = json.loads(MANIFEST_PATH.read_text())
        if cached.get("run_id") == RUN_ID and cached.get("source", {}).get("revision") == REVISION:
            return json.dumps(cached, sort_keys=True)

    VOLUME_ROOT.mkdir(parents=True, exist_ok=True)
    sample_root = VOLUME_ROOT / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    zip_url = f"{BASE_URL}/{LATENT_ZIP}"
    camera_url = f"{BASE_URL}/{CAMERA_FILE}"
    camera_path = VOLUME_ROOT / "source-camera.npz"
    if force or not camera_path.is_file():
        temporary = camera_path.with_suffix(".download")
        with requests.get(camera_url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        os.replace(temporary, camera_path)

    camera = np.load(camera_path, allow_pickle=False)
    ids = [str(value) for value in camera["ids"].tolist()]
    index_by_id = {value: index for index, value in enumerate(ids)}
    ranges = camera["ranges"]
    poses = camera["pose"]
    intrinsics = camera["intrinsics"]
    records: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}

    with RemoteZip(zip_url) as archive:
        members = [item.filename for item in archive.infolist() if item.filename.endswith(".npz")]
        selected = select_scene_disjoint_samples(
            members, train_scenes=16, validation_scenes=4
        )
        for split, split_members in selected.items():
            split_root = sample_root / split
            split_root.mkdir(parents=True, exist_ok=True)
            for member in split_members:
                sample_id = Path(member).stem
                if sample_id not in index_by_id:
                    raise KeyError(f"latent sample absent from camera index: {sample_id}")
                camera_index = index_by_id[sample_id]
                offset, length = (int(value) for value in ranges[camera_index])
                if length < 961:
                    raise ValueError(f"camera trajectory is too short for {sample_id}: {length}")
                latent_archive = np.load(BytesIO(archive.read(member)), allow_pickle=False)
                latent = latent_archive["z"]
                if latent.shape[:2] != (128, 121) or latent.shape[2] < 13 or latent.shape[3] < 22:
                    raise ValueError(f"unexpected latent shape for {sample_id}: {latent.shape}")
                crop = np.asarray(latent[:, :, 9:13, 18:22], dtype=np.float32)
                pose = np.asarray(poses[offset : offset + 961], dtype=np.float32)
                intrinsic = np.asarray(intrinsics[offset : offset + 961], dtype=np.float32)
                destination = split_root / f"{sample_id}.npz"
                np.savez(destination, z=crop, pose=pose, intrinsics=intrinsic)
                records[split].append(
                    {
                        "sample_id": sample_id,
                        "scene_id": scene_id(member),
                        "source_member": member,
                        "prepared_path": str(destination),
                        "sha256": _sha256_file(destination),
                        "latent_shape": list(crop.shape),
                        "camera_frames": int(pose.shape[0]),
                    }
                )

    core_manifest: dict[str, object] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "issue": "CWX-21",
        "source": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "license": "fair-noncommercial-research-license",
            "latent_zip": LATENT_ZIP,
            "camera_file": CAMERA_FILE,
        },
        "selection": {
            "algorithm": "first_sample_per_sorted_scene",
            "train_scenes": 16,
            "validation_scenes": 4,
            "validation_policy": "last_sorted_scenes",
            "scene_disjoint": True,
        },
        "transform": {
            "latent_crop_yx": [9, 18],
            "latent_crop_size": [4, 4],
            "camera_intrinsics": [
                "resize_rgb:1080x1920->720x1280",
                "crop_rgb:left=0,top=8,size=704x1280",
                "crop_rgb:left=576,top=288,size=128x128",
            ],
        },
        "splits": records,
    }
    core_manifest["manifest_sha256"] = _sha256_bytes(_canonical_json(core_manifest))
    MANIFEST_PATH.write_text(json.dumps(core_manifest, indent=2, sort_keys=True) + "\n")
    volume.commit()
    return json.dumps(core_manifest, sort_keys=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={str(VOLUME_ROOT): volume},
    timeout=3600,
    memory=65536,
    cpu=8,
)
def train(manifest_json: str) -> str:
    """Train only from prepared volume data and persist the selected checkpoint."""
    import torch
    import yaml

    os.chdir("/opt/tiny-iwm")
    import sys

    sys.path.insert(0, "/opt/tiny-iwm")
    from algorithms.world_model.flow import FlowMatchSpec, flow_matching_loss
    from algorithms.world_model.models import JointVideoDiT, JointVideoDiTConfig
    from algorithms.world_model.models.prope import (
        TokenCameraProjection,
    )
    from algorithms.world_model.training_batch import StageABatchBuilder
    from core.camera import CameraCondition
    from core.types import VideoBatch
    from core.video_layout import CodecTemporalSpec, VideoLayout
    from core.temporal_config import resolve_temporal_protocol
    from scripts.prepared_stage_data import (
        load_prepared_sample,
        validate_prepared_record,
    )
    from runtime import (
        CheckpointProvenance,
        ExponentialMovingAverage,
        OptimizerSpec,
        SchedulerSpec,
        build_optimizer,
        build_scheduler,
        configure_cuda_math_policy,
        save_checkpoint,
    )

    manifest = json.loads(manifest_json)
    if manifest.get("run_id") != RUN_ID:
        raise ValueError("prepared manifest belongs to a different run")
    config_path = Path("configurations/runs/stage_a_baseline.yaml")
    resolved_config = yaml.safe_load(config_path.read_text())
    config_sha256 = _sha256_bytes(config_path.read_bytes())
    model_values = resolved_config["model"]
    model_config = JointVideoDiTConfig(
        latent_channels=model_values["latent_channels"],
        hidden_size=model_values["hidden_size"],
        depth=model_values["depth"],
        num_heads=model_values["num_heads"],
        patch_size=tuple(model_values["patch_size"]),
        mlp_ratio=model_values["mlp_ratio"],
        qkv_bias=model_values["qkv_bias"],
        prope_camera_dims=model_values["prope_camera_dims"],
    )
    temporal = resolve_temporal_protocol(resolved_config)
    layout = temporal.layout(
        CodecTemporalSpec(8),
        temporal_patch_size=model_config.patch_size[0],
        purpose="train",
    )
    train_records = manifest["splits"]["train"]
    validation_records = manifest["splits"]["validation"]
    for record in (*train_records, *validation_records):
        validate_prepared_record(record, layout)

    if not torch.cuda.is_available():
        raise RuntimeError("CWX-21 training requires CUDA")
    seed = int(resolved_config["seed"])
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    configure_cuda_math_policy()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    model = JointVideoDiT(model_config).to(device=device, dtype=dtype)
    optimizer = build_optimizer(model, OptimizerSpec(learning_rate=1e-4, weight_decay=0.01))
    scheduler = build_scheduler(optimizer, SchedulerSpec("constant"))
    ema = ExponentialMovingAverage(model, 0.9999)
    flow_spec = FlowMatchSpec()
    builder = StageABatchBuilder(flow_spec)
    train_cache = [load_prepared_sample(record, device=device, dtype=dtype, layout=layout) for record in train_records]
    validation_cache = [load_prepared_sample(record, device=device, dtype=dtype, layout=layout) for record in validation_records]

    def loss_for(
        clean: torch.Tensor,
        camera: CameraCondition,
        projection: TokenCameraProjection,
        valid_latent_mask: torch.Tensor,
        *,
        generator: torch.Generator,
        fixed_time: float | None = None,
    ) -> torch.Tensor:
        video = VideoBatch(
            sample_ids=("stage-a",),
            sources=("official-prepared-volume",),
            layout=layout,
            camera=camera,
            latents=clean,
        )
        flow_time = None
        if fixed_time is not None:
            flow_time = torch.tensor([fixed_time], device=device, dtype=dtype)
        batch = builder.build(
            video,
            generator=generator,
            flow_time=flow_time,
            valid_latent_mask=valid_latent_mask,
        )
        prediction = model(
            batch.noisy_latents,
            batch.flow_time,
            camera_projection=projection,
        )
        return flow_matching_loss(
            prediction,
            batch.target_velocity,
            loss_mask=batch.loss_mask,
            time=batch.flow_time,
            spec=flow_spec,
        )

    @torch.no_grad()
    def validate() -> float:
        raw = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }
        ema.copy_to(model)
        model.eval()
        losses = []
        for index, (clean, camera, projection, valid_latent_mask) in enumerate(validation_cache):
            generator = torch.Generator(device=device).manual_seed(21000 + index)
            losses.append(
                float(
                    loss_for(
                        clean,
                        camera,
                        projection,
                        valid_latent_mask,
                        generator=generator,
                        fixed_time=0.5,
                    )
                )
            )
        state = model.state_dict()
        for name, value in raw.items():
            state[name].copy_(value)
        model.train()
        return sum(losses) / len(losses)

    provenance = CheckpointProvenance(
        run_id=RUN_ID,
        stage="A",
        model={
            "architecture": "joint_spatiotemporal_dit",
            "parameters": sum(value.numel() for value in model.parameters()),
            "config": resolved_config["model"],
        },
        codec={
            "identity": "LTX2VAE_diffusers_704x1280_official_latent_cache",
            "source_revision": REVISION,
            "temporal_compression": 8,
            "spatial_compression": 32,
        },
        camera={
            "pose_convention": "camera_to_world",
            "intrinsics_space": "rgb_pixels",
            "preprocessing": manifest["transform"]["camera_intrinsics"],
        },
        resolved_config=resolved_config,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    training_losses: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []
    best_metric = float("inf")
    best_step = 0
    best_checkpoint_id = ""
    started = time.perf_counter()
    model.train()
    for step in range(1, 101):
        clean, camera, projection, valid_latent_mask = train_cache[
            (step - 1) % len(train_cache)
        ]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(
            clean, camera, projection, valid_latent_mask, generator=generator
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)
        if step == 1 or step % 10 == 0:
            training_losses.append({"step": step, "loss": float(loss.detach())})
        if step in (50, 100):
            metric = validate()
            validations.append({"step": step, "ema_flow_matching_mse": metric})
            if metric < best_metric:
                best_metric = metric
                best_step = step
                best_checkpoint_id = save_checkpoint(
                    CHECKPOINT_PATH,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema=ema,
                    global_step=step,
                    epoch=step // len(train_cache),
                    provenance=provenance,
                )
                volume.commit()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    properties = torch.cuda.get_device_properties(device)
    report = {
        "schema_version": 1,
        "issue": "CWX-21",
        "status": "passed",
        "run_id": RUN_ID,
        "seed": seed,
        "config": {"path": str(config_path), "sha256": config_sha256},
        "data": {
            "manifest_sha256": manifest["manifest_sha256"],
            "source_revision": REVISION,
            "train_sample_ids": [value["sample_id"] for value in train_records],
            "validation_sample_ids": [value["sample_id"] for value in validation_records],
            "scene_disjoint": True,
        },
        "training": {
            "steps": 100,
            "batch_size": 1,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "scheduler": "constant",
            "precision": "bfloat16",
            "ema_decay": 0.9999,
            "loss_trace": training_losses,
            "wall_seconds": elapsed,
        },
        "validation": {
            "weights": "ema",
            "deterministic_noise_seed": 21000,
            "flow_time": 0.5,
            "measurements": validations,
            "selection": "lowest_validation_flow_matching_mse",
        },
        "selected_checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "checkpoint_id": best_checkpoint_id,
            "step": best_step,
            "ema_flow_matching_mse": best_metric,
            "resume_scope": [
                "model", "optimizer", "scheduler", "ema", "global_step",
                "epoch", "rng_python", "rng_torch_cpu", "rng_torch_cuda",
            ],
        },
        "hardware": {
            "platform": "Modal",
            "gpu_request": "A10G",
            "device": properties.name,
            "total_memory_bytes": properties.total_memory,
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "python": platform.python_version(),
        },
        "cost_control": {
            "downloads_inside_gpu_function": False,
            "prepared_data_volume": "tiny-iwm-stage-a",
        },
    }
    return json.dumps(report, sort_keys=True)


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=900,
    memory=16384,
    cpu=4,
)
def verify_checkpoint() -> str:
    """Verify the persisted artifact on CPU without starting a GPU."""
    import torch

    volume.reload()
    checkpoint_id = _sha256_file(CHECKPOINT_PATH)
    sidecar = CHECKPOINT_PATH.with_suffix(CHECKPOINT_PATH.suffix + ".sha256")
    expected = "sha256:" + sidecar.read_text().strip()
    if checkpoint_id != expected:
        raise ValueError("checkpoint digest does not match its sidecar")
    payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    required = {
        "model", "optimizer", "scheduler", "ema", "progress", "rng",
        "provenance", "resume_scope",
    }
    missing = sorted(required.difference(payload))
    if payload.get("schema_version") != 1 or missing:
        raise ValueError(f"invalid checkpoint schema; missing={missing}")
    if payload["provenance"]["run_id"] != RUN_ID:
        raise ValueError("checkpoint run identity mismatch")
    return json.dumps(
        {
            "status": "passed",
            "checkpoint_id": checkpoint_id,
            "schema_version": payload["schema_version"],
            "global_step": payload["progress"]["global_step"],
            "epoch": payload["progress"]["epoch"],
            "resume_scope": payload["resume_scope"],
            "sidecar_matches": True,
            "verified_on": "cpu",
        },
        sort_keys=True,
    )


@app.local_entrypoint()
def main(
    report_output: str = str(ROOT / "artifacts" / "m3" / "cwx-21-stage-a-run.json"),
    manifest_output: str = str(ROOT / "artifacts" / "m3" / "cwx-21-stage-a-manifest.json"),
    force_prepare: bool = False,
    verify_only: bool = False,
):
    manifest_json = prepare_data.remote(force_prepare)
    report_path = Path(report_output)
    manifest_path = Path(manifest_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if verify_only:
        report = json.loads(report_path.read_text())
    else:
        report = json.loads(train.remote(manifest_json))
    report["checkpoint_verification"] = json.loads(verify_checkpoint.remote())
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest_path.write_text(json.dumps(json.loads(manifest_json), indent=2, sort_keys=True) + "\n")
    print(report_path)
    print(manifest_path)
