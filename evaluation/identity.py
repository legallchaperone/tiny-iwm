"""Generation identity and collision-safe output directories."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from core.video_layout import VideoLayout

from .selection import canonical_bytes, sha256


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(c in "0123456789abcdef" for c in value[7:])
    )


def generation_identity(
    selection: dict,
    row: dict,
    *,
    checkpoint_id: str,
    weight_flavor: str,
    model_config: dict,
    codec_id: str,
    codec_fingerprint: dict,
    spatial_resolution: tuple[int, int],
    preprocessing_id: str,
    rollout_layout: VideoLayout,
    conditioning: dict,
    implementation_id: str,
    numeric_execution: dict,
    sampler: dict,
) -> dict:
    """Hash every input that can change a generated trajectory."""
    if row not in selection["rows"]:
        raise ValueError("generation row is absent from the frozen selection")
    if selection["selection_id"] != sha256(
        canonical_bytes(
            {key: value for key, value in selection.items() if key != "selection_id"}
        )
    ):
        raise ValueError("selection content differs from its immutable identity")
    if not _is_sha256(checkpoint_id):
        raise ValueError("checkpoint_id must be a SHA-256 digest")
    if weight_flavor not in {"model", "ema"}:
        raise ValueError("weight_flavor must be model or ema")
    if (
        not isinstance(model_config, dict)
        or not {
            "latent_channels",
            "hidden_size",
            "depth",
            "num_heads",
            "patch_size",
            "mlp_ratio",
            "qkv_bias",
            "prope_camera_dims",
        }
        <= model_config.keys()
    ):
        raise ValueError("resolved model configuration must include every DiT setting")
    if not codec_id or not preprocessing_id or not implementation_id or not sampler:
        raise ValueError(
            "codec, preprocessing, implementation, and sampler must be explicit"
        )
    if (
        not isinstance(codec_fingerprint, dict)
        or not {
            "weights_sha256",
            "normalization_sha256",
            "encoding_policy",
            "implementation_revision",
            "execution_dtype",
            "execution_backend",
            "backend_fingerprint",
            "cuda_math_policy",
        }
        <= codec_fingerprint.keys()
        or not all(
            _is_sha256(codec_fingerprint[key])
            for key in ("weights_sha256", "normalization_sha256")
        )
        or not all(
            codec_fingerprint[key]
            for key in (
                "encoding_policy",
                "implementation_revision",
                "execution_dtype",
                "execution_backend",
                "backend_fingerprint",
                "cuda_math_policy",
            )
        )
    ):
        raise ValueError(
            "codec fingerprint must identify weights, normalization, policy, implementation, and execution"
        )
    if (
        not {
            "parameter_dtype",
            "autocast_dtype",
            "latent_dtype",
            "attention_backend",
            "torch_version",
            "tf32_enabled",
            "backend_fingerprint",
        }
        <= numeric_execution.keys()
    ):
        raise ValueError(
            "numeric execution must identify dtypes, attention, torch, and TF32"
        )
    backend = numeric_execution["backend_fingerprint"]
    if (
        not isinstance(backend, dict)
        or not {"device_name", "compute_capability", "cuda_runtime", "cudnn_version"}
        <= backend.keys()
        or any(
            not backend[key]
            for key in (
                "device_name",
                "compute_capability",
                "cuda_runtime",
                "cudnn_version",
            )
        )
    ):
        raise ValueError("numeric execution must identify the hardware backend")
    if len(spatial_resolution) != 2 or any(
        type(value) is not int or value <= 0 for value in spatial_resolution
    ):
        raise ValueError("spatial resolution must be positive integer height and width")
    required_sampler = {
        "solver",
        "steps",
        "cfg_scale",
        "history_policy",
        "initial_history_policy",
        "batch_size",
    }
    if not required_sampler <= sampler.keys():
        raise ValueError("sampler must identify solver, steps, CFG, and history policy")
    if sampler["initial_history_policy"] != "source_image_only":
        raise ValueError("official evaluation permits only the source-image prefix")
    if type(sampler["batch_size"]) is not int or sampler["batch_size"] != 1:
        raise ValueError("official evaluation requires single-row noise streams")
    if not {"camera", "text"} <= conditioning.keys():
        raise ValueError("conditioning must identify camera and text branches")
    camera, text = conditioning["camera"], conditioning["text"]
    if type(camera.get("enabled")) is not bool or type(text.get("enabled")) is not bool:
        raise ValueError("camera and text enabled flags must be explicit booleans")
    if (
        camera["enabled"]
        and not {"method", "translation_scale", "projection_image_size"}
        <= camera.keys()
    ):
        raise ValueError(
            "enabled camera conditioning requires method, scale, and projection size"
        )
    if text["enabled"] and not _is_sha256(text.get("encoder_fingerprint")):
        raise ValueError("enabled text conditioning requires an encoder fingerprint")
    if (
        rollout_layout.valid_rgb_frame_count != row["sanawm_frames"]
        or rollout_layout.fps != row["fps"]
    ):
        raise ValueError("rollout layout differs from the selected official trajectory")
    if (
        rollout_layout.initial_condition_rgb.start != 0
        or rollout_layout.initial_condition_rgb.stop != 1
    ):
        raise ValueError("official evaluation requires one source-image frame")
    layout_identity = {
        "rgb_frame_count": rollout_layout.rgb_frame_count,
        "valid_rgb_frame_count": rollout_layout.valid_rgb_frame_count,
        "latent_frame_count": rollout_layout.latent_frame_count,
        "initial_condition_rgb": [
            rollout_layout.initial_condition_rgb.start,
            rollout_layout.initial_condition_rgb.stop,
        ],
        "latent_to_rgb": [
            [part.start, part.stop] for part in rollout_layout.latent_to_rgb
        ],
        "token_to_latent": list(rollout_layout.token_to_latent),
        "token_to_latent_ranges": [
            [part.start, part.stop] for part in rollout_layout.token_to_latent_ranges
        ],
        "chunk_to_latent": [
            [part.start, part.stop] for part in rollout_layout.chunk_to_latent
        ],
    }
    payload = {
        "schema_version": 1,
        "selection_id": selection["selection_id"],
        "dataset_revision": selection["dataset_revision"],
        "scene_id": row["scene_id"],
        "split": row["split"],
        "seed": row["generation_seed"],
        "checkpoint_id": checkpoint_id,
        "weight_flavor": weight_flavor,
        "model_config": model_config,
        "codec_id": codec_id,
        "codec_fingerprint": codec_fingerprint,
        "spatial_resolution": list(spatial_resolution),
        "preprocessing_id": preprocessing_id,
        "rollout_layout": layout_identity,
        "conditions_sha256": row["conditions_sha256"],
        "conditioning": conditioning,
        "implementation_id": implementation_id,
        "numeric_execution": numeric_execution,
        "sampler": sampler,
    }
    payload = json.loads(canonical_bytes(payload))
    return {"generation_id": sha256(canonical_bytes(payload)), **payload}


def claim_output_directory(root: Path, identity: dict) -> Path:
    """Reserve one directory per identity and reject a conflicting prior claim."""
    expected = sha256(
        canonical_bytes(
            {key: value for key, value in identity.items() if key != "generation_id"}
        )
    )
    if identity.get("generation_id") != expected:
        raise ValueError("generation identity does not match its content")
    for field in ("split", "scene_id"):
        value = identity[field]
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"unsafe {field} in output identity")
    directory = (
        root / identity["split"] / identity["scene_id"] / identity["generation_id"]
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "identity.json"
    if path.is_file():
        verify_output_identity(directory, identity)
        return directory
    data = json.dumps(identity, sort_keys=True, indent=2).encode() + b"\n"
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        # The directory name is the validated content hash; concurrent identical
        # claims can safely publish the same bytes on volumes without hard links.
        os.replace(temporary, path)
        verify_output_identity(directory, identity)
    finally:
        temporary.unlink(missing_ok=True)
    return directory


def verify_output_identity(directory: Path, identity: dict) -> None:
    recorded = json.loads((directory / "identity.json").read_text())
    if recorded != identity:
        raise ValueError("output directory belongs to a different generation identity")
