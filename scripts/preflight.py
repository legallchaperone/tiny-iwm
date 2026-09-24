"""Validate the M0 experiment contract and write run records without training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from utils.provenance import write_run_records
from core.temporal_config import resolve_temporal_protocol
from core.video_layout import CodecTemporalSpec
from experiments.build import resolve_model_config, resolve_recipe_selection


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_DIR = PROJECT_ROOT / "configurations"
PRETRAINED_MODEL_FIELDS = (
    "pretrained",
    "pretrained_checkpoint",
    "pretrained_model_name_or_path",
    "from_pretrained",
    "load_pretrained",
)


def validate_preflight_config(cfg: DictConfig) -> None:
    """Reject unsafe or ambiguous M0 model/checkpoint configuration."""

    if cfg.model.get("initialization") != "random":
        raise ValueError("M0 requires model.initialization=random; implicit pretrained DiT loading is forbidden")

    configured_pretrained = [
        field for field in PRETRAINED_MODEL_FIELDS if cfg.model.get(field) not in (None, False, "")
    ]
    if configured_pretrained:
        names = ", ".join(configured_pretrained)
        raise ValueError(f"implicit pretrained DiT loading is forbidden; configured fields: {names}")

    checkpoint = cfg.get("checkpoint", {})
    init_selectors = {
        name: value
        for name, value in {
            "checkpoint.init_from": checkpoint.get("init_from"),
            "stage.initial_checkpoint": cfg.stage.get("initial_checkpoint"),
            "load": cfg.get("load"),
        }.items()
        if value
    }
    resume_selectors = {
        name: value
        for name, value in {
            "checkpoint.resume_from": checkpoint.get("resume_from"),
            "resume": cfg.get("resume"),
        }.items()
        if value
    }
    if len(init_selectors) > 1:
        raise ValueError(f"multiple initialization selectors configured: {', '.join(init_selectors)}")
    if len(resume_selectors) > 1:
        raise ValueError(f"multiple resume selectors configured: {', '.join(resume_selectors)}")
    if init_selectors and resume_selectors:
        raise ValueError("checkpoint.init_from and checkpoint.resume_from are mutually exclusive")

    _, model_config = resolve_model_config(cfg)
    resolve_recipe_selection(cfg)
    temporal = resolve_temporal_protocol(cfg)
    temporal.layout(
        CodecTemporalSpec(cfg.representation.temporal_compression),
        temporal_patch_size=model_config.patch_size[0],
        purpose="train",
    )
    temporal.layout(
        CodecTemporalSpec(cfg.representation.temporal_compression),
        temporal_patch_size=model_config.patch_size[0],
        purpose="rollout",
    )
    resources = cfg.get("resources", {})
    temporal.validate_resources(
        available_rgb_frames=resources.get("available_rgb_frames"),
        estimated_gpu_gb=resources.get("estimated_gpu_gb"),
        maximum_gpu_gb=resources.get("maximum_gpu_gb"),
    )


def compose_validate_and_write(output_dir: Path, overrides: Sequence[str]):
    """Compose the baseline, validate its contract, and persist local records."""

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides))
    validate_preflight_config(cfg)
    return write_run_records(cfg, output_dir, PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    compose_validate_and_write(args.output_dir, overrides)
    print(f"M0 preflight passed; records written to {args.output_dir}")


if __name__ == "__main__":
    main()
