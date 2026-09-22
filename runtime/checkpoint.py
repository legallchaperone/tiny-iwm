"""Atomic checkpoints with distinct cross-stage init and same-run resume paths."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Callable, Mapping

import torch
from torch import nn

from .ema import ExponentialMovingAverage


SCHEMA_VERSION = 1
RESUME_SCOPE = (
    "model",
    "optimizer",
    "scheduler",
    "ema",
    "global_step",
    "epoch",
    "rng_python",
    "rng_torch_cpu",
    "rng_torch_cuda",
)


@dataclass(frozen=True)
class CheckpointSelection:
    init_from: str | None = None
    resume_from: str | None = None

    def __post_init__(self) -> None:
        if self.init_from and self.resume_from:
            raise ValueError("init_from and resume_from are mutually exclusive")


@dataclass(frozen=True)
class CheckpointProvenance:
    run_id: str
    stage: str
    model: Mapping[str, Any]
    codec: Mapping[str, Any]
    camera: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    parent_checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id or not self.stage:
            raise ValueError("checkpoint run_id and stage must be explicit")
        for name in ("model", "codec", "camera", "resolved_config"):
            value = getattr(self, name)
            if not value:
                raise ValueError(f"checkpoint {name} provenance must be non-empty")
            _json_snapshot(value)

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "model": _json_snapshot(self.model),
            "codec": _json_snapshot(self.codec),
            "camera": _json_snapshot(self.camera),
            "resolved_config": _json_snapshot(self.resolved_config),
            "parent_checkpoint_id": self.parent_checkpoint_id,
        }


@dataclass(frozen=True)
class InitializedStage:
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    ema: ExponentialMovingAverage
    global_step: int
    epoch: int
    parent_checkpoint_id: str
    source_weights: str
    ema_rule: str


@dataclass(frozen=True)
class ResumedRun:
    global_step: int
    epoch: int
    checkpoint_id: str
    restored_scope: tuple[str, ...]
    provenance: Mapping[str, Any]


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: ExponentialMovingAverage,
    global_step: int,
    epoch: int,
    provenance: CheckpointProvenance,
) -> str:
    if global_step < 0 or epoch < 0:
        raise ValueError("checkpoint progress cannot be negative")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance.snapshot(),
        "resume_scope": list(RESUME_SCOPE),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict(),
        "progress": {"global_step": global_step, "epoch": epoch},
        "rng": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    checkpoint_id = _file_id(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        checkpoint_id.removeprefix("sha256:") + "\n", encoding="utf-8"
    )
    return checkpoint_id


def initialize_new_stage(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    scheduler_factory: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler],
    ema_decay: float,
    source_weights: str = "model",
    ema_rule: str = "copy_loaded_model",
) -> InitializedStage:
    """Load weights only, then create fresh state for a different stage."""
    payload, checkpoint_id = _load(path)
    if source_weights == "model":
        state = payload["model"]
    elif source_weights == "ema":
        state = _ema_as_model_state(payload["model"], payload["ema"])
    else:
        raise ValueError("source_weights must be 'model' or 'ema'")
    if ema_rule != "copy_loaded_model":
        raise ValueError("cross-stage EMA rule must be copy_loaded_model")
    model.load_state_dict(state, strict=True)
    optimizer = optimizer_factory(model)
    scheduler = scheduler_factory(optimizer)
    ema = ExponentialMovingAverage(model, ema_decay)
    return InitializedStage(
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        global_step=0,
        epoch=0,
        parent_checkpoint_id=checkpoint_id,
        source_weights=source_weights,
        ema_rule=ema_rule,
    )


def resume_same_run(
    path: str | Path,
    *,
    expected_run_id: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: ExponentialMovingAverage,
) -> ResumedRun:
    """Restore every same-run state component and its recorded scope."""
    payload, checkpoint_id = _load(path)
    provenance = payload["provenance"]
    if provenance["run_id"] != expected_run_id:
        raise ValueError(
            f"resume run_id mismatch: checkpoint={provenance['run_id']}, "
            f"runtime={expected_run_id}"
        )
    if tuple(payload["resume_scope"]) != RESUME_SCOPE:
        raise ValueError("checkpoint resume scope is incomplete or unsupported")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    ema.load_state_dict(payload["ema"])
    random.setstate(payload["rng"]["python"])
    torch.set_rng_state(payload["rng"]["torch_cpu"])
    cuda_rng = payload["rng"]["torch_cuda"]
    if cuda_rng:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint has CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_rng)
    return ResumedRun(
        global_step=int(payload["progress"]["global_step"]),
        epoch=int(payload["progress"]["epoch"]),
        checkpoint_id=checkpoint_id,
        restored_scope=RESUME_SCOPE,
        provenance=provenance,
    )


def _load(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    checkpoint_id = _file_id(source)
    sidecar = source.with_suffix(source.suffix + ".sha256")
    if sidecar.is_file():
        expected = f"sha256:{sidecar.read_text(encoding='utf-8').strip()}"
        if checkpoint_id != expected:
            raise ValueError("checkpoint SHA-256 does not match its sidecar")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported world-model checkpoint schema")
    return payload, checkpoint_id


def _ema_as_model_state(
    model_state: Mapping[str, torch.Tensor], ema_state: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    output = {name: value.detach().clone() for name, value in model_state.items()}
    shadow = ema_state["shadow"]
    for name, value in shadow.items():
        if name not in output:
            raise ValueError(f"EMA key {name} is absent from model state")
        output[name] = value.detach().clone()
    return output


def _file_id(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _json_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(value, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint provenance must be JSON serializable") from error
