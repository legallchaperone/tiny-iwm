"""Capture named DiT features without changing the model's forward result."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import Lock
from typing import Mapping

import torch

from algorithms.world_model.models import JointVideoDiT
from core.types import ProbeEvent


_RECOMPUTING = ContextVar("probe_activation_recomputation", default=False)
_PROCESS_LOCK = Lock()


@contextmanager
def recomputation_context():
    """Use as the second context of non-reentrant activation checkpointing."""
    token = _RECOMPUTING.set(True)
    try:
        yield
    finally:
        _RECOMPUTING.reset(token)


def checkpoint_contexts():
    """Pass to torch.utils.checkpoint.checkpoint(context_fn=...)."""
    return nullcontext(), recomputation_context()


@dataclass(frozen=True)
class TokenSelection:
    index: int
    token_type: str
    physical_position: Mapping[str, float]


@dataclass(frozen=True)
class CaptureContext:
    sample_id: str
    training_seed: int | None
    generation_seed: int | None
    checkpoint_id: str
    config_id: str
    rollout_time: int
    flow_time: float
    branch: str
    forward_purpose: str


class FeatureRecorder:
    """Keep summaries for selected tokens; raw tensors are opt-in and capped."""

    def __init__(
        self,
        root: Path,
        *,
        points: tuple[str, ...] = (),
        tokens: tuple[TokenSelection, ...] = (),
        max_records: int = 256,
        raw_points: frozenset[str] = frozenset(),
        max_raw_bytes: int = 0,
    ) -> None:
        if max_records < 0 or max_raw_bytes < 0 or not raw_points <= set(points):
            raise ValueError("invalid feature capture bounds or raw points")
        if len(set(points)) != len(points):
            raise ValueError("feature capture points must be unique")
        if len({token.index for token in tokens}) != len(tokens) or any(
            token.index < 0 or not token.token_type for token in tokens
        ):
            raise ValueError(
                "token selections require unique non-negative indices and roles"
            )
        self.root = root
        self.points = points
        self.tokens = tokens
        self.max_records = max_records
        self.raw_points = raw_points
        self.max_raw_bytes = max_raw_bytes
        self.record_count = len(list(root.glob("*.json")))
        self.raw_bytes = sum(path.stat().st_size for path in root.glob("*.pt"))
        if self.record_count > max_records or self.raw_bytes > max_raw_bytes:
            raise ValueError("existing feature storage exceeds capture bounds")

    def record(
        self, context: CaptureContext, observations: dict[str, torch.Tensor]
    ) -> None:
        if _RECOMPUTING.get() or not self.points or not self.tokens:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with _PROCESS_LOCK, (self.root / ".capture.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self.record_count = len(list(self.root.glob("*.json")))
                self.raw_bytes = sum(
                    path.stat().st_size for path in self.root.glob("*.pt")
                )
                self._record_locked(context, observations)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _record_locked(
        self, context: CaptureContext, observations: dict[str, torch.Tensor]
    ) -> None:
        count = len(self.points) * len(self.tokens)
        if self.record_count + count > self.max_records:
            raise ValueError("feature record budget exceeded")
        for point in self.points:
            values = observations[point]
            if values.ndim != 3 or values.shape[0] != 1:
                raise ValueError(
                    "feature capture requires batch size one [1, tokens, channels]"
                )
            for token in self.tokens:
                if token.index >= values.shape[1]:
                    raise ValueError("selected token is outside this forward pass")
                feature = (
                    values[0, token.index]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .clone()
                )
                if not torch.isfinite(feature).all():
                    raise ValueError("non-finite captured feature")
                layer, observation_point = (
                    point.rsplit(".", 1)
                    if point.startswith("blocks.")
                    else ("model", point)
                )
                event = ProbeEvent(
                    sample_id=context.sample_id,
                    training_seed=context.training_seed,
                    generation_seed=context.generation_seed,
                    checkpoint_id=context.checkpoint_id,
                    config_id=context.config_id,
                    layer=layer,
                    observation_point=observation_point,
                    rollout_time=context.rollout_time,
                    flow_time=context.flow_time,
                    token_type=token.token_type,
                    physical_position=token.physical_position,
                    branch=context.branch,
                    forward_purpose=context.forward_purpose,
                )
                payload = asdict(event)
                payload["token_index"] = token.index
                payload["channels"] = feature.numel()
                payload["mean"] = float(feature.mean())
                payload["rms"] = float(feature.square().mean().sqrt())
                payload["l2_norm"] = float(torch.linalg.vector_norm(feature))
                event_id = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                raw_path = None
                raw_size = 0
                try:
                    if point in self.raw_points:
                        payload["raw_file"] = f"{event_id}.pt"
                        raw_path = self.root / payload["raw_file"]
                        raw_size = self._write_once(
                            raw_path,
                            feature,
                            max_bytes=self.max_raw_bytes - self.raw_bytes,
                        )
                        self.raw_bytes += raw_size
                    self._write_once(self.root / f"{event_id}.json", payload)
                except Exception:
                    if raw_path is not None and raw_size:
                        raw_path.unlink(missing_ok=True)
                        self.raw_bytes -= raw_size
                    raise
                self.record_count += 1

    @staticmethod
    def _write_once(path: Path, value: object, *, max_bytes: int | None = None) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                if path.suffix == ".json":
                    handle.write(
                        json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
                    )
                else:
                    torch.save(value, handle)
            size = temporary.stat().st_size
            if max_bytes is not None and size > max_bytes:
                raise ValueError("raw feature byte budget exceeded")
            os.link(temporary, path)
            return size
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def capture_forward(
    model: JointVideoDiT,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    *,
    context: CaptureContext,
    recorder: FeatureRecorder,
    **model_kwargs,
):
    """Run the shared model once and return its normal output/cache shape."""
    if not recorder.points or not recorder.tokens or _RECOMPUTING.get():
        return model(latents, timestep, **model_kwargs)
    result = model(latents, timestep, capture=recorder.points, **model_kwargs)
    if model_kwargs.get("return_kv_cache", False):
        velocity, observations, cache = result
        recorder.record(context, observations)
        return velocity, cache
    velocity, observations = result
    recorder.record(context, observations)
    return velocity
