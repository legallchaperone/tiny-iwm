"""Initial framework-independent data objects for the tiny-iwm pipeline.

Video tensors crossing representation boundaries use ``[B, C, T, H, W]``.
Flattened DiT token tensors use ``[B, N, D]``. Adapters must convert upstream
layouts once at their boundary and must not rely on shape guessing downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from core.camera import CameraCondition
from core.video_layout import VideoLayout


class _FrozenDict(dict):
    """A JSON/pickle-friendly dictionary that rejects mutation after creation."""

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("frozen mapping does not support mutation")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __reduce__(self):
        return type(self), (dict(self),)


def _freeze_value(value: Any) -> Any:
    """Snapshot common mutable containers without copying tensor-like leaves."""

    if isinstance(value, Mapping):
        return _FrozenDict({key: _freeze_value(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(child) for child in value)
    if isinstance(value, (set, frozenset)):
        frozen = (_freeze_value(child) for child in value)
        return tuple(sorted(frozen, key=_canonical_sort_key))
    return value


def _canonical_sort_key(value: Any) -> Tuple[str, str]:
    """Order hashable set members consistently across Python processes."""

    value_type = type(value)
    type_name = f"{value_type.__module__}.{value_type.__qualname__}"
    if isinstance(value, tuple):
        return type_name, repr(tuple(_canonical_sort_key(child) for child in value))
    return type_name, repr(value)


@dataclass(frozen=True)
class LatentSpec:
    """Identity and layout of ``[B, C, T, H, W]`` codec latents."""

    codec_id: str
    channels: int
    temporal_compression: int
    spatial_compression: Tuple[int, int]
    normalization: Mapping[str, Any]
    causal: bool
    encoding_policy: str

    def __post_init__(self) -> None:
        if self.channels <= 0 or self.temporal_compression <= 0:
            raise ValueError("latent channels and temporal compression must be positive")
        if len(self.spatial_compression) != 2 or any(
            value <= 0 for value in self.spatial_compression
        ):
            raise ValueError("spatial_compression must contain two positive values")
        if not self.codec_id or not self.encoding_policy:
            raise ValueError("codec_id and encoding_policy must be explicit")


@dataclass(frozen=True)
class VideoBatch:
    """Input samples and conditions before training-batch construction.

    ``video`` and ``latents`` use ``[B, C, T, H, W]``. ``valid_mask`` uses
    ``[B, T, H, W]`` and ``timestamps_seconds`` uses ``[B, T]``.
    """

    sample_ids: Tuple[str, ...]
    sources: Tuple[str, ...]
    layout: VideoLayout
    camera: CameraCondition
    video: Optional[Any] = None
    latents: Optional[Any] = None
    text: Optional[Tuple[str, ...]] = None
    valid_mask: Optional[Any] = None
    timestamps_seconds: Optional[Any] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_ids or len(self.sample_ids) != len(self.sources):
            raise ValueError("sample_ids and sources must be non-empty and aligned")
        if self.text is not None and len(self.text) != len(self.sample_ids):
            raise ValueError("text must align one-to-one with sample_ids")
        if self.video is None and self.latents is None:
            raise ValueError("VideoBatch requires video, latents, or both")


@dataclass(frozen=True)
class TrainingBatch:
    """Flow-matching inputs with explicit visibility and loss semantics.

    Latent-like tensors use ``[B, C, T, H, W]``. ``flow_time`` is ``[B]`` or
    ``[B, T]``; ``loss_mask`` and ``attention_visibility`` document rather than
    imply which targets and attention edges are legal.
    """

    noisy_latents: Any
    clean_condition_latents: Any
    flow_time: Any
    target_velocity: Any
    loss_mask: Any
    attention_visibility: Any
    layout: VideoLayout
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutResult:
    """Immutable identity and outputs of one rollout invocation."""

    output_files: Tuple[str, ...]
    latent_files: Tuple[str, ...]
    checkpoint_id: str
    seed: int
    conditions: Mapping[str, Any]
    inference_settings: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "conditions", _freeze_value(self.conditions))
        object.__setattr__(self, "inference_settings", _freeze_value(self.inference_settings))


@dataclass(frozen=True)
class ProbeEvent:
    """One observation with complete run, time, token, and branch identity.

    Seeds are mandatory fields but may be explicitly ``None`` when the forward
    purpose does not have that seed class (for example, no generation seed during
    training). ``physical_position`` names its coordinates, such as
    ``{"time_seconds": ..., "y": ..., "x": ...}``, instead of relying on tuple
    ordering.
    """

    sample_id: str
    training_seed: Optional[int]
    generation_seed: Optional[int]
    checkpoint_id: str
    config_id: str
    layer: str
    observation_point: str
    rollout_time: int
    flow_time: float
    token_type: str
    physical_position: Mapping[str, float]
    branch: str
    forward_purpose: str
    value: Optional[Any] = None
