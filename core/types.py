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


@dataclass(frozen=True)
class ProbeEvent:
    """One named observation with distinct rollout and flow-matching time."""

    sample_id: str
    layer: str
    observation_point: str
    rollout_time: int
    flow_time: float
    token_type: str
    branch: str
    forward_purpose: str
    value: Optional[Any] = None
