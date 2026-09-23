"""Resolve the end-to-end temporal protocol before data or CUDA is opened."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from core.video_layout import CodecTemporalSpec, VideoLayout


OFFICIAL_RGB_FRAMES = 961
OFFICIAL_FPS = 16.0


@dataclass(frozen=True)
class TemporalProtocol:
    """The unique public configuration contract for temporal geometry."""

    train_window_rgb_frames: int
    chunk_latent_frames: int
    rollout_rgb_frames: int
    initial_condition_rgb_frames: int
    fps: float

    @property
    def train_duration_seconds(self) -> float:
        return (self.train_window_rgb_frames - 1) / self.fps

    @property
    def rollout_duration_seconds(self) -> float:
        return (self.rollout_rgb_frames - 1) / self.fps

    @property
    def is_official_minute(self) -> bool:
        return self.rollout_rgb_frames == OFFICIAL_RGB_FRAMES and self.fps == OFFICIAL_FPS

    @property
    def artifact_kind(self) -> str:
        return "official_minute" if self.is_official_minute else "debug"

    def require_official_minute(self) -> None:
        """Guard the formal research/evaluation handoff."""
        if not self.is_official_minute:
            raise ValueError(
                "formal minute evaluation requires the named 961-frame, 16 FPS preset; "
                "this output is a debug artifact"
            )

    def layout(
        self,
        codec: CodecTemporalSpec,
        *,
        temporal_patch_size: int,
        purpose: str,
    ) -> VideoLayout:
        if purpose not in {"train", "rollout"}:
            raise ValueError("purpose must be 'train' or 'rollout'")
        if self.chunk_latent_frames % temporal_patch_size:
            raise ValueError("latent chunk size must be divisible by temporal patch size")
        frames = (
            self.train_window_rgb_frames if purpose == "train" else self.rollout_rgb_frames
        )
        return VideoLayout.from_codec(
            fps=self.fps,
            rgb_frame_count=frames,
            codec=codec,
            temporal_patch_size=temporal_patch_size,
            latent_chunk_size=self.chunk_latent_frames,
            initial_condition_frames=self.initial_condition_rgb_frames,
        )

    def validate_resources(
        self,
        *,
        available_rgb_frames: int | None = None,
        estimated_gpu_gb: float | None = None,
        maximum_gpu_gb: float | None = None,
    ) -> None:
        """Fail before allocation when data or a supplied memory estimate is insufficient."""
        required = max(self.train_window_rgb_frames, self.rollout_rgb_frames)
        if available_rgb_frames is not None and available_rgb_frames < required:
            raise ValueError(
                f"source has {available_rgb_frames} RGB frames, but protocol requires {required}"
            )
        if (estimated_gpu_gb is None) != (maximum_gpu_gb is None):
            raise ValueError("estimated_gpu_gb and maximum_gpu_gb must be configured together")
        if estimated_gpu_gb is not None:
            if not all(isfinite(value) and value > 0 for value in (estimated_gpu_gb, maximum_gpu_gb)):
                raise ValueError("GPU memory estimates must be finite and positive")
            if estimated_gpu_gb > maximum_gpu_gb:
                raise ValueError(
                    f"estimated GPU memory {estimated_gpu_gb:g} GiB exceeds limit "
                    f"{maximum_gpu_gb:g} GiB"
                )


def resolve_temporal_protocol(config: Mapping[str, Any]) -> TemporalProtocol:
    """Parse the five canonical values, rejecting missing or legacy duplicates."""
    try:
        train = config["train"]
        temporal = config["temporal"]
        rollout = config["rollout"]
        protocol = TemporalProtocol(
            train_window_rgb_frames=int(train["window_rgb_frames"]),
            chunk_latent_frames=int(temporal["chunk_latent_frames"]),
            rollout_rgb_frames=int(rollout["rgb_frames"]),
            initial_condition_rgb_frames=int(temporal["initial_condition_rgb_frames"]),
            fps=float(temporal["fps"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "temporal contract requires train.window_rgb_frames, "
            "temporal.chunk_latent_frames, temporal.initial_condition_rgb_frames, "
            "temporal.fps, and rollout.rgb_frames"
        ) from exc
    integers = {
        "train.window_rgb_frames": protocol.train_window_rgb_frames,
        "temporal.chunk_latent_frames": protocol.chunk_latent_frames,
        "temporal.initial_condition_rgb_frames": protocol.initial_condition_rgb_frames,
        "rollout.rgb_frames": protocol.rollout_rgb_frames,
    }
    if any(value <= 0 for value in integers.values()):
        raise ValueError("temporal frame counts must be positive")
    if not isfinite(protocol.fps) or protocol.fps <= 0:
        raise ValueError("temporal.fps must be finite and positive")
    if protocol.initial_condition_rgb_frames > min(
        protocol.train_window_rgb_frames, protocol.rollout_rgb_frames
    ):
        raise ValueError("initial condition exceeds the train or rollout extent")
    for section in (config.get("dataset", {}), rollout):
        if "fps" in section:
            raise ValueError("fps has one owner: temporal.fps")
    if "rgb_frames" in config.get("dataset", {}):
        raise ValueError("training length has one owner: train.window_rgb_frames")
    if "duration_seconds" in rollout:
        raise ValueError("rollout duration is derived from frames and fps")
    return protocol
