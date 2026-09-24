"""Resolve the end-to-end temporal protocol before data or CUDA is opened."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from core.video_layout import CodecTemporalSpec, VideoLayout


OFFICIAL_RGB_FRAMES = 961
OFFICIAL_FPS = 16.0
LEGACY_RUN_IDS = frozenset({
    "stage-a-sekai-subset-seed21-v1",
    "stage-b-sekai-subset-seed21-v2",
})


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
        if type(temporal_patch_size) is not int or temporal_patch_size <= 0:
            raise ValueError("temporal patch size must be a positive integer")
        if self.chunk_latent_frames % temporal_patch_size:
            raise ValueError("latent chunk size must be divisible by temporal patch size")
        frames = (
            self.train_window_rgb_frames if purpose == "train" else self.rollout_rgb_frames
        )
        layout = VideoLayout.from_codec(
            fps=self.fps,
            rgb_frame_count=frames,
            codec=codec,
            temporal_patch_size=temporal_patch_size,
            latent_chunk_size=self.chunk_latent_frames,
            initial_condition_frames=self.initial_condition_rgb_frames,
        )
        if layout.initial_condition_latent_frame_count % temporal_patch_size:
            raise ValueError(
                "initial condition latent boundary must align with temporal patch size"
            )
        return layout

    def validate_resources(
        self,
        *,
        available_rgb_frames: int | None = None,
        purpose: str = "train",
        estimated_gpu_gb: float | None = None,
        maximum_gpu_gb: float | None = None,
    ) -> None:
        """Fail before allocation when data or a supplied memory estimate is insufficient."""
        if purpose not in {"train", "rollout"}:
            raise ValueError("purpose must be 'train' or 'rollout'")
        required = (
            self.train_window_rgb_frames if purpose == "train" else self.rollout_rgb_frames
        )
        if available_rgb_frames is not None:
            if type(available_rgb_frames) is not int or available_rgb_frames < 0:
                raise ValueError("available_rgb_frames must be a non-negative integer")
            if available_rgb_frames < required:
                raise ValueError(
                    f"source has {available_rgb_frames} RGB frames, but protocol requires {required}"
                )
        if (estimated_gpu_gb is None) != (maximum_gpu_gb is None):
            raise ValueError("estimated_gpu_gb and maximum_gpu_gb must be configured together")
        if estimated_gpu_gb is not None:
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isfinite(value)
                and value > 0
                for value in (estimated_gpu_gb, maximum_gpu_gb)
            ):
                raise ValueError("GPU memory estimates must be finite and positive")
            if estimated_gpu_gb > maximum_gpu_gb:
                raise ValueError(
                    f"estimated GPU memory {estimated_gpu_gb:g} GiB exceeds limit "
                    f"{maximum_gpu_gb:g} GiB"
                )


def resolve_temporal_protocol(config: Mapping[str, Any]) -> TemporalProtocol:
    """Parse the five canonical values without silently truncating user input."""
    # Only the two published runs use this compatibility path. Keep their old
    # config bytes and training geometry stable for checkpoints and provenance.
    if config.get("run_id") in LEGACY_RUN_IDS and not any(
        key in config for key in ("train", "temporal", "rollout")
    ):
        legacy_chunk = config.get("flow_matching", {}).get("latent_frames_per_chunk", 4)
        if type(legacy_chunk) is not int or legacy_chunk <= 0:
            raise ValueError("legacy flow_matching.latent_frames_per_chunk must be positive")
        return TemporalProtocol(
            train_window_rgb_frames=OFFICIAL_RGB_FRAMES,
            chunk_latent_frames=legacy_chunk,
            rollout_rgb_frames=OFFICIAL_RGB_FRAMES,
            initial_condition_rgb_frames=1,
            fps=OFFICIAL_FPS,
        )
    try:
        train = config["train"]
        temporal = config["temporal"]
        rollout = config["rollout"]
        raw_counts = {
            "train.window_rgb_frames": train["window_rgb_frames"],
            "temporal.chunk_latent_frames": temporal["chunk_latent_frames"],
            "temporal.initial_condition_rgb_frames": temporal["initial_condition_rgb_frames"],
            "rollout.rgb_frames": rollout["rgb_frames"],
        }
        for name, value in raw_counts.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        raw_fps = temporal["fps"]
        if isinstance(raw_fps, bool) or not isinstance(raw_fps, (int, float)):
            raise ValueError("temporal.fps must be a number")
        fps = float(raw_fps)
        if not isfinite(fps) or fps <= 0:
            raise ValueError("temporal.fps must be finite and positive")
        protocol = TemporalProtocol(
            train_window_rgb_frames=raw_counts["train.window_rgb_frames"],
            chunk_latent_frames=raw_counts["temporal.chunk_latent_frames"],
            rollout_rgb_frames=raw_counts["rollout.rgb_frames"],
            initial_condition_rgb_frames=raw_counts["temporal.initial_condition_rgb_frames"],
            fps=fps,
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "temporal contract requires train.window_rgb_frames, "
            "temporal.chunk_latent_frames, temporal.initial_condition_rgb_frames, "
            "temporal.fps, and rollout.rgb_frames"
        ) from exc
    if protocol.initial_condition_rgb_frames > min(
        protocol.train_window_rgb_frames, protocol.rollout_rgb_frames
    ):
        raise ValueError("initial condition exceeds the train or rollout extent")

    # Reject old chunk owners explicitly. Otherwise a typo or stale field can
    # look active while the canonical temporal value silently wins.
    legacy_chunk_keys = (
        ("train", "chunk_latent_frames"),
        ("flow_matching", "latent_frames_per_chunk"),
        ("stage", "latent_frames_per_chunk"),
        ("flow_matching", "chunk_duration_seconds"),
        ("stage", "chunk_duration_seconds"),
        ("temporal", "latent_frames_per_chunk"),
        ("temporal", "chunk_duration_seconds"),
        ("rollout", "chunk_duration_seconds"),
    )
    for section_name, key in legacy_chunk_keys:
        section = config.get(section_name)
        if isinstance(section, Mapping) and key in section:
            raise ValueError(
                f"{section_name}.{key} is a legacy chunk setting; "
                "use temporal.chunk_latent_frames only"
            )

    dataset = config.get("dataset", {})
    if isinstance(dataset, Mapping):
        if "fps" in dataset:
            raise ValueError("fps has one owner: temporal.fps")
        if "rgb_frames" in dataset:
            raise ValueError("training length has one owner: train.window_rgb_frames")
    if "fps" in rollout:
        raise ValueError("fps has one owner: temporal.fps")
    if "duration_seconds" in rollout:
        raise ValueError("rollout duration is derived from frames and fps")
    return protocol
