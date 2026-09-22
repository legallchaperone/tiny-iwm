"""Shared, framework-independent contracts for tiny-iwm."""

from core.camera import CameraCondition, IntrinsicsSpace
from core.types import LatentSpec, ProbeEvent, RolloutResult, TrainingBatch, VideoBatch
from core.video_layout import FrameRange, VideoLayout

__all__ = [
    "CameraCondition",
    "FrameRange",
    "IntrinsicsSpace",
    "LatentSpec",
    "ProbeEvent",
    "RolloutResult",
    "TrainingBatch",
    "VideoBatch",
    "VideoLayout",
]
