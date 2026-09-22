"""The sole contract for RGB, latent, token, chunk, and time mappings.

M0 stores explicit mappings supplied by a codec/data adapter. M1 will add the
canonical derivation from codec properties. Other modules must query this object
instead of reimplementing stride, boundary, or padding arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, order=True)
class FrameRange:
    """A non-empty half-open integer range ``[start, stop)``."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("FrameRange must be non-empty with 0 <= start < stop")

    def __len__(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class VideoLayout:
    """Explicit temporal layout shared by readers, cameras, models, and writers.

    ``latent_to_rgb`` and ``chunk_to_latent`` use half-open ranges. A token maps to
    one latent timestep through ``token_to_latent``; spatial token coordinates are
    intentionally outside this initial temporal contract. ``rgb_is_padding`` marks
    padded frames while retaining their indices, so final-frame behavior is never
    inferred from tensor length.
    """

    fps: float
    rgb_frame_count: int
    latent_frame_count: int
    latent_to_rgb: Tuple[FrameRange, ...]
    token_to_latent: Tuple[int, ...]
    chunk_to_latent: Tuple[FrameRange, ...]
    rgb_is_padding: Tuple[bool, ...]
    initial_condition_rgb: FrameRange

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.rgb_frame_count <= 0 or self.latent_frame_count <= 0:
            raise ValueError("frame counts must be positive")
        if len(self.latent_to_rgb) != self.latent_frame_count:
            raise ValueError("latent_to_rgb must contain one range per latent frame")
        if len(self.rgb_is_padding) != self.rgb_frame_count:
            raise ValueError("rgb_is_padding must contain one value per RGB frame")
        if self.initial_condition_rgb.stop > self.rgb_frame_count:
            raise ValueError("initial condition extends beyond RGB frames")
        for rgb_range in self.latent_to_rgb:
            if rgb_range.stop > self.rgb_frame_count:
                raise ValueError("latent_to_rgb range extends beyond RGB frames")
        for latent_index in self.token_to_latent:
            if not 0 <= latent_index < self.latent_frame_count:
                raise ValueError("token_to_latent contains an invalid latent index")
        for latent_range in self.chunk_to_latent:
            if latent_range.stop > self.latent_frame_count:
                raise ValueError("chunk_to_latent range extends beyond latent frames")

    def rgb_range_for_latent(self, latent_index: int) -> FrameRange:
        if not 0 <= latent_index < self.latent_frame_count:
            raise IndexError("latent frame index out of range")
        return self.latent_to_rgb[latent_index]

    def latent_index_for_token(self, token_index: int) -> int:
        if not 0 <= token_index < len(self.token_to_latent):
            raise IndexError("token index out of range")
        return self.token_to_latent[token_index]

    def latent_range_for_chunk(self, chunk_index: int) -> FrameRange:
        if not 0 <= chunk_index < len(self.chunk_to_latent):
            raise IndexError("chunk index out of range")
        return self.chunk_to_latent[chunk_index]

    def rgb_time_seconds(self, rgb_index: int) -> float:
        if not 0 <= rgb_index < self.rgb_frame_count:
            raise IndexError("RGB frame index out of range")
        return rgb_index / self.fps

    def is_padded_rgb(self, rgb_index: int) -> bool:
        if not 0 <= rgb_index < self.rgb_frame_count:
            raise IndexError("RGB frame index out of range")
        return self.rgb_is_padding[rgb_index]
