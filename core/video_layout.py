"""The sole contract for RGB, latent, token, chunk, and time mappings.

M0 stores explicit mappings supplied by a codec/data adapter. M1 will add the
canonical derivation from codec properties. Other modules must query this object
instead of reimplementing stride, boundary, or padding arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
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
class CodecTemporalSpec:
    """Codec properties that determine temporal latent geometry.

    Causal video codecs commonly encode the first RGB frame independently and
    then compress fixed-size groups of later frames. Setting
    ``first_frame_is_independent=False`` describes codecs that compress every
    frame in uniform groups.
    """

    temporal_compression: int
    first_frame_is_independent: bool = True

    def __post_init__(self) -> None:
        if type(self.temporal_compression) is not int or self.temporal_compression <= 0:
            raise ValueError("temporal_compression must be a positive integer")


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
    valid_rgb_frame_count: int | None = None
    valid_latent_frame_count: int | None = None
    token_to_latent_ranges: Tuple[FrameRange, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)) or not isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if any(type(value) is not int or value <= 0 for value in (self.rgb_frame_count, self.latent_frame_count)):
            raise ValueError("frame counts must be positive integers")
        if len(self.latent_to_rgb) != self.latent_frame_count:
            raise ValueError("latent_to_rgb must contain one range per latent frame")
        if len(self.rgb_is_padding) != self.rgb_frame_count:
            raise ValueError("rgb_is_padding must contain one value per RGB frame")
        if self.initial_condition_rgb.stop > self.rgb_frame_count:
            raise ValueError("initial condition extends beyond RGB frames")
        if self.valid_rgb_frame_count is None:
            valid_count = next(
                (index for index, padded in enumerate(self.rgb_is_padding) if padded),
                self.rgb_frame_count,
            )
        else:
            valid_count = self.valid_rgb_frame_count
        if not 0 < valid_count <= self.rgb_frame_count:
            raise ValueError("valid_rgb_frame_count must be within the RGB extent")
        object.__setattr__(self, "valid_rgb_frame_count", valid_count)
        expected_valid_latents = sum(
            rgb_range.start < valid_count for rgb_range in self.latent_to_rgb
        )
        latent_valid_count = self.valid_latent_frame_count
        if latent_valid_count is None:
            latent_valid_count = expected_valid_latents
        if latent_valid_count != expected_valid_latents:
            raise ValueError("valid_latent_frame_count must match the requested RGB extent")
        object.__setattr__(self, "valid_latent_frame_count", latent_valid_count)
        expected_padding = tuple(
            index >= valid_count for index in range(self.rgb_frame_count)
        )
        if self.rgb_is_padding != expected_padding:
            raise ValueError("rgb_is_padding must mark only the padded RGB suffix")
        for rgb_range in self.latent_to_rgb:
            if rgb_range.stop > self.rgb_frame_count:
                raise ValueError("latent_to_rgb range extends beyond RGB frames")
        for latent_index in self.token_to_latent:
            if not 0 <= latent_index < self.latent_frame_count:
                raise ValueError("token_to_latent contains an invalid latent index")
        for latent_range in self.chunk_to_latent:
            if latent_range.stop > self.latent_frame_count:
                raise ValueError("chunk_to_latent range extends beyond latent frames")
        token_ranges = self.token_to_latent_ranges
        if not token_ranges:
            token_ranges = tuple(FrameRange(index, index + 1) for index in self.token_to_latent)
            object.__setattr__(self, "token_to_latent_ranges", token_ranges)
        if len(token_ranges) != len(self.token_to_latent):
            raise ValueError("token range and representative mappings must align")
        for representative, latent_range in zip(self.token_to_latent, token_ranges):
            if latent_range.stop > self.latent_frame_count:
                raise ValueError("token latent range extends beyond latent frames")
            if representative != latent_range.start:
                raise ValueError("token representative must be the start of its latent range")

    @classmethod
    def from_codec(
        cls,
        *,
        fps: float,
        rgb_frame_count: int,
        codec: CodecTemporalSpec,
        temporal_patch_size: int = 1,
        latent_chunk_size: int = 1,
        initial_condition_frames: int = 1,
    ) -> "VideoLayout":
        """Derive every temporal mapping from input length and codec properties.

        The returned RGB extent includes codec-required suffix padding.
        ``valid_rgb_frame_count`` preserves the requested output length, which is
        what writers use after decoding.
        """

        integer_values = {
            "rgb_frame_count": rgb_frame_count,
            "temporal_patch_size": temporal_patch_size,
            "latent_chunk_size": latent_chunk_size,
            "initial_condition_frames": initial_condition_frames,
        }
        if any(type(value) is not int for value in integer_values.values()):
            raise ValueError("frame, patch, and chunk counts must be integers")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        if rgb_frame_count <= 0:
            raise ValueError("rgb_frame_count must be positive")
        if temporal_patch_size <= 0 or latent_chunk_size <= 0:
            raise ValueError("patch and chunk sizes must be positive")
        if latent_chunk_size % temporal_patch_size:
            raise ValueError("latent chunk size must be divisible by temporal patch size")
        if not 0 < initial_condition_frames <= rgb_frame_count:
            raise ValueError("initial_condition_frames must be within the input")

        stride = codec.temporal_compression
        if codec.first_frame_is_independent:
            compressed_count = ceil(max(0, rgb_frame_count - 1) / stride)
            requested_latent_count = 1 + compressed_count
            latent_count = ceil(requested_latent_count / temporal_patch_size) * temporal_patch_size
            padded_rgb_count = 1 + (latent_count - 1) * stride
            latent_ranges = [FrameRange(0, 1)]
            latent_ranges.extend(
                FrameRange(1 + index * stride, 1 + (index + 1) * stride)
                for index in range(latent_count - 1)
            )
        else:
            requested_latent_count = ceil(rgb_frame_count / stride)
            latent_count = ceil(requested_latent_count / temporal_patch_size) * temporal_patch_size
            padded_rgb_count = latent_count * stride
            latent_ranges = [
                FrameRange(index * stride, (index + 1) * stride)
                for index in range(latent_count)
            ]

        token_ranges = tuple(_partition(latent_count, temporal_patch_size))
        chunks = tuple(_partition(latent_count, latent_chunk_size))
        return cls(
            fps=fps,
            rgb_frame_count=padded_rgb_count,
            latent_frame_count=latent_count,
            latent_to_rgb=tuple(latent_ranges),
            token_to_latent=tuple(item.start for item in token_ranges),
            chunk_to_latent=chunks,
            rgb_is_padding=tuple(
                index >= rgb_frame_count for index in range(padded_rgb_count)
            ),
            initial_condition_rgb=FrameRange(0, initial_condition_frames),
            valid_rgb_frame_count=rgb_frame_count,
            valid_latent_frame_count=requested_latent_count,
            token_to_latent_ranges=token_ranges,
        )

    def rgb_range_for_latent(self, latent_index: int) -> FrameRange:
        if not 0 <= latent_index < self.latent_frame_count:
            raise IndexError("latent frame index out of range")
        return self.latent_to_rgb[latent_index]

    def latent_index_for_token(self, token_index: int) -> int:
        if not 0 <= token_index < len(self.token_to_latent):
            raise IndexError("token index out of range")
        return self.token_to_latent[token_index]

    def latent_range_for_token(self, token_index: int) -> FrameRange:
        if not 0 <= token_index < len(self.token_to_latent_ranges):
            raise IndexError("token index out of range")
        return self.token_to_latent_ranges[token_index]

    @property
    def initial_condition_latent_frame_count(self) -> int:
        """Number of complete codec latents covered by the initial RGB prefix."""

        boundary = self.initial_condition_rgb.stop
        count = 0
        for rgb_range in self.latent_to_rgb:
            if rgb_range.start < boundary < rgb_range.stop:
                raise ValueError(
                    "initial condition boundary must align with a codec latent boundary"
                )
            if rgb_range.stop <= boundary:
                count += 1
        return count

    @property
    def valid_target_chunk_indices(self) -> Tuple[int, ...]:
        """Chunks containing at least one real, non-condition latent."""

        condition_end = self.initial_condition_latent_frame_count
        valid_end = self.valid_latent_frame_count
        assert valid_end is not None
        return tuple(
            index
            for index, latent_range in enumerate(self.chunk_to_latent)
            if max(latent_range.start, condition_end)
            < min(latent_range.stop, valid_end)
        )

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

    @property
    def output_rgb_frame_count(self) -> int:
        """Number of unpadded frames that a decoder writer must emit."""

        assert self.valid_rgb_frame_count is not None
        return self.valid_rgb_frame_count

    def camera_rgb_index_for_latent(self, latent_index: int) -> int:
        """Choose the last real RGB observation covered by a latent timestep."""

        rgb_range = self.rgb_range_for_latent(latent_index)
        return min(rgb_range.stop, self.output_rgb_frame_count) - 1

    def camera_rgb_indices_for_chunk(self, chunk_index: int) -> Tuple[int, ...]:
        """Return camera samples for a model chunk via the canonical mapping."""

        latent_range = self.latent_range_for_chunk(chunk_index)
        return tuple(
            self.camera_rgb_index_for_latent(index)
            for index in range(latent_range.start, latent_range.stop)
        )

    def writer_rgb_indices_for_chunk(self, chunk_index: int) -> Tuple[int, ...]:
        """Return decoded, unpadded RGB indices a writer should emit for a chunk."""

        latent_range = self.latent_range_for_chunk(chunk_index)
        start = self.latent_to_rgb[latent_range.start].start
        stop = min(
            self.latent_to_rgb[latent_range.stop - 1].stop,
            self.output_rgb_frame_count,
        )
        return tuple(range(start, stop))

    def token_time_range_seconds(self, token_index: int) -> Tuple[float, float]:
        """Return the physical RGB time interval represented by one temporal token."""

        latent_range = self.latent_range_for_token(token_index)
        first = self.latent_to_rgb[latent_range.start]
        last = self.latent_to_rgb[latent_range.stop - 1]
        return (
            first.start / self.fps,
            min(last.stop, self.output_rgb_frame_count) / self.fps,
        )


def _partition(length: int, group_size: int) -> Tuple[FrameRange, ...]:
    return tuple(
        FrameRange(start, min(start + group_size, length))
        for start in range(0, length, group_size)
    )
