"""Representation contract and complete frozen codec specification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from string import hexdigits
from typing import Tuple

import torch

from core.types import LatentSpec
from representations.normalization import NormalizationStats


@dataclass(frozen=True)
class CodecSpec:
    """Immutable identity, geometry, policy, and statistics for one codec."""

    codec_name: str
    weights_sha256: str
    latent_channels: int
    temporal_compression: int
    spatial_compression: Tuple[int, int]
    causal: bool
    encoding_policy: str
    normalization: NormalizationStats

    def __post_init__(self) -> None:
        digest = self.weights_sha256
        if len(digest) != 64 or any(character not in hexdigits for character in digest):
            raise ValueError("weights_sha256 must be a 64-character SHA-256 digest")
        if not self.codec_name or not self.encoding_policy:
            raise ValueError("codec name and encoding policy must be explicit")
        if type(self.causal) is not bool:
            raise ValueError("causal mode must be a boolean")
        if type(self.latent_channels) is not int or self.latent_channels <= 0:
            raise ValueError("latent channels must be a positive integer")
        if type(self.temporal_compression) is not int or self.temporal_compression <= 0:
            raise ValueError("temporal compression must be a positive integer")
        if len(self.spatial_compression) != 2 or any(
            type(value) is not int or value <= 0 for value in self.spatial_compression
        ):
            raise ValueError("spatial compression must contain two positive integers")

    @property
    def codec_id(self) -> str:
        return f"{self.codec_name}@sha256:{self.weights_sha256.lower()}"

    def as_latent_spec(self) -> LatentSpec:
        return LatentSpec(
            codec_id=self.codec_id,
            channels=self.latent_channels,
            temporal_compression=self.temporal_compression,
            spatial_compression=self.spatial_compression,
            normalization={
                "kind": "frozen_channel_stats",
                "identity": self.normalization.identity,
                "training_data_version": self.normalization.training_data_version,
            },
            causal=self.causal,
            encoding_policy=self.encoding_policy,
        )


class Representation(ABC):
    """Boundary for video tensors and normalized latent tensors."""

    @property
    @abstractmethod
    def spec(self) -> CodecSpec:
        raise NotImplementedError

    @abstractmethod
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
