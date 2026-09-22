"""Canonical latent-cache identity; no cache lookup may omit these inputs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Mapping, Tuple

from representations.base import CodecSpec


@dataclass(frozen=True)
class CacheIdentity:
    data_version: str
    sample_id: str
    frame_indices: Tuple[int, ...]
    preprocessing: Mapping[str, object]
    codec: CodecSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_indices", tuple(self.frame_indices))
        if not self.data_version or not self.sample_id:
            raise ValueError("data version and sample ID must be explicit")
        if not self.frame_indices or any(
            type(index) is not int or index < 0 for index in self.frame_indices
        ):
            raise ValueError("frame_indices must be non-empty non-negative integers")
        if tuple(sorted(set(self.frame_indices))) != self.frame_indices:
            raise ValueError("frame_indices must be unique and strictly increasing")
        if not self.preprocessing:
            raise ValueError("preprocessing must be explicit")
        canonical_preprocessing = _canonical_json_value(self.preprocessing)
        object.__setattr__(
            self, "preprocessing", _freeze_json_value(canonical_preprocessing)
        )

    @property
    def payload(self) -> dict[str, object]:
        stats = self.codec.normalization
        return {
            "schema_version": 1,
            "data_version": self.data_version,
            "sample_id": self.sample_id,
            "frame_indices": self.frame_indices,
            "preprocessing": self.preprocessing,
            "codec": {
                "name": self.codec.codec_name,
                "weights_sha256": self.codec.weights_sha256.lower(),
                "latent_channels": self.codec.latent_channels,
                "temporal_compression": self.codec.temporal_compression,
                "spatial_compression": self.codec.spatial_compression,
                "causal": self.codec.causal,
                "encoding_policy": self.codec.encoding_policy,
            },
            "normalization": {
                "identity": stats.identity,
                "mean": stats.mean,
                "std": stats.std,
                "training_data_version": stats.training_data_version,
                "training_split": stats.training_split,
                "sample_count": stats.sample_count,
                "value_count_per_channel": stats.value_count_per_channel,
            },
        }

    @property
    def key(self) -> str:
        canonical = _canonical_json_value(self.payload)
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()

    @property
    def relative_path(self) -> PurePosixPath:
        return PurePosixPath(self.key[:2], f"{self.key}.pt")


def _canonical_json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError("cache identity cannot contain non-finite floats")
        return value
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("cache identity mapping keys must be strings")
        return {
            key: _canonical_json_value(child)
            for key, child in sorted(value.items())
        }
    raise ValueError(f"unsupported cache identity value: {type(value).__name__}")


def _freeze_json_value(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json_value(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(child) for child in value)
    return value
