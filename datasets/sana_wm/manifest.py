"""Versioned sample selection and source-aware split validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Tuple


Split = Literal["train", "validation", "test"]
_VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class ManifestRecord:
    """One immutable sample selection.

    ``source`` names the upstream collection while ``scene_id`` is the leakage
    boundary. Multiple clips from the same source scene must stay in one split.
    Paths are relative to the dataset root so manifests remain portable.
    """

    sample_id: str
    source: str
    scene_id: str
    split: Split
    data_version: str
    video_path: str
    camera_path: str
    metadata_path: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ManifestRecord":
        required = {
            "sample_id",
            "source",
            "scene_id",
            "split",
            "data_version",
            "video_path",
            "camera_path",
            "metadata_path",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"manifest record is missing fields: {', '.join(missing)}")
        extra = sorted(value.keys() - required)
        if extra:
            raise ValueError(f"manifest record has unknown fields: {', '.join(extra)}")
        record = cls(**{key: value[key] for key in required})  # type: ignore[arg-type]
        record.validate()
        return record

    def validate(self) -> None:
        for name in (
            "sample_id",
            "source",
            "scene_id",
            "data_version",
            "video_path",
            "camera_path",
            "metadata_path",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.split not in _VALID_SPLITS:
            raise ValueError(f"unsupported split {self.split!r}")
        for name in ("video_path", "camera_path", "metadata_path"):
            if Path(getattr(self, name)).is_absolute():
                raise ValueError(f"{name} must be relative to the dataset root")


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    data_version: str
    records: Tuple[ManifestRecord, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(f"unsupported manifest schema_version {self.schema_version}")
        if not isinstance(self.data_version, str) or not self.data_version.strip():
            raise ValueError("manifest data_version must be non-empty")
        if not self.records:
            raise ValueError("manifest must contain at least one record")

        sample_ids: set[str] = set()
        scene_splits: dict[tuple[str, str], str] = {}
        for record in self.records:
            record.validate()
            if record.data_version != self.data_version:
                raise ValueError(
                    f"sample {record.sample_id!r} data_version does not match manifest"
                )
            if record.sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id {record.sample_id!r}")
            sample_ids.add(record.sample_id)

            scene_key = (record.source, record.scene_id)
            previous = scene_splits.setdefault(scene_key, record.split)
            if previous != record.split:
                raise ValueError(
                    "source scene crosses splits: "
                    f"source={record.source!r}, scene_id={record.scene_id!r}, "
                    f"splits={previous!r}/{record.split!r}"
                )

    def for_split(self, split: Split) -> Tuple[ManifestRecord, ...]:
        if split not in _VALID_SPLITS:
            raise ValueError(f"unsupported split {split!r}")
        return tuple(record for record in self.records if record.split == split)


def load_manifest(path: str | Path) -> Manifest:
    """Parse and fully validate a versioned JSON manifest."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"manifest does not exist: {manifest_path}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"manifest cannot be read: {manifest_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest is not valid JSON: {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be an object")
    if set(payload) != {"schema_version", "data_version", "records"}:
        raise ValueError("manifest fields must be schema_version, data_version, and records")
    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise ValueError("manifest records must be a list")
    records = tuple(
        ManifestRecord.from_mapping(record)
        if isinstance(record, dict)
        else _raise_record_type(index)
        for index, record in enumerate(raw_records)
    )
    return Manifest(
        schema_version=payload["schema_version"],  # type: ignore[arg-type]
        data_version=payload["data_version"],  # type: ignore[arg-type]
        records=records,
    )


def _raise_record_type(index: int) -> ManifestRecord:
    raise ValueError(f"manifest record {index} must be an object")


def validate_split_isolation(records: Iterable[ManifestRecord]) -> None:
    """Validate a collection when records are assembled programmatically."""

    materialized = tuple(records)
    if not materialized:
        raise ValueError("records must not be empty")
    Manifest(1, materialized[0].data_version, materialized)
