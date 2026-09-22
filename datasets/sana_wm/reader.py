"""Strict SANA-WM video, camera, and metadata reader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from datasets.sana_wm.manifest import ManifestRecord


class SampleReadError(RuntimeError):
    """A sample cannot be used and should be written to the failure ledger."""

    def __init__(self, sample_id: str, component: str, reason: str) -> None:
        self.sample_id = sample_id
        self.component = component
        self.reason = reason
        super().__init__(f"sample {sample_id!r} {component} error: {reason}")


@dataclass(frozen=True)
class CameraData:
    c2w: np.ndarray
    intrinsics: np.ndarray
    timestamps_seconds: np.ndarray


@dataclass(frozen=True)
class SANASample:
    record: ManifestRecord
    frames_rgb: np.ndarray
    camera: CameraData
    metadata: Mapping[str, Any]


class SANAReader:
    """Read only files named by a validated manifest record."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).resolve()

    def read(self, record: ManifestRecord) -> SANASample:
        frames = self._read_video(record)
        camera = self._read_camera(record)
        metadata = self._read_metadata(record)
        frame_count = frames.shape[0]
        if camera.c2w.shape[0] != frame_count:
            raise SampleReadError(
                record.sample_id,
                "camera",
                f"camera length {camera.c2w.shape[0]} != video length {frame_count}",
            )
        return SANASample(record, frames, camera, metadata)

    def _path(self, record: ManifestRecord, component: str, relative: str) -> Path:
        try:
            path = (self.data_root / relative).resolve()
            is_file = path.is_file()
        except (OSError, RuntimeError) as exc:
            raise SampleReadError(
                record.sample_id, component, f"cannot access path {relative!r}: {exc}"
            ) from exc
        if self.data_root not in path.parents:
            raise SampleReadError(record.sample_id, component, "path escapes data_root")
        if not is_file:
            raise SampleReadError(record.sample_id, component, f"missing file: {relative}")
        return path

    def _read_video(self, record: ManifestRecord) -> np.ndarray:
        path = self._path(record, "video", record.video_path)
        capture = None
        frames: list[np.ndarray] = []
        try:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise SampleReadError(record.sample_id, "video", "cannot open video")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not frames:
                raise SampleReadError(record.sample_id, "video", "no decodable frames")
            return np.stack(frames)
        except SampleReadError:
            raise
        except (cv2.error, ValueError, OSError) as exc:
            raise SampleReadError(record.sample_id, "video", f"decoding failed: {exc}") from exc
        finally:
            if capture is not None:
                capture.release()

    def _read_camera(self, record: ManifestRecord) -> CameraData:
        path = self._path(record, "camera", record.camera_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SampleReadError(record.sample_id, "camera", f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SampleReadError(record.sample_id, "camera", "root must be an object")
        missing = {"c2w", "intrinsics", "timestamps_seconds"} - payload.keys()
        if missing:
            raise SampleReadError(
                record.sample_id, "camera", f"missing fields: {', '.join(sorted(missing))}"
            )
        for field in ("c2w", "intrinsics", "timestamps_seconds"):
            if not _is_numeric_json_array(payload[field]):
                raise SampleReadError(
                    record.sample_id,
                    "camera",
                    f"{field} must contain only JSON numbers",
                )
        try:
            c2w = np.asarray(payload["c2w"], dtype=np.float64)
            intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64)
            timestamps = np.asarray(payload["timestamps_seconds"], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SampleReadError(record.sample_id, "camera", f"non-numeric values: {exc}") from exc
        if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
            raise SampleReadError(record.sample_id, "camera", "c2w must have shape [T, 4, 4]")
        if c2w.shape[0] == 0:
            raise SampleReadError(record.sample_id, "camera", "camera trajectory must not be empty")
        if intrinsics.shape != (c2w.shape[0], 3, 3):
            raise SampleReadError(record.sample_id, "camera", "intrinsics must have shape [T, 3, 3]")
        if timestamps.shape != (c2w.shape[0],):
            raise SampleReadError(record.sample_id, "camera", "timestamps_seconds must have shape [T]")
        if not all(np.isfinite(value).all() for value in (c2w, intrinsics, timestamps)):
            raise SampleReadError(record.sample_id, "camera", "values must be finite")
        if not np.allclose(c2w[:, 3], np.array([0.0, 0.0, 0.0, 1.0])):
            raise SampleReadError(record.sample_id, "camera", "c2w has invalid homogeneous row")
        rotations = c2w[:, :3, :3]
        identity = np.broadcast_to(np.eye(3), rotations.shape)
        if not np.allclose(rotations @ np.swapaxes(rotations, 1, 2), identity, atol=1e-5):
            raise SampleReadError(record.sample_id, "camera", "c2w rotation must be orthonormal")
        if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-5):
            raise SampleReadError(record.sample_id, "camera", "c2w rotation must have determinant +1")
        if np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0):
            raise SampleReadError(record.sample_id, "camera", "camera focal lengths must be positive")
        if not np.allclose(intrinsics[:, 2], np.array([0.0, 0.0, 1.0])):
            raise SampleReadError(record.sample_id, "camera", "intrinsics has invalid homogeneous row")
        if np.any(np.abs(np.linalg.det(intrinsics)) <= 1e-12):
            raise SampleReadError(record.sample_id, "camera", "intrinsics must be nonsingular")
        if np.any(np.diff(timestamps) <= 0):
            raise SampleReadError(record.sample_id, "camera", "timestamps must increase strictly")
        return CameraData(c2w, intrinsics, timestamps)

    def _read_metadata(self, record: ManifestRecord) -> Mapping[str, Any]:
        path = self._path(record, "metadata", record.metadata_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SampleReadError(record.sample_id, "metadata", f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SampleReadError(record.sample_id, "metadata", "root must be an object")
        return payload


def _is_numeric_json_array(value: object) -> bool:
    """Accept nested JSON arrays whose leaves are numbers, excluding booleans."""

    if isinstance(value, list):
        return all(_is_numeric_json_array(item) for item in value)
    return isinstance(value, (int, float)) and not isinstance(value, bool)
