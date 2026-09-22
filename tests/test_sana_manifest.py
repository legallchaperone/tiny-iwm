import json
from dataclasses import replace

import numpy as np
import pytest

from datasets.sana_wm.manifest import Manifest, ManifestRecord, load_manifest
from datasets.sana_wm.reader import SANAReader, SampleReadError
from datasets.sana_wm.transforms import resize_crop_intrinsics


def _record(**changes):
    record = ManifestRecord(
        sample_id="clip-a",
        source="sana-wm",
        scene_id="scene-a",
        split="train",
        data_version="data-v1",
        video_path="video/a.mp4",
        camera_path="camera/a.json",
        metadata_path="metadata/a.json",
    )
    return replace(record, **changes)


def test_fixed_smoke_manifest_records_identity_source_split_and_version():
    manifest = load_manifest("data/manifests/sana_wm_m1_smoke_v1.json")

    assert manifest.schema_version == 1
    assert manifest.data_version == "sana-wm-m1-smoke-v1"
    assert {record.split for record in manifest.records} == {"train", "validation"}
    assert all(record.source == "sana-wm" for record in manifest.records)
    assert len({record.sample_id for record in manifest.records}) == 2


def test_source_scene_cannot_cross_train_validation_split():
    with pytest.raises(ValueError, match="source scene crosses splits"):
        Manifest(1, "data-v1", (_record(), _record(sample_id="clip-b", split="validation")))


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_manifest_requires_exact_integer_schema_version(schema_version):
    with pytest.raises(ValueError, match="unsupported manifest schema_version"):
        Manifest(schema_version, "data-v1", (_record(),))


def test_resize_crop_updates_focal_length_and_principal_point():
    intrinsics = np.array([[[100.0, 4.0, 50.0], [2.0, 80.0, 40.0], [0.0, 0.0, 1.0]]])

    transformed = resize_crop_intrinsics(
        intrinsics,
        input_size=(100, 200),
        resized_size=(200, 300),
        crop_top_left=(20, 30),
        output_size=(160, 240),
    )

    np.testing.assert_allclose(
        transformed[0],
        [[150.0, 6.0, 45.0], [4.0, 160.0, 60.0], [0.0, 0.0, 1.0]],
    )
    np.testing.assert_array_equal(intrinsics[0], [[100, 4, 50], [2, 80, 40], [0, 0, 1]])


def test_missing_camera_is_an_explicit_sample_failure(tmp_path):
    reader = SANAReader(tmp_path)
    record = _record()
    (tmp_path / "video").mkdir()
    (tmp_path / record.video_path).write_bytes(b"not needed for direct camera check")

    with pytest.raises(SampleReadError, match="camera error: missing file") as raised:
        reader._read_camera(record)
    assert raised.value.sample_id == "clip-a"
    assert raised.value.component == "camera"


def test_malformed_camera_is_an_explicit_sample_failure(tmp_path):
    record = _record()
    camera_path = tmp_path / record.camera_path
    camera_path.parent.mkdir()
    camera_path.write_text(json.dumps({"c2w": []}), encoding="utf-8")

    with pytest.raises(SampleReadError, match="missing fields"):
        SANAReader(tmp_path)._read_camera(record)


def test_non_utf8_camera_is_an_explicit_sample_failure(tmp_path):
    record = _record()
    camera_path = tmp_path / record.camera_path
    camera_path.parent.mkdir()
    camera_path.write_bytes(b"\xff\xfe")

    with pytest.raises(SampleReadError, match="invalid JSON"):
        SANAReader(tmp_path)._read_camera(record)


def test_singular_camera_transform_is_rejected(tmp_path):
    record = _record()
    camera_path = tmp_path / record.camera_path
    camera_path.parent.mkdir()
    camera_path.write_text(
        json.dumps(
            {
                "c2w": [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]],
                "intrinsics": [[[100, 0, 50], [0, 100, 50], [0, 0, 1]]],
                "timestamps_seconds": [0],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SampleReadError, match="rotation must be orthonormal"):
        SANAReader(tmp_path)._read_camera(record)


def test_singular_intrinsics_are_rejected(tmp_path):
    record = _record()
    camera_path = tmp_path / record.camera_path
    camera_path.parent.mkdir()
    camera_path.write_text(
        json.dumps(
            {
                "c2w": [np.eye(4).tolist()],
                "intrinsics": [[[1, 1, 50], [1, 1, 50], [0, 0, 1]]],
                "timestamps_seconds": [0],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SampleReadError, match="intrinsics must be nonsingular"):
        SANAReader(tmp_path)._read_camera(record)


def test_manifest_import_does_not_load_opencv_in_fresh_interpreter():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import datasets.sana_wm.manifest; "
            "assert 'cv2' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_inconsistent_video_frames_become_sample_failure(tmp_path, monkeypatch):
    record = _record()
    video_path = tmp_path / record.video_path
    video_path.parent.mkdir()
    video_path.write_bytes(b"fake")

    class FakeCapture:
        def __init__(self):
            self.frames = [
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.zeros((3, 2, 3), dtype=np.uint8),
            ]

        def isOpened(self):
            return True

        def read(self):
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def release(self):
            pass

    monkeypatch.setattr("datasets.sana_wm.reader.cv2.VideoCapture", lambda _path: FakeCapture())
    monkeypatch.setattr("datasets.sana_wm.reader.cv2.cvtColor", lambda frame, _code: frame)

    with pytest.raises(SampleReadError, match="video error: decoding failed"):
        SANAReader(tmp_path)._read_video(record)


def test_path_resolution_failure_becomes_sample_failure(tmp_path, monkeypatch):
    from pathlib import Path

    reader = SANAReader(tmp_path)
    record = _record()

    def fail_resolve(_path):
        raise RuntimeError("symlink loop")

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    with pytest.raises(SampleReadError, match="camera error: cannot access path"):
        reader._path(record, "camera", record.camera_path)


def test_file_status_failure_becomes_sample_failure(tmp_path, monkeypatch):
    from pathlib import Path

    reader = SANAReader(tmp_path)
    record = _record()

    def fail_is_file(_path):
        raise OSError("network filesystem unavailable")

    monkeypatch.setattr(Path, "is_file", fail_is_file)
    with pytest.raises(SampleReadError, match="camera error: cannot access path"):
        reader._path(record, "camera", record.camera_path)


@pytest.mark.parametrize("output_size", [(0, 8), (8, 0), (-1, 8), (8, -1)])
def test_frame_transform_rejects_nonpositive_output_size(output_size):
    from datasets.sana_wm.transforms import resize_crop_frames

    frames = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="image sizes must be positive"):
        resize_crop_frames(
            frames,
            resized_size=(8, 8),
            crop_top_left=(0, 0),
            output_size=output_size,
        )
