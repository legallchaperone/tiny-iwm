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


def test_resize_crop_updates_focal_length_and_principal_point():
    intrinsics = np.array([[[100.0, 0.0, 50.0], [0.0, 80.0, 40.0], [0.0, 0.0, 1.0]]])

    transformed = resize_crop_intrinsics(
        intrinsics,
        input_size=(100, 200),
        resized_size=(200, 300),
        crop_top_left=(20, 30),
        output_size=(160, 240),
    )

    np.testing.assert_allclose(
        transformed[0],
        [[150.0, 0.0, 45.0], [0.0, 160.0, 60.0], [0.0, 0.0, 1.0]],
    )
    np.testing.assert_array_equal(intrinsics[0], [[100, 0, 50], [0, 80, 40], [0, 0, 1]])


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
