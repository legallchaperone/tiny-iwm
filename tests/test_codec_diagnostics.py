import json

import numpy as np
import pytest
import torch

from core.video_layout import CodecTemporalSpec, FrameRange, VideoLayout
from datasets.sana_wm.manifest import ManifestRecord
from datasets.sana_wm.reader import CameraData, SANASample
from representations.base import CodecSpec, Representation
from representations.diagnostics import CodecGateSettings, validate_complete_sample
from representations.normalization import NormalizationStats
from scripts.validate_m1_codec import run_gate


class CausalFixtureCodec(Representation):
    """Small exact-shape codec used to exercise the gate, not a SANA fallback."""

    def __init__(self, *, leak_future: bool = False, bad_shape: str | None = None) -> None:
        self.leak_future = leak_future
        self.bad_shape = bad_shape
        self._spec = CodecSpec(
            codec_name="test-only-causal-fixture",
            weights_sha256="d" * 64,
            latent_channels=3,
            temporal_compression=4,
            spatial_compression=(1, 1),
            causal=True,
            encoding_policy="first_frame_then_nonoverlapping_groups_v1",
            execution_dtype="float32",
            execution_backend="cpu",
            backend_fingerprint="test-only-cpu",
            cuda_math_policy="strict_no_tf32_no_reduced_reduction_v1",
            normalization=NormalizationStats(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                training_data_version="fixture-train-v1",
                sample_count=1,
                value_count_per_channel=1,
            ),
        )

    @property
    def spec(self) -> CodecSpec:
        return self._spec

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        first = video[:, :, :1]
        remainder = video[:, :, 1:]
        batch, channels, _, height, width = remainder.shape
        grouped = remainder.reshape(batch, channels, 240, 4, height, width).mean(dim=3)
        latents = torch.cat((first, grouped), dim=2)
        if self.leak_future:
            latents = latents + video[:, :, -1:].mean(dim=(2, 3, 4), keepdim=True)
        if self.bad_shape == "batch":
            latents = torch.cat((latents, latents), dim=0)
        elif self.bad_shape == "channels":
            latents = latents[:, :2]
        return latents

    def decode(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                normalized_latents[:, :, :1],
                normalized_latents[:, :, 1:].repeat_interleave(4, dim=2),
            ),
            dim=2,
        )


class ExtremeLatentCodec(CausalFixtureCodec):
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        latents = super().encode(video).double()
        latents[:, :, 0, 0, 0] = torch.finfo(torch.float64).max
        return latents

    def decode(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        batch, _, _, height, width = normalized_latents.shape
        return torch.zeros(
            (batch, 3, 961, height, width),
            dtype=torch.float64,
            device=normalized_latents.device,
        )


class UniformPaddingLeakCodec(CausalFixtureCodec):
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        batch, channels, _, height, width = video.shape
        grouped = video.reshape(batch, channels, 241, 4, height, width).mean(dim=3)
        padded_final_frame = video[:, :, -1:].mean(dim=(2, 3, 4), keepdim=True)
        return grouped + padded_final_frame

    def decode(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        return normalized_latents.repeat_interleave(4, dim=2)


def _record(**changes) -> ManifestRecord:
    values = {
        "sample_id": "held-out-961",
        "source": "fixture",
        "scene_id": "fixture-scene",
        "split": "validation",
        "data_version": "fixture-v1",
        "video_path": "videos/held-out.mp4",
        "camera_path": "camera/held-out.json",
        "metadata_path": "metadata/held-out.json",
    }
    values.update(changes)
    return ManifestRecord(**values)


def _sample() -> SANASample:
    frames = np.arange(961 * 4 * 4 * 3, dtype=np.uint32)
    frames = (frames % 256).astype(np.uint8).reshape(961, 4, 4, 3)
    c2w = np.broadcast_to(np.eye(4), (961, 4, 4)).copy()
    intrinsics = np.broadcast_to(np.eye(3), (961, 3, 3)).copy()
    timestamps = np.arange(961, dtype=np.float64) / 16.0
    return SANASample(
        record=_record(),
        frames_rgb=frames,
        camera=CameraData(c2w, intrinsics, timestamps),
        metadata={"fixture": True},
    )


def _layout() -> VideoLayout:
    return VideoLayout.from_codec(
        fps=16.0,
        rgb_frame_count=961,
        codec=CodecTemporalSpec(temporal_compression=4),
        latent_chunk_size=64,
    )


def test_complete_961_frame_gate_records_layout_camera_metrics_and_cache():
    report = validate_complete_sample(
        _sample(),
        codec=CausalFixtureCodec(),
        layout=_layout(),
        preprocessing={"color_space": "RGB", "range": "0_to_1"},
    )

    assert report["status"] == "passed"
    assert report["sample"]["rgb_shape_thwc"] == [961, 4, 4, 3]
    assert report["sample"]["duration_seconds"] == 60.0
    assert report["layout"]["latent_frame_count"] == 241
    assert report["layout"]["first_latent_rgb_range"] == {"start": 0, "stop": 1}
    assert report["layout"]["last_latent_rgb_range"] == {"start": 957, "stop": 961}
    assert report["layout"]["last_writer_rgb_index"] == 960
    assert report["camera_alignment"]["last_latent_camera_rgb_index"] == 960
    assert len(report["camera_alignment"]["latent_camera_rgb_indices"]) == 241
    assert report["camera_alignment"]["latent_camera_rgb_indices"][:2] == [0, 4]
    assert report["camera_alignment"]["latent_camera_timestamps_seconds"][-1] == 60.0
    assert report["camera_alignment"]["last_timestamp_seconds"] == 60.0
    assert report["causality"]["history_latent_count"] == 121
    assert report["causality"]["max_abs_history_latent_delta"] == 0.0
    assert report["latents"]["shape_bcthw"] == [1, 3, 241, 4, 4]
    assert report["cache"]["frame_count"] == 961
    assert len(report["cache"]["key"]) == 64
    json.dumps(report, allow_nan=False)


def test_gate_rejects_a_codec_that_leaks_future_rgb_into_history_latents():
    with pytest.raises(ValueError, match="future RGB changed history latents"):
        validate_complete_sample(
            _sample(),
            codec=CausalFixtureCodec(leak_future=True),
            layout=_layout(),
            preprocessing={"color_space": "RGB"},
        )


@pytest.mark.parametrize(
    ("bad_shape", "message"),
    [("batch", "batch size"), ("channels", "latent channels")],
)
def test_gate_rejects_latents_incompatible_with_codec_identity(bad_shape, message):
    with pytest.raises(ValueError, match=message):
        validate_complete_sample(
            _sample(),
            codec=CausalFixtureCodec(bad_shape=bad_shape),
            layout=_layout(),
            preprocessing={"color_space": "RGB"},
        )


def test_gate_rejects_nonfinite_statistics_derived_from_finite_extreme_latents():
    with pytest.raises(ValueError, match="derived latent statistics must be finite"):
        validate_complete_sample(
            _sample(),
            codec=ExtremeLatentCodec(),
            layout=_layout(),
            preprocessing={"color_space": "RGB"},
        )


def test_future_perturbation_regenerates_layout_padding_before_causality_check():
    uniform_layout = VideoLayout.from_codec(
        fps=16.0,
        rgb_frame_count=961,
        codec=CodecTemporalSpec(
            temporal_compression=4,
            first_frame_is_independent=False,
        ),
        latent_chunk_size=64,
    )

    with pytest.raises(ValueError, match="future RGB changed history latents"):
        validate_complete_sample(
            _sample(),
            codec=UniformPaddingLeakCodec(),
            layout=uniform_layout,
            preprocessing={"color_space": "RGB"},
        )


def test_gate_rejects_camera_timestamps_not_aligned_to_16_fps():
    sample = _sample()
    shifted = CameraData(
        sample.camera.c2w,
        sample.camera.intrinsics,
        sample.camera.timestamps_seconds + 0.01,
    )
    misaligned = SANASample(sample.record, sample.frames_rgb, shifted, sample.metadata)

    with pytest.raises(ValueError, match="timestamps are not aligned"):
        validate_complete_sample(
            misaligned,
            codec=CausalFixtureCodec(),
            layout=_layout(),
            preprocessing={"color_space": "RGB"},
        )


@pytest.mark.parametrize("atol", [float("nan"), float("inf"), -1.0])
def test_gate_settings_reject_nonfinite_or_negative_causality_tolerance(atol):
    with pytest.raises(ValueError, match="finite and non-negative"):
        CodecGateSettings(causality_atol=atol)


def test_manifest_runner_writes_structured_reader_failures(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_version": "fixture-v1",
                "records": [
                    {
                        "sample_id": "held-out-961",
                        "source": "fixture",
                        "scene_id": "fixture-scene",
                        "split": "validation",
                        "data_version": "fixture-v1",
                        "video_path": "videos/missing.mp4",
                        "camera_path": "camera/missing.json",
                        "metadata_path": "metadata/missing.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_gate(
        manifest_path=manifest_path,
        data_root=tmp_path,
        codec_factory=CausalFixtureCodec,
        split="validation",
        preprocessing={"color_space": "RGB"},
        settings=CodecGateSettings(),
        first_frame_is_independent=True,
    )

    assert report["status"] == "failed"
    assert report["passed_samples"] == []
    assert report["failures"] == [
        {
            "sample_id": "held-out-961",
            "component": "video",
            "reason": "missing file: videos/missing.mp4",
        }
    ]
