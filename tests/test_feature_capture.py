import json
from dataclasses import replace

import pytest
import torch

from algorithms.world_model.models import JointVideoDiT, JointVideoDiTConfig
from probing.capture import (
    CaptureContext,
    FeatureRecorder,
    TokenSelection,
    capture_forward,
    recomputation_context,
)


def _model():
    torch.manual_seed(31)
    return JointVideoDiT(
        JointVideoDiTConfig(4, 24, 2, 4, (1, 2, 2), mlp_ratio=2.0)
    ).eval()


def _context():
    return CaptureContext(
        sample_id="scene-1",
        training_seed=21,
        generation_seed=42,
        checkpoint_id="sha256:checkpoint",
        config_id="sha256:config",
        rollout_time=8,
        flow_time=0.5,
        branch="conditional",
        forward_purpose="denoise",
    )


def test_observational_capture_preserves_output_rng_and_bounds_storage(tmp_path):
    model = _model()
    latents = torch.randn(1, 4, 2, 4, 4)
    time = torch.tensor([0.5])
    before = torch.random.get_rng_state()
    expected = model(latents, time)
    assert torch.equal(torch.random.get_rng_state(), before)
    recorder = FeatureRecorder(
        tmp_path,
        points=("blocks.0.post_attention",),
        tokens=(
            TokenSelection(0, "history", {"time_seconds": 0.0, "y": 0.0, "x": 0.0}),
        ),
        max_records=1,
        raw_points=frozenset({"blocks.0.post_attention"}),
        max_raw_bytes=4096,
    )
    actual = capture_forward(
        model, latents, time, context=_context(), recorder=recorder
    )
    assert torch.equal(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), before)
    records = list(tmp_path.glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert (
        record["sample_id"],
        record["generation_seed"],
        record["rollout_time"],
        record["flow_time"],
        record["token_type"],
        record["branch"],
        record["forward_purpose"],
    ) == ("scene-1", 42, 8, 0.5, "history", "conditional", "denoise")
    raw = torch.load(tmp_path / record["raw_file"], weights_only=True)
    assert raw.shape == (24,)
    assert raw.untyped_storage().nbytes() == 24 * 4
    assert recorder.raw_bytes == (tmp_path / record["raw_file"]).stat().st_size
    with pytest.raises(ValueError, match="record budget"):
        capture_forward(model, latents, time, context=_context(), recorder=recorder)


def test_recomputation_does_not_double_record_or_change_output(tmp_path):
    model = _model()
    latents = torch.randn(1, 4, 2, 4, 4)
    time = torch.tensor([0.5])
    recorder = FeatureRecorder(
        tmp_path,
        points=("final_norm",),
        tokens=(TokenSelection(0, "target", {"time_seconds": 0.25}),),
        max_records=1,
    )
    first = capture_forward(model, latents, time, context=_context(), recorder=recorder)
    with recomputation_context():
        second = capture_forward(
            model, latents, time, context=_context(), recorder=recorder
        )
    assert torch.equal(first, second)
    assert recorder.record_count == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_raw_capture_is_disabled_by_default(tmp_path):
    model = _model()
    recorder = FeatureRecorder(
        tmp_path,
        points=("patch_tokens",),
        tokens=(TokenSelection(0, "target", {"time_seconds": 0.0}),),
        max_records=1,
    )
    capture_forward(
        model,
        torch.randn(1, 4, 2, 4, 4),
        torch.tensor([0.5]),
        context=_context(),
        recorder=recorder,
    )
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert list(tmp_path.glob("*.pt")) == []


def test_capture_detaches_records_but_keeps_model_gradient(tmp_path):
    model = _model()
    latents = torch.randn(1, 4, 2, 4, 4, requires_grad=True)
    recorder = FeatureRecorder(
        tmp_path,
        points=("blocks.1.block_output",),
        tokens=(TokenSelection(0, "target", {"time_seconds": 0.0}),),
        max_records=1,
        raw_points=frozenset({"blocks.1.block_output"}),
        max_raw_bytes=4096,
    )
    output = capture_forward(
        model, latents, torch.tensor([0.5]), context=_context(), recorder=recorder
    )
    output.square().mean().backward()
    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert not torch.load(
        tmp_path / record["raw_file"], weights_only=True
    ).requires_grad


def test_failed_serialization_cleans_temporary_file(tmp_path, monkeypatch):
    def fail_save(*args, **kwargs):
        raise OSError("volume full")

    monkeypatch.setattr("probing.capture.torch.save", fail_save)
    with pytest.raises(OSError, match="volume full"):
        FeatureRecorder._write_once(tmp_path / "raw.pt", torch.ones(2))
    assert list(tmp_path.iterdir()) == []


def test_existing_files_count_toward_new_recorder_budget(tmp_path):
    (tmp_path / "existing.json").write_text("{}")
    with pytest.raises(ValueError, match="existing feature storage"):
        FeatureRecorder(tmp_path, max_records=0)


def test_separate_recorders_share_one_storage_budget(tmp_path):
    model = _model()
    settings = {
        "points": ("final_norm",),
        "tokens": (TokenSelection(0, "target", {"time_seconds": 0.25}),),
        "max_records": 1,
    }
    first = FeatureRecorder(tmp_path, **settings)
    second = FeatureRecorder(tmp_path, **settings)
    latents = torch.randn(1, 4, 2, 4, 4)
    time = torch.tensor([0.5])
    capture_forward(model, latents, time, context=_context(), recorder=first)
    with pytest.raises(ValueError, match="record budget"):
        capture_forward(
            model,
            latents,
            time,
            context=replace(_context(), sample_id="second-scene"),
            recorder=second,
        )
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_failed_metadata_publication_removes_raw_tensor(tmp_path, monkeypatch):
    original = FeatureRecorder._write_once

    def fail_metadata(path, value, **kwargs):
        if path.suffix == ".json":
            raise OSError("metadata write failed")
        return original(path, value, **kwargs)

    monkeypatch.setattr(FeatureRecorder, "_write_once", staticmethod(fail_metadata))
    recorder = FeatureRecorder(
        tmp_path,
        points=("final_norm",),
        tokens=(TokenSelection(0, "target", {"time_seconds": 0.25}),),
        max_records=1,
        raw_points=frozenset({"final_norm"}),
        max_raw_bytes=4096,
    )
    with pytest.raises(OSError, match="metadata write failed"):
        capture_forward(
            _model(),
            torch.randn(1, 4, 2, 4, 4),
            torch.tensor([0.5]),
            context=_context(),
            recorder=recorder,
        )
    assert recorder.raw_bytes == 0
    assert list(tmp_path.glob("*.pt")) == []
    assert list(tmp_path.glob("*.json")) == []


def test_duplicate_points_are_rejected_and_empty_tokens_skip_capture(tmp_path):
    with pytest.raises(ValueError, match="points must be unique"):
        FeatureRecorder(tmp_path, points=("final_norm", "final_norm"))
    model = _model()
    recorder = FeatureRecorder(tmp_path, points=("final_norm",), tokens=())
    latents = torch.randn(1, 4, 2, 4, 4)
    time = torch.tensor([0.5])
    expected = model(latents, time)
    actual = capture_forward(
        model, latents, time, context=_context(), recorder=recorder
    )
    assert torch.equal(actual, expected)
    assert list(tmp_path.glob("*.json")) == []
