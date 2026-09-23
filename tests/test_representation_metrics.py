import json

import pytest
import torch

from probing.metrics import (
    compare_capture_roots,
    compare_features,
    describe_features,
    history_alignment,
    pair_captures,
    summarize_capture,
)


def _write(root, purpose, time, scene, value):
    stem = f"{purpose}-{time}-{scene}"
    torch.save(torch.tensor(value, dtype=torch.float32), root / f"{stem}.pt")
    record = {
        "sample_id": scene,
        "training_seed": 21,
        "generation_seed": 42,
        "checkpoint_id": "checkpoint",
        "config_id": "config",
        "layer": "blocks.0",
        "observation_point": "block_output",
        "rollout_time": time,
        "flow_time": 0.5,
        "token_index": 0,
        "token_type": "history",
        "physical_position": {"time_seconds": time / 16},
        "branch": "conditional",
        "forward_purpose": purpose,
        "channels": len(value),
        "raw_file": f"{stem}.pt",
    }
    (root / f"{stem}.json").write_text(json.dumps(record))


def test_centered_spectrum_norms_and_cka_have_declared_axes():
    matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    stats = describe_features(matrix)
    assert stats["axes"]["centering"] == "subtract per-channel mean across rows"
    assert stats["samples"] == 3 and stats["channels"] == 2
    assert len(stats["singular_values"]) == 2
    assert 1 < stats["effective_rank"] <= 2
    comparison = compare_features(matrix, matrix)
    assert comparison["centered_linear_cka"] == pytest.approx(1)
    assert comparison["mean_row_cosine"] == pytest.approx(1)
    assert comparison["mean_row_l2_drift"] == 0
    wider = torch.cat((matrix, matrix), dim=1)
    assert compare_features(matrix, wider)["centered_linear_cka"] == pytest.approx(1)
    assert compare_features(matrix, wider)["mean_row_cosine"] is None
    assert describe_features(torch.tensor([[1.0], [3.0]]))["norm"]["median"] == 2.0


def test_controlled_history_alignment_keeps_rollout_and_fm_time_separate(tmp_path):
    for time in (8, 16):
        for scene, index in (("scene-a", 0), ("scene-b", 1)):
            _write(
                tmp_path, "controlled_gt_history", time, scene, [1 + index, time / 16]
            )
            _write(
                tmp_path,
                "controlled_generated_history",
                time,
                scene,
                [1 + index, time / 16 + 0.25],
            )
    report = summarize_capture(tmp_path)
    assert len(report["groups"]) == 4
    assert all(group["raw_files"] for group in report["groups"])
    alignment = history_alignment(tmp_path)
    assert [item["rollout_time"] for item in alignment["per_group"]] == [8, 16]
    assert all(
        item["mean_row_l2_drift"] == pytest.approx(0.25)
        for item in alignment["per_group"]
    )
    assert "flow_time" in alignment["fixed_fields"]
    assert "rollout_time" in alignment["fixed_fields"]


def test_history_alignment_keeps_model_provenance_separate(tmp_path):
    for scene in ("a", "b", "c", "d"):
        for purpose in ("controlled_gt_history", "controlled_generated_history"):
            _write(tmp_path, purpose, 8, scene, [1.0, float(ord(scene))])
            if scene in ("c", "d"):
                path = tmp_path / f"{purpose}-8-{scene}.json"
                record = json.loads(path.read_text())
                record.update(
                    checkpoint_id="other", config_id="other", training_seed=22
                )
                path.write_text(json.dumps(record))
    groups = history_alignment(tmp_path)["per_group"]
    assert len(groups) == 2
    assert {item["checkpoint_id"] for item in groups} == {"checkpoint", "other"}
    assert all(item["matched_observations"] == 2 for item in groups)


def test_mixed_feature_widths_are_grouped_before_stacking(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for root in (left, right):
        for scene in ("a", "b", "c", "d"):
            value = [1.0, 2.0] if scene in ("a", "b") else [1.0, 2.0, 3.0]
            for purpose in ("controlled_gt_history", "controlled_generated_history"):
                _write(root, purpose, 8, scene, value)
                if scene in ("c", "d"):
                    path = root / f"{purpose}-8-{scene}.json"
                    record = json.loads(path.read_text())
                    record["observation_point"] = "velocity_tokens"
                    path.write_text(json.dumps(record))
    assert len(history_alignment(left)["per_group"]) == 2
    assert len(compare_capture_roots(left, right)["per_group"]) == 4


def test_nullable_training_seed_groups_sort_deterministically(tmp_path):
    for scene in ("a", "b"):
        _write(tmp_path, "denoise", 8, scene, [1.0, 2.0])
    path = tmp_path / "denoise-8-a.json"
    record = json.loads(path.read_text())
    record["training_seed"] = None
    path.write_text(json.dumps(record))
    assert len(summarize_capture(tmp_path)["groups"]) == 2


def test_mismatched_capture_coordinates_are_rejected():
    record = {
        "sample_id": "a",
        "generation_seed": 42,
        "rollout_time": 8,
        "flow_time": 0.5,
        "token_index": 0,
        "token_type": "history",
        "layer": "blocks.0",
        "observation_point": "block_output",
        "branch": "conditional",
        "physical_position": {"time_seconds": 0.5},
    }
    shifted = dict(record, flow_time=0.75)
    with pytest.raises(ValueError, match="fixed sample/time/token"):
        pair_captures([(record, torch.ones(2))], [(shifted, torch.ones(2))])


def test_variant_comparison_requires_matched_token_rows(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for scene, index in (("scene-a", 0), ("scene-b", 1)):
        _write(left, "denoise", 8, scene, [1 + index, 2 + index])
        _write(right, "denoise", 8, scene, [2 + index, 4 + index])
    result = compare_capture_roots(left, right)
    assert result["per_group"][0]["matched_observations"] == 2
    assert result["left_capture_identity"] != result["right_capture_identity"]
    _write(right, "denoise", 8, "scene-c", [3, 5])
    with pytest.raises(ValueError, match="fixed sample/time/token"):
        compare_capture_roots(left, right)


def test_variant_comparison_rejects_mixed_right_provenance(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for root in (left, right):
        for scene in ("a", "b"):
            _write(root, "denoise", 8, scene, [1.0, 2.0])
    path = right / "denoise-8-b.json"
    record = json.loads(path.read_text())
    record["checkpoint_id"] = "other-checkpoint"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="right capture provenance"):
        compare_capture_roots(left, right)


def test_capture_rejects_symlinked_raw_file(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    inside = tmp_path / "inside"
    inside.mkdir()
    _write(outside, "denoise", 8, "a", [1.0, 2.0])
    record = json.loads((outside / "denoise-8-a.json").read_text())
    (inside / "denoise-8-a.json").write_text(json.dumps(record))
    (inside / record["raw_file"]).symlink_to(outside / record["raw_file"])
    with pytest.raises(ValueError, match="unsafe raw feature"):
        summarize_capture(inside)
