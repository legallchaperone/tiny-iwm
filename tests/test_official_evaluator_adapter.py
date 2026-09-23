import json

import pytest

from core.video_layout import CodecTemporalSpec, VideoLayout
from evaluation.identity import claim_output_directory, generation_identity
from evaluation.official import OFFICIAL_COMMIT, metric_commands, stage_official_inputs
from evaluation.selection import canonical_bytes, sha256


def _fixture(tmp_path):
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    (benchmark / "kept.txt").write_bytes(b"pinned official metadata")
    rows = [
        {
            "scene_id": scene,
            "split": "simple_60s",
            "generation_seed": 42,
            "conditions_sha256": "sha256:" + character * 64,
            "sanawm_frames": 961,
            "fps": 16,
        }
        for scene, character in (("game_style_001", "a"), ("indoor_001", "b"))
    ]
    payload = {
        "dataset_revision": "0" * 40,
        "source_sha256": {"kept.txt": sha256(b"pinned official metadata")},
        "rows": rows,
    }
    selection = {"selection_id": sha256(canonical_bytes(payload)), **payload}
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    outputs = tmp_path / "outputs"
    layout = VideoLayout.from_codec(
        fps=16,
        rgb_frame_count=961,
        codec=CodecTemporalSpec(8),
        latent_chunk_size=4,
        initial_condition_frames=1,
    )
    for row in rows:
        identity = generation_identity(
            selection,
            row,
            checkpoint_id="sha256:" + "c" * 64,
            weight_flavor="ema",
            codec_id="codec",
            spatial_resolution=(704, 1280),
            preprocessing_id="crop-v1",
            rollout_layout=layout,
            sampler={
                "solver": "euler",
                "steps": 25,
                "cfg_scale": 1,
                "history_policy": "clean_cached",
            },
        )
        directory = claim_output_directory(outputs, identity)
        (directory / "video.mp4").write_bytes(b"fake video for directory adapter test")
    return selection_path, benchmark, outputs, tmp_path / "method"


def test_staging_preserves_separate_identity_and_official_output_names(tmp_path):
    paths = _fixture(tmp_path)
    report = stage_official_inputs(*paths, seed=42)
    assert report["official_source"]["commit"] == OFFICIAL_COMMIT
    assert report["official_source"]["local_modifications"] == []
    assert len(report["staged"]) == 2
    for item in report["staged"]:
        link = paths[3] / item["split"] / f"{item['scene_id']}_generated.mp4"
        assert link.is_symlink()
        assert link.read_bytes() == b"fake video for directory adapter test"
        assert item["official_video"] == str(link.absolute())
    assert stage_official_inputs(*paths, seed=42) == report


def test_staging_rejects_changed_metadata_or_missing_identity(tmp_path):
    paths = _fixture(tmp_path)
    (paths[1] / "kept.txt").write_bytes(b"different")
    with pytest.raises(ValueError, match="metadata hash differs"):
        stage_official_inputs(*paths, seed=42)
    (paths[1] / "kept.txt").write_bytes(b"pinned official metadata")
    (next(paths[2].glob("simple_60s/game_style_001/*/identity.json"))).unlink()
    with pytest.raises(ValueError, match="expected one generation identity"):
        stage_official_inputs(*paths, seed=42)


def test_metric_commands_use_official_entry_points_and_pinned_protocol(tmp_path):
    metric, camera = metric_commands(
        tmp_path / "Sana",
        tmp_path / "benchmark",
        tmp_path / "method",
        "simple_60s",
        python="python",
    )
    assert metric[1].endswith("tools/metrics/sana_wm/eval_unified.py")
    assert camera[1].endswith("tools/metrics/sana_wm/eval_benchmark_poses.py")
    assert metric[metric.index("--max_pairs") + 1] == "5"
    assert metric[metric.index("--ref_fps") + 1] == "16"
    assert metric[metric.index("--skip_first_frame") + 1] == "no"
    assert "--revisit_lpips" in metric
    assert "--vbench_dims" in metric
