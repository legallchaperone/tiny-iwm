import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.video_layout import CodecTemporalSpec, VideoLayout
from evaluation.identity import (
    claim_output_directory,
    finalize_output_video,
    generation_identity,
)
from evaluation.official import (
    OFFICIAL_COMMIT,
    RUN_FIELDS,
    _validate_video,
    _verify_scored_split,
    metric_commands,
    score_official,
    stage_official_inputs,
)
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
            "sanawm_frames": 961,
            "fps": 16,
        }
        for scene in ("game_style_001", "indoor_001")
    ]
    source_hashes = {"kept.txt": sha256(b"pinned official metadata")}
    for row in rows:
        scene = row["scene_id"]
        conditions = {}
        for kind, relative in (
            ("image", f"images/{scene}.png"),
            ("camera", f"benchmark_v2_smooth_60s/sanawm_export_v2/{scene}.npz"),
        ):
            asset = benchmark / relative
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_bytes(relative.encode())
            conditions[f"{kind}_path"] = relative
            conditions[f"{kind}_sha256"] = sha256(asset.read_bytes())
            source_hashes[relative] = conditions[f"{kind}_sha256"]
        row["conditions"] = conditions
        row["conditions_sha256"] = sha256(canonical_bytes(conditions))
    payload = {
        "dataset_revision": "0" * 40,
        "source_sha256": source_hashes,
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
    backend = {
        "device_name": "NVIDIA A10G",
        "compute_capability": "8.6",
        "cuda_runtime": "12.8",
        "cudnn_version": 91002,
    }
    for row in rows:
        identity = generation_identity(
            selection,
            row,
            source_root=benchmark,
            checkpoint_id="sha256:" + "c" * 64,
            weight_flavor="ema",
            model_config={
                "latent_channels": 128,
                "hidden_size": 832,
                "depth": 24,
                "num_heads": 16,
                "patch_size": [1, 2, 2],
                "mlp_ratio": 4.0,
                "qkv_bias": True,
                "prope_camera_dims": 48,
            },
            codec_id="codec",
            codec_fingerprint={
                "weights_sha256": "sha256:" + "d" * 64,
                "normalization_sha256": "sha256:" + "e" * 64,
                "encoding_policy": "bidirectional_mode_v1",
                "implementation_revision": "diffusers:0.37.0",
                "execution_dtype": "bfloat16",
                "execution_backend": "cuda",
                "backend_fingerprint": backend,
                "cuda_math_policy": "strict_no_tf32",
            },
            spatial_resolution=(704, 1280),
            preprocessing_id="crop-v1",
            rollout_layout=layout,
            conditioning={"camera": {"enabled": False}, "text": {"enabled": False}},
            implementation_id="test-implementation",
            numeric_execution={
                "parameter_dtype": "float32",
                "autocast_dtype": "bfloat16",
                "latent_dtype": "bfloat16",
                "attention_backend": "math",
                "torch_version": "2.8.0",
                "tf32_enabled": False,
                "backend_fingerprint": backend,
            },
            sampler={
                "solver": "euler",
                "steps": 25,
                "cfg_scale": 1,
                "history_policy": "clean_cached",
                "initial_history_policy": "source_image_only",
                "batch_size": 1,
            },
        )
        directory = claim_output_directory(outputs, identity)
        video = directory / "video.mp4"
        video.write_bytes(b"fake video for directory adapter test")
        finalize_output_video(directory, identity, {})
    return selection_path, benchmark, outputs, tmp_path / "method"


def test_staging_preserves_separate_identity_and_official_output_names(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("evaluation.official._validate_video", lambda *_: None)
    paths = _fixture(tmp_path)
    report = stage_official_inputs(*paths, seed=42)
    assert report["official_source"]["commit"] == OFFICIAL_COMMIT
    assert report["official_source"]["local_modifications"] == []
    assert len(report["staged"]) == 2
    assert all(
        item["video_sha256"] == sha256(b"fake video for directory adapter test")
        for item in report["staged"]
    )
    for item in report["staged"]:
        link = paths[3] / item["split"] / f"{item['scene_id']}_generated.mp4"
        assert link.is_symlink()
        assert link.read_bytes() == b"fake video for directory adapter test"
        assert item["official_video"] == str(link.absolute())
    assert stage_official_inputs(*paths, seed=42) == report


def test_staging_rejects_changed_metadata_or_missing_identity(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.official._validate_video", lambda *_: None)
    paths = _fixture(tmp_path)
    (paths[1] / "kept.txt").write_bytes(b"different")
    with pytest.raises(ValueError, match="metadata hash differs"):
        stage_official_inputs(*paths, seed=42)
    (paths[1] / "kept.txt").write_bytes(b"pinned official metadata")
    (next(paths[2].glob("simple_60s/game_style_001/*/identity.json"))).unlink()
    with pytest.raises(ValueError, match="expected one generation identity"):
        stage_official_inputs(*paths, seed=42)


def test_staging_failure_leaves_no_partial_links(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.official._validate_video", lambda *_: None)
    paths = _fixture(tmp_path)
    next(paths[2].glob("simple_60s/indoor_001/*/video.mp4")).unlink()
    with pytest.raises(FileNotFoundError):
        stage_official_inputs(*paths, seed=42)
    assert not list(paths[3].rglob("*_generated.mp4"))


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
    assert metric[metric.index("--metrics") + 1 : metric.index("--vbench_dims")] == [
        "vbench",
        "revisit",
        "camera",
        "temporal",
    ]


def test_scoring_consumes_raw_pose_results_in_official_summary(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    (tmp_path / "Sana").mkdir()
    monkeypatch.setattr("evaluation.official._validate_video", lambda *_: None)
    monkeypatch.setattr("evaluation.official._verify_official_checkout", lambda _: None)
    monkeypatch.setattr("evaluation.official._verify_scored_split", lambda *_: None)
    commands = []
    monkeypatch.setattr(
        "evaluation.official.subprocess.run",
        lambda command, **_: commands.append(command),
    )
    score_official(*paths, tmp_path / "Sana", seed=42)
    assert commands[0][1].endswith("eval_benchmark_poses.py")
    assert commands[1][1].endswith("eval_unified.py")


def test_scored_split_requires_all_scenes_and_revisit_pairs(tmp_path):
    root = tmp_path / "eval/simple_60s"
    root.mkdir(parents=True)
    split = tmp_path / "simple_60s"
    split.mkdir()
    scenes = ("game_style_001", "indoor_001")
    rows = [{"scene_id": scene, "evaluation_pair_count": 1} for scene in scenes]
    (split / "eval_poses.json").write_text(json.dumps({scene: {} for scene in scenes}))
    (root / "camera_accuracy.json").write_text(
        json.dumps({scene: {} for scene in scenes})
    )
    revisit = {
        "per_scene": {scene: {"n_pairs": 1} for scene in scenes},
        "summary": {"n_total_pairs": 2},
    }
    (root / "revisit_consistency.json").write_text(json.dumps(revisit))
    (root / "vbench_scores.json").write_text(
        json.dumps({"raw_scores": {str(index): 0.5 for index in range(9)}})
    )
    (root / "temporal_degradation.json").write_text(
        json.dumps({"windows": {"w0": {"score": 0.5}}})
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "n_videos": 2,
                "split": "simple_60s",
                "camera": {"n_scenes": 2},
                "vbench": {"n_dimensions": 9},
                "temporal_degradation": {"score": 0.5},
            }
        )
    )
    _verify_scored_split(tmp_path, "simple_60s", rows)
    revisit["per_scene"][scenes[1]]["n_pairs"] = 0
    (root / "revisit_consistency.json").write_text(json.dumps(revisit))
    with pytest.raises(ValueError, match="incomplete coverage"):
        _verify_scored_split(tmp_path, "simple_60s", rows)


def test_staging_rejects_unselected_video(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.official._validate_video", lambda *_: None)
    paths = _fixture(tmp_path)
    extra = paths[3] / "simple_60s" / "other_generated.mp4"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"stale video")
    with pytest.raises(ValueError, match="video set differs"):
        stage_official_inputs(*paths, seed=42)


def test_staging_selects_one_run_from_multiple_generation_identities(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("evaluation.official._validate_video", lambda *_: None)
    paths = _fixture(tmp_path)
    original = json.loads(
        next(paths[2].glob("simple_60s/game_style_001/*/identity.json")).read_text()
    )
    alternate = dict(original, sampler={**original["sampler"], "steps": 26})
    alternate["generation_id"] = sha256(
        canonical_bytes(
            {key: value for key, value in alternate.items() if key != "generation_id"}
        )
    )
    alternate_dir = claim_output_directory(paths[2], alternate)
    (alternate_dir / "video.mp4").write_bytes(b"alternate run")
    foreign = dict(original, selection_id="sha256:" + "f" * 64)
    foreign["generation_id"] = sha256(
        canonical_bytes(
            {key: value for key, value in foreign.items() if key != "generation_id"}
        )
    )
    claim_output_directory(paths[2], foreign)
    with pytest.raises(ValueError, match="available run IDs"):
        stage_official_inputs(*paths, seed=42)
    run_id = sha256(canonical_bytes({field: original[field] for field in RUN_FIELDS}))
    report = stage_official_inputs(*paths, seed=42, run_id=run_id)
    assert report["run_id"] == run_id
    assert len(report["staged"]) == 2


def test_staging_rejects_video_changed_after_generation(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.official._validate_video", lambda *_: None)
    paths = _fixture(tmp_path)
    video = next(paths[2].glob("simple_60s/game_style_001/*/video.mp4"))
    video.write_bytes(b"different complete video")
    with pytest.raises(ValueError, match="differs from its identity metadata"):
        stage_official_inputs(*paths, seed=42)


def test_scoring_rechecks_video_between_official_metrics(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    (tmp_path / "Sana").mkdir()
    monkeypatch.setattr("evaluation.official._validate_video", lambda *_: None)
    monkeypatch.setattr("evaluation.official._verify_official_checkout", lambda _: None)
    video = next(paths[2].glob("simple_60s/game_style_001/*/video.mp4"))

    def run(command, **kwargs):
        video.write_bytes(b"changed during scoring")

    monkeypatch.setattr("evaluation.official.subprocess.run", run)
    with pytest.raises(ValueError, match="changed before official scoring"):
        score_official(*paths, tmp_path / "Sana", seed=42)


def test_video_probe_rejects_wrong_length_before_scoring(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command[0])
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "nb_read_frames": "960",
                            "avg_frame_rate": "16/1",
                            "width": 128,
                            "height": 128,
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr("evaluation.official.subprocess.run", run)
    with pytest.raises(ValueError, match="wrong frames, FPS, or resolution"):
        _validate_video(tmp_path / "video.mp4", {"sanawm_frames": 961, "fps": 16})
    assert calls == ["ffprobe"]

    calls.clear()

    def valid_run(command, **kwargs):
        calls.append(command[0])
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "nb_read_frames": "961",
                            "avg_frame_rate": "16/1",
                            "width": 128,
                            "height": 128,
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr("evaluation.official.subprocess.run", valid_run)
    _validate_video(tmp_path / "video.mp4", {"sanawm_frames": 961, "fps": 16})
    assert calls == ["ffprobe", "ffmpeg"]


def test_video_probe_rejects_wrong_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "evaluation.official.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "nb_read_frames": "961",
                            "avg_frame_rate": "16/1",
                            "width": 256,
                            "height": 256,
                        }
                    ]
                }
            )
        ),
    )
    with pytest.raises(ValueError, match="resolution"):
        _validate_video(tmp_path / "video.mp4", {"sanawm_frames": 961, "fps": 16})


def test_metric_commands_resolve_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metric, camera = metric_commands(
        Path("Sana"), Path("benchmark"), Path("method"), "simple_60s"
    )
    assert metric[1] == str(tmp_path / "Sana/tools/metrics/sana_wm/eval_unified.py")
    assert camera[camera.index("--result_folder") + 1] == str(
        tmp_path / "method/simple_60s"
    )
