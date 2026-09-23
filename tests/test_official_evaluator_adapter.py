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
    TEMPORAL_DIMS,
    VBenCH_DIMS,
    _validate_video,
    _verify_official_checkout,
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
            spatial_resolution=(128, 128),
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
    assert report["official_source"]["checkout_verified"] is False
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


def test_stage_normalizes_method_path_before_recording_links(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.official._validate_video", lambda *_: None)
    paths = _fixture(tmp_path)
    paths[3].mkdir()
    alias = tmp_path / "method-alias"
    alias.symlink_to(paths[3], target_is_directory=True)
    report = stage_official_inputs(*paths[:3], alias, seed=42)
    assert all(str(paths[3]) in row["official_video"] for row in report["staged"])
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


def test_official_checkout_rejects_changes_outside_metric_directory(
    tmp_path, monkeypatch
):
    (tmp_path / "LICENSE").write_text("Apache-2.0")

    def run(command, **kwargs):
        if "rev-parse" in command:
            return SimpleNamespace(stdout=OFFICIAL_COMMIT + "\n")
        if "status" in command:
            assert "--ignore-submodules=none" in command
            return SimpleNamespace(stdout=" M local_libs/VBench/vbench/__init__.py\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr("evaluation.official.subprocess.run", run)
    with pytest.raises(ValueError, match="unrecorded local modifications"):
        _verify_official_checkout(tmp_path)


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
    scoring = json.loads((paths[3] / "scoring.json").read_text())
    assert scoring["official_source"]["checkout_verified"] is True
    assert scoring["official_source"]["local_modifications"] == []
    assert commands[0][1].endswith("eval_benchmark_poses.py")
    assert commands[1][1].endswith("eval_unified.py")


def test_scored_split_requires_all_scenes_and_revisit_pairs(tmp_path):
    root = tmp_path / "eval/simple_60s"
    root.mkdir(parents=True)
    split = tmp_path / "simple_60s"
    split.mkdir()
    scenes = ("game_style_001", "indoor_001")
    benchmark = tmp_path / "benchmark/benchmark_v2_smooth_60s"
    benchmark.mkdir(parents=True)
    selected_pairs = [{"frame_a": 1, "frame_b": 2}]
    (benchmark / "scene_trajectories_v2.json").write_text(
        json.dumps(
            {
                "scenes": [
                    {"scene_id": scene, "evaluation_pairs": selected_pairs}
                    for scene in scenes
                ]
            }
        )
    )
    rows = [
        {
            "scene_id": scene,
            "evaluation_pair_count": 1,
            "evaluation_pairs_sha256": sha256(canonical_bytes(selected_pairs)),
            "sanawm_frames": 961,
            "official_scoring_frames": 960,
            "fps": 16,
        }
        for scene in scenes
    ]
    pose = {
        "RotErr": 1.0,
        "RotErr_unit": "deg",
        "TransErr_rel": 0.2,
        "CamMC_rel": 0.3,
        "n_frames": 241,
        "skip_first_frame": False,
    }
    (split / "eval_poses.json").write_text(
        json.dumps({scene: pose for scene in scenes})
    )
    (root / "camera_accuracy.json").write_text(
        json.dumps({scene: pose for scene in scenes})
    )
    revisit = {
        "per_scene": {
            scene: {
                "n_pairs": 1,
                "mean_psnr": 20.0,
                "mean_ssim": 0.8,
                "mean_lpips": 0.1,
                "pairs": [
                    {
                        "frame_a": 1,
                        "frame_b": 2,
                        "psnr": 20.0,
                        "ssim": 0.8,
                        "lpips": 0.1,
                    }
                ],
            }
            for scene in scenes
        },
        "summary": {
            "n_total_pairs": 2,
            "n_lpips_pairs": 2,
            "overall_mean_psnr": 20.0,
            "overall_mean_ssim": 0.8,
            "overall_mean_lpips": 0.1,
        },
    }
    (root / "revisit_consistency.json").write_text(json.dumps(revisit))
    (root / "vbench_scores.json").write_text(
        json.dumps({"raw_scores": {dimension: 0.5 for dimension in VBenCH_DIMS}})
    )
    windows = {f"w{index}_{index * 10}s-{(index + 1) * 10}s" for index in range(6)}
    trend = {
        dimension: {
            "first_window": 0.5,
            "last_window": 0.5,
            "degradation": 0.0,
            "per_window": [0.5] * 6,
        }
        for dimension in TEMPORAL_DIMS
    }
    (root / "temporal_degradation.json").write_text(
        json.dumps(
            {
                "windows": {
                    window: {dimension: 0.5 for dimension in TEMPORAL_DIMS}
                    for window in windows
                },
                "trend": trend,
            }
        )
    )
    for result_root, dimensions in [
        (root, VBenCH_DIMS),
        *(
            (root / "temporal/temporal_results" / window, TEMPORAL_DIMS)
            for window in windows
        ),
    ]:
        result_root.mkdir(parents=True, exist_ok=True)
        for dimension in dimensions:
            (result_root / f"eval_{dimension}_eval_results.json").write_text(
                json.dumps(
                    {
                        dimension: [
                            0.5,
                            [
                                {
                                    "video_path": f"{scene}_generated.mp4",
                                    "video_results": 0.5,
                                }
                                for scene in scenes
                            ],
                        ]
                    }
                )
            )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "n_videos": 2,
                "split": "simple_60s",
                "camera": {
                    "n_scenes": 2,
                    "mean_rot_err_deg": 1.0,
                    "mean_trans_err_rel": 0.2,
                },
                "vbench": {
                    "n_dimensions": 9,
                    "quality_score": 0.5,
                    "semantic_score": 0.5,
                    "total_score": 0.5,
                },
                "temporal_degradation": trend,
            }
        )
    )
    _verify_scored_split(tmp_path, tmp_path / "benchmark", "simple_60s", rows)
    revisit["per_scene"][scenes[1]]["pairs"][0]["frame_b"] = 3
    (root / "revisit_consistency.json").write_text(json.dumps(revisit))
    with pytest.raises(ValueError, match="incomplete coverage"):
        _verify_scored_split(tmp_path, tmp_path / "benchmark", "simple_60s", rows)
    revisit["per_scene"][scenes[1]]["pairs"][0]["frame_b"] = 2
    (root / "revisit_consistency.json").write_text(json.dumps(revisit))
    vbench = {"raw_scores": {dimension: 0.5 for dimension in VBenCH_DIMS}}
    vbench["raw_scores"][VBenCH_DIMS[0]] = None
    (root / "vbench_scores.json").write_text(json.dumps(vbench))
    with pytest.raises(ValueError, match="incomplete coverage"):
        _verify_scored_split(tmp_path, tmp_path / "benchmark", "simple_60s", rows)
    vbench["raw_scores"][VBenCH_DIMS[0]] = 0.5
    (root / "vbench_scores.json").write_text(json.dumps(vbench))
    pose["n_frames"] = 1
    (split / "eval_poses.json").write_text(
        json.dumps({scene: pose for scene in scenes})
    )
    with pytest.raises(ValueError, match="incomplete coverage"):
        _verify_scored_split(tmp_path, tmp_path / "benchmark", "simple_60s", rows)
    pose["n_frames"] = 241
    (split / "eval_poses.json").write_text(
        json.dumps({scene: pose for scene in scenes})
    )
    revisit["per_scene"][scenes[1]]["n_pairs"] = 0
    (root / "revisit_consistency.json").write_text(json.dumps(revisit))
    with pytest.raises(ValueError, match="incomplete coverage"):
        _verify_scored_split(tmp_path, tmp_path / "benchmark", "simple_60s", rows)
    revisit["per_scene"][scenes[1]]["n_pairs"] = 1
    revisit["per_scene"][scenes[1]]["mean_psnr"] = float("nan")
    (root / "revisit_consistency.json").write_text(json.dumps(revisit))
    with pytest.raises(ValueError, match="incomplete coverage"):
        _verify_scored_split(tmp_path, tmp_path / "benchmark", "simple_60s", rows)
    revisit["per_scene"][scenes[1]]["mean_psnr"] = 20.0
    (root / "revisit_consistency.json").write_text(json.dumps(revisit))
    path = root / f"eval_{VBenCH_DIMS[0]}_eval_results.json"
    path.write_text(
        json.dumps(
            {
                VBenCH_DIMS[0]: [
                    0.5,
                    [
                        {
                            "video_path": f"{scenes[0]}_generated.mp4",
                            "video_results": 0.5,
                        }
                    ],
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="incomplete VBench per-video"):
        _verify_scored_split(tmp_path, tmp_path / "benchmark", "simple_60s", rows)


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


def test_staging_ignores_other_generation_seed(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.official._validate_video", lambda *_: None)
    paths = _fixture(tmp_path)
    original = json.loads(
        next(paths[2].glob("simple_60s/game_style_001/*/identity.json")).read_text()
    )
    other_seed = dict(original, seed=43)
    other_seed["generation_id"] = sha256(
        canonical_bytes(
            {key: value for key, value in other_seed.items() if key != "generation_id"}
        )
    )
    claim_output_directory(paths[2], other_seed)
    report = stage_official_inputs(*paths, seed=42)
    assert report["generation_seed"] == 42
    assert report["run_id"] == sha256(
        canonical_bytes({field: original[field] for field in RUN_FIELDS})
    )


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
    with pytest.raises(ValueError, match="changed during official scoring"):
        score_official(*paths, tmp_path / "Sana", seed=42)


def test_scoring_rechecks_video_after_final_metric(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    (tmp_path / "Sana").mkdir()
    monkeypatch.setattr("evaluation.official._validate_video", lambda *_: None)
    monkeypatch.setattr("evaluation.official._verify_official_checkout", lambda _: None)
    video = next(paths[2].glob("simple_60s/game_style_001/*/video.mp4"))
    calls = 0

    def run(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            video.write_bytes(b"changed during final scorer")

    monkeypatch.setattr("evaluation.official.subprocess.run", run)
    with pytest.raises(ValueError, match="changed during official scoring"):
        score_official(*paths, tmp_path / "Sana", seed=42)


def test_scoring_rechecks_staged_link_after_metric(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    (tmp_path / "Sana").mkdir()
    monkeypatch.setattr("evaluation.official._validate_video", lambda *_: None)
    monkeypatch.setattr("evaluation.official._verify_official_checkout", lambda _: None)
    other = tmp_path / "other.mp4"
    other.write_bytes(b"fake video for directory adapter test")
    link = paths[3] / "simple_60s/game_style_001_generated.mp4"

    def run(command, **kwargs):
        link.unlink()
        link.symlink_to(other)

    monkeypatch.setattr("evaluation.official.subprocess.run", run)
    with pytest.raises(
        ValueError, match="staged video link changed during official scoring"
    ):
        score_official(*paths, tmp_path / "Sana", seed=42)


def test_scoring_rechecks_generated_video_set_after_metric(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    (tmp_path / "Sana").mkdir()
    monkeypatch.setattr("evaluation.official._validate_video", lambda *_: None)
    monkeypatch.setattr("evaluation.official._verify_official_checkout", lambda _: None)

    def run(command, **kwargs):
        (paths[3] / "simple_60s/extra_generated.mp4").write_bytes(b"extra")

    monkeypatch.setattr("evaluation.official.subprocess.run", run)
    with pytest.raises(ValueError, match="official video set differs"):
        score_official(*paths, tmp_path / "Sana", seed=42)


def test_failed_rescore_removes_previous_completion_record(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    (tmp_path / "Sana").mkdir()
    monkeypatch.setattr("evaluation.official._validate_video", lambda *_: None)
    monkeypatch.setattr("evaluation.official._verify_official_checkout", lambda _: None)
    monkeypatch.setattr("evaluation.official._verify_scored_split", lambda *_: None)
    monkeypatch.setattr("evaluation.official.subprocess.run", lambda *_, **__: None)
    score_official(*paths, tmp_path / "Sana", seed=42)
    assert (paths[3] / "scoring.json").is_file()

    def fail(*args, **kwargs):
        raise RuntimeError("scorer failed")

    monkeypatch.setattr("evaluation.official.subprocess.run", fail)
    with pytest.raises(RuntimeError, match="scorer failed"):
        score_official(*paths, tmp_path / "Sana", seed=42)
    assert not (paths[3] / "scoring.json").exists()


def test_rescore_removes_previous_metric_outputs(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    (tmp_path / "Sana").mkdir()
    monkeypatch.setattr("evaluation.official._validate_video", lambda *_: None)
    monkeypatch.setattr("evaluation.official._verify_official_checkout", lambda _: None)
    poses = paths[3] / "simple_60s/eval_poses.json"
    poses.parent.mkdir(parents=True)
    poses.write_text("old poses")
    summary = paths[3] / "eval/simple_60s/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("old summary")

    def fail(*args, **kwargs):
        raise RuntimeError("scorer failed")

    monkeypatch.setattr("evaluation.official.subprocess.run", fail)
    with pytest.raises(RuntimeError, match="scorer failed"):
        score_official(*paths, tmp_path / "Sana", seed=42)
    assert not poses.exists()
    assert not summary.exists()


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
                            "time_base": "1/16",
                            "width": 128,
                            "height": 128,
                        }
                    ],
                    "frames": [
                        {"best_effort_timestamp": str(index)} for index in range(961)
                    ],
                }
            )
        )

    monkeypatch.setattr("evaluation.official.subprocess.run", valid_run)
    _validate_video(tmp_path / "video.mp4", {"sanawm_frames": 961, "fps": 16})
    assert calls == ["ffprobe", "ffmpeg"]


def test_video_probe_rejects_variable_frame_timing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "evaluation.official.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "nb_read_frames": "3",
                            "avg_frame_rate": "16/1",
                            "time_base": "1/16",
                            "width": 128,
                            "height": 128,
                        }
                    ],
                    "frames": [
                        {"best_effort_timestamp": value} for value in ("0", "1", "3")
                    ],
                }
            )
        ),
    )
    with pytest.raises(ValueError, match="variable frame timing"):
        _validate_video(tmp_path / "video.mp4", {"sanawm_frames": 3, "fps": 16})


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
