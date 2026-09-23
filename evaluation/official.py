"""Stage identity-checked videos and invoke pinned official SANA-WM metrics."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .identity import verify_output_identity
from .selection import SPLIT_DIRS, canonical_bytes, sha256, write_immutable


OFFICIAL_REPO = "https://github.com/NVlabs/Sana"
OFFICIAL_COMMIT = "f9178744c096dcf2a2ea773da183e341bcbeb044"
VBenCH_DIMS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "overall_consistency",
    "temporal_style",
)
TEMPORAL_DIMS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "imaging_quality",
)
RUN_FIELDS = (
    "selection_id",
    "checkpoint_id",
    "weight_flavor",
    "model_config",
    "codec_id",
    "codec_fingerprint",
    "spatial_resolution",
    "preprocessing_id",
    "conditioning",
    "implementation_id",
    "numeric_execution",
    "sampler",
    "rollout_layout",
)


def _selection(path: Path) -> dict:
    value = json.loads(path.read_text())
    content = {key: item for key, item in value.items() if key != "selection_id"}
    if value["selection_id"] != sha256(canonical_bytes(content)):
        raise ValueError("evaluation selection content differs from its identity")
    return value


def _verify_metadata(selection: dict, benchmark_root: Path) -> None:
    for relative, expected in selection["source_sha256"].items():
        actual = sha256((benchmark_root / relative).read_bytes())
        if actual != expected:
            raise ValueError(f"official benchmark metadata hash differs: {relative}")


def _verify_identity(identity: dict, selection: dict, row: dict) -> None:
    content = {key: value for key, value in identity.items() if key != "generation_id"}
    if identity["generation_id"] != sha256(canonical_bytes(content)):
        raise ValueError("generated video identity content has changed")
    for field, expected in (
        ("selection_id", selection["selection_id"]),
        ("dataset_revision", selection["dataset_revision"]),
        ("scene_id", row["scene_id"]),
        ("split", row["split"]),
        ("seed", row["generation_seed"]),
        ("conditions_sha256", row["conditions_sha256"]),
    ):
        if identity[field] != expected:
            raise ValueError(f"generation identity has wrong {field}")


def _link_once(source: Path, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source.resolve(strict=True), destination)
        return True
    except FileExistsError:
        if not destination.is_symlink() or destination.resolve(
            strict=True
        ) != source.resolve(strict=True):
            raise FileExistsError(
                f"official input already belongs to another output: {destination}"
            ) from None
        return False


def _validate_video(path: Path, row: dict) -> None:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,avg_frame_rate,width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if (
        len(streams) != 1
        or int(streams[0].get("nb_read_frames", 0)) != row["sanawm_frames"]
        or Fraction(streams[0]["avg_frame_rate"]) != row["fps"]
        or (streams[0].get("width"), streams[0].get("height")) != (128, 128)
    ):
        raise ValueError(
            f"generated video has wrong frames, FPS, or resolution: {path}"
        )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def stage_official_inputs(
    selection_path: Path,
    benchmark_root: Path,
    outputs_root: Path,
    method_dir: Path,
    *,
    seed: int,
    run_id: str | None = None,
) -> dict:
    """Adapt directory format only; never edit metric code or generated videos."""
    selection = _selection(selection_path)
    _verify_metadata(selection, benchmark_root)
    chosen = [row for row in selection["rows"] if row["generation_seed"] == seed]
    if not chosen:
        raise ValueError("generation seed absent from frozen selection")
    seen = set()
    shared_run = None
    staged = []
    for row in chosen:
        key = (row["split"], row["scene_id"])
        if key in seen:
            raise ValueError(f"duplicate selected scene/split: {key}")
        seen.add(key)
        candidates = sorted(
            (outputs_root / row["split"] / row["scene_id"]).glob("*/identity.json")
        )
        matched = []
        available = set()
        for candidate in candidates:
            identity = json.loads(candidate.read_text())
            verify_output_identity(candidate.parent, identity)
            content = {
                key: value for key, value in identity.items() if key != "generation_id"
            }
            if identity["generation_id"] != sha256(canonical_bytes(content)):
                raise ValueError("candidate generation identity content has changed")
            run = {field: identity[field] for field in RUN_FIELDS}
            candidate_run_id = sha256(canonical_bytes(run))
            available.add(candidate_run_id)
            if run_id is not None and candidate_run_id != run_id:
                continue
            if identity["selection_id"] != selection["selection_id"]:
                continue
            _verify_identity(identity, selection, row)
            matched.append((candidate.parent, identity, run, candidate_run_id))
        if len(matched) != 1:
            raise ValueError(
                f"expected one generation identity for {key}, found {len(matched)}; "
                f"available run IDs: {sorted(available)}"
            )
        directory, identity, run, selected_run_id = matched[0]
        if shared_run is None:
            shared_run = run
        elif run != shared_run:
            raise ValueError(
                "different model/sampler/layout identities cannot share a score run"
            )
        source = directory / "video.mp4"
        if not source.is_file():
            raise FileNotFoundError(source)
        metadata = json.loads((directory / "metadata.json").read_text())
        video_sha256 = _file_sha256(source)
        if (
            metadata.get("generation_id") != identity["generation_id"]
            or metadata.get("video_sha256") != video_sha256
        ):
            raise ValueError("generated video differs from its identity metadata")
        _validate_video(source, row)
        target = method_dir / row["split"] / f"{row['scene_id']}_generated.mp4"
        staged.append(
            {
                "scene_id": row["scene_id"],
                "split": row["split"],
                "generation_id": identity["generation_id"],
                "video_sha256": video_sha256,
                "source_video": str(source.resolve()),
                "official_video": str(target.absolute()),
            }
        )
    for split in {row["split"] for row in chosen}:
        expected = {
            f"{row['scene_id']}_generated.mp4"
            for row in chosen
            if row["split"] == split
        }
        actual = {path.name for path in (method_dir / split).glob("*_generated.mp4")}
        if actual - expected:
            raise ValueError(
                f"official video set differs from staged selection: {split}"
            )
    for row in staged:
        source = Path(row["source_video"])
        target = Path(row["official_video"])
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source:
                raise FileExistsError(
                    f"official input already belongs to another output: {target}"
                )
    payload = {
        "schema_version": 1,
        "selection_id": selection["selection_id"],
        "dataset_revision": selection["dataset_revision"],
        "generation_seed": seed,
        "run_id": selected_run_id,
        "run_identity": shared_run,
        "official_source": {
            "repository": OFFICIAL_REPO,
            "commit": OFFICIAL_COMMIT,
            "license": "Apache-2.0",
            "local_modifications": [],
        },
        "staged": staged,
    }
    created = []
    try:
        for row in staged:
            target = Path(row["official_video"])
            if _link_once(Path(row["source_video"]), target):
                created.append(target)
        write_immutable(method_dir / "staging.json", payload)
    except Exception:
        for target in created:
            target.unlink(missing_ok=True)
        raise
    return payload


def _verify_official_checkout(repo: Path) -> None:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    if git("rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise ValueError("official evaluator checkout differs from pinned commit")
    if git(
        "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"
    ):
        raise ValueError("official evaluator has unrecorded local modifications")
    if any(
        line.startswith(("-", "+", "U"))
        for line in git("submodule", "status", "--recursive").splitlines()
    ):
        raise ValueError("official evaluator submodule differs from pinned checkout")
    if not (repo / "LICENSE").is_file():
        raise FileNotFoundError(repo / "LICENSE")


def metric_commands(
    official_repo: Path,
    benchmark_root: Path,
    method_dir: Path,
    split: str,
    *,
    python: str = sys.executable,
) -> tuple[list[str], list[str]]:
    """Build official entry-point commands without modifying metric semantics."""
    if split not in SPLIT_DIRS:
        raise ValueError("unsupported official split")
    official_repo = official_repo.resolve()
    benchmark_root = benchmark_root.resolve()
    method_dir = method_dir.resolve()
    source = benchmark_root / SPLIT_DIRS[split]
    metric = [
        python,
        str(official_repo / "tools/metrics/sana_wm/eval_unified.py"),
        "--method_dir",
        str(method_dir),
        "--split",
        split,
        "--benchmark_meta",
        str(source / "scene_trajectories_v2.json"),
        "--metrics",
        "vbench",
        "revisit",
        "camera",
        "temporal",
        "--vbench_dims",
        *VBenCH_DIMS,
        "--revisit_lpips",
        "--window_sec",
        "10",
        "--max_pairs",
        "5",
        "--ref_fps",
        "16",
        "--skip_first_frame",
        "no",
    ]
    camera = [
        python,
        str(official_repo / "tools/metrics/sana_wm/eval_benchmark_poses.py"),
        "--result_folder",
        str(method_dir / split),
        "--manifest",
        str(source / "sanawm_export_v2/run_manifest.jsonl"),
        "--interval",
        "4",
        "--batch_size",
        "1",
        "--output",
        str(method_dir / split / "eval_poses.json"),
        "--skip_first_frame",
        "no",
    ]
    return metric, camera


def _verify_scored_split(method_dir: Path, split: str, rows: list[dict]) -> None:
    expected = {row["scene_id"] for row in rows}
    expected_pairs = {
        row["scene_id"]: min(5, row["evaluation_pair_count"]) for row in rows
    }
    root = method_dir / "eval" / split
    poses = json.loads((method_dir / split / "eval_poses.json").read_text())
    revisit = json.loads((root / "revisit_consistency.json").read_text())
    camera = json.loads((root / "camera_accuracy.json").read_text())
    vbench = json.loads((root / "vbench_scores.json").read_text())
    temporal = json.loads((root / "temporal_degradation.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    per_scene = revisit["per_scene"]
    window_counts = {row["sanawm_frames"] // (10 * row["fps"]) for row in rows}
    if len(window_counts) != 1:
        raise ValueError("selected scenes have mixed temporal window counts")
    expected_windows = {
        f"w{index}_{index * 10}s-{(index + 1) * 10}s"
        for index in range(window_counts.pop())
    }
    if (
        set(poses) != expected
        or set(camera) != expected
        or set(per_scene) != expected
        or any(
            per_scene[scene]["n_pairs"] != count
            for scene, count in expected_pairs.items()
        )
        or revisit["summary"]["n_total_pairs"] != sum(expected_pairs.values())
        or summary["n_videos"] != len(expected)
        or summary["split"] != split
        or summary["camera"]["n_scenes"] != len(expected)
        or summary["vbench"]["n_dimensions"] != len(VBenCH_DIMS)
        or set(vbench["raw_scores"]) != set(VBenCH_DIMS)
        or set(temporal["windows"]) != expected_windows
        or "temporal_degradation" not in summary
    ):
        raise ValueError(f"official scorer returned incomplete coverage for {split}")
    for dimension in VBenCH_DIMS:
        _verify_vbench_scene_results(root, dimension, expected)
    for window in expected_windows:
        for dimension in TEMPORAL_DIMS:
            _verify_vbench_scene_results(
                root / "temporal/temporal_results" / window, dimension, expected
            )


def _verify_vbench_scene_results(
    root: Path, dimension: str, expected: set[str]
) -> None:
    candidates = (
        root / f"eval_{dimension}_eval_results.json",
        root / f"eval_{dimension}_{dimension}_eval_results.json",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise ValueError(f"missing VBench per-video results: {root} {dimension}")
    result = json.loads(path.read_text())[dimension]
    if len(result) < 2 or not math.isfinite(float(result[0])):
        raise ValueError(f"invalid VBench results: {path}")
    videos = result[1]
    scenes = {
        Path(item["video_path"]).stem.removesuffix("_generated") for item in videos
    }
    if (
        len(videos) != len(expected)
        or scenes != expected
        or not all(math.isfinite(float(item["video_results"])) for item in videos)
    ):
        raise ValueError(f"incomplete VBench per-video results: {path}")


def score_official(
    selection_path: Path,
    benchmark_root: Path,
    outputs_root: Path,
    method_dir: Path,
    official_repo: Path,
    *,
    seed: int,
    run_id: str | None = None,
) -> None:
    """Rerun scoring from existing videos without invoking generation."""
    official_repo = official_repo.resolve(strict=True)
    benchmark_root = benchmark_root.resolve(strict=True)
    method_dir = method_dir.resolve()
    _verify_official_checkout(official_repo)
    staged = stage_official_inputs(
        selection_path,
        benchmark_root,
        outputs_root,
        method_dir,
        seed=seed,
        run_id=run_id,
    )
    selected_rows = [
        row
        for row in _selection(selection_path)["rows"]
        if row["generation_seed"] == seed
    ]
    for split in sorted({row["split"] for row in staged["staged"]}):
        metric, camera = metric_commands(
            official_repo, benchmark_root, method_dir, split
        )
        for command in (camera, metric):
            for row in staged["staged"]:
                if (
                    row["split"] == split
                    and _file_sha256(Path(row["source_video"])) != row["video_sha256"]
                ):
                    raise ValueError("staged video changed before official scoring")
            subprocess.run(command, cwd=official_repo, check=True)
        _verify_scored_split(
            method_dir, split, [row for row in selected_rows if row["split"] == split]
        )
    _verify_official_checkout(official_repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "score"))
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--method-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", help="SHA-256 of the shared run identity")
    parser.add_argument("--official-repo", type=Path)
    args = parser.parse_args()
    paths = (args.selection, args.benchmark_root, args.outputs_root, args.method_dir)
    if args.mode == "stage":
        print(
            json.dumps(
                stage_official_inputs(*paths, seed=args.seed, run_id=args.run_id),
                sort_keys=True,
            )
        )
    else:
        if args.official_repo is None:
            parser.error("--official-repo is required for score")
        score_official(*paths, args.official_repo, seed=args.seed, run_id=args.run_id)


if __name__ == "__main__":
    main()
