"""Freeze a reproducible official SANA-WM scene/split/seed subset."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any


DATASET = "Efficient-Large-Model/SANA-WM-Bench"
SPLIT_DIRS = {
    "simple_60s": "benchmark_v2_smooth_60s",
    "hard_60s": "benchmark_v2_hard_60s",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _source_files(root: Path, split: str) -> tuple[Path, Path]:
    split_dir = root / SPLIT_DIRS[split]
    return (
        split_dir / "sanawm_export_v2/run_manifest.jsonl",
        split_dir / "scene_trajectories_v2.json",
    )


def _index_unique(rows: list[dict], key: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        value = str(row[key])
        if value in indexed:
            raise ValueError(f"duplicate official {key}: {value}")
        indexed[value] = row
    return indexed


def build_selection(
    source_root: Path,
    *,
    revision: str,
    splits: tuple[str, ...] = ("simple_60s", "hard_60s"),
    scene_ids: tuple[str, ...] = (),
    per_category: int = 1,
    selection_seed: int = 2026,
    generation_seeds: tuple[int, ...] = (42,),
) -> dict:
    """Select scene count only; preserve each official trajectory and pair list."""
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("dataset revision must be a 40-character lowercase commit SHA")
    if (
        not splits
        or len(set(splits)) != len(splits)
        or any(s not in SPLIT_DIRS for s in splits)
    ):
        raise ValueError("splits must be unique official 60-second split names")
    if (
        not generation_seeds
        or len(set(generation_seeds)) != len(generation_seeds)
        or any(s < 0 for s in generation_seeds)
    ):
        raise ValueError("generation seeds must be unique non-negative integers")
    if per_category <= 0:
        raise ValueError("per_category must be positive")

    kept_file = source_root / "scene_set/kept_scenes.txt"
    kept = tuple(
        line.strip() for line in kept_file.read_text().splitlines() if line.strip()
    )
    if len(kept) != 80 or len(set(kept)) != 80:
        raise ValueError("official kept_scenes must contain 80 unique IDs")
    if scene_ids:
        chosen = tuple(sorted(set(scene_ids)))
        if len(chosen) != len(scene_ids) or not set(chosen) <= set(kept):
            raise ValueError("explicit scene IDs must be unique members of kept_scenes")
        method = "explicit_ids"
    else:
        categories: dict[str, list[str]] = defaultdict(list)
        for scene in kept:
            categories[scene.rsplit("_", 1)[0]].append(scene)
        if len(categories) != 4 or any(
            len(group) != 20 for group in categories.values()
        ):
            raise ValueError(
                "official 80-scene categories must contain four groups of 20"
            )
        rng = random.Random(selection_seed)
        chosen = tuple(
            sorted(
                scene
                for category in sorted(categories)
                for scene in rng.sample(sorted(categories[category]), per_category)
            )
        )
        method = "stratified_seeded"

    source_hashes = {"scene_set/kept_scenes.txt": sha256(kept_file.read_bytes())}
    rows = []
    for split in splits:
        manifest_file, trajectory_file = _source_files(source_root, split)
        source_hashes[str(manifest_file.relative_to(source_root))] = sha256(
            manifest_file.read_bytes()
        )
        source_hashes[str(trajectory_file.relative_to(source_root))] = sha256(
            trajectory_file.read_bytes()
        )
        manifest = _index_unique(
            [
                json.loads(line)
                for line in manifest_file.read_text().splitlines()
                if line.strip()
            ],
            "id",
        )
        trajectory = _index_unique(
            json.loads(trajectory_file.read_text())["scenes"], "scene_id"
        )
        if set(manifest) != set(kept) or set(trajectory) != set(kept):
            raise ValueError(
                f"official {split} manifest/trajectory scene set differs from kept_scenes"
            )
        for scene in chosen:
            item, motion = manifest[scene], trajectory[scene]
            expected_camera = f"{SPLIT_DIRS[split]}/sanawm_export_v2/{scene}.npz"
            if (
                item["camera_path"] != expected_camera
                or item["image_path"] != f"images/{scene}.png"
            ):
                raise ValueError(f"unexpected official asset path for {split}/{scene}")
            if (
                item["sanawm_frames"],
                item["expected_fps"],
                item["expected_video_frames"],
            ) != (961, 16.0, 960):
                raise ValueError(
                    f"official minute protocol changed for {split}/{scene}"
                )
            if item["trajectory_id"] != motion["trajectory_id"]:
                raise ValueError(f"trajectory identity disagrees for {split}/{scene}")
            pairs = motion["evaluation_pairs"]
            if not pairs or any(
                not (0 <= pair["frame_a"] < 961 and 0 <= pair["frame_b"] < 961)
                for pair in pairs
            ):
                raise ValueError(
                    f"invalid official revisit pair index for {split}/{scene}"
                )
            conditions = {
                "dataset_revision": revision,
                "image_path": item["image_path"],
                "camera_path": item["camera_path"],
                "prompt": item["prompt"],
                "trajectory_id": item["trajectory_id"],
                "official_manifest_row_sha256": sha256(canonical_bytes(item)),
                "official_trajectory_row_sha256": sha256(canonical_bytes(motion)),
            }
            for seed in generation_seeds:
                rows.append(
                    {
                        "scene_id": scene,
                        "category": item["scene_type"],
                        "split": split,
                        "generation_seed": seed,
                        "conditions": conditions,
                        "conditions_sha256": sha256(canonical_bytes(conditions)),
                        "sanawm_frames": 961,
                        "fps": 16,
                        "official_scoring_frames": item["expected_video_frames"],
                        "official_trim_to_frames": item["trim_to_frames"],
                        "evaluation_pair_count": len(pairs),
                        "evaluation_pair_max_frame": max(
                            max(pair["frame_a"], pair["frame_b"]) for pair in pairs
                        ),
                        "evaluation_pairs_sha256": sha256(canonical_bytes(pairs)),
                    }
                )
    payload = {
        "schema_version": 1,
        "protocol": "sana_wm_official_60s",
        "dataset": DATASET,
        "dataset_revision": revision,
        "source_sha256": source_hashes,
        "selection": {
            "method": method,
            "selection_seed": selection_seed,
            "scenes": list(chosen),
            "splits": list(splits),
            "generation_seeds": list(generation_seeds),
        },
        "rows": rows,
    }
    return {"selection_id": sha256(canonical_bytes(payload)), **payload}


def write_immutable(path: Path, value: dict) -> None:
    """Never silently replace a frozen selection with changed content."""
    data = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(
                f"immutable selection already exists with different content: {path}"
            )
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise FileExistsError(
                    f"immutable selection already exists with different content: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", action="append", choices=tuple(SPLIT_DIRS))
    parser.add_argument("--scene-id", action="append")
    parser.add_argument("--per-category", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--generation-seed", type=int, action="append")
    args = parser.parse_args()
    selection = build_selection(
        args.source_root,
        revision=args.revision,
        splits=tuple(args.split or SPLIT_DIRS),
        scene_ids=tuple(args.scene_id or ()),
        per_category=args.per_category,
        selection_seed=args.selection_seed,
        generation_seeds=tuple(args.generation_seed or (42,)),
    )
    write_immutable(args.output, selection)
    print(selection["selection_id"])


if __name__ == "__main__":
    main()
