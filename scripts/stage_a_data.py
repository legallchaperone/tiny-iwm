"""Deterministic data selection helpers for the Stage A baseline."""

from collections.abc import Iterable
from pathlib import PurePosixPath


def scene_id(member: str) -> str:
    """Return the official Sekai recording id at the start of a latent filename."""
    return PurePosixPath(member).name.split("_", 1)[0]


def select_scene_disjoint_samples(
    members: Iterable[str], *, train_scenes: int, validation_scenes: int
) -> dict[str, list[str]]:
    """Select one stable sample from each of disjoint sorted scene groups."""
    if train_scenes <= 0 or validation_scenes <= 0:
        raise ValueError("train_scenes and validation_scenes must be positive")
    grouped: dict[str, list[str]] = {}
    for member in sorted(set(members)):
        if member.endswith(".npz"):
            grouped.setdefault(scene_id(member), []).append(member)
    scenes = sorted(grouped)
    if len(scenes) < train_scenes + validation_scenes:
        raise ValueError("official latent archive does not contain enough scene groups")
    train_ids = scenes[:train_scenes]
    validation_ids = scenes[-validation_scenes:]
    if set(train_ids) & set(validation_ids):
        raise ValueError("training and validation scene selections overlap")
    return {
        "train": [grouped[value][0] for value in train_ids],
        "validation": [grouped[value][0] for value in validation_ids],
    }
