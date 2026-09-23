"""Generation identity and collision-safe output directories."""

from __future__ import annotations

import json
from pathlib import Path

from .selection import canonical_bytes, sha256, write_immutable


def generation_identity(
    selection: dict,
    row: dict,
    *,
    checkpoint_id: str,
    weight_flavor: str,
    codec_id: str,
    sampler: dict,
) -> dict:
    """Hash every input that can change a generated trajectory."""
    if row not in selection["rows"]:
        raise ValueError("generation row is absent from the frozen selection")
    if selection["selection_id"] != sha256(
        canonical_bytes(
            {key: value for key, value in selection.items() if key != "selection_id"}
        )
    ):
        raise ValueError("selection content differs from its immutable identity")
    if (
        not checkpoint_id.startswith("sha256:")
        or len(checkpoint_id) != 71
        or any(c not in "0123456789abcdef" for c in checkpoint_id[7:])
    ):
        raise ValueError("checkpoint_id must be a SHA-256 digest")
    if weight_flavor not in {"model", "ema"}:
        raise ValueError("weight_flavor must be model or ema")
    if not codec_id or not sampler:
        raise ValueError("codec and sampler must be explicit")
    required_sampler = {"solver", "steps", "cfg_scale", "history_policy"}
    if not required_sampler <= sampler.keys():
        raise ValueError("sampler must identify solver, steps, CFG, and history policy")
    payload = {
        "schema_version": 1,
        "selection_id": selection["selection_id"],
        "dataset_revision": selection["dataset_revision"],
        "scene_id": row["scene_id"],
        "split": row["split"],
        "seed": row["generation_seed"],
        "checkpoint_id": checkpoint_id,
        "weight_flavor": weight_flavor,
        "codec_id": codec_id,
        "conditions_sha256": row["conditions_sha256"],
        "sampler": sampler,
    }
    return {"generation_id": sha256(canonical_bytes(payload)), **payload}


def claim_output_directory(root: Path, identity: dict) -> Path:
    """Reserve one directory per identity and reject a conflicting prior claim."""
    expected = sha256(
        canonical_bytes(
            {key: value for key, value in identity.items() if key != "generation_id"}
        )
    )
    if identity.get("generation_id") != expected:
        raise ValueError("generation identity does not match its content")
    for field in ("split", "scene_id"):
        value = identity[field]
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"unsafe {field} in output identity")
    directory = (
        root / identity["split"] / identity["scene_id"] / identity["generation_id"]
    )
    write_immutable(directory / "identity.json", identity)
    return directory


def verify_output_identity(directory: Path, identity: dict) -> None:
    recorded = json.loads((directory / "identity.json").read_text())
    if recorded != identity:
        raise ValueError("output directory belongs to a different generation identity")
