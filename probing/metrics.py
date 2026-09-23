"""Offline representation statistics from explicitly retained probe tokens."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import torch


GROUP_FIELDS = (
    "checkpoint_id",
    "config_id",
    "training_seed",
    "layer",
    "observation_point",
    "token_type",
    "branch",
    "forward_purpose",
    "flow_time",
    "rollout_time",
)
PAIR_FIELDS = (
    "sample_id",
    "generation_seed",
    "rollout_time",
    "flow_time",
    "token_index",
    "token_type",
    "layer",
    "observation_point",
    "branch",
    "physical_position",
)
CONTROLLED_FIELDS = PAIR_FIELDS + ("checkpoint_id", "config_id", "training_seed")
VARIANT_FIELDS = PAIR_FIELDS + ("forward_purpose",)
ALIGNMENT_GROUP_FIELDS = (
    "checkpoint_id",
    "config_id",
    "training_seed",
    "layer",
    "observation_point",
    "token_type",
    "branch",
    "rollout_time",
    "flow_time",
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def load_raw_capture(root: Path) -> list[tuple[dict, torch.Tensor]]:
    """Read selected raw features; reject missing/non-finite tensors."""
    records = []
    for path in sorted(root.glob("*.json")):
        record = json.loads(path.read_text())
        if "raw_file" not in record:
            continue
        raw = root / record["raw_file"]
        if raw.parent != root or raw.is_symlink() or not raw.is_file():
            raise ValueError(f"missing or unsafe raw feature: {raw}")
        feature = torch.load(raw, map_location="cpu", weights_only=True)
        if feature.ndim != 1 or feature.numel() != record["channels"]:
            raise ValueError(f"raw feature shape differs from capture record: {raw}")
        if not torch.isfinite(feature).all():
            raise ValueError(f"non-finite raw feature: {raw}")
        records.append((record, feature.to(dtype=torch.float64)))
    if not records:
        raise ValueError("no raw features retained; enable selected raw_points")
    return records


def describe_features(matrix: torch.Tensor) -> dict:
    """Rows are sampled observations, columns are feature channels."""
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("feature matrix must be nonempty [samples, channels]")
    if not torch.isfinite(matrix).all():
        raise ValueError("feature matrix must be finite")
    matrix = matrix.to(dtype=torch.float64)
    norms = torch.linalg.vector_norm(matrix, dim=1)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    if float(energy.sum()) == 0:
        effective_rank = 0.0
    else:
        weights = energy[energy > 0] / energy.sum()
        effective_rank = float(torch.exp(-(weights * weights.log()).sum()))
    return {
        "axes": {
            "rows": "selected sample/token observations",
            "columns": "feature channels",
            "centering": "subtract per-channel mean across rows",
        },
        "samples": matrix.shape[0],
        "channels": matrix.shape[1],
        "norm": {
            "mean": float(norms.mean()),
            "minimum": float(norms.min()),
            "median": float(torch.quantile(norms, 0.5)),
            "p95": float(torch.quantile(norms, 0.95)),
            "maximum": float(norms.max()),
        },
        "channel_mean_mean": float(matrix.mean(dim=0).mean()),
        "channel_std_mean": float(matrix.std(dim=0, unbiased=False).mean()),
        "singular_values": [float(value) for value in singular],
        "effective_rank": effective_rank,
    }


def compare_features(left: torch.Tensor, right: torch.Tensor) -> dict:
    """Centered linear CKA permits different channel counts; cosine requires equal ones."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("paired matrices need equal observation rows")
    if left.shape[0] < 2:
        raise ValueError("CKA needs at least two matched observations")
    x = left.to(dtype=torch.float64) - left.to(dtype=torch.float64).mean(dim=0)
    y = right.to(dtype=torch.float64) - right.to(dtype=torch.float64).mean(dim=0)
    numerator = torch.linalg.matrix_norm(x.T @ y).square()
    denominator = torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y)
    cka = None if float(denominator) == 0 else float(numerator / denominator)
    cosine = None
    cosine_valid_rows = None
    l2_drift = None
    if left.shape[1] == right.shape[1]:
        left64, right64 = left.to(torch.float64), right.to(torch.float64)
        left_norm = torch.linalg.vector_norm(left64, dim=1)
        right_norm = torch.linalg.vector_norm(right64, dim=1)
        valid = (left_norm > 0) & (right_norm > 0)
        cosine_valid_rows = int(valid.sum())
        if cosine_valid_rows:
            cosine = float(
                (
                    (left64[valid] / left_norm[valid, None])
                    * (right64[valid] / right_norm[valid, None])
                )
                .sum(dim=1)
                .mean()
            )
        l2_drift = float(torch.linalg.vector_norm(left - right, dim=1).mean())
    return {
        "matched_observations": left.shape[0],
        "centered_linear_cka": cka,
        "mean_row_cosine": cosine,
        "cosine_valid_rows": cosine_valid_rows,
        "mean_row_l2_drift": l2_drift,
    }


def _key(record: dict, fields: tuple[str, ...]) -> tuple:
    return tuple(_canonical(record[field]) for field in fields)


def _sort_key(record: dict, fields: tuple[str, ...]) -> tuple:
    def part(value):
        if value is None:
            return (0, "")
        if isinstance(value, (int, float)):
            return (1, value)
        if isinstance(value, str):
            return (2, value)
        return (3, _canonical(value))

    return tuple(part(record[field]) for field in fields)


def pair_captures(
    left: list[tuple[dict, torch.Tensor]],
    right: list[tuple[dict, torch.Tensor]],
    *,
    fixed_fields: tuple[str, ...] = PAIR_FIELDS,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict], list[dict]]:
    """Join on every fixed variable; provenance/purpose may differ deliberately."""
    left_by_key = {
        _key(record, fixed_fields): (record, value) for record, value in left
    }
    right_by_key = {
        _key(record, fixed_fields): (record, value) for record, value in right
    }
    if len(left_by_key) != len(left) or len(right_by_key) != len(right):
        raise ValueError("duplicate capture identity in comparison")
    if left_by_key.keys() != right_by_key.keys():
        raise ValueError("capture rows differ in fixed sample/time/token coordinates")
    keys = sorted(left_by_key)
    return (
        [left_by_key[key][1] for key in keys],
        [right_by_key[key][1] for key in keys],
        [left_by_key[key][0] for key in keys],
        [right_by_key[key][0] for key in keys],
    )


def summarize_capture(root: Path) -> dict:
    records = load_raw_capture(root)
    groups = defaultdict(list)
    for record, feature in records:
        groups[_key(record, GROUP_FIELDS)].append((record, feature))
    summaries = []
    for members in groups.values():
        meta = {field: members[0][0][field] for field in GROUP_FIELDS}
        matrix = torch.stack([value for _, value in members])
        summaries.append(
            {
                **meta,
                "statistics": describe_features(matrix),
                "raw_files": [record["raw_file"] for record, _ in members],
            }
        )
    summaries.sort(key=lambda item: _sort_key(item, GROUP_FIELDS))
    identity_parts = [
        {
            "record": record,
            "raw_sha256": hashlib.sha256(
                (root / record["raw_file"]).read_bytes()
            ).hexdigest(),
        }
        for record, _ in records
    ]
    identity = hashlib.sha256(_canonical(identity_parts)).hexdigest()
    return {
        "schema_version": 1,
        "capture_identity": "sha256:" + identity,
        "source_directory": str(root),
        "groups": summaries,
    }


def history_alignment(root: Path) -> dict:
    """Pair controlled GT/generated history forwards at each rollout time."""
    records = load_raw_capture(root)
    gt = [
        (record, value)
        for record, value in records
        if record["forward_purpose"] == "controlled_gt_history"
    ]
    generated = [
        (record, value)
        for record, value in records
        if record["forward_purpose"] == "controlled_generated_history"
    ]
    if not gt or not generated:
        raise ValueError("controlled GT and generated history records are required")
    left, right, matched, _ = pair_captures(
        gt, generated, fixed_fields=CONTROLLED_FIELDS
    )
    groups = defaultdict(list)
    for index, record in enumerate(matched):
        groups[_key(record, ALIGNMENT_GROUP_FIELDS)].append(index)
    by_group = []
    for indices in groups.values():
        if len(indices) < 2:
            raise ValueError(
                "each layer/time group needs two matched observations for CKA"
            )
        by_group.append(
            {
                **{
                    field: matched[indices[0]][field]
                    for field in ALIGNMENT_GROUP_FIELDS
                },
                **compare_features(
                    torch.stack([left[index] for index in indices]),
                    torch.stack([right[index] for index in indices]),
                ),
            }
        )
    by_group.sort(key=lambda item: _sort_key(item, ALIGNMENT_GROUP_FIELDS))
    return {
        "comparison": "controlled_gt_vs_generated_history",
        "fixed_fields": list(CONTROLLED_FIELDS),
        "per_group": by_group,
    }


def compare_capture_roots(left_root: Path, right_root: Path) -> dict:
    """Compare variants only where sample, time, token, and purpose match."""
    left, right, records, right_records = pair_captures(
        load_raw_capture(left_root),
        load_raw_capture(right_root),
        fixed_fields=VARIANT_FIELDS,
    )
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[_key(record, GROUP_FIELDS)].append(index)
    comparisons = []
    for indices in groups.values():
        if len(indices) < 2:
            raise ValueError("each comparable layer/time group needs two observations")
        right_provenance = {
            _key(right_records[index], ("checkpoint_id", "config_id", "training_seed"))
            for index in indices
        }
        if len(right_provenance) != 1:
            raise ValueError("right capture provenance differs within comparison group")
        comparisons.append(
            {
                **{field: records[indices[0]][field] for field in GROUP_FIELDS},
                "right_provenance": {
                    field: right_records[indices[0]][field]
                    for field in ("checkpoint_id", "config_id", "training_seed")
                },
                "right_raw_files": [
                    right_records[index]["raw_file"] for index in indices
                ],
                **compare_features(
                    torch.stack([left[index] for index in indices]),
                    torch.stack([right[index] for index in indices]),
                ),
            }
        )
    comparisons.sort(key=lambda item: _sort_key(item, GROUP_FIELDS))
    return {
        "fixed_fields": list(VARIANT_FIELDS),
        "left_capture_identity": summarize_capture(left_root)["capture_identity"],
        "right_capture_identity": summarize_capture(right_root)["capture_identity"],
        "right_source_directory": str(right_root),
        "per_group": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-alignment", action="store_true")
    parser.add_argument("--compare-root", type=Path)
    args = parser.parse_args()
    report = summarize_capture(args.capture_root)
    if args.history_alignment:
        report["history_alignment"] = history_alignment(args.capture_root)
    if args.compare_root:
        report["variant_comparison"] = compare_capture_roots(
            args.capture_root, args.compare_root
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
