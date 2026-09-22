"""Run the M1 961-frame reader, codec, camera, and causality gate."""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
from typing import Callable, Sequence

from core.video_layout import CodecTemporalSpec, VideoLayout
from datasets.sana_wm import SANAReader, SampleReadError, load_manifest
from representations import Representation
from representations.diagnostics import CodecGateSettings, validate_complete_sample


def _load_factory(reference: str) -> Callable[[], Representation]:
    try:
        module_name, attribute = reference.split(":", 1)
    except ValueError as exc:
        raise ValueError("codec factory must use module.path:callable syntax") from exc
    factory = getattr(import_module(module_name), attribute, None)
    if not callable(factory):
        raise ValueError(f"codec factory is not callable: {reference}")
    return factory


def run_gate(
    *,
    manifest_path: Path,
    data_root: Path,
    codec_factory: Callable[[], Representation],
    split: str,
    preprocessing: dict[str, object],
    settings: CodecGateSettings,
    first_frame_is_independent: bool,
) -> dict[str, object]:
    manifest = load_manifest(manifest_path)
    records = manifest.for_split(split)  # type: ignore[arg-type]
    if not records:
        raise ValueError(f"manifest contains no {split!r} records")
    codec = codec_factory()
    if not isinstance(codec, Representation):
        raise TypeError("codec factory must return a Representation")
    if not codec.spec.causal:
        raise ValueError("M1 gate requires a codec declared as causal")
    layout = VideoLayout.from_codec(
        fps=settings.fps,
        rgb_frame_count=settings.rgb_frame_count,
        codec=CodecTemporalSpec(
            temporal_compression=codec.spec.temporal_compression,
            first_frame_is_independent=first_frame_is_independent,
        ),
        temporal_patch_size=1,
        latent_chunk_size=64,
    )
    reader = SANAReader(data_root)
    passed: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for record in records:
        try:
            sample = reader.read(record)
            passed.append(
                validate_complete_sample(
                    sample,
                    codec=codec,
                    layout=layout,
                    preprocessing=preprocessing,
                    settings=settings,
                )
            )
        except SampleReadError as exc:
            failures.append(
                {
                    "sample_id": exc.sample_id,
                    "component": exc.component,
                    "reason": exc.reason,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "sample_id": record.sample_id,
                    "component": "codec_gate",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "schema_version": 1,
        "status": "passed" if passed and not failures else "failed",
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "split": split,
        "passed_samples": passed,
        "failures": failures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--codec-factory",
        required=True,
        help="zero-argument factory in module.path:callable form",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--rgb-frames", type=int, default=961)
    parser.add_argument("--future-start-rgb", type=int, default=481)
    parser.add_argument("--causality-atol", type=float, default=0.0)
    parser.add_argument(
        "--preprocessing-json",
        default='{"color_space":"RGB","range":"0_to_1","resize":"none","crop":"none"}',
    )
    parser.add_argument(
        "--first-frame-independent",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)
    preprocessing = json.loads(args.preprocessing_json)
    if not isinstance(preprocessing, dict) or not preprocessing:
        parser.error("--preprocessing-json must decode to a non-empty object")
    settings = CodecGateSettings(
        fps=args.fps,
        rgb_frame_count=args.rgb_frames,
        future_start_rgb=args.future_start_rgb,
        causality_atol=args.causality_atol,
    )
    report = run_gate(
        manifest_path=args.manifest,
        data_root=args.data_root,
        codec_factory=_load_factory(args.codec_factory),
        split=args.split,
        preprocessing=preprocessing,
        settings=settings,
        first_frame_is_independent=args.first_frame_independent,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
