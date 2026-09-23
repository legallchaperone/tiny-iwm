import json
from dataclasses import replace
from pathlib import Path

import pytest

from core.video_layout import CodecTemporalSpec, FrameRange, VideoLayout
from evaluation.identity import (
    claim_output_directory,
    generation_identity,
    verify_output_identity,
)
from evaluation.selection import canonical_bytes, sha256, write_immutable


MANIFEST = Path("data/manifests/sana_wm_eval_subset_v1.json")
BACKEND = {
    "device_name": "NVIDIA A10G",
    "compute_capability": "8.6",
    "cuda_runtime": "12.8",
    "cudnn_version": 91002,
}


def _selection():
    return json.loads(MANIFEST.read_text())


def _identity(selection, row, **overrides):
    inputs = {
        "checkpoint_id": "sha256:" + "a" * 64,
        "weight_flavor": "ema",
        "model_config": {
            "latent_channels": 128,
            "hidden_size": 832,
            "depth": 24,
            "num_heads": 16,
            "patch_size": [1, 2, 2],
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "prope_camera_dims": 48,
        },
        "codec_id": "LTX2VAE_diffusers_704x1280_official_latent_cache",
        "codec_fingerprint": {
            "weights_sha256": "sha256:" + "c" * 64,
            "normalization_sha256": "sha256:" + "d" * 64,
            "encoding_policy": "bidirectional_mode_v1",
            "implementation_revision": "diffusers:0.37.0",
            "execution_dtype": "bfloat16",
            "execution_backend": "cuda",
            "backend_fingerprint": BACKEND,
            "cuda_math_policy": "strict_no_tf32_no_reduced_reduction_v1",
        },
        "spatial_resolution": (704, 1280),
        "preprocessing_id": "official_center_crop_v1",
        "conditioning": {
            "camera": {
                "enabled": True,
                "method": "tiled_prope",
                "translation_scale": 1.0,
                "projection_image_size": [128, 128],
            },
            "text": {"enabled": False},
        },
        "implementation_id": "git:checkpoint-compatible-revision",
        "numeric_execution": {
            "parameter_dtype": "float32",
            "autocast_dtype": "bfloat16",
            "latent_dtype": "bfloat16",
            "attention_backend": "math",
            "torch_version": "2.8.0",
            "tf32_enabled": False,
            "backend_fingerprint": BACKEND,
        },
        "rollout_layout": VideoLayout.from_codec(
            fps=16,
            rgb_frame_count=961,
            codec=CodecTemporalSpec(8),
            latent_chunk_size=4,
            initial_condition_frames=1,
        ),
        "sampler": {
            "solver": "euler",
            "steps": 25,
            "cfg_scale": 1.0,
            "history_policy": "clean_cached",
            "initial_history_policy": "source_image_only",
            "batch_size": 1,
        },
    }
    inputs.update(overrides)
    return generation_identity(selection, row, **inputs)


def test_frozen_subset_preserves_official_minute_protocol():
    selection = _selection()
    payload = {key: value for key, value in selection.items() if key != "selection_id"}
    assert selection["selection_id"] == sha256(canonical_bytes(payload))
    assert len(selection["rows"]) == 8
    assert {row["split"] for row in selection["rows"]} == {"simple_60s", "hard_60s"}
    assert {row["category"] for row in selection["rows"]} == {
        "game_style",
        "indoor",
        "outdoor_city",
        "outdoor_nature",
    }
    assert len({row["scene_id"] for row in selection["rows"]}) == 4
    assert all(
        (
            row["sanawm_frames"],
            row["fps"],
            row["official_scoring_frames"],
            row["official_trim_to_frames"],
        )
        == (961, 16, 960, 960)
        and row["evaluation_pair_count"] > 0
        and row["evaluation_pair_max_frame"] < 960
        for row in selection["rows"]
    )
    assert all(
        row["conditions"][f"{kind}_sha256"]
        == selection["source_sha256"][row["conditions"][f"{kind}_path"]]
        and row["conditions_sha256"] == sha256(canonical_bytes(row["conditions"]))
        for row in selection["rows"]
        for kind in ("image", "camera")
    )


def test_generation_identity_separates_all_variable_inputs(tmp_path):
    selection = _selection()
    row = selection["rows"][0]
    baseline = _identity(selection, row)
    layout = VideoLayout.from_codec(
        fps=16,
        rgb_frame_count=961,
        codec=CodecTemporalSpec(8),
        latent_chunk_size=4,
        initial_condition_frames=1,
    )
    changes = [
        _identity(selection, row, checkpoint_id="sha256:" + "b" * 64),
        _identity(selection, row, weight_flavor="model"),
        _identity(
            selection,
            row,
            model_config={**baseline["model_config"], "prope_camera_dims": 32},
        ),
        _identity(selection, row, codec_id="other-codec"),
        _identity(
            selection,
            row,
            codec_fingerprint={
                **baseline["codec_fingerprint"],
                "weights_sha256": "sha256:" + "e" * 64,
            },
        ),
        _identity(
            selection,
            row,
            codec_fingerprint={
                **baseline["codec_fingerprint"],
                "execution_dtype": "float32",
            },
        ),
        _identity(selection, row, spatial_resolution=(128, 128)),
        _identity(selection, row, preprocessing_id="other-crop"),
        _identity(
            selection,
            row,
            numeric_execution={
                "parameter_dtype": "float32",
                "autocast_dtype": "none",
                "latent_dtype": "float32",
                "attention_backend": "math",
                "torch_version": "2.8.0",
                "tf32_enabled": False,
                "backend_fingerprint": BACKEND,
            },
        ),
        _identity(
            selection,
            row,
            numeric_execution={
                **baseline["numeric_execution"],
                "backend_fingerprint": {**BACKEND, "device_name": "NVIDIA H100"},
            },
        ),
        _identity(
            selection,
            row,
            conditioning={
                "camera": {
                    "enabled": True,
                    "method": "tiled_prope",
                    "translation_scale": 2.0,
                    "projection_image_size": [128, 128],
                },
                "text": {"enabled": False},
            },
        ),
        _identity(
            selection,
            row,
            conditioning={
                **baseline["conditioning"],
                "text": {"enabled": True, "encoder_fingerprint": "sha256:" + "f" * 64},
            },
        ),
        _identity(
            selection,
            row,
            rollout_layout=VideoLayout.from_codec(
                fps=16,
                rgb_frame_count=961,
                codec=CodecTemporalSpec(8),
                latent_chunk_size=8,
                initial_condition_frames=1,
            ),
        ),
        _identity(
            selection,
            row,
            rollout_layout=replace(
                layout,
                token_to_latent_ranges=(
                    FrameRange(0, 2),
                    *layout.token_to_latent_ranges[1:],
                ),
            ),
        ),
        _identity(
            selection,
            row,
            sampler={
                "solver": "euler",
                "steps": 26,
                "cfg_scale": 1.0,
                "history_policy": "clean_cached",
                "initial_history_policy": "source_image_only",
                "batch_size": 1,
            },
        ),
        _identity(selection, selection["rows"][1]),
    ]
    assert len({item["generation_id"] for item in [baseline, *changes]}) == 17
    tuple_config = {**baseline["model_config"], "patch_size": (1, 2, 2)}
    assert _identity(selection, row, model_config=tuple_config) == baseline
    directory = claim_output_directory(tmp_path, baseline)
    assert claim_output_directory(tmp_path, baseline) == directory
    verify_output_identity(directory, baseline)
    with pytest.raises(ValueError, match="different generation identity"):
        verify_output_identity(directory, changes[0])


def test_tampering_and_replacement_are_rejected(tmp_path):
    selection = _selection()
    row = selection["rows"][0]
    identity = _identity(selection, row)
    with pytest.raises(ValueError, match="source-image prefix"):
        _identity(
            selection,
            row,
            sampler={**identity["sampler"], "initial_history_policy": "prefilled"},
        )
    with pytest.raises(ValueError, match="single-row noise streams"):
        _identity(selection, row, sampler={**identity["sampler"], "batch_size": 2})
    changed_selection = dict(selection)
    changed_selection["dataset_revision"] = "0" * 40
    with pytest.raises(ValueError, match="selection content differs"):
        _identity(changed_selection, row)
    changed_identity = dict(identity, checkpoint_id="sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="does not match"):
        claim_output_directory(tmp_path, changed_identity)
    path = tmp_path / "frozen.json"
    write_immutable(path, selection)
    write_immutable(path, selection)
    with pytest.raises(FileExistsError, match="immutable selection"):
        write_immutable(path, changed_selection)


def test_enabled_text_and_codec_require_immutable_fingerprints():
    selection = _selection()
    row = selection["rows"][0]
    baseline = _identity(selection, row)
    with pytest.raises(ValueError, match="encoder fingerprint"):
        _identity(
            selection,
            row,
            conditioning={**baseline["conditioning"], "text": {"enabled": True}},
        )
    with pytest.raises(ValueError, match="codec fingerprint"):
        _identity(selection, row, codec_fingerprint={"weights_sha256": "label"})


def test_identical_concurrent_claim_is_idempotent(tmp_path, monkeypatch):
    def another_worker_wins(source, destination):
        destination.write_bytes(source.read_bytes())
        raise FileExistsError(destination)

    monkeypatch.setattr("evaluation.selection.os.link", another_worker_wins)
    path = tmp_path / "selection.json"
    write_immutable(path, _selection())
    assert json.loads(path.read_text()) == _selection()


def test_generation_claim_works_on_volume_without_hard_links(tmp_path, monkeypatch):
    def unsupported_hard_link(source, destination):
        raise PermissionError("hard links unsupported")

    monkeypatch.setattr("evaluation.selection.os.link", unsupported_hard_link)
    selection = _selection()
    identity = _identity(selection, selection["rows"][0])
    directory = claim_output_directory(tmp_path, identity)
    assert claim_output_directory(tmp_path, identity) == directory
    verify_output_identity(directory, identity)
