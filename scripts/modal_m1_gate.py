"""Run the CWX-14 official SANA-WM causal VAE gate on Modal."""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
SANA_COMMIT = "f9178744c096dcf2a2ea773da183e341bcbeb044"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.8.0",
        "diffusers>=0.37.0",
        "accelerate>=1.0.1",
        "huggingface-hub>=0.34.0",
        "safetensors>=0.5.0",
        "numpy>=2.0.0",
        "opencv-python-headless>=4.10.0",
        "pillow>=11.0.0",
    )
    .run_commands(
        "git clone --filter=blob:none --no-checkout https://github.com/NVlabs/Sana.git /opt/Sana",
        f"cd /opt/Sana && git checkout {SANA_COMMIT}",
    )
    .add_local_dir(REPO, remote_path="/opt/tiny-iwm")
)

app = modal.App("tiny-iwm-cwx-14-codec-gate")
cache = modal.Volume.from_name("tiny-iwm-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/cache": cache},
    timeout=900,
    cpu=4,
)
def prepare_assets() -> dict[str, str]:
    import cv2
    from huggingface_hub import hf_hub_download, snapshot_download
    import numpy as np
    from PIL import Image

    vae_snapshot = Path(
        snapshot_download(
            "Efficient-Large-Model/SANA-WM_streaming",
            allow_patterns=["ltx2_causal_vae/*"],
            cache_dir="/cache/huggingface",
        )
    )
    vae_path = vae_snapshot / "ltx2_causal_vae"
    dataset_repo = "Efficient-Large-Model/SANA-WM-Bench"
    image_path = Path(
        hf_hub_download(
            dataset_repo,
            "images/game_style_001.png",
            repo_type="dataset",
            cache_dir="/cache/huggingface",
        )
    )
    trajectory_path = Path(
        hf_hub_download(
            dataset_repo,
            "benchmark_v2_smooth_60s/sanawm_export_v2/game_style_001.npz",
            repo_type="dataset",
            cache_dir="/cache/huggingface",
        )
    )

    data_root = Path("/cache/m1-data")
    data_root.mkdir(parents=True, exist_ok=True)
    source = np.asarray(Image.open(image_path).convert("RGB"))
    source_h, source_w = source.shape[:2]
    side = min(source_h, source_w)
    top = (source_h - side) // 2
    left = (source_w - side) // 2
    frame_size = 128
    frame = np.asarray(
        Image.fromarray(source[top : top + side, left : left + side]).resize(
            (frame_size, frame_size), Image.Resampling.LANCZOS
        )
    )
    video_path = data_root / "game_style_001_hold_961.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 16.0, (frame_size, frame_size)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the validation MP4")
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for _ in range(961):
        writer.write(bgr)
    writer.release()

    trajectory = np.load(trajectory_path)
    c2w = np.asarray(trajectory["c2w"], dtype=np.float64)
    intrinsics = np.asarray(trajectory["intrinsics"], dtype=np.float64).copy()
    scale = frame_size / side
    intrinsics[:, 0, 0] *= scale
    intrinsics[:, 1, 1] *= scale
    intrinsics[:, 0, 2] = (intrinsics[:, 0, 2] - left) * scale
    intrinsics[:, 1, 2] = (intrinsics[:, 1, 2] - top) * scale
    camera_path = data_root / "game_style_001_camera.json"
    camera_path.write_text(
        json.dumps(
            {
                "c2w": c2w.tolist(),
                "intrinsics": intrinsics.tolist(),
                "timestamps_seconds": (np.arange(961, dtype=np.float64) / 16.0).tolist(),
            }
        ),
        encoding="utf-8",
    )
    metadata_path = data_root / "game_style_001_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source": dataset_repo,
                "scene": "game_style_001",
                "trajectory": "benchmark_v2_smooth_60s",
                "derivation": "official held-out first frame repeated for codec-only validation",
                "original_image_size_hw": [source_h, source_w],
                "validation_size_hw": [frame_size, frame_size],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_path = data_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_version": "SANA-WM-Bench@0e527925-game_style_001-hold-v1",
                "records": [
                    {
                        "sample_id": "game_style_001_hold_961",
                        "source": dataset_repo,
                        "scene_id": "game_style_001",
                        "split": "validation",
                        "data_version": "SANA-WM-Bench@0e527925-game_style_001-hold-v1",
                        "video_path": video_path.name,
                        "camera_path": camera_path.name,
                        "metadata_path": metadata_path.name,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    cache.commit()
    return {
        "vae_path": str(vae_path),
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/cache": cache},
    memory=65536,
    cpu=8,
)
def run_gate(assets: dict[str, str]) -> dict:
    import sys

    sys.path.insert(0, "/opt/tiny-iwm")
    os.chdir("/opt/tiny-iwm")
    cache.reload()

    os.environ.update(
        {
            "SANA_WM_VAE_PATH": assets["vae_path"],
            "SANA_WM_UPSTREAM_PATH": "/opt/Sana",
            "SANA_WM_CODEC_BACKEND": "cuda",
            "SANA_WM_CODEC_DTYPE": "bfloat16",
        }
    )
    from scripts.validate_m1_codec import main

    output = Path("/tmp/m1-codec-gate.json")
    exit_code = main(
        [
            "--manifest",
            assets["manifest_path"],
            "--data-root",
            assets["data_root"],
            "--codec-factory",
            "representations.sana_wm_streaming:create_official_sana_wm_streaming_codec",
            "--output",
            str(output),
            "--causality-atol",
            "0.00001",
            "--preprocessing-json",
            json.dumps(
                {
                    "color_space": "RGB",
                    "range": "0_to_1",
                    "resize": [128, 128],
                    "crop": "center_square",
                    "video_derivation": "held_out_first_frame_hold",
                },
                separators=(",", ":"),
            ),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    report["modal"] = {
        "gpu": os.environ.get("MODAL_GPU_TYPE", "A10G"),
        "sana_commit": SANA_COMMIT,
        "exit_code": exit_code,
    }
    if exit_code:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main(output: str = str(ROOT / "artifacts" / "m1" / "cwx-14-m1-codec-gate.json")):
    assets = prepare_assets.remote()
    report = run_gate.remote(assets)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(destination)
