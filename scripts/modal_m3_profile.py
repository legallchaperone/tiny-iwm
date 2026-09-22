"""Run CWX-19 GPU-only profiling on Modal; no asset download is needed."""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0")
    .add_local_dir(ROOT, remote_path="/opt/tiny-iwm")
)
app = modal.App("tiny-iwm-cwx-19-full-profile")


@app.function(image=image, gpu="A10G", timeout=1800, memory=65536, cpu=8)
def run_profile() -> str:
    import sys

    sys.path.insert(0, "/opt/tiny-iwm")
    os.chdir("/opt/tiny-iwm")
    from scripts.profile_m3_full_length import profile_cuda

    return json.dumps(
        profile_cuda(warmup_steps=1, measured_steps=3, sampling_steps=16),
        sort_keys=True,
    )


@app.local_entrypoint()
def main(
    output: str = str(ROOT / "artifacts" / "m3" / "cwx-19-full-length-profile.json"),
):
    report = json.loads(run_profile.remote())
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(destination)
