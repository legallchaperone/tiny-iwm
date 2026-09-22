"""Factory for the pinned official SANA-WM streaming causal VAE.

The factory deliberately requires explicit local paths.  Downloading weights
or importing an arbitrary checkout is an orchestration concern, not a model
fallback hidden inside the training process.
"""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import os
from pathlib import Path
import sys

import torch
from torch import nn

from representations.base import CodecSpec, Representation
from representations.normalization import ChannelNormalizer, NormalizationStats
from representations.video_vae import SanaCausalVideoVAEAdapter, _backend_fingerprint


_OFFICIAL_WEIGHTS_SHA256 = (
    "3694526c1da964fcf1ebccb70edb09895bf9e0a723b3ac48e0b5adf5bf97cc4a"
)


class _OfficialSanaCausalVAE(nn.Module):
    """Expose the upstream Diffusers VAE through tensor-only methods."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        self.vae.clear_encoder_cache()
        model_input = video.mul(2.0).sub(1.0)
        posterior = self.vae.encode(model_input, causal=True).latent_dist
        return posterior.mode()

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.vae.clear_decoder_cache()
        decoded = self.vae.decode(
            latents, temb=None, causal=True, return_dict=False
        )[0]
        return decoded.add(1.0).div(2.0)


def create_official_sana_wm_streaming_codec() -> Representation:
    """Load the verified 128-channel, 8x32x32 causal LTX-2 codec.

    Required environment variables:

    ``SANA_WM_VAE_PATH``
        Directory containing ``config.json`` and the released safetensors file.
    ``SANA_WM_UPSTREAM_PATH``
        NVlabs/Sana checkout pinned by ``third_party/UPSTREAMS.yaml``.

    Optional variables select ``cuda``, ``mps``, or ``cpu`` and a floating
    dtype.  CUDA bfloat16 is the canonical released execution mode.
    """

    checkpoint = _required_directory("SANA_WM_VAE_PATH")
    upstream = _required_directory("SANA_WM_UPSTREAM_PATH")
    _verify_checkpoint(checkpoint)
    source = upstream / "diffusion/model/ltx2/causal_vae.py"
    if not source.is_file():
        raise RuntimeError(f"pinned SANA causal VAE source is missing: {source}")
    module_spec = importlib.util.spec_from_file_location(
        "_tiny_iwm_official_sana_causal_vae", source
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load official SANA causal VAE source: {source}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    vae_class = getattr(module, "AutoencoderKLCausalLTX2Video")
    backend = os.environ.get(
        "SANA_WM_CODEC_BACKEND", "cuda" if torch.cuda.is_available() else "mps"
    )
    if backend == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("SANA_WM_CODEC_BACKEND=mps but MPS is unavailable")
    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SANA_WM_CODEC_BACKEND=cuda but CUDA is unavailable")
    dtype_name = os.environ.get(
        "SANA_WM_CODEC_DTYPE", "bfloat16" if backend == "cuda" else "float16"
    )
    dtype = getattr(torch, dtype_name, None)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"unsupported SANA_WM_CODEC_DTYPE {dtype_name!r}")
    device = torch.device(backend)

    vae = vae_class.from_pretrained(
        checkpoint, torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    vae.requires_grad_(False)
    vae.enable_tiling(
        tile_sample_min_height=int(os.environ.get("SANA_WM_TILE_HEIGHT", "256")),
        tile_sample_min_width=int(os.environ.get("SANA_WM_TILE_WIDTH", "256")),
        tile_sample_min_num_frames=int(
            os.environ.get("SANA_WM_TILE_FRAMES", "96")
        ),
        tile_sample_stride_num_frames=int(
            os.environ.get("SANA_WM_TILE_STRIDE_FRAMES", "64")
        ),
    )

    mean = tuple(float(value) for value in vae.latents_mean.detach().float().cpu())
    std = tuple(float(value) for value in vae.latents_std.detach().float().cpu())
    normalization = NormalizationStats(
        mean=mean,
        std=std,
        training_data_version="SANA-WM_streaming@c7694938c854",
        # The upstream checkpoint does not publish fitting counts.  These
        # positive sentinels identify one published statistics vector rather
        # than claiming a reconstructed sample population.
        sample_count=1,
        value_count_per_channel=1,
    )
    spec = CodecSpec(
        codec_name="sana-wm-streaming-ltx2-causal-vae",
        weights_sha256=_OFFICIAL_WEIGHTS_SHA256,
        latent_channels=128,
        temporal_compression=8,
        spatial_compression=(32, 32),
        causal=True,
        encoding_policy="official_causal_mode_temporal_tile_96_stride_64_v1",
        execution_dtype=dtype_name,
        execution_backend=backend,
        backend_fingerprint=_backend_fingerprint(device),
        cuda_math_policy="strict_no_tf32_no_reduced_reduction_v1",
        normalization=normalization,
    )
    return SanaCausalVideoVAEAdapter(
        _OfficialSanaCausalVAE(vae),
        spec=spec,
        normalizer=ChannelNormalizer(normalization),
        snapshot_model=False,
    )


def _required_directory(variable: str) -> Path:
    raw = os.environ.get(variable)
    if not raw:
        raise RuntimeError(f"{variable} must name an explicit local directory")
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"{variable} is not a directory: {path}")
    return path


def _verify_checkpoint(checkpoint: Path) -> None:
    weights = checkpoint / "diffusion_pytorch_model.safetensors"
    config = checkpoint / "config.json"
    if not config.is_file() or not weights.is_file():
        raise RuntimeError("SANA_WM_VAE_PATH must contain config.json and safetensors")
    if os.environ.get("SANA_WM_VERIFY_WEIGHTS", "1") == "0":
        return
    digest = sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != _OFFICIAL_WEIGHTS_SHA256:
        raise RuntimeError("SANA-WM causal VAE weights do not match the pinned SHA-256")
