"""One configurable joint spatiotemporal diffusion transformer."""

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .blocks import OBSERVATION_POINTS, _norm, build_block
from .attention import AttentionKV
from .conditioning import build_conditioner
from .latent_io import build_latent_io
from .position import token_coordinates
from .prope import TokenCameraProjection


@dataclass(frozen=True)
class JointVideoDiTConfig:
    latent_channels: int
    hidden_size: int
    depth: int
    num_heads: int
    patch_size: tuple[int, int, int]
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    prope_camera_dims: int | None = None
    block_kind: str = "dit_modulated"
    attention_kind: str = "joint_spatiotemporal_softmax"
    ffn_kind: str = "gelu_tanh"
    norm_kind: str = "layer_norm"
    position_kind: str = "rope_3d"
    conditioner_kind: str = "sinusoidal_timestep"
    latent_io_kind: str = "conv3d_patch"

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.hidden_size <= 0 or self.num_heads <= 0:
            raise ValueError("hidden_size and num_heads must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        supported = {
            "block_kind": {"dit_modulated"},
            "attention_kind": {"joint_spatiotemporal_softmax"},
            "ffn_kind": {"gelu_tanh", "swiglu"},
            "norm_kind": {"layer_norm", "rms_norm"},
            "position_kind": {"rope_3d", "none"},
            "conditioner_kind": {"sinusoidal_timestep"},
            "latent_io_kind": {"conv3d_patch"},
        }
        for field, choices in supported.items():
            if getattr(self, field) not in choices:
                raise ValueError(f"unsupported {field}: {getattr(self, field)!r}; choose {sorted(choices)}")


class JointVideoDiT(nn.Module):
    """Predict latent velocity with one model for all visibility patterns.

    The module never loads a checkpoint.  Stage A, Stage B, and inference use
    this same class and select their semantics with ``visibility_mask``.
    """

    GLOBAL_OBSERVATION_POINTS = ("patch_tokens", "final_norm", "velocity_tokens")

    def __init__(self, config: JointVideoDiTConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_io = build_latent_io(
            config.latent_io_kind, config.latent_channels,
            config.hidden_size, config.patch_size,
        )
        self.timestep = build_conditioner(config.conditioner_kind, config.hidden_size)
        self.blocks = nn.ModuleList(
            build_block(
                config.block_kind,
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                prope_camera_dims=config.prope_camera_dims,
                ffn_kind=config.ffn_kind,
                norm_kind=config.norm_kind,
                position_kind=config.position_kind,
                attention_kind=config.attention_kind,
            )
            for _ in range(config.depth)
        )
        self.final_norm = _norm(config.norm_kind, config.hidden_size)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_size, config.hidden_size * 2)
        )
        self.reset_parameters()

    @property
    def observation_points(self) -> tuple[str, ...]:
        per_block = tuple(
            f"blocks.{index}.{point}"
            for index in range(self.config.depth)
            for point in OBSERVATION_POINTS
        )
        return self.GLOBAL_OBSERVATION_POINTS + per_block

    def reset_parameters(self) -> None:
        """Randomly initialize this DiT without any pretrained-weight path."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv3d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        visibility_mask: torch.Tensor | None = None,
        time_offset: int = 0,
        capture: Iterable[str] = (),
        camera_projection: TokenCameraProjection | None = None,
        kv_cache: tuple[AttentionKV, ...] | None = None,
        return_kv_cache: bool = False,
    ) -> torch.Tensor | tuple:
        if timestep.shape != (latents.shape[0],) and (
            timestep.ndim != 2 or timestep.shape[0] != latents.shape[0]
        ):
            raise ValueError("timestep must have shape [B] or [B, N]")
        requested = frozenset(capture)
        unknown = requested.difference(self.observation_points)
        if unknown:
            raise ValueError(f"unknown observation points: {sorted(unknown)}")

        tokens, layout = self.latent_io.patchify(latents)
        if timestep.ndim == 2:
            if timestep.shape[1] == layout.grid_shape[0]:
                timestep = timestep.repeat_interleave(
                    layout.grid_shape[1] * layout.grid_shape[2], dim=1
                )
            if timestep.shape[1] != tokens.shape[1]:
                raise ValueError("token timestep count must match patched latents")
        coordinates = token_coordinates(
            layout.grid_shape, time_offset=time_offset, device=tokens.device
        )
        if (
            camera_projection is not None
            and camera_projection.token_count != tokens.shape[1]
        ):
            raise ValueError(
                "camera projection token count does not match patched latents"
            )
        if kv_cache is not None and len(kv_cache) != len(self.blocks):
            raise ValueError("KV cache must contain one entry per block")
        condition = self.timestep(timestep).to(dtype=tokens.dtype)
        observations: dict[str, torch.Tensor] = {}
        if "patch_tokens" in requested:
            observations["patch_tokens"] = tokens

        updated_caches = []
        for index, block in enumerate(self.blocks):
            result = block(
                tokens,
                condition,
                coordinates,
                visibility_mask,
                camera_projection,
                kv_cache=None if kv_cache is None else kv_cache[index],
                return_kv_cache=return_kv_cache,
            )
            if return_kv_cache:
                tokens, block_observations, updated_cache = result
                updated_caches.append(updated_cache)
            else:
                tokens, block_observations = result
            for point, value in block_observations.items():
                name = f"blocks.{index}.{point}"
                if name in requested:
                    observations[name] = value

        modulation = self.final_modulation(condition)
        if modulation.ndim == 2:
            modulation = modulation.unsqueeze(1)
        shift, scale = modulation.chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + scale) + shift
        if "final_norm" in requested:
            observations["final_norm"] = tokens
        velocity_tokens = self.latent_io.output_projection(tokens)
        if "velocity_tokens" in requested:
            observations["velocity_tokens"] = velocity_tokens
        # Avoid applying output_projection twice while retaining one LatentPatchIO owner.
        batch = velocity_tokens.shape[0]
        gt, gh, gw = layout.grid_shape
        pt, ph, pw = layout.patch_size
        velocity = (
            velocity_tokens.view(
                batch, gt, gh, gw, self.config.latent_channels, pt, ph, pw
            )
            .permute(0, 4, 1, 5, 2, 6, 3, 7)
            .reshape(
                batch,
                self.config.latent_channels,
                gt * pt,
                gh * ph,
                gw * pw,
            )
        )
        time, height, width = layout.original_shape
        velocity = velocity[:, :, :time, :height, :width]
        if return_kv_cache:
            if requested:
                return velocity, observations, tuple(updated_caches)
            return velocity, tuple(updated_caches)
        if requested:
            return velocity, observations
        return velocity
