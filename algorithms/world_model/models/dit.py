"""One configurable joint spatiotemporal diffusion transformer."""

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .blocks import DiTBlock, OBSERVATION_POINTS
from .conditioning import TimestepConditioner
from .latent_io import LatentPatchIO
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

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.hidden_size <= 0 or self.num_heads <= 0:
            raise ValueError("hidden_size and num_heads must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")


class JointVideoDiT(nn.Module):
    """Predict latent velocity with one model for all visibility patterns.

    The module never loads a checkpoint.  Stage A, Stage B, and inference use
    this same class and select their semantics with ``visibility_mask``.
    """

    GLOBAL_OBSERVATION_POINTS = ("patch_tokens", "final_norm", "velocity_tokens")

    def __init__(self, config: JointVideoDiTConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_io = LatentPatchIO(
            config.latent_channels, config.hidden_size, config.patch_size
        )
        self.timestep = TimestepConditioner(config.hidden_size)
        self.blocks = nn.ModuleList(
            DiTBlock(
                config.hidden_size,
                config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                prope_camera_dims=config.prope_camera_dims,
            )
            for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
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
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if timestep.shape != (latents.shape[0],):
            raise ValueError("timestep must have shape [B]")
        requested = frozenset(capture)
        unknown = requested.difference(self.observation_points)
        if unknown:
            raise ValueError(f"unknown observation points: {sorted(unknown)}")

        tokens, layout = self.latent_io.patchify(latents)
        coordinates = token_coordinates(
            layout.grid_shape, time_offset=time_offset, device=tokens.device
        )
        if camera_projection is not None and camera_projection.token_count != tokens.shape[1]:
            raise ValueError("camera projection token count does not match patched latents")
        condition = self.timestep(timestep).to(dtype=tokens.dtype)
        observations: dict[str, torch.Tensor] = {}
        if "patch_tokens" in requested:
            observations["patch_tokens"] = tokens

        for index, block in enumerate(self.blocks):
            tokens, block_observations = block(
                tokens, condition, coordinates, visibility_mask, camera_projection
            )
            for point, value in block_observations.items():
                name = f"blocks.{index}.{point}"
                if name in requested:
                    observations[name] = value

        shift, scale = self.final_modulation(condition).unsqueeze(1).chunk(2, dim=-1)
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
        velocity = velocity_tokens.view(
            batch, gt, gh, gw, self.config.latent_channels, pt, ph, pw
        ).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
            batch,
            self.config.latent_channels,
            gt * pt,
            gh * ph,
            gw * pw,
        )
        time, height, width = layout.original_shape
        velocity = velocity[:, :, :time, :height, :width]
        if requested:
            return velocity, observations
        return velocity
