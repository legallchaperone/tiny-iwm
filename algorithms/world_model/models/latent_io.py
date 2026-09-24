"""Shape-safe latent patching for the joint video DiT.

The adapter owns the trainable latent projections.  It pads only the high end
of each axis, records that padding, and crops it after unpatching so callers do
not need lengths that happen to divide the patch size.
"""

from dataclasses import dataclass
from math import prod

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PatchLayout:
    original_shape: tuple[int, int, int]
    grid_shape: tuple[int, int, int]
    patch_size: tuple[int, int, int]

    @property
    def token_count(self) -> int:
        return prod(self.grid_shape)


class LatentPatchIO(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        hidden_size: int,
        patch_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        if latent_channels <= 0 or hidden_size <= 0:
            raise ValueError("latent_channels and hidden_size must be positive")
        if len(patch_size) != 3 or any(size <= 0 for size in patch_size):
            raise ValueError("patch_size must contain three positive integers")
        self.latent_channels = latent_channels
        self.patch_size = tuple(patch_size)
        self.input_projection = nn.Conv3d(
            latent_channels,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.output_projection = nn.Linear(
            hidden_size, latent_channels * prod(self.patch_size)
        )

    def patchify(self, latents: torch.Tensor) -> tuple[torch.Tensor, PatchLayout]:
        if latents.ndim != 5:
            raise ValueError("latents must use [B, C, T, H, W]")
        if latents.shape[1] != self.latent_channels:
            raise ValueError("latent channel count does not match the model")
        original = tuple(int(size) for size in latents.shape[2:])
        padding = tuple((-size) % patch for size, patch in zip(original, self.patch_size))
        # torch padding order is W, H, T.
        padded = F.pad(latents, (0, padding[2], 0, padding[1], 0, padding[0]))
        projected = self.input_projection(padded)
        grid = tuple(int(size) for size in projected.shape[2:])
        tokens = projected.flatten(2).transpose(1, 2)
        return tokens, PatchLayout(original, grid, self.patch_size)

    def unpatchify(self, tokens: torch.Tensor, layout: PatchLayout) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must use [B, N, D]")
        if tokens.shape[1] != layout.token_count:
            raise ValueError("token count does not match PatchLayout")
        patches = self.output_projection(tokens)
        batch = patches.shape[0]
        gt, gh, gw = layout.grid_shape
        pt, ph, pw = layout.patch_size
        patches = patches.view(batch, gt, gh, gw, self.latent_channels, pt, ph, pw)
        latents = patches.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
            batch, self.latent_channels, gt * pt, gh * ph, gw * pw
        )
        time, height, width = layout.original_shape
        return latents[:, :, :time, :height, :width]


def build_latent_io(
    kind: str,
    latent_channels: int,
    hidden_size: int,
    patch_size: tuple[int, int, int],
) -> LatentPatchIO:
    if kind != "conv3d_patch":
        raise ValueError(f"unsupported latent I/O component: {kind}")
    return LatentPatchIO(latent_channels, hidden_size, patch_size)
