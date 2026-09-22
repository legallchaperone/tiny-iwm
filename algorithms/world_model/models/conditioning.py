"""Diffusion-time conditioning without camera-specific side channels."""

import math

import torch
from torch import nn


def sinusoidal_timestep_embedding(timestep: torch.Tensor, width: int) -> torch.Tensor:
    if timestep.ndim != 1:
        raise ValueError("timestep must have shape [B]")
    half = width // 2
    if half == 0:
        return timestep[:, None]
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timestep.to(torch.float32)[:, None] * frequencies[None]
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if width % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


class TimestepConditioner(nn.Module):
    def __init__(self, hidden_size: int, frequency_width: int = 256) -> None:
        super().__init__()
        self.frequency_width = frequency_width
        self.mlp = nn.Sequential(
            nn.Linear(frequency_width, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        embedding = sinusoidal_timestep_embedding(timestep, self.frequency_width)
        # Trigonometric features are intentionally computed in float32, then
        # cross the model precision boundary exactly once before the MLP.
        embedding = embedding.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(embedding)
