"""Wan-style modulated transformer block with named observation points."""

import torch
from torch import nn

from .attention import AttentionKV, JointSelfAttention
from .prope import TokenCameraProjection


OBSERVATION_POINTS = ("post_attention", "post_mlp", "block_output")


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        prope_camera_dims: int | None = None,
    ) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        if mlp_hidden <= 0:
            raise ValueError("mlp_ratio must produce a positive hidden width")
        self.attention_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.attention = JointSelfAttention(
            hidden_size,
            num_heads,
            qkv_bias=qkv_bias,
            prope_camera_dims=prope_camera_dims,
        )
        self.mlp_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        coordinates: torch.Tensor,
        visibility_mask: torch.Tensor | None = None,
        camera_projection: TokenCameraProjection | None = None,
        kv_cache: AttentionKV | None = None,
        return_kv_cache: bool = False,
    ) -> (
        tuple[torch.Tensor, dict[str, torch.Tensor]]
        | tuple[torch.Tensor, dict[str, torch.Tensor], AttentionKV]
    ):
        modulation = self.modulation(condition)
        if modulation.ndim == 2:
            modulation = modulation.unsqueeze(1)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            modulation.chunk(6, dim=-1)
        )
        normalized = self.attention_norm(tokens) * (1 + scale_attn) + shift_attn
        attended = self.attention(
            normalized,
            coordinates,
            visibility_mask,
            camera_projection,
            kv_cache=kv_cache,
            return_kv_cache=return_kv_cache,
        )
        if return_kv_cache:
            attended, updated_cache = attended
        tokens = tokens + gate_attn * attended
        post_attention = tokens
        normalized = self.mlp_norm(tokens) * (1 + scale_mlp) + shift_mlp
        tokens = tokens + gate_mlp * self.mlp(normalized)
        observations = {
            "post_attention": post_attention,
            "post_mlp": tokens,
            "block_output": tokens,
        }
        if return_kv_cache:
            return tokens, observations, updated_cache
        return tokens, observations
