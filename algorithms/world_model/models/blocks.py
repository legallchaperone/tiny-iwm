"""Wan-style modulated transformer block with named observation points."""

import torch
from torch import nn

from .attention import AttentionKV, build_attention
from .prope import TokenCameraProjection


OBSERVATION_POINTS = ("post_attention", "post_mlp", "block_output")


class SwiGLUFFN(nn.Module):
    """Gated feed-forward alternative with its own checkpoint identity."""

    def __init__(self, hidden_size: int, inner_size: int) -> None:
        super().__init__()
        self.gate_value = nn.Linear(hidden_size, inner_size * 2)
        self.output = nn.Linear(inner_size, hidden_size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, content = self.gate_value(value).chunk(2, dim=-1)
        return self.output(torch.nn.functional.silu(gate) * content)


def _norm(kind: str, hidden_size: int) -> nn.Module:
    if kind == "layer_norm":
        return nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    if kind == "rms_norm":
        return nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    raise ValueError(f"unsupported norm component: {kind}")


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        prope_camera_dims: int | None = None,
        ffn_kind: str = "gelu_tanh",
        norm_kind: str = "layer_norm",
        position_kind: str = "rope_3d",
        attention_kind: str = "joint_spatiotemporal_softmax",
    ) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        if mlp_hidden <= 0:
            raise ValueError("mlp_ratio must produce a positive hidden width")
        self.attention_norm = _norm(norm_kind, hidden_size)
        self.attention = build_attention(
            attention_kind,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            prope_camera_dims=prope_camera_dims,
            position_kind=position_kind,
        )
        self.mlp_norm = _norm(norm_kind, hidden_size)
        self.mlp = build_ffn(ffn_kind, hidden_size, mlp_hidden)
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


def build_ffn(kind: str, hidden_size: int, inner_size: int) -> nn.Module:
    if kind == "gelu_tanh":
        return nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner_size, hidden_size),
        )
    if kind == "swiglu":
        return SwiGLUFFN(hidden_size, inner_size)
    raise ValueError(f"unsupported FFN component: {kind}")


def build_block(kind: str, **kwargs) -> DiTBlock:
    if kind != "dit_modulated":
        raise ValueError(f"unsupported block component: {kind}")
    return DiTBlock(**kwargs)
