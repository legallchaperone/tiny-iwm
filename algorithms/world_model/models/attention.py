"""Joint spatiotemporal self-attention shared by bidirectional and causal use."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .position import apply_position
from .prope import TokenCameraProjection, apply_camera_projection


@dataclass(frozen=True)
class AttentionKV:
    """Post-RoPE/PRoPE keys and values from committed clean history."""

    key: torch.Tensor
    value: torch.Tensor


class JointSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        prope_camera_dims: int | None = None,
        position_kind: str = "rope_3d",
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.prope_camera_dims = prope_camera_dims
        if position_kind not in {"rope_3d", "none"}:
            raise ValueError(f"unsupported position component: {position_kind}")
        self.position_kind = position_kind
        if prope_camera_dims is not None and not 0 < prope_camera_dims <= self.head_dim:
            raise ValueError("prope_camera_dims must fit within each attention head")
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        tokens: torch.Tensor,
        coordinates: torch.Tensor,
        visibility_mask: torch.Tensor | None = None,
        camera_projection: TokenCameraProjection | None = None,
        kv_cache: AttentionKV | None = None,
        return_kv_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionKV]:
        batch, token_count, hidden = tokens.shape
        qkv = self.qkv(tokens).view(
            batch, token_count, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        query, key = apply_position(self.position_kind, query, key, coordinates)
        if camera_projection is not None:
            if self.prope_camera_dims is None:
                raise ValueError(
                    "camera projection requires configured prope_camera_dims"
                )
            query = apply_camera_projection(
                query, camera_projection.transpose, camera_dims=self.prope_camera_dims
            )
            key = apply_camera_projection(
                key, camera_projection.inverse, camera_dims=self.prope_camera_dims
            )
            value = apply_camera_projection(
                value, camera_projection.inverse, camera_dims=self.prope_camera_dims
            )
        if kv_cache is not None:
            if (
                kv_cache.key.shape[:2] != (batch, self.num_heads)
                or kv_cache.key.shape[-1] != self.head_dim
            ):
                raise ValueError("KV cache does not match attention batch/head layout")
            if kv_cache.key.shape != kv_cache.value.shape:
                raise ValueError("cached keys and values must have identical shapes")
            if kv_cache.key.device != key.device or kv_cache.key.dtype != key.dtype:
                raise ValueError(
                    "KV cache device and dtype must match the current tokens"
                )
            key = torch.cat((kv_cache.key, key), dim=2)
            value = torch.cat((kv_cache.value, value), dim=2)
        if kv_cache is not None and visibility_mask is not None:
            raise ValueError("cached attention accepts only an unmasked clean prefix")
        mask = _attention_mask(visibility_mask, batch, token_count, tokens.device)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        if camera_projection is not None:
            attended = apply_camera_projection(
                attended,
                camera_projection.projection,
                camera_dims=self.prope_camera_dims,
            )
        attended = attended.transpose(1, 2).reshape(batch, token_count, hidden)
        output = self.output(attended)
        if return_kv_cache:
            return output, AttentionKV(key, value)
        return output


def build_attention(kind: str, **kwargs) -> JointSelfAttention:
    if kind != "joint_spatiotemporal_softmax":
        raise ValueError(f"unsupported attention component: {kind}")
    return JointSelfAttention(**kwargs)


def _attention_mask(
    mask: torch.Tensor | None,
    batch: int,
    token_count: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Validate an allowed-edge mask; ``True`` means visible."""
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        raise ValueError("visibility_mask must be boolean with True meaning visible")
    if mask.ndim == 2:
        if mask.shape != (token_count, token_count):
            raise ValueError("2D visibility_mask must have shape [N, N]")
        return mask.to(device=device)[None, None]
    if mask.ndim == 3:
        if mask.shape != (batch, token_count, token_count):
            raise ValueError("3D visibility_mask must have shape [B, N, N]")
        return mask.to(device=device)[:, None]
    raise ValueError("visibility_mask must have shape [N, N] or [B, N, N]")
