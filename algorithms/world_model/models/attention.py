"""Joint spatiotemporal self-attention shared by bidirectional and causal use."""

import torch
from torch import nn
from torch.nn import functional as F

from .position import apply_3d_rope


class JointSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        tokens: torch.Tensor,
        coordinates: torch.Tensor,
        visibility_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, token_count, hidden = tokens.shape
        qkv = self.qkv(tokens).view(batch, token_count, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        query, key = apply_3d_rope(query, key, coordinates)
        mask = _attention_mask(visibility_mask, batch, token_count, tokens.device)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        attended = attended.transpose(1, 2).reshape(batch, token_count, hidden)
        return self.output(attended)


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

