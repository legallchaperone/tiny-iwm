"""Variable-length three-dimensional rotary position encoding."""

import torch


def apply_position(
    kind: str,
    query: torch.Tensor,
    key: torch.Tensor,
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kind == "rope_3d":
        return apply_3d_rope(query, key, coordinates)
    if kind == "none":
        return query, key
    raise ValueError(f"unsupported position component: {kind}")


def token_coordinates(
    grid_shape: tuple[int, int, int],
    *,
    time_offset: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return flattened ``(time, row, column)`` coordinates for a patch grid."""
    if len(grid_shape) != 3 or any(size <= 0 for size in grid_shape):
        raise ValueError("grid_shape must contain three positive integers")
    time = torch.arange(grid_shape[0], device=device) + time_offset
    row = torch.arange(grid_shape[1], device=device)
    column = torch.arange(grid_shape[2], device=device)
    return torch.stack(torch.meshgrid(time, row, column, indexing="ij"), dim=-1).reshape(-1, 3)


def apply_3d_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply separable 3D RoPE to ``[B, heads, tokens, head_dim]`` Q/K.

    Complete feature pairs are distributed across time, height, and width.
    Any final unpaired feature is intentionally left unchanged.
    """
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("query and key must have equal [B, H, N, D] shapes")
    if coordinates.shape != (query.shape[2], 3):
        raise ValueError("coordinates must have shape [N, 3]")
    pair_count = query.shape[-1] // 2
    if pair_count == 0:
        return query, key
    pair_axes = torch.arange(pair_count, device=query.device) % 3
    pair_indices = torch.arange(pair_count, device=query.device) // 3
    exponent = (2.0 * pair_indices.to(torch.float32)) / max(query.shape[-1], 1)
    inverse_frequency = torch.pow(
        torch.tensor(base, device=query.device, dtype=torch.float32), -exponent
    )
    positions = coordinates.to(device=query.device, dtype=torch.float32)[:, pair_axes]
    angles = positions * inverse_frequency
    cosine = angles.cos().to(dtype=query.dtype)[None, None]
    sine = angles.sin().to(dtype=query.dtype)[None, None]

    def rotate(value: torch.Tensor) -> torch.Tensor:
        paired = value[..., : pair_count * 2].reshape(*value.shape[:-1], pair_count, 2)
        first, second = paired.unbind(dim=-1)
        rotated = torch.stack(
            (first * cosine - second * sine, first * sine + second * cosine), dim=-1
        ).flatten(-2)
        return torch.cat((rotated, value[..., pair_count * 2 :]), dim=-1)

    return rotate(query), rotate(key)
