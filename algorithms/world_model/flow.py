"""The single native Flow Matching convention used by training and sampling."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlowMatchSpec:
    """Explicit time sampling and loss weighting for native linear FM."""

    time_distribution: str = "uniform"
    time_min: float = 0.0
    time_max: float = 1.0
    loss_weighting: str = "uniform"

    def __post_init__(self) -> None:
        if self.time_distribution != "uniform":
            raise ValueError("only explicit uniform FM time sampling is supported")
        if not 0.0 <= self.time_min < self.time_max <= 1.0:
            raise ValueError("FM time bounds must satisfy 0 <= min < max <= 1")
        if self.loss_weighting != "uniform":
            raise ValueError("only explicit uniform FM loss weighting is supported")

    def sample_time(
        self,
        batch_size: int,
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        unit = torch.rand(batch_size, device=device, generator=generator)
        return self.time_min + unit * (self.time_max - self.time_min)

    def loss_weights(self, time: torch.Tensor) -> torch.Tensor:
        _validate_time(time)
        return torch.ones_like(time)


def interpolate(clean: torch.Tensor, noise: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Return ``z_t = (1-t) * clean + t * noise`` with t=0 clean."""
    _validate_pair(clean, noise)
    coefficient = _broadcast_time(time, clean)
    return (1 - coefficient) * clean + coefficient * noise


def target_velocity(clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Return the constant native-FM velocity ``noise - clean``."""
    _validate_pair(clean, noise)
    return noise - clean


def euler_step(
    state: torch.Tensor,
    velocity: torch.Tensor,
    *,
    time: torch.Tensor,
    next_time: torch.Tensor,
) -> torch.Tensor:
    """Integrate either direction; sampling passes decreasing times from 1 to 0."""
    _validate_pair(state, velocity)
    step = _broadcast_time(next_time, state) - _broadcast_time(time, state)
    return state + step * velocity


def flow_matching_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_mask: torch.Tensor,
    time: torch.Tensor,
    spec: FlowMatchSpec,
) -> torch.Tensor:
    """Masked per-element MSE with the spec's explicit time weighting."""
    _validate_pair(prediction, target)
    if loss_mask.dtype != torch.bool:
        raise ValueError("loss_mask must be boolean")
    try:
        expanded_mask = loss_mask.expand_as(prediction)
    except RuntimeError as error:
        raise ValueError("loss_mask must broadcast to the prediction shape") from error
    if not expanded_mask.any():
        raise ValueError("loss_mask must select at least one prediction element")
    weights = _broadcast_time(spec.loss_weights(time), prediction)
    squared_error = (prediction - target).square() * weights
    return squared_error.masked_select(expanded_mask).mean()


def _broadcast_time(time: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    _validate_time(time)
    if time.shape != (reference.shape[0],):
        raise ValueError("FM time must have shape [B]")
    return time.to(device=reference.device, dtype=reference.dtype).view(
        reference.shape[0], *([1] * (reference.ndim - 1))
    )


def _validate_time(time: torch.Tensor) -> None:
    if not isinstance(time, torch.Tensor) or not time.is_floating_point() or time.ndim != 1:
        raise ValueError("FM time must be a floating tensor with shape [B]")
    if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
        raise ValueError("FM time must be finite and within [0, 1]")


def _validate_pair(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.shape != right.shape:
        raise ValueError("Flow Matching tensors must have identical shapes")
    if not left.is_floating_point() or not right.is_floating_point():
        raise ValueError("Flow Matching tensors must be floating point")
    if left.device != right.device or left.dtype != right.dtype:
        raise ValueError("Flow Matching tensors must share device and dtype")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Flow Matching tensors must be finite")
