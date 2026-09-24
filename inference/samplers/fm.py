"""Euler integration for native velocity Flow Matching."""

from dataclasses import dataclass

import torch

from algorithms.world_model.flow import euler_step


@dataclass(frozen=True)
class FMEulerSampler:
    name: str = "fm_euler"
    objective_name: str = "native_fm"
    prediction_type: str = "velocity"

    def initial_state(
        self, shape: tuple[int, ...], *, device: torch.device,
        dtype: torch.dtype, generator: torch.Generator,
    ) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def grid(self, steps: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if steps <= 0:
            raise ValueError("FM sampling steps must be positive")
        return torch.linspace(1, 0, steps + 1, device=device, dtype=dtype)

    def step(
        self, state: torch.Tensor, prediction: torch.Tensor,
        *, time: torch.Tensor, next_time: torch.Tensor,
    ) -> torch.Tensor:
        return euler_step(state, prediction, time=time, next_time=next_time)
