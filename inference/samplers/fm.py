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
        accumulator_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        return torch.randn(
            shape, generator=generator, device=device, dtype=accumulator_dtype
        )

    def grid(self, steps: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if type(steps) is not int or steps <= 0:
            raise ValueError("FM sampling steps must be positive")
        # Keep the time coordinate distinct even for bfloat16 latent states.
        grid = torch.linspace(1, 0, steps + 1, device=device, dtype=torch.float32)
        if not torch.all(grid[:-1] > grid[1:]):
            raise ValueError("FM sampling grid loses distinct float32 timesteps")
        return grid

    def step(
        self, state: torch.Tensor, prediction: torch.Tensor,
        *, time: torch.Tensor, next_time: torch.Tensor,
    ) -> torch.Tensor:
        # euler_step contains the native FM update. Run it at time-grid
        # precision before rounding the stored latent state once per step.
        compute_dtype = torch.float64 if state.dtype == torch.float64 else torch.float32
        updated = euler_step(
            state.to(compute_dtype), prediction.to(compute_dtype),
            time=time, next_time=next_time,
        )
        return updated.to(state.dtype)
