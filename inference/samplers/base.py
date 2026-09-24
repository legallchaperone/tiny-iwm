"""Small sampler protocol used by the shared rollout orchestration."""

from typing import Protocol

import torch


class Sampler(Protocol):
    name: str
    objective_name: str
    prediction_type: str

    def initial_state(
        self, shape: tuple[int, ...], *, device: torch.device,
        dtype: torch.dtype, generator: torch.Generator,
    ) -> torch.Tensor: ...

    def grid(self, steps: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor: ...

    def step(
        self, state: torch.Tensor, prediction: torch.Tensor,
        *, time: torch.Tensor, next_time: torch.Tensor,
    ) -> torch.Tensor: ...
