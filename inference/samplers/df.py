"""Deterministic DDIM update for the minimal cosine epsilon DF recipe."""

from dataclasses import dataclass, field

import torch

from algorithms.world_model.objectives.df import CosineDFSchedule


@dataclass(frozen=True)
class DFDDIMSampler:
    schedule: CosineDFSchedule = field(default_factory=CosineDFSchedule)
    name: str = "df_ddim"
    objective_name: str = "minimal_df"
    prediction_type: str = "epsilon"

    def initial_state(
        self, shape: tuple[int, ...], *, device: torch.device,
        dtype: torch.dtype, generator: torch.Generator,
    ) -> torch.Tensor:
        # The schedule floors alpha at t=1, so pure normal noise is an
        # approximation of the terminal marginal for this minimal recipe.
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def grid(self, steps: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if type(steps) is not int or not 0 < steps <= self.schedule.train_steps:
            raise ValueError("DF sampling steps must be within the training schedule")
        # Each chosen time lies on the discrete training grid.
        indices = torch.linspace(
            self.schedule.train_steps, 0, steps + 1, device=device
        ).round().to(torch.int64)
        if not torch.all(indices[:-1] > indices[1:]):
            raise ValueError("DF sampling grid must have strictly decreasing timesteps")
        # Time conditioning must retain the discrete training levels even when
        # the latent state is bfloat16. The model accepts fp32 timestep input.
        grid = indices.to(torch.float32) / self.schedule.train_steps
        if not torch.all(grid[:-1] > grid[1:]):
            raise ValueError("DF sampling grid loses distinct timesteps in float32")
        return grid

    def step(
        self, state: torch.Tensor, prediction: torch.Tensor,
        *, time: torch.Tensor, next_time: torch.Tensor,
    ) -> torch.Tensor:
        if state.shape != prediction.shape or time.shape != (state.shape[0],) or next_time.shape != time.shape:
            raise ValueError("DDIM state, epsilon prediction, and time shapes must align")
        if torch.any(next_time >= time):
            raise ValueError("DDIM requires decreasing timesteps")
        # BF16 schedule coefficients can coincide at adjacent timesteps even
        # when their fp32 values differ. Perform the DDIM update in fp32, then
        # round only the resulting latent state back to its storage dtype.
        compute_dtype = torch.float64 if state.dtype == torch.float64 else torch.float32
        alpha = self.schedule.alpha_bar(time).to(device=state.device, dtype=compute_dtype)
        next_alpha = self.schedule.alpha_bar(next_time).to(device=state.device, dtype=compute_dtype)
        shape = (state.shape[0],) + (1,) * (state.ndim - 1)
        alpha = alpha.reshape(shape)
        next_alpha = next_alpha.reshape(shape)
        state_fp = state.to(compute_dtype)
        prediction_fp = prediction.to(compute_dtype)
        clean_estimate = (state_fp - (1 - alpha).sqrt() * prediction_fp) / alpha.sqrt()
        updated = next_alpha.sqrt() * clean_estimate + (1 - next_alpha).sqrt() * prediction_fp
        return updated.to(state.dtype)
