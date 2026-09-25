"""Minimal discrete cosine diffusion-forcing objective with epsilon targets."""

from dataclasses import dataclass
from math import cos, pi

import torch

from algorithms.world_model.training_policies import TrainingPolicy
from core.types import ObjectiveBatch, VideoBatch


@dataclass(frozen=True)
class CosineDFSchedule:
    train_steps: int = 1000
    offset: float = 0.008
    alpha_floor: float = 1e-5

    def __post_init__(self) -> None:
        if type(self.train_steps) is not int or self.train_steps <= 1:
            raise ValueError("DF train_steps must be an integer greater than one")
        if not 0 <= self.offset < 1 or not 0 < self.alpha_floor < 1:
            raise ValueError("invalid DF cosine schedule parameters")

    def alpha_bar(self, time: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(time) or not torch.isfinite(time).all() or torch.any(
            (time < 0) | (time > 1)
        ):
            raise ValueError("DF time must be finite within [0, 1]")
        angle = ((time.to(torch.float32) + self.offset) / (1 + self.offset)) * (pi / 2)
        origin = cos(self.offset / (1 + self.offset) * pi / 2) ** 2
        alpha = (angle.cos().square() / origin).clamp(self.alpha_floor, 1)
        return torch.where(time == 0, torch.ones_like(alpha), alpha)


class CosineDFObjective:
    name = "minimal_df"
    prediction_type = "epsilon"

    def __init__(self, schedule: CosineDFSchedule, policy: TrainingPolicy) -> None:
        if policy.name != "teacher_forced_causal":
            raise ValueError("minimal DF requires teacher_forced_causal policy")
        self.schedule = schedule
        self.policy = policy

    def build_batch(
        self,
        video: VideoBatch,
        *,
        target_chunk: int,
        noise: torch.Tensor | None = None,
        timestep_indices: torch.Tensor | None = None,
        valid_latent_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> ObjectiveBatch:
        clean = video.latents
        selection = self.policy.select_target(
            video, target_chunk=target_chunk, valid_latent_mask=valid_latent_mask
        )
        if noise is None:
            noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        if (noise.shape != clean.shape or noise.device != clean.device
                or noise.dtype != clean.dtype or not torch.isfinite(noise).all()):
            raise ValueError("DF noise must be finite and match clean latents, device, and dtype")
        token_count = len(video.layout.token_to_latent_ranges)
        if timestep_indices is None:
            timestep_indices = torch.randint(
                1, self.schedule.train_steps + 1, (clean.shape[0], token_count),
                device=clean.device, generator=generator,
            )
        if (
            timestep_indices.shape != (clean.shape[0], token_count)
            or timestep_indices.device != clean.device
            or timestep_indices.dtype not in (torch.int32, torch.int64)
            or torch.any((timestep_indices < 1) | (timestep_indices > self.schedule.train_steps))
        ):
            raise ValueError("DF timestep_indices must be [B, temporal tokens] within the schedule")
        model_time = torch.zeros(
            (clean.shape[0], token_count), device=clean.device, dtype=torch.float32
        )
        alpha = torch.ones((clean.shape[0], clean.shape[2]), device=clean.device, dtype=clean.dtype)
        for index, latent_range in enumerate(video.layout.token_to_latent_ranges):
            selected = selection.target_time[latent_range.start:latent_range.stop]
            if selected.any() and not selected.all():
                raise ValueError("DF temporal patch mixes clean and noisy latents")
            if selected.all():
                time = timestep_indices[:, index] / self.schedule.train_steps
                model_time[:, index] = time
                alpha[:, latent_range.start:latent_range.stop] = self.schedule.alpha_bar(time).to(clean.dtype)[:, None]
        coefficient = alpha[:, None, :, None, None]
        noisy = coefficient.sqrt() * clean + (1 - coefficient).sqrt() * noise
        model_input = torch.where(
            selection.valid_mask,
            torch.where(selection.target_mask, noisy, clean),
            torch.zeros_like(clean),
        )
        target = torch.where(selection.loss_mask, noise, torch.zeros_like(noise))
        return ObjectiveBatch(
            model_input=model_input,
            noise_condition=model_time,
            prediction_target=target,
            prediction_type=self.prediction_type,
            loss_mask=selection.loss_mask,
            attention_visibility=selection.visibility,
            layout=video.layout,
            metadata={
                "objective": self.name,
                "policy": self.policy.name,
                "target_chunk": target_chunk,
                "timestep_indices": timestep_indices.detach().cpu().tolist(),
                "schedule": "cosine_discrete_v1",
            },
        )

    def loss(self, prediction: torch.Tensor, batch: ObjectiveBatch) -> torch.Tensor:
        if batch.prediction_type != self.prediction_type:
            raise ValueError("DF loss requires epsilon predictions")
        if prediction.shape != batch.prediction_target.shape:
            raise ValueError("DF prediction must match epsilon target shape")
        mask = batch.loss_mask.expand_as(prediction)
        if not mask.any():
            raise ValueError("DF loss mask must select target elements")
        return (prediction - batch.prediction_target).square().masked_select(mask).mean()
