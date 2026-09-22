"""Stage A bidirectional training-batch semantics."""

from dataclasses import dataclass

import torch

from core.types import TrainingBatch, VideoBatch

from .flow import FlowMatchSpec, interpolate, target_velocity


@dataclass(frozen=True)
class AttentionVisibilitySpec:
    """Length-independent visibility semantics expanded at the model boundary."""

    mode: str

    def __post_init__(self) -> None:
        if self.mode != "bidirectional":
            raise ValueError("Stage A visibility must be bidirectional")


class StageABatchBuilder:
    """Mix target latents with noise while keeping known condition latents clean."""

    def __init__(self, flow_spec: FlowMatchSpec) -> None:
        self.flow_spec = flow_spec

    def build(
        self,
        batch: VideoBatch,
        *,
        noise: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
        valid_latent_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> TrainingBatch:
        clean = batch.latents
        if not isinstance(clean, torch.Tensor) or clean.ndim != 5:
            raise ValueError("Stage A requires latents with shape [B, C, T, H, W]")
        if clean.shape[2] != batch.layout.latent_frame_count:
            raise ValueError("latent time dimension must match VideoLayout")
        if not clean.is_floating_point() or not torch.isfinite(clean).all():
            raise ValueError("Stage A latents must be finite floating-point values")

        condition_time = _condition_latent_mask(batch)
        condition_mask = condition_time[None, None, :, None, None].expand(
            clean.shape[0], 1, clean.shape[2], clean.shape[3], clean.shape[4]
        )
        valid_mask = _valid_mask(clean, valid_latent_mask)
        loss_mask = valid_mask & ~condition_mask
        if not loss_mask.any():
            raise ValueError("Stage A batch has no valid target latents")

        if noise is None:
            noise = torch.randn(
                clean.shape,
                device=clean.device,
                dtype=clean.dtype,
                generator=generator,
            )
        elif noise.shape != clean.shape:
            raise ValueError("noise must match the clean latent shape")
        if flow_time is None:
            flow_time = self.flow_spec.sample_time(
                clean.shape[0], device=clean.device, generator=generator
            ).to(dtype=clean.dtype)
        else:
            self.flow_spec.loss_weights(flow_time)
            if torch.any(
                (flow_time < self.flow_spec.time_min)
                | (flow_time > self.flow_spec.time_max)
            ):
                raise ValueError("flow_time falls outside the configured sampling bounds")

        mixed_target = interpolate(clean, noise, flow_time)
        model_input = torch.where(condition_mask, clean, mixed_target)
        clean_condition = torch.where(condition_mask, clean, torch.zeros_like(clean))
        velocity = target_velocity(clean, noise)
        velocity = torch.where(loss_mask, velocity, torch.zeros_like(velocity))
        return TrainingBatch(
            noisy_latents=model_input,
            clean_condition_latents=clean_condition,
            flow_time=flow_time,
            target_velocity=velocity,
            loss_mask=loss_mask,
            attention_visibility=AttentionVisibilitySpec("bidirectional"),
            layout=batch.layout,
            metadata={
                "stage": "A",
                "attention": "bidirectional",
                "attention_visibility_axis": "latent_time",
                "flow_convention": "t0_clean_t1_noise",
                "time_distribution": self.flow_spec.time_distribution,
                "loss_weighting": self.flow_spec.loss_weighting,
                "condition_latent_indices": tuple(
                    condition_time.nonzero(as_tuple=False).flatten().tolist()
                ),
            },
        )


def _condition_latent_mask(batch: VideoBatch) -> torch.Tensor:
    boundary = batch.layout.initial_condition_rgb.stop
    mask: list[bool] = []
    for rgb_range in batch.layout.latent_to_rgb:
        if rgb_range.start < boundary < rgb_range.stop:
            raise ValueError("initial condition boundary must align with a latent boundary")
        mask.append(rgb_range.stop <= boundary)
    return torch.tensor(mask, dtype=torch.bool, device=batch.latents.device)


def _valid_mask(
    clean: torch.Tensor, valid_latent_mask: torch.Tensor | None
) -> torch.Tensor:
    if valid_latent_mask is None:
        return torch.ones(
            clean.shape[0], 1, clean.shape[2], clean.shape[3], clean.shape[4],
            dtype=torch.bool,
            device=clean.device,
        )
    expected = (clean.shape[0], clean.shape[2], clean.shape[3], clean.shape[4])
    if valid_latent_mask.shape != expected or valid_latent_mask.dtype != torch.bool:
        raise ValueError("valid_latent_mask must be boolean [B, T, H, W]")
    return valid_latent_mask[:, None].to(device=clean.device)
