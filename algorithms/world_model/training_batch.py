"""Stage A bidirectional and Stage B chunk-causal training semantics."""

from dataclasses import dataclass

import torch

from core.types import TrainingBatch, VideoBatch
from core.video_layout import VideoLayout

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
        model_input = torch.where(
            valid_mask,
            torch.where(condition_mask, clean, mixed_target),
            torch.zeros_like(clean),
        )
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


@dataclass(frozen=True)
class ChunkCausalVisibility:
    """A causal mask with bidirectional attention inside each temporal chunk."""

    temporal_token_chunks: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.temporal_token_chunks or any(
            value < 0 for value in self.temporal_token_chunks
        ):
            raise ValueError("temporal token chunk ids must be non-empty and non-negative")
        if tuple(sorted(self.temporal_token_chunks)) != self.temporal_token_chunks:
            raise ValueError("temporal token chunk ids must be monotonic")

    @classmethod
    def from_layout(cls, layout: VideoLayout) -> "ChunkCausalVisibility":
        condition_boundary = layout.initial_condition_rgb.stop
        condition_latents = 0
        for rgb_range in layout.latent_to_rgb:
            if rgb_range.start < condition_boundary < rgb_range.stop:
                raise ValueError("initial condition boundary must align with a latent boundary")
            if rgb_range.stop <= condition_boundary:
                condition_latents += 1
        latent_chunk: dict[int, int] = {}
        for chunk_index, latent_range in enumerate(layout.chunk_to_latent):
            for latent_index in range(latent_range.start, latent_range.stop):
                latent_chunk[latent_index] = 0 if latent_index < condition_latents else chunk_index + 1
        token_chunks: list[int] = []
        for token_range in layout.token_to_latent_ranges:
            chunks = {
                latent_chunk[index]
                for index in range(token_range.start, token_range.stop)
            }
            if len(chunks) != 1:
                raise ValueError("temporal patches must not cross Stage B chunk boundaries")
            token_chunks.append(chunks.pop())
        return cls(tuple(token_chunks))

    def materialize(self, *, spatial_tokens_per_temporal_token: int) -> torch.Tensor:
        """Return [N,N] allowed edges; rows are queries and columns are keys."""
        if spatial_tokens_per_temporal_token <= 0:
            raise ValueError("spatial token count must be positive")
        chunks = torch.tensor(self.temporal_token_chunks, dtype=torch.long).repeat_interleave(
            spatial_tokens_per_temporal_token
        )
        return chunks[:, None] >= chunks[None, :]


class StageBBatchBuilder:
    """Build one teacher-forced noisy target chunk over clean causal branches."""

    def __init__(self, flow_spec: FlowMatchSpec) -> None:
        self.flow_spec = flow_spec

    def build(
        self,
        batch: VideoBatch,
        *,
        target_chunk: int,
        noise: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
        valid_latent_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> TrainingBatch:
        clean = batch.latents
        if not isinstance(clean, torch.Tensor) or clean.ndim != 5:
            raise ValueError("Stage B requires latents with shape [B, C, T, H, W]")
        if clean.shape[2] != batch.layout.latent_frame_count:
            raise ValueError("latent time dimension must match VideoLayout")
        if not clean.is_floating_point() or not torch.isfinite(clean).all():
            raise ValueError("Stage B latents must be finite floating-point values")
        if not 0 <= target_chunk < len(batch.layout.chunk_to_latent):
            raise IndexError("target_chunk is outside VideoLayout")

        target_range = batch.layout.latent_range_for_chunk(target_chunk)
        target_time = torch.zeros(clean.shape[2], dtype=torch.bool, device=clean.device)
        target_time[target_range.start : target_range.stop] = True
        condition_time = _condition_latent_mask(batch)
        target_time &= ~condition_time
        if not target_time.any():
            raise ValueError("target chunk contains no non-condition latent")
        target_mask = target_time[None, None, :, None, None].expand(
            clean.shape[0], 1, clean.shape[2], clean.shape[3], clean.shape[4]
        )
        valid_mask = _valid_mask(clean, valid_latent_mask)
        loss_mask = valid_mask & target_mask
        if not loss_mask.any():
            raise ValueError("Stage B target chunk has no valid target latents")

        if noise is None:
            noise = torch.randn(
                clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
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

        mixed = interpolate(clean, noise, flow_time)
        model_input = torch.where(
            valid_mask,
            torch.where(target_mask, mixed, clean),
            torch.zeros_like(clean),
        )
        clean_branches = torch.where(target_mask, torch.zeros_like(clean), clean)
        velocity = torch.where(
            loss_mask, target_velocity(clean, noise), torch.zeros_like(clean)
        )
        visibility = ChunkCausalVisibility.from_layout(batch.layout)
        model_time = torch.zeros(
            (clean.shape[0], len(batch.layout.token_to_latent_ranges)),
            device=clean.device, dtype=flow_time.dtype,
        )
        for token_index, latent_range in enumerate(batch.layout.token_to_latent_ranges):
            selected = target_time[latent_range.start:latent_range.stop]
            if selected.any() and not selected.all():
                raise ValueError("target and clean condition cannot share a temporal patch")
            if selected.all():
                model_time[:, token_index] = flow_time
        return TrainingBatch(
            noisy_latents=model_input,
            clean_condition_latents=clean_branches,
            flow_time=flow_time,
            target_velocity=velocity,
            loss_mask=loss_mask,
            attention_visibility=visibility,
            layout=batch.layout,
            model_time=model_time,
            metadata={
                "stage": "B",
                "teacher_forcing": True,
                "attention": "chunk_causal",
                "chunk_internal_attention": "bidirectional",
                "target_chunk": target_chunk,
                "target_latent_range": (target_range.start, target_range.stop),
                "target_latent_indices": tuple(
                    target_time.nonzero(as_tuple=False).flatten().tolist()
                ),
                "future_clean_present_but_causally_invisible": True,
                "flow_convention": "t0_clean_t1_noise",
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
