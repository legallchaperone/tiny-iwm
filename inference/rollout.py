"""One autoregressive rollout loop using an objective-compatible sampler."""

import torch

from algorithms.world_model.models.dit import JointVideoDiT
from algorithms.world_model.models.prope import TokenCameraProjection
from core.video_layout import VideoLayout
from inference.samplers import FMEulerSampler
from inference.samplers.base import Sampler

from .history import HistoryIdentity, HistorySession


@torch.no_grad()
def rollout_latents(
    model: JointVideoDiT,
    layout: VideoLayout,
    identity: HistoryIdentity,
    initial_chunks: tuple[torch.Tensor, ...],
    *,
    steps: int,
    seed: int,
    mode: str = "cached",
    initial_history_policy: str = "allow_clean_prefill",
    camera_projection: TokenCameraProjection | None = None,
    sampler: Sampler | None = None,
    objective_name: str | None = None,
) -> torch.Tensor:
    """Generate all remaining chunks and return the full latent trajectory.

    The first tensor is the exact initial-condition prefix from VideoLayout.
    Additional tensors, when provided, are complete clean leading chunks.
    """
    if steps <= 0 or not initial_chunks:
        raise ValueError(
            "steps must be positive and at least one clean chunk is required"
        )
    if initial_history_policy not in {"allow_clean_prefill", "source_image_only"}:
        raise ValueError("unknown initial history policy")
    if initial_history_policy == "source_image_only" and len(initial_chunks) != 1:
        raise ValueError("source-image-only rollout rejects extra clean chunks")
    if mode not in {"cached", "reference"}:
        raise ValueError("mode must be 'cached' or 'reference'")
    if sampler is None:
        sampler = FMEulerSampler()
        objective_name = "native_fm" if objective_name is None else objective_name
    if objective_name != sampler.objective_name:
        raise ValueError(
            f"sampler {sampler.name} requires objective {sampler.objective_name}; "
            f"received {objective_name!r}"
        )
    session = HistorySession(
        model, layout, identity, camera_projection=camera_projection
    )
    session.commit_initial_prefix(identity, initial_chunks[0])
    for chunk in initial_chunks[1:]:
        session.commit_clean(identity, session.next_chunk, chunk)
    sample = initial_chunks[0]
    generator = torch.Generator(device=sample.device).manual_seed(seed)
    time_grid = sampler.grid(steps, device=sample.device, dtype=sample.dtype)
    for index in range(session.next_chunk, len(layout.chunk_to_latent)):
        target_range = layout.latent_range_for_chunk(index)
        length = target_range.stop - sum(chunk.shape[2] for chunk in session.history)
        shape = (
            sample.shape[0],
            sample.shape[1],
            length,
            sample.shape[3],
            sample.shape[4],
        )
        target = sampler.initial_state(
            shape, generator=generator, device=sample.device, dtype=sample.dtype
        )
        for step in range(steps):
            time = time_grid[step].expand(sample.shape[0])
            next_time = time_grid[step + 1].expand(sample.shape[0])
            # Keep the sampler accumulator at its own precision across steps.
            prediction = session.predict(
                identity, index, target.to(sample.dtype), time, mode=mode
            )
            target = sampler.step(target, prediction, time=time, next_time=next_time)
        session.commit_clean(identity, index, target.to(sample.dtype))
    return torch.cat(session.history, dim=2)
