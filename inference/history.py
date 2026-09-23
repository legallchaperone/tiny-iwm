"""Clean-history KV lifecycle for reference and cached chunk inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from algorithms.world_model.models.attention import AttentionKV
from algorithms.world_model.models.dit import JointVideoDiT
from algorithms.world_model.models.prope import TokenCameraProjection
from algorithms.world_model.training_batch import ChunkCausalVisibility
from core.video_layout import VideoLayout

if TYPE_CHECKING:
    from probing.capture import CaptureContext, FeatureRecorder


@dataclass(frozen=True)
class HistoryIdentity:
    episode_id: str
    checkpoint_id: str
    condition_id: str
    cfg_branch: str

    def __post_init__(self) -> None:
        if not all(
            (self.episode_id, self.checkpoint_id, self.condition_id, self.cfg_branch)
        ):
            raise ValueError("history identity fields must be non-empty")


class HistorySession:
    """One episode/weight/condition/CFG branch; denoising never mutates its KV."""

    def __init__(
        self,
        model: JointVideoDiT,
        layout: VideoLayout,
        identity: HistoryIdentity,
        *,
        camera_projection: TokenCameraProjection | None = None,
    ) -> None:
        if model.training:
            raise ValueError("history inference requires model.eval()")
        self.model = model
        self.layout = layout
        self.identity = identity
        self.camera_projection = camera_projection
        self.visibility = ChunkCausalVisibility.from_layout(layout)
        self.history: list[torch.Tensor] = []
        self.kv_cache: tuple[AttentionKV, ...] | None = None
        self._spatial_tokens: int | None = None
        self._next_chunk = 0
        self._prefix_latents = 0
        boundary = layout.initial_condition_rgb.stop
        self._condition_latents = sum(
            rgb_range.stop <= boundary for rgb_range in layout.latent_to_rgb
        )

    @property
    def next_chunk(self) -> int:
        return self._next_chunk

    def _check(
        self, identity: HistoryIdentity, chunk: torch.Tensor, chunk_index: int
    ) -> tuple[int, int]:
        if identity != self.identity:
            raise ValueError("history identity mismatch; start a new session")
        if chunk_index != self.next_chunk:
            raise ValueError("chunks must be processed in VideoLayout order")
        if (
            chunk_index == 0
            and not self.history
            and self._condition_latents < self.layout.chunk_to_latent[0].stop
        ):
            raise ValueError("commit the initial condition prefix before the first target")
        if chunk.ndim != 5 or chunk.shape[1] != self.model.config.latent_channels:
            raise ValueError("chunk must have model-compatible [B, C, T, H, W] shape")
        expected = self.layout.latent_range_for_chunk(chunk_index)
        start_latent = max(self._prefix_latents, expected.start)
        if chunk.shape[2] != expected.stop - start_latent:
            raise ValueError("chunk latent length does not match VideoLayout")
        patch_time, patch_height, patch_width = self.model.config.patch_size
        if start_latent % patch_time or expected.stop % patch_time:
            raise ValueError("chunk boundaries must align with temporal patches")
        if chunk.shape[3] % patch_height or chunk.shape[4] % patch_width:
            raise ValueError("spatial chunk dimensions must align with patches")
        spatial = (chunk.shape[3] // patch_height) * (chunk.shape[4] // patch_width)
        if self._spatial_tokens is not None and spatial != self._spatial_tokens:
            raise ValueError("spatial token geometry changed within an episode")
        if self.history and chunk.shape[:2] + chunk.shape[3:] != (
            self.history[0].shape[:2] + self.history[0].shape[3:]
        ):
            raise ValueError("chunk batch, channel, or spatial geometry changed")
        token_start = (start_latent // patch_time) * spatial
        token_stop = (expected.stop // patch_time) * spatial
        if (
            self.camera_projection is not None
            and self.camera_projection.token_count < token_stop
        ):
            raise ValueError("camera projection does not cover the target chunk")
        return token_start, token_stop

    @torch.no_grad()
    def commit_initial_prefix(self, identity: HistoryIdentity, clean_prefix: torch.Tensor) -> None:
        """Commit the real initial condition, including a partial first chunk."""
        if identity != self.identity or self.history:
            raise ValueError("initial prefix requires a fresh matching history session")
        prefix_count = self._condition_latents
        if clean_prefix.ndim != 5 or clean_prefix.shape[2] != prefix_count:
            raise ValueError("initial prefix length must match VideoLayout condition")
        if prefix_count % self.model.config.patch_size[0]:
            raise ValueError("initial condition must align with temporal patches")
        if clean_prefix.shape[1] != self.model.config.latent_channels:
            raise ValueError("initial prefix channels do not match the model")
        if clean_prefix.shape[3] % self.model.config.patch_size[1] or clean_prefix.shape[4] % self.model.config.patch_size[2]:
            raise ValueError("initial prefix spatial dimensions must align with patches")
        spatial = (clean_prefix.shape[3] // self.model.config.patch_size[1]) * (
            clean_prefix.shape[4] // self.model.config.patch_size[2]
        )
        token_count = (prefix_count // self.model.config.patch_size[0]) * spatial
        zero_time = torch.zeros(
            (clean_prefix.shape[0], token_count), device=clean_prefix.device,
            dtype=clean_prefix.dtype,
        )
        _, updated = self.model(
            clean_prefix, zero_time, camera_projection=self._camera(0, token_count),
            return_kv_cache=True,
        )
        self.history.append(clean_prefix.detach().clone())
        self.kv_cache = tuple(
            AttentionKV(item.key.detach(), item.value.detach()) for item in updated
        )
        self._spatial_tokens = spatial
        self._prefix_latents = prefix_count
        self._next_chunk = sum(
            chunk.stop <= prefix_count for chunk in self.layout.chunk_to_latent
        )

    def _camera(self, start: int, stop: int) -> TokenCameraProjection | None:
        if self.camera_projection is None:
            return None
        projection = self.camera_projection
        return TokenCameraProjection(
            projection.projection[:, start:stop],
            projection.transpose[:, start:stop],
            projection.inverse[:, start:stop],
        )

    @torch.no_grad()
    def predict(
        self,
        identity: HistoryIdentity,
        chunk_index: int,
        noisy_chunk: torch.Tensor,
        flow_time: torch.Tensor,
        *,
        mode: str = "cached",
        recorder: FeatureRecorder | None = None,
        context: CaptureContext | None = None,
    ) -> torch.Tensor:
        """Predict target velocity without admitting its temporary KV to history."""
        start, stop = self._check(identity, noisy_chunk, chunk_index)
        if flow_time.shape != (noisy_chunk.shape[0],):
            raise ValueError("flow_time must have shape [B]")
        if not torch.isfinite(flow_time).all() or torch.any(
            (flow_time < 0) | (flow_time > 1)
        ):
            raise ValueError("flow_time must be finite within [0, 1]")
        if mode == "cached":
            if recorder is not None:
                raise ValueError("feature capture for replay requires reference mode")
            token_time = flow_time[:, None].expand(-1, stop - start)
            spatial = (noisy_chunk.shape[3] // self.model.config.patch_size[1]) * (
                noisy_chunk.shape[4] // self.model.config.patch_size[2]
            )
            return self.model(
                noisy_chunk,
                token_time,
                time_offset=start // spatial,
                camera_projection=self._camera(start, stop),
                kv_cache=self.kv_cache,
            )
        if mode != "reference":
            raise ValueError("mode must be 'cached' or 'reference'")
        full = torch.cat((*self.history, noisy_chunk), dim=2)
        token_time = torch.zeros(
            (noisy_chunk.shape[0], stop),
            device=noisy_chunk.device,
            dtype=flow_time.dtype,
        )
        token_time[:, start:stop] = flow_time[:, None]
        spatial = (noisy_chunk.shape[3] // self.model.config.patch_size[1]) * (
            noisy_chunk.shape[4] // self.model.config.patch_size[2]
        )
        mask = self.visibility.materialize(spatial_tokens_per_temporal_token=spatial)
        if recorder is None:
            output = self.model(
                full,
                token_time,
                visibility_mask=mask[:stop, :stop],
                camera_projection=self._camera(0, stop),
            )
        else:
            if context is None:
                raise ValueError("captured replay requires a capture context")
            from probing.capture import capture_forward

            output = capture_forward(
                self.model,
                full,
                token_time,
                context=context,
                recorder=recorder,
                visibility_mask=mask[:stop, :stop],
                camera_projection=self._camera(0, stop),
            )
        return output[:, :, -noisy_chunk.shape[2] :]

    @torch.no_grad()
    def commit_clean(
        self, identity: HistoryIdentity, chunk_index: int, clean_chunk: torch.Tensor
    ) -> None:
        """Recompute the finished chunk at t=0 and persist only clean KV."""
        start, stop = self._check(identity, clean_chunk, chunk_index)
        clean_time = torch.zeros(
            (clean_chunk.shape[0], stop - start),
            device=clean_chunk.device,
            dtype=clean_chunk.dtype,
        )
        _, updated = self.model(
            clean_chunk,
            clean_time,
            time_offset=start // (
                (clean_chunk.shape[3] // self.model.config.patch_size[1])
                * (clean_chunk.shape[4] // self.model.config.patch_size[2])
            ),
            camera_projection=self._camera(start, stop),
            kv_cache=self.kv_cache,
            return_kv_cache=True,
        )
        self.history.append(clean_chunk.detach().clone())
        self.kv_cache = tuple(
            AttentionKV(item.key.detach(), item.value.detach()) for item in updated
        )
        self._spatial_tokens = (stop - start) // (
            clean_chunk.shape[2] // self.model.config.patch_size[0]
        )
        self._next_chunk = chunk_index + 1
