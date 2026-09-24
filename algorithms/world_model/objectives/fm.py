"""Native Flow Matching adapter; all math remains in ``flow.py``."""

import torch

from algorithms.world_model.flow import FlowMatchSpec, flow_matching_loss
from algorithms.world_model.training_batch import StageABatchBuilder, StageBBatchBuilder
from algorithms.world_model.training_policies import TrainingPolicy
from core.types import ObjectiveBatch, VideoBatch


class FMObjective:
    name = "native_fm"
    prediction_type = "velocity"

    def __init__(self, spec: FlowMatchSpec, policy: TrainingPolicy) -> None:
        self.spec = spec
        self.policy = policy

    def build_batch(
        self,
        video: VideoBatch,
        *,
        target_chunk: int | None = None,
        noise: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
        valid_latent_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> ObjectiveBatch:
        if self.policy.requires_target_chunk:
            if target_chunk is None:
                raise ValueError("teacher-forced causal policy requires target_chunk")
            legacy = StageBBatchBuilder(self.spec).build(
                video, target_chunk=target_chunk, noise=noise, flow_time=flow_time,
                valid_latent_mask=valid_latent_mask, generator=generator,
            )
        else:
            if target_chunk is not None:
                raise ValueError("bidirectional policy does not select a target chunk")
            legacy = StageABatchBuilder(self.spec).build(
                video, noise=noise, flow_time=flow_time,
                valid_latent_mask=valid_latent_mask, generator=generator,
            )
        return ObjectiveBatch(
            model_input=legacy.noisy_latents,
            noise_condition=legacy.model_time if legacy.model_time is not None else legacy.flow_time,
            prediction_target=legacy.target_velocity,
            prediction_type=self.prediction_type,
            loss_mask=legacy.loss_mask,
            attention_visibility=legacy.attention_visibility,
            layout=legacy.layout,
            metadata={**legacy.metadata, "objective": self.name, "policy": self.policy.name},
            loss_time=legacy.flow_time,
        )

    def loss(self, prediction: torch.Tensor, batch: ObjectiveBatch) -> torch.Tensor:
        if batch.prediction_type != self.prediction_type or batch.loss_time is None:
            raise ValueError("FM loss requires velocity target and scalar FM loss time")
        # Stage B computes the model under bf16 autocast but evaluates its loss
        # in fp32. Preserve that path while leaving Stage A's native precision
        # unchanged when prediction and target already share a dtype.
        target = batch.prediction_target.to(dtype=prediction.dtype)
        time = batch.loss_time.to(dtype=prediction.dtype)
        return flow_matching_loss(
            prediction, target, loss_mask=batch.loss_mask,
            time=time, spec=self.spec,
        )
