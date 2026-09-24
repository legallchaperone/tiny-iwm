"""Selection and visibility policies independent of the prediction objective."""

from dataclasses import dataclass

import torch

from algorithms.world_model.training_batch import TeacherForcedSelection, teacher_forced_selection
from core.types import VideoBatch


@dataclass(frozen=True)
class TrainingPolicy:
    name: str
    requires_target_chunk: bool

    def select_target(
        self, batch: VideoBatch, *, target_chunk: int,
        valid_latent_mask: torch.Tensor | None = None,
    ) -> TeacherForcedSelection:
        if not self.requires_target_chunk:
            raise ValueError(f"policy {self.name} does not select a causal target chunk")
        return teacher_forced_selection(
            batch, target_chunk=target_chunk, valid_latent_mask=valid_latent_mask
        )


def build_training_policy(name: str) -> TrainingPolicy:
    if name == "bidirectional":
        return TrainingPolicy(name, False)
    if name == "teacher_forced_causal":
        return TrainingPolicy(name, True)
    raise ValueError(f"unsupported training policy: {name}")
