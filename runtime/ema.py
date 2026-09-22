"""Explicit model EMA state with no hidden framework lifecycle."""

from collections.abc import Mapping

import torch
from torch import nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must satisfy 0 <= decay < 1")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        if self.shadow.keys() != {
            name for name, value in current.items() if value.is_floating_point()
        }:
            raise ValueError("EMA and model floating state keys do not match")
        for name, shadow in self.shadow.items():
            value = current[name].detach().to(device=shadow.device, dtype=shadow.dtype)
            shadow.lerp_(value, 1.0 - self.decay)
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, shadow in self.shadow.items():
            state[name].copy_(shadow.to(device=state[name].device, dtype=state[name].dtype))

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {name: value.detach().clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        decay = float(state["decay"])
        if decay != self.decay:
            raise ValueError(f"EMA decay mismatch: checkpoint={decay}, runtime={self.decay}")
        shadow = state["shadow"]
        if not isinstance(shadow, Mapping) or shadow.keys() != self.shadow.keys():
            raise ValueError("EMA checkpoint keys do not match the model")
        for name, value in shadow.items():
            if not isinstance(value, torch.Tensor) or value.shape != self.shadow[name].shape:
                raise ValueError(f"invalid EMA tensor for {name}")
            self.shadow[name].copy_(value)
        self.num_updates = int(state["num_updates"])

