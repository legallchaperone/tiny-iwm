"""Small explicit factories for optimizer and scheduler lifecycle."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RuntimeSpec:
    accelerator: str = "gpu"
    devices: int = 1
    num_nodes: int = 1
    precision: str = "bfloat16"
    distributed_strategy: str = "single_device"

    def __post_init__(self) -> None:
        if self.accelerator not in {"cpu", "gpu"}:
            raise ValueError("runtime accelerator must be cpu or gpu")
        if self.devices <= 0 or self.num_nodes <= 0:
            raise ValueError("runtime devices and nodes must be positive")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("runtime precision must be float32 or bfloat16")
        if self.distributed_strategy == "single_device" and (
            self.devices != 1 or self.num_nodes != 1
        ):
            raise ValueError("single_device runtime requires one device on one node")

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "bfloat16": torch.bfloat16}[self.precision]


def configure_cuda_math_policy() -> None:
    """Apply the recorded strict CUDA policy before model construction."""
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


@dataclass(frozen=True)
class OptimizerSpec:
    name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)

    def __post_init__(self) -> None:
        if self.name != "adamw":
            raise ValueError("only the configured AdamW optimizer is supported")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer learning rate must be positive and decay non-negative")


@dataclass(frozen=True)
class SchedulerSpec:
    name: str = "constant"

    def __post_init__(self) -> None:
        if self.name != "constant":
            raise ValueError("only the explicit constant scheduler is currently supported")


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
        betas=spec.betas,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, spec: SchedulerSpec
) -> torch.optim.lr_scheduler.LRScheduler:
    del spec
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
