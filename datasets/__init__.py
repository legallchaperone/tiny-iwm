"""Dataset entry points without importing optional example dependencies eagerly."""

from __future__ import annotations

from typing import Any

__all__ = ["CIFAR10Dataset"]


def __getattr__(name: str) -> Any:
    if name == "CIFAR10Dataset":
        from .example_classification import CIFAR10Dataset

        return CIFAR10Dataset
    raise AttributeError(name)
