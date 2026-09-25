"""Named objective adapters with distinct prediction semantics."""

from .fm import FMObjective
from .df import CosineDFObjective, CosineDFSchedule

__all__ = ["FMObjective", "CosineDFObjective", "CosineDFSchedule"]
