"""Configuration-only M0 experiment shell.

Executable world-model tasks are deliberately added in later milestones.  An
empty task list lets the baseline validate the complete Hydra contract without
accidentally beginning an expensive training run.
"""

from .exp_base import BaseExperiment
from .build import BuiltModel, build_model


class WorldModelExperiment(BaseExperiment):
    compatible_algorithms = {}

    def build_model(self) -> BuiltModel:
        """Use the same resolved model construction as launchers and inference."""
        return build_model(self.root_cfg)
