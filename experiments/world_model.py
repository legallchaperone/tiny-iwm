"""Configuration-only M0 experiment shell.

Executable world-model tasks are deliberately added in later milestones.  An
empty task list lets the baseline validate the complete Hydra contract without
accidentally beginning an expensive training run.
"""

from .exp_base import BaseExperiment


class WorldModelExperiment(BaseExperiment):
    compatible_algorithms = {}
