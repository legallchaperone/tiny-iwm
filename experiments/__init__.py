from __future__ import annotations

from importlib import import_module
from typing import Optional, TYPE_CHECKING, Union
import pathlib

if TYPE_CHECKING:
    from omegaconf import DictConfig

# each key has to be a yaml file under '[project_root]/configurations/experiment' without .yaml suffix
exp_registry = dict(
    world_model="experiments.world_model:WorldModelExperiment",
    example_classification="experiments.example_classification:ClassificationExperiment",
    example_helloworld="experiments.example_helloworld:HelloWorldExperiment",
)


def build_experiment(
    cfg: DictConfig, logger: Optional[object] = None, ckpt_path: Optional[Union[str, pathlib.Path]] = None
):
    """
    Build an experiment instance based on registry
    :param cfg: configuration file
    :param logger: optional logger for the experiment
    :param ckpt_path: optional checkpoint path for saving and loading
    :return:
    """
    if cfg.experiment._name not in exp_registry:
        raise ValueError(
            f"Experiment {cfg.experiment._name} not found in registry {list(exp_registry.keys())}. "
            "Make sure you register it correctly in 'experiments/__init__.py' under the same name as yaml file."
        )

    module_name, class_name = exp_registry[cfg.experiment._name].split(":")
    experiment_class = getattr(import_module(module_name), class_name)
    return experiment_class(cfg, logger, ckpt_path)
