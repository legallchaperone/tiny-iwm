from omegaconf import OmegaConf
import pytest

from experiments.exp_base import BaseExperiment


class ExampleExperiment(BaseExperiment):
    compatible_algorithms = {}

    def main(self):
        self.task_executed = True


def test_exec_task_calls_named_method(tmp_path):
    cfg = OmegaConf.create({"debug": False, "experiment": {"tasks": ["main"]}})
    experiment = ExampleExperiment(cfg, output_dir=tmp_path)
    experiment.task_executed = False

    experiment.exec_task("main")

    assert experiment.task_executed is True


def test_exec_task_rejects_unknown_task(tmp_path):
    cfg = OmegaConf.create({"debug": False, "experiment": {"tasks": ["main"]}})
    experiment = ExampleExperiment(cfg, output_dir=tmp_path)

    with pytest.raises(ValueError, match="Specified task 'missing'"):
        experiment.exec_task("missing")
