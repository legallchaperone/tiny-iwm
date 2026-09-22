from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = Path(__file__).parents[1] / "configurations"


def test_default_configuration_composes():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config")

    assert cfg.algorithm.arch == "resnet18"
    assert cfg.dataset.data_dir == "data/cifar10"
    assert cfg.experiment.tasks == ["training", "test"]
    assert cfg.experiment.training.batch_size == 32


def test_command_line_style_overrides_compose():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "+name=smoke-test",
                "wandb.mode=disabled",
                "experiment.training.max_epochs=1",
            ],
        )

    assert cfg.name == "smoke-test"
    assert cfg.wandb.mode == "disabled"
    assert cfg.experiment.training.max_epochs == 1
