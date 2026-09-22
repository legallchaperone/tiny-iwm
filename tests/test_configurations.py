from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = Path(__file__).parents[1] / "configurations"


def test_default_configuration_composes():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config")

    assert cfg.model.architecture == "joint_spatiotemporal_dit"
    assert cfg.dataset.rgb_frames == 961
    assert cfg.experiment.tasks == []
    assert cfg.wandb.mode == "disabled"
    assert cfg.algorithm.training.batch_size is None
    assert cfg.rollout.history_policy == "full"


def test_command_line_style_overrides_compose():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "+name=smoke-test",
                "wandb.mode=disabled",
                "algorithm.training.max_steps=1",
            ],
        )

    assert cfg.name == "smoke-test"
    assert cfg.wandb.mode == "disabled"
    assert cfg.algorithm.training.max_steps == 1


def test_every_baseline_group_is_present():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config")

    expected = {
        "experiment", "algorithm", "model", "stage", "representation",
        "conditioning", "dataset", "rollout", "evaluation", "probing", "runtime",
    }
    assert expected <= set(cfg.keys())
