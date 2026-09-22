from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest
import yaml

import main


CONFIG_DIR = Path(__file__).parents[1] / "configurations"


def test_default_configuration_composes():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config")

    assert cfg.model.architecture == "joint_spatiotemporal_dit"
    assert cfg.dataset.rgb_frames == 961
    assert cfg.experiment.tasks == []
    assert cfg.wandb.mode == "disabled"
    assert cfg.algorithm.training.batch_size == 1
    assert cfg.algorithm.training.gradient_accumulation == 1
    assert cfg.algorithm.ema.enabled is True
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


def test_cloud_checkpoint_requires_wandb_entity():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "+name=cloud-load-test",
                "+load=abcdefgh",
                "wandb.mode=disabled",
                "wandb.entity=null",
            ],
        )

    with pytest.raises(ValueError, match="cloud checkpoint loading"):
        main.run.__wrapped__(cfg)


def test_harvard_cluster_reads_node_count_from_runtime():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "cluster=harvard_fas",
                "runtime=distributed",
                "runtime.num_nodes=2",
            ],
        )

    assert "#SBATCH --nodes=2" in cfg.cluster.launch_template


def test_example_sweep_selects_compatible_example_groups():
    sweep = yaml.safe_load((CONFIG_DIR / "sweep" / "example_sweep.yaml").read_text())
    parameters = sweep["parameters"]

    assert parameters["experiment"]["value"] == "example_classification"
    assert parameters["dataset"]["value"] == "example_cifar10"
    assert parameters["algorithm"]["value"] == "example_classifier"

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=example_classification",
                "dataset=example_cifar10",
                "algorithm=example_classifier",
                "algorithm.lr=0.001",
                "experiment.training.batch_size=32",
            ],
        )

    assert cfg.algorithm.lr == 0.001
    assert cfg.experiment.training.batch_size == 32
