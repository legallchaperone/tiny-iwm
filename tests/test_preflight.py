from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
import pytest

from scripts.preflight import CONFIG_DIR, compose_validate_and_write, validate_preflight_config


def _baseline_config():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=["+name=preflight-test"])


def test_cpu_preflight_writes_records_without_wandb_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    provenance = compose_validate_and_write(
        tmp_path / "run", ["+name=preflight-test", "runtime.accelerator=cpu"]
    )

    resolved = OmegaConf.load(tmp_path / "run" / "resolved_config.yaml")
    assert resolved.wandb.mode == "disabled"
    assert resolved.runtime.accelerator == "cpu"
    assert resolved.runtime.devices == 1
    assert resolved.experiment.tasks == []
    assert (tmp_path / "run" / "provenance.json").is_file()
    assert provenance["git"]["revision"]


@pytest.mark.parametrize("initialization", ["pretrained", "from_pretrained"])
def test_preflight_rejects_nonrandom_dit_initialization(initialization):
    cfg = _baseline_config()
    cfg.model.initialization = initialization

    with pytest.raises(ValueError, match="implicit pretrained DiT loading"):
        validate_preflight_config(cfg)


def test_preflight_rejects_pretrained_dit_fields():
    cfg = _baseline_config()
    with open_dict(cfg.model):
        cfg.model.pretrained_checkpoint = "vendor/model.ckpt"

    with pytest.raises(ValueError, match="pretrained_checkpoint"):
        validate_preflight_config(cfg)


def test_preflight_rejects_init_and_resume_together():
    cfg = _baseline_config()
    cfg.checkpoint.init_from = "stage-a.ckpt"
    cfg.checkpoint.resume_from = "interrupted.ckpt"

    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_preflight_config(cfg)


def test_preflight_accepts_each_checkpoint_mode_individually():
    init_cfg = _baseline_config()
    init_cfg.checkpoint.init_from = "stage-a.ckpt"
    validate_preflight_config(init_cfg)

    resume_cfg = _baseline_config()
    resume_cfg.checkpoint.resume_from = "interrupted.ckpt"
    validate_preflight_config(resume_cfg)


@pytest.mark.parametrize(
    ("legacy_selector", "checkpoint_selector"),
    [
        ("resume", "init_from"),
        ("load", "resume_from"),
    ],
)
def test_preflight_rejects_legacy_selector_mixed_with_opposite_mode(
    legacy_selector, checkpoint_selector
):
    cfg = _baseline_config()
    cfg[legacy_selector] = "legacy-checkpoint"
    cfg.checkpoint[checkpoint_selector] = "explicit-checkpoint"

    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_preflight_config(cfg)


@pytest.mark.parametrize(
    ("legacy_selector", "checkpoint_selector", "message"),
    [
        ("load", "init_from", "multiple initialization selectors"),
        ("resume", "resume_from", "multiple resume selectors"),
    ],
)
def test_preflight_rejects_duplicate_legacy_and_explicit_selectors(
    legacy_selector, checkpoint_selector, message
):
    cfg = _baseline_config()
    cfg[legacy_selector] = "legacy-checkpoint"
    cfg.checkpoint[checkpoint_selector] = "explicit-checkpoint"

    with pytest.raises(ValueError, match=message):
        validate_preflight_config(cfg)
