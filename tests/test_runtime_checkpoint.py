import random

import pytest
import torch
from torch import nn

from runtime import (
    CheckpointCompatibility,
    CheckpointProvenance,
    CheckpointSelection,
    ExponentialMovingAverage,
    OptimizerSpec,
    RuntimeSpec,
    SchedulerSpec,
    build_optimizer,
    build_scheduler,
    initialize_new_stage,
    resume_same_run,
    save_checkpoint,
)


def _model():
    torch.manual_seed(4)
    return nn.Sequential(nn.Linear(3, 5), nn.SiLU(), nn.Linear(5, 2))


def _provenance(run_id="stage-a-run", parent=None):
    return CheckpointProvenance(
        run_id=run_id,
        stage="A",
        model={"architecture": "tiny-test", "width": 5},
        codec={"id": "codec@sha256:test", "causal": True},
        camera={"method": "tiled_prope", "pose": "c2w"},
        resolved_config={"seed": 4, "precision": "float32"},
        parent_checkpoint_id=parent,
    )


def _runtime(model):
    optimizer = build_optimizer(model, OptimizerSpec())
    scheduler = build_scheduler(optimizer, SchedulerSpec())
    ema = ExponentialMovingAverage(model, 0.9)
    return optimizer, scheduler, ema


def _one_step(model, optimizer, scheduler, ema):
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    ema.update(model)


def test_bf16_model_ema_accumulates_small_updates_in_fp32():
    model = nn.Linear(1, 1, bias=False).to(dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(1)
    ema = ExponentialMovingAverage(model, 0.9999)
    with torch.no_grad():
        model.weight.fill_(2)
    ema.update(model)
    assert ema.shadow["weight"].dtype is torch.float32
    assert float(ema.shadow["weight"][0, 0]) > 1.00005


def test_checkpoint_selection_rejects_ambiguous_lifecycle():
    with pytest.raises(ValueError, match="mutually exclusive"):
        CheckpointSelection(init_from="stage-a.pt", resume_from="same-run.pt")


def test_runtime_spec_makes_precision_and_single_device_scope_explicit():
    spec = RuntimeSpec()
    assert spec.torch_dtype is torch.bfloat16
    with pytest.raises(ValueError, match="single_device"):
        RuntimeSpec(devices=2, distributed_strategy="single_device")


def test_resume_restores_full_same_run_scope(tmp_path):
    model = _model()
    optimizer, scheduler, ema = _runtime(model)
    _one_step(model, optimizer, scheduler, ema)
    expected_model = {name: value.clone() for name, value in model.state_dict().items()}
    expected_ema = {name: value.clone() for name, value in ema.shadow.items()}
    path = tmp_path / "resume.pt"
    checkpoint_id = save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        global_step=17,
        epoch=3,
        provenance=_provenance(),
    )
    assert path.with_suffix(".pt.sha256").read_text().strip() in checkpoint_id

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    optimizer.state.clear()
    ema.shadow = {name: torch.zeros_like(value) for name, value in ema.shadow.items()}
    random.random()
    torch.rand(2)
    resumed = resume_same_run(
        path,
        expected_run_id="stage-a-run",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
    )

    assert resumed.global_step == 17 and resumed.epoch == 3
    assert resumed.checkpoint_id == checkpoint_id
    assert set(resumed.restored_scope) >= {"model", "optimizer", "scheduler", "ema"}
    assert optimizer.state
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected_model[name])
    for name, value in ema.shadow.items():
        assert torch.equal(value, expected_ema[name])


def test_resume_rejects_a_different_run_identity(tmp_path):
    model = _model()
    optimizer, scheduler, ema = _runtime(model)
    path = tmp_path / "resume.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        global_step=0,
        epoch=0,
        provenance=_provenance(),
    )
    with pytest.raises(ValueError, match="run_id mismatch"):
        resume_same_run(
            path,
            expected_run_id="another-run",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
        )


@pytest.mark.parametrize("source_weights", ["model", "ema"])
def test_init_from_loads_only_selected_weights_and_resets_training_state(
    tmp_path, source_weights
):
    source = _model()
    source_optimizer, source_scheduler, source_ema = _runtime(source)
    _one_step(source, source_optimizer, source_scheduler, source_ema)
    path = tmp_path / "stage-a.pt"
    checkpoint_id = save_checkpoint(
        path,
        model=source,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        ema=source_ema,
        global_step=99,
        epoch=8,
        provenance=_provenance(),
    )

    target = _model()
    initialized = initialize_new_stage(
        path,
        model=target,
        optimizer_factory=lambda model: build_optimizer(model, OptimizerSpec()),
        scheduler_factory=lambda optimizer: build_scheduler(optimizer, SchedulerSpec()),
        ema_decay=0.95,
        source_weights=source_weights,
    )
    expected = (
        source.state_dict()
        if source_weights == "model"
        else {
            **source.state_dict(),
            **source_ema.shadow,
        }
    )
    assert initialized.global_step == 0 and initialized.epoch == 0
    assert initialized.parent_checkpoint_id == checkpoint_id
    assert initialized.ema_rule == "copy_loaded_model"
    assert not initialized.optimizer.state
    assert initialized.scheduler.last_epoch == 0
    assert initialized.ema.num_updates == 0
    for name, value in target.state_dict().items():
        assert torch.equal(value, expected[name])
        if value.is_floating_point():
            assert torch.equal(initialized.ema.shadow[name], value)


def test_stage_transition_checks_compatibility_before_loading_weights(tmp_path):
    source = _model()
    optimizer, scheduler, ema = _runtime(source)
    path = tmp_path / "stage-a.pt"
    save_checkpoint(
        path,
        model=source,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        global_step=4,
        epoch=1,
        provenance=_provenance(),
    )
    target = _model()
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="codec.id"):
        initialize_new_stage(
            path,
            model=target,
            optimizer_factory=lambda model: build_optimizer(model, OptimizerSpec()),
            scheduler_factory=lambda value: build_scheduler(value, SchedulerSpec()),
            ema_decay=0.95,
            source_weights="ema",
            expected_compatibility=CheckpointCompatibility(
                model={"architecture": "tiny-test", "width": 5},
                codec={"id": "wrong-codec", "causal": True},
                camera={"method": "tiled_prope", "pose": "c2w"},
            ),
        )
    for name, value in target.state_dict().items():
        assert torch.equal(value, before[name])


def test_stage_transition_accepts_subset_compatibility(tmp_path):
    source = _model()
    optimizer, scheduler, ema = _runtime(source)
    path = tmp_path / "stage-a.pt"
    save_checkpoint(
        path,
        model=source,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        global_step=4,
        epoch=1,
        provenance=_provenance(),
    )
    target = _model()
    initialized = initialize_new_stage(
        path,
        model=target,
        optimizer_factory=lambda model: build_optimizer(model, OptimizerSpec()),
        scheduler_factory=lambda value: build_scheduler(value, SchedulerSpec()),
        ema_decay=0.95,
        source_weights="ema",
        expected_compatibility=CheckpointCompatibility(
            model={"architecture": "tiny-test"},
            codec={"causal": True},
            camera={"pose": "c2w"},
        ),
    )
    assert initialized.global_step == 0
    assert initialized.parent_checkpoint_id.startswith("sha256:")


def test_checkpoint_requires_complete_json_provenance():
    with pytest.raises(ValueError, match="codec provenance"):
        CheckpointProvenance(
            run_id="run",
            stage="A",
            model={"name": "model"},
            codec={},
            camera={"method": "prope"},
            resolved_config={"seed": 1},
        )


def test_checkpoint_load_rejects_file_corruption(tmp_path):
    model = _model()
    optimizer, scheduler, ema = _runtime(model)
    path = tmp_path / "corrupt.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        global_step=0,
        epoch=0,
        provenance=_provenance(),
    )
    with path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="SHA-256"):
        resume_same_run(
            path,
            expected_run_id="stage-a-run",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
        )
