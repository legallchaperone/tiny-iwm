"""E2 model selection, component overrides, and shared execution smoke."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch

from algorithms.world_model.flow import FlowMatchSpec, flow_matching_loss
from algorithms.world_model.models import JointVideoDiT, JointVideoDiTConfig
from algorithms.world_model.models.blocks import SwiGLUFFN
from algorithms.world_model.training_batch import StageABatchBuilder
from core.camera import CameraCondition, IntrinsicsSpace
from core.types import VideoBatch
from core.video_layout import CodecTemporalSpec, VideoLayout
from experiments.build import build_model, resolve_model_config
from experiments.world_model import WorldModelExperiment
from inference.history import HistoryIdentity
from inference.rollout import rollout_latents
from scripts.preflight import validate_preflight_config


CONFIG_DIR = Path(__file__).parents[1] / "configurations"


def _small_root(*, architecture="joint_spatiotemporal_dit", **components):
    return {
        "model": {
            "architecture": architecture,
            "initialization": "random",
            "latent_channels": 2,
            "hidden_size": 24,
            "depth": 1,
            "num_heads": 4,
            "patch_size": [1, 1, 1],
            "mlp_ratio": 2.0,
            "qkv_bias": True,
            "prope_camera_dims": None,
            "components": components,
        },
        "conditioning": {"camera": {"enabled": False}},
    }


def test_default_builder_preserves_checkpoint_keys():
    built = build_model(_small_root())
    direct = JointVideoDiT(
        JointVideoDiTConfig(
            latent_channels=2, hidden_size=24, depth=1, num_heads=4,
            patch_size=(1, 1, 1), mlp_ratio=2.0, prope_camera_dims=None,
        )
    )
    assert built.architecture == "joint_spatiotemporal_dit"
    assert built.capabilities.kv_cache
    assert set(built.model.state_dict()) == set(direct.state_dict())


def test_swiglu_structure_trains_generates_and_exposes_probe_features():
    torch.manual_seed(4)
    built = build_model(_small_root(
        architecture="joint_spatiotemporal_dit_swiglu", ffn="swiglu", norm="rms_norm"
    ))
    model = built.model
    assert isinstance(model.blocks[0].mlp, SwiGLUFFN)
    assert isinstance(model.blocks[0].attention_norm, torch.nn.RMSNorm)
    layout = VideoLayout.from_codec(
        fps=8, rgb_frame_count=5, codec=CodecTemporalSpec(1),
        latent_chunk_size=2, initial_condition_frames=1,
    )
    camera = CameraCondition(
        c2w=torch.eye(4).repeat(1, 5, 1, 1),
        intrinsics=torch.eye(3).repeat(1, 5, 1, 1),
        timestamps_seconds=torch.arange(5)[None] / 8,
        intrinsics_space=IntrinsicsSpace.RGB_PIXELS,
    )
    clean = torch.randn(1, 2, 5, 2, 2)
    batch = StageABatchBuilder(FlowMatchSpec()).build(
        VideoBatch(("fixture",), ("fixture",), layout, camera, latents=clean),
        flow_time=torch.tensor([0.5]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    prediction, features = model(
        batch.noisy_latents, batch.flow_time, capture=("blocks.0.post_mlp",)
    )
    assert features["blocks.0.post_mlp"].shape[-1] == 24
    loss = flow_matching_loss(
        prediction, batch.target_velocity, loss_mask=batch.loss_mask,
        time=batch.flow_time, spec=FlowMatchSpec(),
    )
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    model.eval()
    generated = rollout_latents(
        model, layout, HistoryIdentity("episode", "ckpt", "source", "conditional"),
        (clean[:, :, :1],), steps=2, seed=3, mode="reference",
    )
    assert generated.shape == clean.shape
    assert torch.isfinite(generated).all()


@pytest.mark.parametrize("component", [
    "block", "attention", "ffn", "norm", "position", "conditioner", "latent_io"
])
def test_unknown_component_fails_before_model_allocation(component):
    root = _small_root(**{component: "unimplemented"})
    with pytest.raises(ValueError, match="unsupported|requires"):
        resolve_model_config(root)


def test_unknown_model_and_camera_mismatch_fail_explicitly():
    with pytest.raises(ValueError, match="unsupported model architecture"):
        resolve_model_config(_small_root(architecture="unknown"))
    root = _small_root()
    root["model"]["prope_camera_dims"] = 4
    with pytest.raises(ValueError, match="camera.enabled=false"):
        resolve_model_config(root)
    root["conditioning"]["camera"]["enabled"] = True
    root["model"]["prope_camera_dims"] = None
    with pytest.raises(ValueError, match="camera.enabled=true"):
        resolve_model_config(root)


def test_conflicting_attention_selectors_fail_before_build():
    root = _small_root()
    root["model"]["attention"] = "joint_spatiotemporal_softmax"
    root["model"]["attention_kind"] = "unimplemented"
    with pytest.raises(ValueError, match="attention conflicts"):
        resolve_model_config(root)


def test_hydra_model_variant_passes_cpu_preflight_with_small_shape():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=[
            "model=joint_dit_swiglu", "model.latent_channels=2",
            "model.hidden_size=24", "model.depth=1", "model.num_heads=4",
            "model.patch_size=[1,1,1]", "model.prope_camera_dims=null",
            "conditioning.camera.enabled=false",
        ])
    validate_preflight_config(cfg)
    assert build_model(cfg).architecture == "joint_spatiotemporal_dit_swiglu"


def test_world_model_experiment_uses_shared_builder(tmp_path):
    root = _small_root(architecture="joint_spatiotemporal_dit_swiglu", ffn="swiglu")
    root.update({"debug": False, "experiment": {"tasks": []}})
    experiment = WorldModelExperiment(OmegaConf.create(root), tmp_path)
    assert isinstance(experiment.build_model().model.blocks[0].mlp, SwiGLUFFN)
