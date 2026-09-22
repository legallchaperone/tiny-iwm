import pytest
import torch

from algorithms.world_model.models import JointVideoDiT, JointVideoDiTConfig
from algorithms.world_model.models.latent_io import LatentPatchIO
from algorithms.world_model.models.position import token_coordinates


def _tiny_model(**overrides):
    values = dict(
        latent_channels=4,
        hidden_size=24,
        depth=2,
        num_heads=4,
        patch_size=(1, 2, 2),
        mlp_ratio=2.0,
    )
    values.update(overrides)
    torch.manual_seed(7)
    return JointVideoDiT(JointVideoDiTConfig(**values))


def test_joint_dit_preserves_arbitrary_latent_shape_and_has_gradients():
    model = _tiny_model()
    latents = torch.randn(2, 4, 3, 5, 7, requires_grad=True)
    velocity = model(latents, torch.tensor([0.25, 0.75]))

    assert velocity.shape == latents.shape
    velocity.square().mean().backward()
    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()


def test_one_model_supports_bidirectional_and_causal_visibility():
    model = _tiny_model(depth=1).eval()
    latents = torch.randn(1, 4, 2, 2, 2)
    timestep = torch.tensor([0.5])
    bidirectional = model(latents, timestep)
    causal_mask = torch.ones(2, 2, dtype=torch.bool).tril()
    causal = model(latents, timestep, visibility_mask=causal_mask)

    assert bidirectional.shape == causal.shape == latents.shape
    assert not torch.allclose(bidirectional, causal)


def test_named_observations_are_stable_and_keep_gradients():
    model = _tiny_model()
    requested = ("patch_tokens", "blocks.0.post_attention", "blocks.1.block_output")
    velocity, observations = model(
        torch.randn(1, 4, 2, 4, 4), torch.tensor([0.3]), capture=requested
    )

    assert tuple(observations) == requested
    assert all(value.requires_grad for value in observations.values())
    assert set(requested) <= set(model.observation_points)
    with pytest.raises(ValueError, match="unknown observation"):
        model(torch.randn(1, 4, 1, 2, 2), torch.tensor([0.2]), capture=("hidden",))


def test_patch_io_round_trip_layout_has_no_fixed_length_assumption():
    adapter = LatentPatchIO(3, 12, (2, 3, 2))
    source = torch.randn(1, 3, 5, 7, 9)
    tokens, layout = adapter.patchify(source)
    restored = adapter.unpatchify(tokens, layout)

    assert layout.grid_shape == (3, 3, 5)
    assert tokens.shape == (1, 45, 12)
    assert restored.shape == source.shape


def test_global_time_offset_changes_only_time_coordinate():
    base = token_coordinates((2, 2, 2), time_offset=0)
    offset = token_coordinates((2, 2, 2), time_offset=17)
    assert torch.equal(offset[:, 0], base[:, 0] + 17)
    assert torch.equal(offset[:, 1:], base[:, 1:])


def test_constructor_exposes_no_pretrained_checkpoint_argument():
    with pytest.raises(TypeError):
        JointVideoDiT(_tiny_model().config, pretrained_checkpoint="weights.pt")

