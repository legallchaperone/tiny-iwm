from scripts.profile_m3_full_length import _variant_costs
import torch

from algorithms.world_model.models.conditioning import TimestepConditioner


def test_variant_costs_preserve_physical_time_and_expose_attention_tradeoff():
    variants = _variant_costs(depth=24, hidden_size=832, dtype_bytes=2)
    by_name = {item["name"]: item for item in variants}

    assert all(item["physical_rgb_frames"] == 961 for item in variants)
    assert all(item["physical_duration_seconds"] == 60 for item in variants)
    assert by_name["128px_patch2_selected"]["tokens"] == 484
    assert by_name["128px_patch1"]["tokens"] == 1936
    assert by_name["256px_patch2"]["tokens"] == 1936
    assert by_name["256px_patch4"]["tokens"] == 484
    assert by_name["128px_patch1"]["dense_attention_ratio_vs_selected"] == 16


def test_timestep_conditioner_crosses_the_bfloat16_model_boundary():
    conditioner = TimestepConditioner(32).to(dtype=torch.bfloat16)
    output = conditioner(torch.tensor([0.25, 0.75], dtype=torch.bfloat16))
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
