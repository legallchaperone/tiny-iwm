from dataclasses import replace

import pytest
import torch
from torch import nn

from representations import (
    CacheIdentity,
    ChannelNormalizer,
    CodecSpec,
    SanaCausalVideoVAEAdapter,
)


def _normalizer() -> ChannelNormalizer:
    training_latents = torch.tensor([1.0, 2.0, 3.0, 6.0]).view(2, 2, 1, 1, 1)
    return ChannelNormalizer.fit_from_training(
        [training_latents], split="train", data_version="sana-train-v1"
    )


def _codec(normalizer: ChannelNormalizer, **changes) -> CodecSpec:
    spec = CodecSpec(
        codec_name="sana-causal-video-vae",
        weights_sha256="a" * 64,
        latent_channels=2,
        temporal_compression=4,
        spatial_compression=(8, 8),
        causal=True,
        encoding_policy="prefix_causal_v1",
        normalization=normalizer.stats,
    )
    return replace(spec, **changes)


def _identity(codec: CodecSpec, **changes) -> CacheIdentity:
    identity = CacheIdentity(
        data_version="sana-data-v1",
        sample_id="scene-1-clip-1",
        frame_indices=(0, 1, 2, 3, 4),
        preprocessing={"resize": [320, 512], "crop": [0, 0, 320, 512]},
        codec=codec,
    )
    return replace(identity, **changes)


def test_statistics_are_training_only_frozen_and_reversible():
    normalizer = _normalizer()
    values = torch.tensor([1.0, 2.0, 3.0, 6.0]).view(2, 2, 1, 1, 1)

    normalized = normalizer.normalize(values)

    torch.testing.assert_close(
        normalized, torch.tensor([-1.0, -1.0, 1.0, 1.0]).view_as(values)
    )
    torch.testing.assert_close(normalizer.denormalize(normalized), values)
    assert normalizer.stats.training_split == "train"
    assert normalizer.stats.sample_count == 2
    with pytest.raises(ValueError, match="train split"):
        ChannelNormalizer.fit_from_training(
            [values], split="validation", data_version="sana-data-v1"
        )


def test_cache_identity_separates_codec_preprocessing_frames_and_statistics():
    normalizer = _normalizer()
    codec = _codec(normalizer)
    baseline = _identity(codec)

    changed_codec = _identity(replace(codec, weights_sha256="b" * 64))
    changed_preprocessing = _identity(
        codec, preprocessing={"resize": [256, 512], "crop": [0, 0, 256, 512]}
    )
    changed_frames = _identity(codec, frame_indices=(1, 2, 3, 4, 5))
    other_stats = replace(
        normalizer.stats,
        mean=(normalizer.stats.mean[0] + 0.1, normalizer.stats.mean[1]),
    )
    changed_statistics = _identity(replace(codec, normalization=other_stats))

    assert len(
        {
            baseline.key,
            changed_codec.key,
            changed_preprocessing.key,
            changed_frames.key,
            changed_statistics.key,
        }
    ) == 5
    assert baseline.payload["codec"]["encoding_policy"] == "prefix_causal_v1"
    assert baseline.payload["normalization"]["training_split"] == "train"


def test_cache_identity_is_stable_across_mapping_insertion_order():
    codec = _codec(_normalizer())
    preprocessing = {"resize": [320, 512], "crop": [0, 0, 320, 512]}
    first = _identity(codec, preprocessing=preprocessing)
    second = _identity(codec, preprocessing={"crop": [0, 0, 320, 512], "resize": [320, 512]})

    preprocessing["resize"][0] = 999
    assert first.key == second.key
    assert str(first.relative_path) == f"{first.key[:2]}/{first.key}.pt"


class _OfficialCodecStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return video * self.scale

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents / self.scale


def test_sana_adapter_freezes_codec_and_round_trips_through_normalization():
    normalizer = _normalizer()
    model = _OfficialCodecStub()
    adapter = SanaCausalVideoVAEAdapter(
        model, spec=_codec(normalizer), normalizer=normalizer
    )
    video = torch.tensor([0.5, 1.0, 1.5, 3.0]).view(2, 2, 1, 1, 1)

    normalized = adapter.encode(video)
    decoded = adapter.decode(normalized)

    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert not normalized.requires_grad
    torch.testing.assert_close(decoded, video)
    assert adapter.spec.as_latent_spec().codec_id.endswith("a" * 64)
