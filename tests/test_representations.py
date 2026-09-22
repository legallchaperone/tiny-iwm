from dataclasses import replace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from representations import (
    CacheIdentity,
    ChannelNormalizer,
    CodecSpec,
    NormalizationStats,
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
        execution_dtype="float32",
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


def test_statistics_keep_small_variance_around_a_large_offset():
    values = torch.tensor([1e8 - 1, 1e8 + 1], dtype=torch.float64).view(2, 1, 1, 1, 1)

    normalizer = ChannelNormalizer.fit_from_training(
        [values], split="train", data_version="large-offset-v1"
    )

    assert normalizer.stats.mean == (1e8,)
    assert normalizer.stats.std == (1.0,)


def test_half_precision_normalization_promotes_to_avoid_overflow():
    stats = NormalizationStats(
        mean=(0.0,),
        std=(1e-6,),
        training_data_version="small-scale-v1",
        sample_count=1,
        value_count_per_channel=1,
    )

    normalized = ChannelNormalizer(stats).normalize(
        torch.ones((1, 1, 1, 1, 1), dtype=torch.float16)
    )

    assert normalized.dtype == torch.float32
    assert torch.isfinite(normalized).all()


def test_cache_identity_separates_codec_preprocessing_frames_and_statistics():
    normalizer = _normalizer()
    codec = _codec(normalizer)
    baseline = _identity(codec)

    changed_codec = _identity(replace(codec, weights_sha256="b" * 64))
    changed_precision = _identity(replace(codec, execution_dtype="float16"))
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
            changed_precision.key,
            changed_preprocessing.key,
            changed_frames.key,
            changed_statistics.key,
        }
    ) == 6
    assert baseline.payload["codec"]["encoding_policy"] == "prefix_causal_v1"
    assert baseline.payload["codec"]["execution_dtype"] == "float32"
    assert baseline.payload["normalization"]["training_split"] == "train"


def test_cache_identity_is_stable_across_mapping_insertion_order():
    codec = _codec(_normalizer())
    preprocessing = {"resize": [320, 512], "crop": [0, 0, 320, 512]}
    first = _identity(codec, preprocessing=preprocessing)
    second = _identity(codec, preprocessing={"crop": [0, 0, 320, 512], "resize": [320, 512]})

    preprocessing["resize"][0] = 999
    assert first.key == second.key
    assert str(first.relative_path) == f"{first.key[:2]}/{first.key}.pt"


def test_cache_identity_snapshots_yaml_backed_sequences():
    means = [2.0, 4.0]
    stds = [1.0, 2.0]
    spatial_compression = [8, 8]
    frames = [0, 1, 2]
    stats = NormalizationStats(
        mean=means,
        std=stds,
        training_data_version="sana-train-v1",
        sample_count=2,
        value_count_per_channel=2,
    )
    codec = _codec(
        ChannelNormalizer(stats),
        spatial_compression=spatial_compression,
        normalization=stats,
    )
    identity = _identity(codec, frame_indices=frames)
    original_key = identity.key

    means[0] = 999.0
    stds[0] = 999.0
    spatial_compression[0] = 999
    frames[0] = 999

    assert identity.key == original_key


def test_cache_identity_accepts_composed_hydra_preprocessing():
    preprocessing = OmegaConf.create(
        {"resize": [320, 512], "crop": [0, 0, 320, 512]}
    )

    identity = _identity(_codec(_normalizer()), preprocessing=preprocessing)

    assert len(identity.key) == 64


class _OfficialCodecStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return video * self.scale

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.last_decode_dtype = latents.dtype
        return latents / self.scale


class _AutocastSensitiveCodecStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv3d(2, 2, kernel_size=1, bias=False)

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        encoded = self.projection(video)
        self.last_encode_dtype = encoded.dtype
        return encoded

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = self.projection(latents)
        self.last_decode_dtype = decoded.dtype
        return decoded


def test_sana_adapter_freezes_codec_and_round_trips_through_normalization():
    normalizer = _normalizer()
    model = _OfficialCodecStub()
    adapter = SanaCausalVideoVAEAdapter(
        model, spec=_codec(normalizer), normalizer=normalizer
    )
    video = torch.tensor([0.5, 1.0, 1.5, 3.0]).view(2, 2, 1, 1, 1)

    model.train()
    normalized = adapter.encode(video)
    model.train()
    decoded = adapter.decode(normalized)

    assert not adapter._model.training
    assert all(not parameter.requires_grad for parameter in adapter._model.parameters())
    assert not normalized.requires_grad
    torch.testing.assert_close(decoded, video)
    assert adapter.spec.as_latent_spec().codec_id.endswith("a" * 64)


def test_sana_adapter_snapshots_injected_normalizer():
    normalizer = _normalizer()
    adapter = SanaCausalVideoVAEAdapter(
        _OfficialCodecStub(), spec=_codec(normalizer), normalizer=normalizer
    )
    video = torch.tensor([0.5, 1.0, 1.5, 3.0]).view(2, 2, 1, 1, 1)
    expected = torch.tensor([-1.0, -1.0, 1.0, 1.0]).view_as(video)

    normalizer._stats = replace(normalizer.stats, mean=(999.0, 999.0))

    torch.testing.assert_close(adapter.encode(video), expected)


def test_sana_adapter_snapshots_injected_codec_state():
    normalizer = _normalizer()
    model = _OfficialCodecStub()
    adapter = SanaCausalVideoVAEAdapter(
        model, spec=_codec(normalizer), normalizer=normalizer
    )
    video = torch.tensor([0.5, 1.0, 1.5, 3.0]).view(2, 2, 1, 1, 1)
    expected = torch.tensor([-1.0, -1.0, 1.0, 1.0]).view_as(video)

    with torch.no_grad():
        model.scale.fill_(10.0)

    torch.testing.assert_close(adapter.encode(video), expected)


def test_sana_adapter_restores_reduced_codec_dtype_before_decode():
    normalizer = _normalizer()
    adapter = SanaCausalVideoVAEAdapter(
        _OfficialCodecStub().half(),
        spec=_codec(normalizer, execution_dtype="float16"),
        normalizer=normalizer,
    )
    video = torch.tensor([0.5, 1.0, 1.5, 3.0], dtype=torch.float16).view(
        2, 2, 1, 1, 1
    )

    normalized = adapter.encode(video)
    decoded = adapter.decode(normalized)

    assert normalized.dtype == torch.float32
    assert adapter._model.last_decode_dtype == torch.float16
    torch.testing.assert_close(decoded, video)


def test_sana_adapter_rejects_overflow_when_restoring_codec_dtype():
    stats = NormalizationStats(
        mean=(0.0, 0.0),
        std=(2.0, 2.0),
        training_data_version="wide-latent-v1",
        sample_count=1,
        value_count_per_channel=1,
    )
    normalizer = ChannelNormalizer(stats)
    adapter = SanaCausalVideoVAEAdapter(
        _OfficialCodecStub().half(),
        spec=_codec(
            normalizer, normalization=stats, execution_dtype="float16"
        ),
        normalizer=normalizer,
    )
    normalized = torch.full((1, 2, 1, 1, 1), 40_000.0)

    with pytest.raises(ValueError, match="codec latents must contain finite"):
        adapter.decode(normalized)


def test_codec_spec_rejects_normalization_channel_mismatch():
    normalizer = _normalizer()

    with pytest.raises(ValueError, match="normalization channels"):
        _codec(normalizer, latent_channels=3)


def test_sana_adapter_rejects_execution_dtype_mismatch():
    normalizer = _normalizer()

    with pytest.raises(ValueError, match="execution dtype"):
        SanaCausalVideoVAEAdapter(
            _OfficialCodecStub().half(),
            spec=_codec(normalizer),
            normalizer=normalizer,
        )


def test_sana_adapter_disables_ambient_autocast_for_codec_calls():
    normalizer = _normalizer()
    adapter = SanaCausalVideoVAEAdapter(
        _AutocastSensitiveCodecStub(),
        spec=_codec(normalizer),
        normalizer=normalizer,
    )
    video = torch.ones((1, 2, 1, 1, 1))

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        normalized = adapter.encode(video)
        adapter.decode(normalized)

    assert adapter._model.last_encode_dtype == torch.float32
    assert adapter._model.last_decode_dtype == torch.float32
