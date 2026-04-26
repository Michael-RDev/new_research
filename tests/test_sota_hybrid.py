import sys
import types
from pathlib import Path

import numpy as np
import torch

from aoede.audio.latent_stats import LatentStats
from aoede.audio.speaker import build_speaker_encoder, speaker_cache_key
from aoede.config import ModelConfig
from aoede.data.sota_distill import (
    SotaDistillDataset,
    SotaDistillEntry,
    collate_sota_distill,
)
from aoede.model.residualflow import SotaResidualFlowModel
from aoede.providers import get_provider
from aoede.text.tokenizer import UnicodeTokenizer


def test_voxcpm_provider_maps_clone_arguments(monkeypatch):
    calls = {}

    class FakeModel:
        tts_model = types.SimpleNamespace(sample_rate=48000)

        def generate(self, **kwargs):
            calls.update(kwargs)
            return np.ones(128, dtype=np.float32) * 0.1

    class FakeVoxCPM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model_id"] = model_id
            calls["load_kwargs"] = kwargs
            return FakeModel()

    monkeypatch.setitem(sys.modules, "voxcpm", types.SimpleNamespace(VoxCPM=FakeVoxCPM))

    provider = get_provider("voxcpm2", device="cpu", model_id="fake/VoxCPM2")
    result = provider.synthesize(
        text="hello",
        reference_audio="voice.wav",
        language="en",
        prompt_text="reference words",
    )

    assert result.provider == "voxcpm2"
    assert result.sample_rate == 48000
    assert calls["model_id"] == "fake/VoxCPM2"
    assert calls["text"] == "hello"
    assert calls["reference_wav_path"] == "voice.wav"
    assert calls["prompt_wav_path"] == "voice.wav"
    assert calls["prompt_text"] == "reference words"


def test_qwen3_provider_maps_official_clone_arguments(monkeypatch):
    calls = {}

    class FakeQwenModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model_id"] = model_id
            calls["load_kwargs"] = kwargs
            return FakeQwenModel()

        def create_voice_clone_prompt(self, **kwargs):
            calls["prompt"] = kwargs
            return {"cached": True}

        def generate_voice_clone(self, **kwargs):
            calls["generate"] = kwargs
            return [np.ones(64, dtype=np.float32)], 24000

    qwen_module = types.SimpleNamespace(Qwen3TTSModel=FakeQwenModel)
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_module)

    provider = get_provider("qwen3", device="cpu", model_id="fake/Qwen3")
    result = provider.synthesize(
        text="hello",
        reference_audio="voice.wav",
        language="en",
        prompt_text="reference words",
    )

    assert result.provider == "qwen3"
    assert calls["model_id"] == "fake/Qwen3"
    assert calls["prompt"]["ref_audio"] == "voice.wav"
    assert calls["prompt"]["ref_text"] == "reference words"
    assert calls["prompt"]["x_vector_only_mode"] is False
    assert calls["generate"]["language"] == "English"
    assert calls["generate"]["voice_clone_prompt"] == {"cached": True}


def test_speechbrain_ecapa_encoder_shape_and_cache_key(monkeypatch):
    class FakeClassifier:
        def encode_batch(self, signal):
            return torch.ones(1, 1, 192)

    class FakeEncoderClassifier:
        @staticmethod
        def from_hparams(**kwargs):
            return FakeClassifier()

    speaker_module = types.SimpleNamespace(EncoderClassifier=FakeEncoderClassifier)
    inference_module = types.SimpleNamespace(speaker=speaker_module)
    speechbrain_module = types.SimpleNamespace(inference=inference_module)
    monkeypatch.setitem(sys.modules, "speechbrain", speechbrain_module)
    monkeypatch.setitem(sys.modules, "speechbrain.inference", inference_module)
    monkeypatch.setitem(sys.modules, "speechbrain.inference.speaker", speaker_module)

    encoder = build_speaker_encoder("ecapa", device="cpu")
    embedding = encoder.encode(np.zeros(16000, dtype=np.float32), sample_rate=16000)

    assert embedding.shape == (192,)
    assert np.isclose(np.linalg.norm(embedding), 1.0)
    assert speaker_cache_key("speechbrain", 192).startswith("ecapa_")


def test_latent_stats_normalize_round_trip():
    stats = LatentStats(
        mean=torch.tensor([1.0, -1.0]),
        std=torch.tensor([2.0, 4.0]),
        count=8,
    )
    latents = torch.tensor([[[3.0, 3.0]]])

    normalized = stats.normalize(latents)
    restored = stats.denormalize(normalized)

    assert torch.allclose(restored, latents)


def test_sota_residualflow_loss_and_refine_shapes():
    config = ModelConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_text_layers=1,
        n_decoder_layers=1,
        codec_latent_dim=8,
        speaker_dim=16,
        max_text_tokens=16,
        architecture_variant="sota_residualflow",
    )
    model = SotaResidualFlowModel(config)
    batch = 2
    length = 5
    token_ids = torch.randint(0, config.vocab_size, (batch, 6))
    language_ids = torch.tensor([1, 2])
    speaker = torch.randn(batch, config.speaker_dim)
    target = torch.randn(batch, length, config.codec_latent_dim)
    teacher = torch.randn(batch, length, config.codec_latent_dim)
    reference = torch.randn(batch, 3, config.codec_latent_dim)
    mask = torch.ones(batch, length, dtype=torch.bool)

    losses = model.loss(
        token_ids,
        language_ids,
        speaker,
        target,
        teacher,
        reference,
        target_mask=mask,
    )
    refined = model.refine(
        token_ids,
        language_ids,
        speaker,
        teacher,
        reference,
        steps=2,
    )

    assert losses["loss"].shape == ()
    assert torch.isfinite(losses["loss"])
    assert refined.shape == teacher.shape


def test_sota_distill_dataset_collates_normalized_latents(tmp_path: Path):
    tokenizer = UnicodeTokenizer()
    tokenizer.fit(["hello"], ["en"])
    stats = LatentStats(mean=torch.zeros(4), std=torch.ones(4), count=10)
    real = tmp_path / "real.pt"
    teacher = tmp_path / "teacher.pt"
    reference = tmp_path / "reference.pt"
    speaker = tmp_path / "speaker.pt"
    torch.save(torch.ones(3, 4), real)
    torch.save(torch.zeros(3, 4), teacher)
    torch.save(torch.ones(2, 4), reference)
    torch.save(torch.ones(192), speaker)
    entry = SotaDistillEntry(
        item_id="a",
        text="hello",
        language_code="en",
        audio_path="a.wav",
        speaker_ref=None,
        real_latents_path=str(real),
        teacher_latents_path=str(teacher),
        reference_latents_path=str(reference),
        speaker_embedding_path=str(speaker),
    )

    dataset = SotaDistillDataset([entry], tokenizer, stats)
    batch = collate_sota_distill([dataset[0]])

    assert batch["target_latents"].shape == (1, 3, 4)
    assert batch["teacher_latents"].shape == (1, 3, 4)
    assert batch["reference_mask"].tolist() == [[True, True]]


def test_sota_distill_dataset_does_not_grow_tokenizer_for_unknown_text(tmp_path: Path):
    tokenizer = UnicodeTokenizer()
    tokenizer.fit(["hello"], ["en"])
    initial_size = tokenizer.size
    stats = LatentStats(mean=torch.zeros(4), std=torch.ones(4), count=10)
    real = tmp_path / "real.pt"
    teacher = tmp_path / "teacher.pt"
    reference = tmp_path / "reference.pt"
    speaker = tmp_path / "speaker.pt"
    torch.save(torch.ones(3, 4), real)
    torch.save(torch.zeros(3, 4), teacher)
    torch.save(torch.ones(2, 4), reference)
    torch.save(torch.ones(192), speaker)
    entry = SotaDistillEntry(
        item_id="a",
        text="hello 🐉",
        language_code="en",
        audio_path="a.wav",
        speaker_ref=None,
        real_latents_path=str(real),
        teacher_latents_path=str(teacher),
        reference_latents_path=str(reference),
        speaker_embedding_path=str(speaker),
    )

    dataset = SotaDistillDataset([entry], tokenizer, stats)
    example = dataset[0]

    assert tokenizer.size == initial_size
    assert int(example["token_ids"].max()) < initial_size
    assert tokenizer.unk_id in example["token_ids"].tolist()
