import sys
import types

import torch
import torch.nn as nn

from aoede.audio.codec import DacAudioCodec, FrozenAudioCodec, build_audio_codec, codec_cache_key
from aoede.config import ModelConfig


class FakeDacModel(nn.Module):
    sample_rate = 24000
    latent_dim = 1024
    hop_length = 512

    def preprocess(self, audio_data, sample_rate):
        assert sample_rate == self.sample_rate
        return audio_data

    def encode(self, audio_data):
        frames = max(1, audio_data.shape[-1] // self.hop_length)
        pooled = audio_data[..., : frames * self.hop_length].reshape(
            audio_data.shape[0],
            1,
            frames,
            self.hop_length,
        )
        values = pooled.mean(dim=-1)
        z = values.expand(audio_data.shape[0], self.latent_dim, frames).contiguous()
        codes = torch.zeros(audio_data.shape[0], 1, frames, dtype=torch.long)
        return z, codes, z, z.new_zeros(()), z.new_zeros(())

    def decode(self, z):
        audio = z[:, :1].repeat_interleave(self.hop_length, dim=-1)
        return audio.clamp(-1.0, 1.0)


def install_fake_dac(monkeypatch):
    fake_dac = types.SimpleNamespace(
        DAC=types.SimpleNamespace(load=lambda path: FakeDacModel()),
        utils=types.SimpleNamespace(download=lambda model_type: f"{model_type}.dac"),
    )
    monkeypatch.setitem(sys.modules, "dac", fake_dac)


def test_build_audio_codec_defaults_to_frozen_backend():
    config = ModelConfig(codec_latent_dim=24)

    codec = build_audio_codec(config)
    latents = codec.encode(torch.zeros(1, 2048))

    assert isinstance(codec, FrozenAudioCodec)
    assert latents.shape[-1] == 24


def test_dac_backend_round_trips_continuous_latents(monkeypatch):
    install_fake_dac(monkeypatch)
    config = ModelConfig(
        codec_backend="dac",
        codec_latent_dim=1024,
        codec_hop_length=512,
    )

    codec = build_audio_codec(config, device="cpu")
    codec.validate()
    latents = codec.encode(torch.ones(1, 2048))
    waveform = codec.decode(latents)

    assert isinstance(codec, DacAudioCodec)
    assert latents.shape == (1, 4, 1024)
    assert waveform.shape == (1, 2048)
    assert codec.state_dict() == {}


def test_codec_cache_key_separates_real_and_fallback_backends():
    frozen = ModelConfig(codec_backend="frozen", codec_latent_dim=128, speaker_dim=192)
    dac = ModelConfig(
        codec_backend="dac",
        codec_model_type="24khz",
        codec_latent_dim=1024,
        codec_hop_length=512,
        speaker_dim=192,
    )

    assert codec_cache_key(frozen) != codec_cache_key(dac)
    assert codec_cache_key(dac).startswith("dac_24khz_codec1024_hop512")
