import io
import wave
from pathlib import Path

import numpy as np

from aoede.config import AppConfig, ModelConfig, TrainingConfig
from aoede.schemas import SynthesisRequest
from aoede.service import AoedeService


def make_wav_bytes(duration_s: float = 0.7, sample_rate: int = 24000):
    t = np.linspace(0.0, duration_s, num=int(sample_rate * duration_s), endpoint=False)
    signal = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)
    pcm = (signal * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def test_torch_runtime_enrollment_persists_atlasflow_memory(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AOEDE_RUNTIME", "torch")
    monkeypatch.delenv("AOEDE_DISABLE_TORCH", raising=False)
    config = AppConfig(
        project_root=tmp_path,
        model=ModelConfig(
            vocab_size=256,
            d_model=96,
            n_heads=4,
            n_text_layers=3,
            n_decoder_layers=3,
            style_dim=24,
            speaker_dim=48,
            codec_latent_dim=48,
            max_text_tokens=64,
            max_latent_frames=80,
            architecture_variant="atlasflow",
            speaker_memory_tokens=4,
            planner_stride=4,
            planner_dim=32,
            memory_conditioning_heads=4,
            composer_layers=2,
        ),
        training=TrainingConfig(mixed_precision=False),
    )
    config.ensure_directories()
    service = AoedeService(config)
    assert service.runtime_name == "torch"

    response = service.enroll(
        make_wav_bytes(),
        voice_id="atlas-service",
        metadata={"filename": "atlas.wav"},
    )
    profile = service.profile_store.load(response.voice_id)
    assert profile is not None
    assert profile.speaker_memory is not None
    assert profile.speaker_summary is not None

    audio_bytes, duration = service.synthesize(
        SynthesisRequest(
            text="atlas flow synthesis",
            language_code="en",
            voice_id=response.voice_id,
            sampling_steps=4,
        )
    )
    assert duration > 0
    assert len(audio_bytes) > 44
