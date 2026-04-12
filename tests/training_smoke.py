from pathlib import Path
import sys
import tempfile
import wave

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aoede.config import AppConfig, ModelConfig, TrainingConfig
from aoede.audio.codec import FrozenAudioCodec
from aoede.audio.speaker import FrozenSpeakerEncoder
from aoede.data.dataset import ManifestDataset, collate_training_examples
from aoede.data.manifest import ManifestEntry
from aoede.model.core import AoedeModel
from aoede.text.tokenizer import UnicodeTokenizer
from aoede.training.trainer import Trainer


def write_wav(path: Path, freq: float, duration_s: float = 0.7, sample_rate: int = 24000):
    t = np.linspace(0.0, duration_s, num=int(sample_rate * duration_s), endpoint=False)
    signal = 0.2 * np.sin(2.0 * np.pi * freq * t)
    pcm = (signal * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        audio_a = root / "a.wav"
        audio_b = root / "b.wav"
        write_wav(audio_a, 220.0)
        write_wav(audio_b, 330.0)

        tokenizer = UnicodeTokenizer()
        tokenizer.fit(["hello world", "hola mundo"], ["en", "es"])

        config = AppConfig(
            project_root=root,
            model=ModelConfig(
                vocab_size=512,
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
            training=TrainingConfig(batch_size=2, mixed_precision=False),
        )
        config.ensure_directories()

        codec = FrozenAudioCodec(
            sample_rate=config.model.sample_rate,
            latent_dim=config.model.codec_latent_dim,
            frame_size=config.model.codec_frame_size,
            hop_length=config.model.codec_hop_length,
        )
        entries = [
            ManifestEntry(item_id="a", audio_path=str(audio_a), text="hello world", language_code="en", speaker_ref=str(audio_b)),
            ManifestEntry(item_id="b", audio_path=str(audio_b), text="hola mundo", language_code="es"),
        ]
        dataset = ManifestDataset(
            entries,
            tokenizer=tokenizer,
            codec=codec,
            speaker_encoder=FrozenSpeakerEncoder(embedding_dim=config.model.speaker_dim),
            cache_dir=root / "cache",
            planner_stride=config.model.planner_stride,
            planner_dim=config.model.planner_dim,
        )
        batch = collate_training_examples([dataset[0], dataset[1]])

        model = AoedeModel(config.model)
        trainer = Trainer(model, config, device="cpu")
        metrics = trainer.train_step(batch)

        assert metrics["loss"] > 0
        assert "planner_loss" in metrics
        assert "memory_speaker_loss" in metrics
        assert batch["has_reference"].tolist() == [True, False]
        print("training smoke test passed")


if __name__ == "__main__":
    main()
