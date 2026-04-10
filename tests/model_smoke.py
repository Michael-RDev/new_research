from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aoede.config import ModelConfig
from aoede.model.core import AoedeModel


def main():
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=256,
        d_model=96,
        n_heads=4,
        n_text_layers=3,
        n_decoder_layers=3,
        codec_latent_dim=48,
        max_text_tokens=64,
        max_latent_frames=80,
    )
    model = AoedeModel(config)
    batch = 2
    token_ids = torch.randint(0, config.vocab_size, (batch, 12))
    language_ids = torch.tensor([1, 2], dtype=torch.long)
    speaker_embedding = torch.randn(batch, config.speaker_dim)
    target_latents = torch.randn(batch, 40, config.codec_latent_dim)
    target_durations = torch.randint(1, 5, (batch, 12))

    output = model(
        token_ids=token_ids,
        language_ids=language_ids,
        speaker_embedding=speaker_embedding,
        target_latents=target_latents,
        target_durations=target_durations,
    )
    assert output["loss"].shape == ()

    style_latent = torch.randn(batch, config.style_dim)
    waveform, latents = model.synthesize(
        token_ids=token_ids[:1],
        language_ids=language_ids[:1],
        speaker_embedding=speaker_embedding[:1],
        style_latent=style_latent[:1],
        sampling_steps=4,
    )
    assert waveform.dim() == 2
    assert latents.shape[-1] == config.codec_latent_dim
    print("model smoke test passed")


if __name__ == "__main__":
    main()
