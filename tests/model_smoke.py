from pathlib import Path
import math
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
    )
    model = AoedeModel(config)
    batch = 2
    token_ids = torch.randint(0, config.vocab_size, (batch, 12))
    language_ids = torch.tensor([1, 2], dtype=torch.long)
    speaker_embedding = torch.randn(batch, config.speaker_dim)
    target_latents = torch.randn(batch, 40, config.codec_latent_dim)
    reference_latents = torch.randn(batch, 10, config.codec_latent_dim)
    reference_mask = torch.ones(batch, 10, dtype=torch.bool)
    prosody_targets = torch.randn(
        batch,
        math.ceil(target_latents.shape[1] / config.planner_stride),
        config.planner_dim,
    )
    target_durations = torch.randint(1, 5, (batch, 12))

    output = model(
        token_ids=token_ids,
        language_ids=language_ids,
        speaker_embedding=speaker_embedding,
        target_latents=target_latents,
        target_durations=target_durations,
        reference_latents=reference_latents,
        reference_mask=reference_mask,
        prosody_targets=prosody_targets,
        has_reference=torch.ones(batch, dtype=torch.bool),
    )
    assert output["loss"].shape == ()
    assert torch.isfinite(output["planner_loss"])
    assert torch.isfinite(output["memory_speaker_loss"])

    style_latent = torch.randn(batch, config.style_dim)
    speaker_memory, speaker_summary, _ = model.infer_reference_memory(
        reference_latents[:1],
        reference_mask[:1],
    )
    waveform, latents = model.synthesize(
        token_ids=token_ids[:1],
        language_ids=language_ids[:1],
        speaker_embedding=speaker_embedding[:1],
        style_latent=style_latent[:1],
        sampling_steps=4,
        speaker_memory=speaker_memory,
        speaker_summary=speaker_summary,
    )
    assert waveform.dim() == 2
    assert latents.shape[-1] == config.codec_latent_dim
    print("model smoke test passed")


if __name__ == "__main__":
    main()
