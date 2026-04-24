import torch

from aoede.config import ModelConfig
from aoede.model.core import AoedeModel
from aoede.model.mosaicflow import build_masked_semantic_input


def build_config():
    return ModelConfig(
        vocab_size=128,
        d_model=64,
        n_heads=4,
        n_text_layers=2,
        n_decoder_layers=3,
        semantic_layers=2,
        style_dim=16,
        speaker_dim=24,
        codec_latent_dim=20,
        semantic_dim=18,
        semantic_stride=2,
        prompt_token_count=4,
        max_text_tokens=64,
        max_latent_frames=64,
        architecture_variant="mosaicflow",
        planner_stride=2,
        planner_dim=12,
    )


def test_mosaicflow_forward_returns_factorized_losses():
    config = build_config()
    model = AoedeModel(config)
    token_ids = torch.randint(0, config.vocab_size, (2, 10))
    language_ids = torch.tensor([1, 2], dtype=torch.long)
    speaker_embedding = torch.randn(2, config.speaker_dim)
    target_latents = torch.randn(2, 24, config.codec_latent_dim)
    reference_latents = torch.randn(2, 8, config.codec_latent_dim)
    reference_mask = torch.ones(2, 8, dtype=torch.bool)
    prosody_targets = torch.randn(2, 12, config.planner_dim)
    target_durations = torch.randint(1, 4, (2, 10))

    output = model(
        token_ids=token_ids,
        language_ids=language_ids,
        speaker_embedding=speaker_embedding,
        target_latents=target_latents,
        target_durations=target_durations,
        reference_latents=reference_latents,
        reference_mask=reference_mask,
        prosody_targets=prosody_targets,
        has_reference=torch.ones(2, dtype=torch.bool),
    )

    assert output["loss"].shape == ()
    assert torch.isfinite(output["semantic_loss"])
    assert torch.isfinite(output["prompt_loss"])
    assert torch.isfinite(output["coverage_loss"])


def test_masked_semantic_input_expands_mask_without_aliasing():
    targets = torch.randn(3, 12, 8)

    masked, mask = build_masked_semantic_input(targets)

    assert masked.shape == targets.shape
    assert mask.shape == targets.shape[:2]
    assert mask.any(dim=1).all()


def test_mosaicflow_can_extract_prompt_memory_and_synthesize():
    config = build_config()
    model = AoedeModel(config)
    reference_latents = torch.randn(1, 8, config.codec_latent_dim)
    reference_mask = torch.ones(1, 8, dtype=torch.bool)
    style_latent = model.infer_style(reference_latents, reference_mask)
    prompt_tokens, prompt_summary, valid = model.infer_reference_memory(
        reference_latents,
        reference_mask,
    )
    waveform, latents = model.synthesize(
        token_ids=torch.randint(0, config.vocab_size, (1, 8)),
        language_ids=torch.tensor([1], dtype=torch.long),
        speaker_embedding=torch.randn(1, config.speaker_dim),
        style_latent=style_latent,
        sampling_steps=3,
        speaker_memory=prompt_tokens,
        speaker_summary=prompt_summary,
    )

    assert valid.tolist() == [True]
    assert waveform.dim() == 2
    assert latents.shape[-1] == config.codec_latent_dim
    assert torch.isfinite(waveform).all()
