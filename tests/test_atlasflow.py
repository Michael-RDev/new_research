import math

import torch

from aoede.config import ModelConfig
from aoede.model.atlasflow import AtlasComposer, ContinuousProsodyPlanner, SpeakerMemoryEncoder
from aoede.model.core import AoedeModel


def build_atlas_config():
    return ModelConfig(
        vocab_size=128,
        d_model=64,
        n_heads=4,
        n_text_layers=2,
        n_decoder_layers=3,
        style_dim=16,
        speaker_dim=24,
        codec_latent_dim=20,
        max_text_tokens=64,
        max_latent_frames=64,
        architecture_variant="atlasflow",
        speaker_memory_tokens=4,
        planner_stride=3,
        planner_dim=18,
        memory_conditioning_heads=4,
        composer_layers=2,
    )


def test_speaker_memory_encoder_shapes_and_null_fallback():
    config = build_atlas_config()
    encoder = SpeakerMemoryEncoder(
        latent_dim=config.codec_latent_dim,
        hidden_size=config.d_model,
        num_memory_tokens=config.speaker_memory_tokens,
        num_heads=config.memory_conditioning_heads,
        dropout=config.memory_dropout,
    )
    latents = torch.randn(2, 7, config.codec_latent_dim)
    mask = torch.tensor(
        [
            [True, True, True, True, True, True, True],
            [False, False, False, False, False, False, False],
        ]
    )
    memory, summary, valid = encoder(latents, mask)
    assert memory.shape == (2, config.speaker_memory_tokens, config.d_model)
    assert summary.shape == (2, config.d_model)
    assert valid.tolist() == [True, False]
    assert torch.isfinite(memory).all()
    assert torch.isfinite(summary).all()


def test_prosody_planner_uses_stride_for_sequence_length():
    config = build_atlas_config()
    planner = ContinuousProsodyPlanner(
        hidden_size=config.d_model,
        planner_dim=config.planner_dim,
        stride=config.planner_stride,
        num_heads=config.memory_conditioning_heads,
        dropout=config.memory_dropout,
    )
    frame_states = torch.randn(2, 10, config.d_model)
    speaker_memory = torch.randn(2, config.speaker_memory_tokens, config.d_model)
    targets = torch.randn(2, math.ceil(frame_states.shape[1] / config.planner_stride), config.planner_dim)
    plan_vectors, planner_loss = planner(frame_states, speaker_memory, targets)
    assert plan_vectors.shape == (2, 4, config.planner_dim)
    assert planner_loss is not None
    assert torch.isfinite(planner_loss)


def test_atlas_composer_returns_composed_states_gate_and_decoder_residuals():
    config = build_atlas_config()
    composer = AtlasComposer(
        hidden_size=config.d_model,
        planner_dim=config.planner_dim,
        decoder_layers=config.n_decoder_layers,
        num_heads=config.memory_conditioning_heads,
        composer_layers=config.composer_layers,
        dropout=config.memory_dropout,
    )
    frame_states = torch.randn(2, 9, config.d_model)
    speaker_memory = torch.randn(2, config.speaker_memory_tokens, config.d_model)
    speaker_summary = torch.randn(2, config.d_model)
    plan_vectors = torch.randn(2, 3, config.planner_dim)
    composed, gate, residuals = composer(
        frame_states,
        speaker_memory,
        speaker_summary,
        plan_vectors,
        planner_stride=config.planner_stride,
    )
    assert composed.shape == frame_states.shape
    assert gate.shape == (2, 9)
    assert residuals.shape == (2, config.n_decoder_layers, config.d_model)
    assert torch.all(gate >= 0.0)
    assert torch.all(gate <= 1.0)


def test_atlasflow_model_handles_null_memory_during_synthesis():
    config = build_atlas_config()
    model = AoedeModel(config)
    token_ids = torch.randint(0, config.vocab_size, (1, 8))
    language_ids = torch.tensor([1], dtype=torch.long)
    speaker_embedding = torch.randn(1, config.speaker_dim)
    style_latent = torch.randn(1, config.style_dim)
    waveform, latents = model.synthesize(
        token_ids=token_ids,
        language_ids=language_ids,
        speaker_embedding=speaker_embedding,
        style_latent=style_latent,
        sampling_steps=3,
    )
    assert waveform.dim() == 2
    assert latents.shape[-1] == config.codec_latent_dim
    assert torch.isfinite(waveform).all()


def test_atlasflow_model_loads_baseline_checkpoint_with_strict_false():
    atlas_config = build_atlas_config()
    baseline_config = build_atlas_config()
    baseline_config.architecture_variant = "baseline"
    baseline_model = AoedeModel(baseline_config)
    atlas_model = AoedeModel(atlas_config)
    incompatible = atlas_model.load_state_dict(baseline_model.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert any("speaker_memory_encoder" in key for key in incompatible.missing_keys)
