from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aoede.audio.codec import FrozenAudioCodec
from aoede.config import ModelConfig
from aoede.model.atlasflow import (
    AtlasComposer,
    ContinuousProsodyPlanner,
    SpeakerMemoryEncoder,
)
from aoede.model.modules import (
    ConditionedTransformerBlock,
    DiTBlock,
    SinusoidalTimeEmbedding,
)


def length_regulate(hidden: torch.Tensor, durations: torch.Tensor):
    durations = durations.round().long().clamp_min(1)
    outputs = []
    for batch_index in range(hidden.shape[0]):
        frames = torch.repeat_interleave(
            hidden[batch_index], durations[batch_index], dim=0
        )
        outputs.append(frames)
    max_len = max(item.shape[0] for item in outputs)
    result = hidden.new_zeros(hidden.shape[0], max_len, hidden.shape[-1])
    for batch_index, item in enumerate(outputs):
        result[batch_index, : item.shape[0]] = item
    return result


class TextEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=0
        )
        self.language_embedding = nn.Embedding(512, config.d_model)
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.max_text_tokens, config.d_model) * 0.01
        )
        self.blocks = nn.ModuleList(
            [
                ConditionedTransformerBlock(
                    config.d_model, config.n_heads, config.d_model
                )
                for _ in range(config.n_text_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, token_ids: torch.Tensor, language_ids: torch.Tensor):
        x = (
            self.token_embedding(token_ids)
            + self.position_embedding[:, : token_ids.shape[1]]
        )
        language_condition = self.language_embedding(language_ids)
        for block in self.blocks:
            x = block(x, language_condition)
        return self.output_norm(x)


class DurationPredictor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        cond_dim = config.d_model
        self.speaker_proj = nn.Linear(config.speaker_dim, cond_dim)
        self.style_proj = nn.Linear(config.style_dim, cond_dim)
        self.language_proj = nn.Embedding(512, cond_dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.d_model),
                    nn.Linear(config.d_model, config.d_model),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                for _ in range(config.duration_predictor_layers)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1), nn.Softplus()
        )

    def forward(
        self,
        text_states: torch.Tensor,
        speaker_embedding: torch.Tensor,
        style_latent: torch.Tensor,
        language_ids: torch.Tensor,
    ):
        condition = (
            self.speaker_proj(speaker_embedding)
            + self.style_proj(style_latent)
            + self.language_proj(language_ids)
        )
        x = text_states
        for layer in self.layers:
            x = x + layer(x) + condition.unsqueeze(1)
        return self.out(x).squeeze(-1) + 1e-3


class StyleEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.latent_proj = nn.Linear(config.codec_latent_dim, config.d_model)
        self.blocks = nn.ModuleList(
            [
                ConditionedTransformerBlock(
                    config.d_model, config.n_heads, config.d_model
                )
                for _ in range(2)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.style_dim),
        )
        self.anchor = nn.Parameter(torch.zeros(config.d_model))

    def forward(self, latents: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.latent_proj(latents)
        condition = self.anchor.unsqueeze(0).expand(x.shape[0], -1)
        for block in self.blocks:
            x = block(x, condition)
        if mask is None:
            pooled = x.mean(dim=1)
        else:
            weights = mask.to(x.dtype).unsqueeze(-1)
            pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.out(pooled)


class FlowMatchingDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        cond_dim = config.d_model
        self.n_layers = config.n_decoder_layers
        self.latents_in = nn.Linear(config.codec_latent_dim, config.d_model)
        self.time_emb = SinusoidalTimeEmbedding(config.d_model)
        self.speaker_proj = nn.Linear(config.speaker_dim, cond_dim)
        self.style_proj = nn.Linear(config.style_dim, cond_dim)
        self.language_proj = nn.Embedding(512, cond_dim)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(config.d_model, config.n_heads, cond_dim)
                for _ in range(config.n_decoder_layers)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.codec_latent_dim),
        )

    def _condition(
        self,
        t: torch.Tensor,
        speaker_embedding: torch.Tensor,
        style_latent: torch.Tensor,
        language_ids: torch.Tensor,
    ):
        return (
            self.time_emb(t)
            + self.speaker_proj(speaker_embedding)
            + self.style_proj(style_latent)
            + self.language_proj(language_ids)
        )

    def forward(
        self,
        noisy_latents: torch.Tensor,
        frame_states: torch.Tensor,
        speaker_embedding: torch.Tensor,
        style_latent: torch.Tensor,
        language_ids: torch.Tensor,
        t: torch.Tensor,
        condition_residuals: Optional[torch.Tensor] = None,
    ):
        x = self.latents_in(noisy_latents)
        condition = self._condition(t, speaker_embedding, style_latent, language_ids)
        for layer_index, block in enumerate(self.blocks):
            layer_condition = condition
            if condition_residuals is not None:
                layer_condition = layer_condition + condition_residuals[:, layer_index]
            x = block(x, frame_states, layer_condition)
        return self.out(x)

    def flow_loss(
        self,
        frame_states: torch.Tensor,
        target_latents: torch.Tensor,
        speaker_embedding: torch.Tensor,
        style_latent: torch.Tensor,
        language_ids: torch.Tensor,
        condition_residuals: Optional[torch.Tensor] = None,
    ):
        batch_size = target_latents.shape[0]
        t = torch.rand(batch_size, device=target_latents.device)
        noise = torch.randn_like(target_latents)
        xt = (1.0 - t[:, None, None]) * noise + t[:, None, None] * target_latents
        target_vector_field = target_latents - noise
        prediction = self.forward(
            xt,
            frame_states,
            speaker_embedding,
            style_latent,
            language_ids,
            t,
            condition_residuals=condition_residuals,
        )
        return F.mse_loss(prediction, target_vector_field)

    @torch.no_grad()
    def sample(
        self,
        frame_states: torch.Tensor,
        speaker_embedding: torch.Tensor,
        style_latent: torch.Tensor,
        language_ids: torch.Tensor,
        steps: int = 18,
        latent_dim: int = 128,
        condition_residuals: Optional[torch.Tensor] = None,
    ):
        x = torch.randn(
            frame_states.shape[0],
            frame_states.shape[1],
            latent_dim,
            device=frame_states.device,
        )
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full(
                (frame_states.shape[0],), step / steps, device=frame_states.device
            )
            x = x + dt * self.forward(
                x,
                frame_states,
                speaker_embedding,
                style_latent,
                language_ids,
                t,
                condition_residuals=condition_residuals,
            )
        return x


class AoedeModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.codec = FrozenAudioCodec(
            sample_rate=config.sample_rate,
            latent_dim=config.codec_latent_dim,
            frame_size=config.codec_frame_size,
            hop_length=config.codec_hop_length,
        )
        self.text_encoder = TextEncoder(config)
        self.duration_predictor = DurationPredictor(config)
        self.style_encoder = StyleEncoder(config)
        self.decoder = FlowMatchingDecoder(config)
        self.speaker_head = nn.Linear(config.d_model, config.speaker_dim)
        self.memory_speaker_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
        )
        if self._atlasflow_enabled:
            self.speaker_memory_encoder = SpeakerMemoryEncoder(
                latent_dim=config.codec_latent_dim,
                hidden_size=config.d_model,
                num_memory_tokens=config.speaker_memory_tokens,
                num_heads=config.memory_conditioning_heads,
                dropout=config.memory_dropout,
            )
            self.prosody_planner = ContinuousProsodyPlanner(
                hidden_size=config.d_model,
                planner_dim=config.planner_dim,
                stride=config.planner_stride,
                num_heads=config.memory_conditioning_heads,
                dropout=config.memory_dropout,
            )
            self.atlas_composer = AtlasComposer(
                hidden_size=config.d_model,
                planner_dim=config.planner_dim,
                decoder_layers=config.n_decoder_layers,
                num_heads=config.memory_conditioning_heads,
                composer_layers=config.composer_layers,
                dropout=config.memory_dropout,
            )
        else:
            self.speaker_memory_encoder = None
            self.prosody_planner = None
            self.atlas_composer = None

    @property
    def _atlasflow_enabled(self):
        return self.config.architecture_variant == "atlasflow"

    def align_frame_states(self, frame_states: torch.Tensor, frame_count: int):
        if frame_states.shape[1] > frame_count:
            return frame_states[:, :frame_count]
        if frame_states.shape[1] < frame_count:
            pad = frame_count - frame_states.shape[1]
            return F.pad(frame_states, (0, 0, 0, pad))
        return frame_states

    def infer_style(
        self,
        reference_latents: torch.Tensor,
        reference_mask: Optional[torch.Tensor] = None,
    ):
        return self.style_encoder(reference_latents, mask=reference_mask)

    def infer_reference_memory(
        self,
        reference_latents: torch.Tensor,
        reference_mask: Optional[torch.Tensor] = None,
    ):
        if not self._atlasflow_enabled or self.speaker_memory_encoder is None:
            return None, None, None
        return self.speaker_memory_encoder(reference_latents, reference_mask)

    def _atlasflow_context(
        self,
        frame_states: torch.Tensor,
        reference_latents: torch.Tensor,
        reference_mask: Optional[torch.Tensor] = None,
        prosody_targets: Optional[torch.Tensor] = None,
        speaker_memory: Optional[torch.Tensor] = None,
        speaker_summary: Optional[torch.Tensor] = None,
    ):
        if not self._atlasflow_enabled:
            return frame_states, None, None, None, None

        if speaker_memory is None or speaker_summary is None:
            assert self.speaker_memory_encoder is not None
            speaker_memory, speaker_summary, valid_reference = (
                self.speaker_memory_encoder(
                    reference_latents,
                    reference_mask,
                )
            )
        else:
            valid_reference = reference_latents.new_ones(
                reference_latents.shape[0],
                dtype=torch.bool,
            )

        assert self.prosody_planner is not None
        assert self.atlas_composer is not None
        plan_vectors, planner_loss = self.prosody_planner(
            frame_states,
            speaker_memory,
            prosody_targets=prosody_targets,
        )
        composed_frame_states, speaker_gate, condition_residuals = self.atlas_composer(
            frame_states,
            speaker_memory,
            speaker_summary,
            plan_vectors,
            planner_stride=self.config.planner_stride,
        )
        return (
            composed_frame_states,
            condition_residuals,
            planner_loss,
            speaker_summary,
            valid_reference,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        language_ids: torch.Tensor,
        speaker_embedding: torch.Tensor,
        target_latents: torch.Tensor,
        target_durations: Optional[torch.Tensor] = None,
        reference_latents: Optional[torch.Tensor] = None,
        reference_mask: Optional[torch.Tensor] = None,
        prosody_targets: Optional[torch.Tensor] = None,
        has_reference: Optional[torch.Tensor] = None,
    ):
        text_states = self.text_encoder(token_ids, language_ids)
        reference_latents = (
            reference_latents if reference_latents is not None else target_latents
        )
        style_latent = self.style_encoder(reference_latents, mask=reference_mask)
        duration_prediction = self.duration_predictor(
            text_states, speaker_embedding, style_latent, language_ids
        )

        if target_durations is None:
            total_frames = target_latents.shape[1]
            target_durations = torch.full_like(
                duration_prediction,
                fill_value=max(total_frames // max(token_ids.shape[1], 1), 1),
            )
        frame_states = length_regulate(text_states, target_durations)
        frame_states = self.align_frame_states(frame_states, target_latents.shape[1])

        if reference_mask is None:
            reference_mask = torch.ones(
                reference_latents.shape[0],
                reference_latents.shape[1],
                dtype=torch.bool,
                device=reference_latents.device,
            )

        (
            composed_frame_states,
            condition_residuals,
            planner_loss,
            speaker_summary,
            valid_reference,
        ) = self._atlasflow_context(
            frame_states,
            reference_latents,
            reference_mask=reference_mask,
            prosody_targets=prosody_targets,
        )
        decoder_context = (
            composed_frame_states if self._atlasflow_enabled else frame_states
        )

        flow_loss = self.decoder.flow_loss(
            decoder_context,
            target_latents,
            speaker_embedding,
            style_latent,
            language_ids,
            condition_residuals=condition_residuals,
        )
        duration_loss = F.l1_loss(duration_prediction, target_durations.float())
        style_reg = style_latent.pow(2).mean()
        planner_loss = (
            planner_loss if planner_loss is not None else flow_loss.new_zeros(())
        )

        if self._atlasflow_enabled and speaker_summary is not None:
            target_summary = F.normalize(
                self.memory_speaker_head(decoder_context.mean(dim=1)),
                dim=-1,
            )
            reference_summary = F.normalize(speaker_summary, dim=-1)
            per_item_memory_loss = 1.0 - F.cosine_similarity(
                target_summary,
                reference_summary,
                dim=-1,
            )
            valid_mask = valid_reference
            if has_reference is not None:
                valid_mask = valid_mask & has_reference.to(
                    device=per_item_memory_loss.device, dtype=torch.bool
                )
            if valid_mask.any():
                memory_speaker_loss = per_item_memory_loss[valid_mask].mean()
            else:
                memory_speaker_loss = per_item_memory_loss.new_zeros(())
            speaker_loss = memory_speaker_loss
            total_loss = (
                flow_loss
                + 0.1 * duration_loss
                + 0.01 * style_reg
                + self.config.planner_loss_weight * planner_loss
                + self.config.memory_speaker_loss_weight * memory_speaker_loss
            )
        else:
            speaker_prediction = self.speaker_head(frame_states.mean(dim=1))
            speaker_loss = (
                1.0
                - F.cosine_similarity(
                    speaker_prediction, speaker_embedding, dim=-1
                ).mean()
            )
            memory_speaker_loss = flow_loss.new_zeros(())
            total_loss = (
                flow_loss + 0.1 * duration_loss + 0.1 * speaker_loss + 0.01 * style_reg
            )

        return {
            "loss": total_loss,
            "flow_loss": flow_loss,
            "duration_loss": duration_loss,
            "speaker_loss": speaker_loss,
            "planner_loss": planner_loss,
            "memory_speaker_loss": memory_speaker_loss,
            "style_reg": style_reg,
        }

    @torch.no_grad()
    def synthesize(
        self,
        token_ids: torch.Tensor,
        language_ids: torch.Tensor,
        speaker_embedding: torch.Tensor,
        style_latent: torch.Tensor,
        sampling_steps: int = 18,
        speaker_memory: Optional[torch.Tensor] = None,
        speaker_summary: Optional[torch.Tensor] = None,
    ):
        text_states = self.text_encoder(token_ids, language_ids)
        durations = self.duration_predictor(
            text_states, speaker_embedding, style_latent, language_ids
        )
        frame_states = length_regulate(text_states, durations.clamp(1, 20))
        frame_states = frame_states[:, : self.config.max_latent_frames]
        condition_residuals = None
        decoder_context = frame_states
        if self._atlasflow_enabled:
            assert self.speaker_memory_encoder is not None
            if speaker_memory is None or speaker_summary is None:
                null_memory, null_summary = self.speaker_memory_encoder.null_context(
                    frame_states.shape[0],
                    frame_states.device,
                )
                if speaker_memory is None:
                    speaker_memory = null_memory
                if speaker_summary is None:
                    speaker_summary = null_summary
            if speaker_memory.dim() == 2:
                speaker_memory = speaker_memory.unsqueeze(0)
            if speaker_summary.dim() == 1:
                speaker_summary = speaker_summary.unsqueeze(0)
            assert self.prosody_planner is not None
            assert self.atlas_composer is not None
            plan_vectors, _ = self.prosody_planner(frame_states, speaker_memory)
            decoder_context, _, condition_residuals = self.atlas_composer(
                frame_states,
                speaker_memory,
                speaker_summary,
                plan_vectors,
                planner_stride=self.config.planner_stride,
            )
        latents = self.decoder.sample(
            decoder_context,
            speaker_embedding,
            style_latent,
            language_ids,
            steps=sampling_steps,
            latent_dim=self.config.codec_latent_dim,
            condition_residuals=condition_residuals,
        )
        waveform = self.codec.decode(latents)
        return waveform, latents
