from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aoede.audio.codec import build_audio_codec
from aoede.config import ModelConfig
from aoede.model.atlasflow import (
    AtlasComposer,
    ContinuousProsodyPlanner,
    SpeakerMemoryEncoder,
)
from aoede.model.mosaicflow import (
    OccupancyPredictor,
    PromptFactorEncoder,
    SemanticCanvasGenerator,
    SemanticUpsampler,
    align_sequence_length,
    build_masked_semantic_input,
    build_semantic_targets,
    pool_sequence,
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
        self.codec = build_audio_codec(config)
        self.text_encoder = TextEncoder(config)
        self.decoder = FlowMatchingDecoder(config)
        self.speaker_head = nn.Linear(config.d_model, config.speaker_dim)
        if self._mosaicflow_enabled:
            self.duration_predictor = OccupancyPredictor(config)
            self.style_encoder = None
            self.memory_speaker_head = nn.Identity()
            self.prompt_factor_encoder = PromptFactorEncoder(config)
            self.semantic_generator = SemanticCanvasGenerator(config)
            self.semantic_upsampler = SemanticUpsampler(config)
            self.prosody_head = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.planner_dim),
            )
            self.speaker_memory_encoder = None
            self.prosody_planner = None
            self.atlas_composer = None
        else:
            self.duration_predictor = DurationPredictor(config)
            self.style_encoder = StyleEncoder(config)
            self.memory_speaker_head = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
            )
            self.prompt_factor_encoder = None
            self.semantic_generator = None
            self.semantic_upsampler = None
            self.prosody_head = None

        if self.config.architecture_variant == "atlasflow":
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
        return self.config.architecture_variant in {"atlasflow", "mosaicflow"}

    @property
    def _mosaicflow_enabled(self):
        return self.config.architecture_variant == "mosaicflow"

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
        if self._mosaicflow_enabled:
            assert self.prompt_factor_encoder is not None
            style_latent, _, _, _ = self.prompt_factor_encoder(
                reference_latents,
                reference_mask,
            )
            return style_latent
        assert self.style_encoder is not None
        return self.style_encoder(reference_latents, mask=reference_mask)

    def infer_reference_memory(
        self,
        reference_latents: torch.Tensor,
        reference_mask: Optional[torch.Tensor] = None,
    ):
        if self._mosaicflow_enabled:
            assert self.prompt_factor_encoder is not None
            _, prompt_tokens, prompt_summary, valid = self.prompt_factor_encoder(
                reference_latents,
                reference_mask,
            )
            return prompt_tokens, prompt_summary, valid
        if not self._atlasflow_enabled or self.speaker_memory_encoder is None:
            return None, None, None
        return self.speaker_memory_encoder(reference_latents, reference_mask)

    def _default_reference_mask(self, reference_latents: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            reference_latents.shape[0],
            reference_latents.shape[1],
            dtype=torch.bool,
            device=reference_latents.device,
        )

    def _semantic_alignment_targets(
        self,
        text_length: int,
        target_latents: torch.Tensor,
        target_durations: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if target_durations is not None:
            return target_durations
        total_frames = target_latents.shape[1]
        return torch.full(
            (target_latents.shape[0], text_length),
            fill_value=max(total_frames // max(text_length, 1), 1),
            dtype=torch.long,
            device=target_latents.device,
        )

    def _mosaicflow_context(
        self,
        text_states: torch.Tensor,
        language_ids: torch.Tensor,
        speaker_embedding: torch.Tensor,
        target_latents: torch.Tensor,
        target_durations: Optional[torch.Tensor],
        reference_latents: torch.Tensor,
        reference_mask: torch.Tensor,
        prosody_targets: Optional[torch.Tensor],
        has_reference: Optional[torch.Tensor],
    ):
        assert self.prompt_factor_encoder is not None
        assert self.semantic_generator is not None
        assert self.semantic_upsampler is not None
        assert self.prosody_head is not None

        style_latent, prompt_tokens, prompt_summary, valid_reference = (
            self.prompt_factor_encoder(reference_latents, reference_mask)
        )
        duration_prediction = self.duration_predictor(
            text_states,
            speaker_embedding,
            style_latent,
            language_ids,
            prompt_summary,
        )
        target_durations = self._semantic_alignment_targets(
            text_length=text_states.shape[1],
            target_latents=target_latents,
            target_durations=target_durations,
        )
        frame_states = length_regulate(text_states, target_durations)
        frame_states = self.align_frame_states(frame_states, target_latents.shape[1])

        semantic_canvas = pool_sequence(frame_states, self.config.semantic_stride)
        semantic_targets = build_semantic_targets(
            target_latents,
            semantic_dim=self.config.semantic_dim,
            semantic_stride=self.config.semantic_stride,
        )
        semantic_input, semantic_mask = build_masked_semantic_input(semantic_targets)
        predicted_semantics = self.semantic_generator(
            semantic_canvas,
            semantic_input,
            prompt_tokens,
            prompt_summary,
            language_ids,
        )
        semantic_frame_states = self.semantic_upsampler(
            predicted_semantics,
            prompt_tokens,
            prompt_summary,
            language_ids,
            target_length=target_latents.shape[1],
            residual_text=frame_states,
        )

        semantic_error = (predicted_semantics - semantic_targets).pow(2).mean(dim=-1)
        semantic_loss = semantic_error[semantic_mask].mean()
        if not torch.isfinite(semantic_loss):
            semantic_loss = semantic_error.mean()

        target_length = float(target_latents.shape[1])
        coverage_loss = (
            duration_prediction.sum(dim=1) - target_length
        ).abs().mean() / max(target_length, 1.0)

        speaker_prediction = self.speaker_head(semantic_frame_states.mean(dim=1))
        speaker_loss = 1.0 - F.cosine_similarity(
            speaker_prediction,
            speaker_embedding,
            dim=-1,
        ).mean()

        prompt_loss = semantic_loss.new_zeros(())
        if prosody_targets is not None:
            prompt_predictions = pool_sequence(
                semantic_frame_states,
                self.config.planner_stride,
            )
            prompt_predictions = self.prosody_head(prompt_predictions)
            prompt_predictions = align_sequence_length(
                prompt_predictions,
                prosody_targets.shape[1],
            )
            prompt_loss = F.mse_loss(prompt_predictions, prosody_targets)

        if has_reference is not None:
            valid_reference = valid_reference & has_reference.to(
                device=valid_reference.device,
                dtype=torch.bool,
            )

        return {
            "style_latent": style_latent,
            "prompt_tokens": prompt_tokens,
            "prompt_summary": prompt_summary,
            "valid_reference": valid_reference,
            "duration_prediction": duration_prediction,
            "target_durations": target_durations,
            "frame_states": frame_states,
            "decoder_context": semantic_frame_states,
            "predicted_semantics": predicted_semantics,
            "semantic_targets": semantic_targets,
            "semantic_loss": semantic_loss,
            "speaker_loss": speaker_loss,
            "prompt_loss": prompt_loss,
            "coverage_loss": coverage_loss,
        }

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
        if reference_mask is None:
            reference_mask = self._default_reference_mask(reference_latents)

        if self._mosaicflow_enabled:
            mosaic = self._mosaicflow_context(
                text_states=text_states,
                language_ids=language_ids,
                speaker_embedding=speaker_embedding,
                target_latents=target_latents,
                target_durations=target_durations,
                reference_latents=reference_latents,
                reference_mask=reference_mask,
                prosody_targets=prosody_targets,
                has_reference=has_reference,
            )
            flow_loss = self.decoder.flow_loss(
                mosaic["decoder_context"],
                target_latents,
                speaker_embedding,
                mosaic["style_latent"],
                language_ids,
            )
            duration_loss = F.l1_loss(
                mosaic["duration_prediction"],
                mosaic["target_durations"].float(),
            )
            style_reg = mosaic["style_latent"].pow(2).mean()
            total_loss = (
                flow_loss
                + 0.1 * duration_loss
                + 0.1 * mosaic["speaker_loss"]
                + 0.01 * style_reg
                + self.config.semantic_loss_weight * mosaic["semantic_loss"]
                + self.config.prompt_loss_weight * mosaic["prompt_loss"]
                + self.config.coverage_loss_weight * mosaic["coverage_loss"]
            )
            return {
                "loss": total_loss,
                "flow_loss": flow_loss,
                "duration_loss": duration_loss,
                "speaker_loss": mosaic["speaker_loss"],
                "planner_loss": mosaic["prompt_loss"],
                "memory_speaker_loss": mosaic["speaker_loss"],
                "style_reg": style_reg,
                "semantic_loss": mosaic["semantic_loss"],
                "prompt_loss": mosaic["prompt_loss"],
                "coverage_loss": mosaic["coverage_loss"],
            }

        assert self.style_encoder is not None
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
        if self._mosaicflow_enabled:
            assert self.prompt_factor_encoder is not None
            assert self.semantic_generator is not None
            assert self.semantic_upsampler is not None
            durations = self.duration_predictor(
                text_states,
                speaker_embedding,
                style_latent,
                language_ids,
                (
                    speaker_summary
                    if speaker_summary is not None
                    else text_states.mean(dim=1)
                ),
            )
            frame_states = length_regulate(text_states, durations.clamp(1, 20))
            frame_states = frame_states[:, : self.config.max_latent_frames]
            if speaker_memory is None or speaker_summary is None:
                null_memory, null_summary = self.prompt_factor_encoder.null_context(
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

            semantic_canvas = pool_sequence(frame_states, self.config.semantic_stride)
            semantic_input = frame_states.new_zeros(
                frame_states.shape[0],
                semantic_canvas.shape[1],
                self.config.semantic_dim,
            )
            predicted_semantics = self.semantic_generator(
                semantic_canvas,
                semantic_input,
                speaker_memory,
                speaker_summary,
                language_ids,
            )
            predicted_semantics = self.semantic_generator(
                semantic_canvas,
                0.5 * predicted_semantics,
                speaker_memory,
                speaker_summary,
                language_ids,
            )
            decoder_context = self.semantic_upsampler(
                predicted_semantics,
                speaker_memory,
                speaker_summary,
                language_ids,
                target_length=frame_states.shape[1],
                residual_text=frame_states,
            )
            latents = self.decoder.sample(
                decoder_context,
                speaker_embedding,
                style_latent,
                language_ids,
                steps=sampling_steps,
                latent_dim=self.config.codec_latent_dim,
            )
            waveform = self.codec.decode(latents)
            return waveform, latents

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
