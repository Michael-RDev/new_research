from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aoede.config import ModelConfig
from aoede.model.modules import ConditionedTransformerBlock


def masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=1)
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def pool_sequence(sequence: torch.Tensor, stride: int) -> torch.Tensor:
    if stride <= 1:
        return sequence
    pad = (-sequence.shape[1]) % stride
    if pad:
        sequence = F.pad(sequence, (0, 0, 0, pad))
    batch, length, width = sequence.shape
    return sequence.view(batch, length // stride, stride, width).mean(dim=2)


def align_sequence_length(sequence: torch.Tensor, target_length: int) -> torch.Tensor:
    if sequence.shape[1] > target_length:
        return sequence[:, :target_length]
    if sequence.shape[1] < target_length:
        pad = target_length - sequence.shape[1]
        return F.pad(sequence, (0, 0, 0, pad))
    return sequence


def repeat_to_length(sequence: torch.Tensor, repeat_factor: int, target_length: int) -> torch.Tensor:
    if repeat_factor > 1:
        sequence = torch.repeat_interleave(sequence, repeat_factor, dim=1)
    return align_sequence_length(sequence, target_length)


def build_semantic_targets(
    latents: torch.Tensor,
    semantic_dim: int,
    semantic_stride: int,
) -> torch.Tensor:
    pooled = pool_sequence(latents, semantic_stride)
    if pooled.shape[1] == 0:
        return latents.new_zeros(latents.shape[0], 0, semantic_dim)

    shifted = F.pad(pooled[:, :-1], (0, 0, 1, 0))
    delta = pooled - shifted
    energy = pool_sequence(latents.abs().mean(dim=-1, keepdim=True), semantic_stride)
    targets = torch.cat([pooled, delta, energy], dim=-1)

    if targets.shape[-1] > semantic_dim:
        return targets[..., :semantic_dim]
    if targets.shape[-1] < semantic_dim:
        return F.pad(targets, (0, semantic_dim - targets.shape[-1]))
    return targets


def build_masked_semantic_input(
    semantic_targets: torch.Tensor,
    mask_ratio: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor]:
    if semantic_targets.shape[1] == 0:
        mask = semantic_targets.new_zeros(
            semantic_targets.shape[:2],
            dtype=torch.bool,
        )
        return semantic_targets, mask

    mask = torch.rand(
        semantic_targets.shape[0],
        semantic_targets.shape[1],
        device=semantic_targets.device,
    ) < mask_ratio
    if semantic_targets.shape[1] > 1:
        mask[:, 1:] |= mask[:, :-1]
    missing_any = ~mask.any(dim=1)
    if missing_any.any():
        mask[missing_any, 0] = True

    noise = torch.randn_like(semantic_targets) * 0.15
    masked = torch.where(mask.unsqueeze(-1), noise, semantic_targets)
    return masked, mask


class PromptFactorEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.prompt_token_count = config.prompt_token_count
        self.latents_in = nn.Linear(config.codec_latent_dim, config.d_model)
        self.blocks = nn.ModuleList(
            [
                ConditionedTransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_model,
                    dropout=config.memory_dropout,
                )
                for _ in range(2)
            ]
        )
        self.query_tokens = nn.Parameter(
            torch.randn(1 + config.prompt_token_count, config.d_model) * 0.02
        )
        self.query_attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.memory_dropout,
            batch_first=True,
        )
        self.summary_norm = nn.LayerNorm(config.d_model)
        self.prompt_norm = nn.LayerNorm(config.d_model)
        self.style_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.style_dim),
        )

    def forward(
        self,
        reference_latents: torch.Tensor,
        reference_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.latents_in(reference_latents)
        valid = torch.ones(
            reference_latents.shape[0],
            dtype=torch.bool,
            device=reference_latents.device,
        )
        key_padding_mask = None
        if reference_mask is not None:
            safe_mask = reference_mask.clone()
            valid = safe_mask.any(dim=1)
            if safe_mask.shape[1] > 0:
                safe_mask[~valid, 0] = True
            x = x * safe_mask.unsqueeze(-1).to(x.dtype)
            key_padding_mask = ~safe_mask

        condition = masked_mean(x, reference_mask)
        for block in self.blocks:
            x = block(x, condition, key_padding_mask=key_padding_mask)

        queries = self.query_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
        pooled, _ = self.query_attn(
            queries,
            x,
            x,
            key_padding_mask=key_padding_mask,
        )
        summary = self.summary_norm(pooled[:, 0])
        prompt_tokens = self.prompt_norm(pooled[:, 1:])
        style_latent = self.style_head(summary)
        return style_latent, prompt_tokens, summary, valid

    def null_context(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_tokens = torch.zeros(
            batch_size,
            self.prompt_token_count,
            self.query_tokens.shape[-1],
            device=device,
        )
        summary = torch.zeros(
            batch_size,
            self.query_tokens.shape[-1],
            device=device,
        )
        return prompt_tokens, summary


class OccupancyPredictor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.speaker_proj = nn.Linear(config.speaker_dim, config.d_model)
        self.style_proj = nn.Linear(config.style_dim, config.d_model)
        self.prompt_proj = nn.Linear(config.d_model, config.d_model)
        self.language_proj = nn.Embedding(512, config.d_model)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.d_model),
                    nn.Linear(config.d_model, config.d_model),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                for _ in range(max(2, config.duration_predictor_layers))
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        text_states: torch.Tensor,
        speaker_embedding: torch.Tensor,
        style_latent: torch.Tensor,
        language_ids: torch.Tensor,
        prompt_summary: torch.Tensor,
    ) -> torch.Tensor:
        condition = (
            self.speaker_proj(speaker_embedding)
            + self.style_proj(style_latent)
            + self.prompt_proj(prompt_summary)
            + self.language_proj(language_ids)
        )
        x = text_states
        for layer in self.layers:
            x = x + layer(x + condition.unsqueeze(1))
        return self.out(x).squeeze(-1) + 1e-3


class SemanticCanvasGenerator(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.canvas_proj = nn.Linear(config.d_model, config.d_model)
        self.semantic_in = nn.Linear(config.semantic_dim, config.d_model)
        self.prompt_attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.memory_dropout,
            batch_first=True,
        )
        self.language_proj = nn.Embedding(512, config.d_model)
        self.summary_proj = nn.Linear(config.d_model, config.d_model)
        self.blocks = nn.ModuleList(
            [
                ConditionedTransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_model,
                    dropout=config.memory_dropout,
                )
                for _ in range(max(2, config.semantic_layers))
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.semantic_dim),
        )

    def forward(
        self,
        canvas_states: torch.Tensor,
        semantic_input: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_summary: torch.Tensor,
        language_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = self.canvas_proj(canvas_states) + self.semantic_in(semantic_input)
        prompt_context, _ = self.prompt_attn(x, prompt_tokens, prompt_tokens)
        x = x + prompt_context
        condition = self.summary_proj(prompt_summary) + self.language_proj(language_ids)
        for block in self.blocks:
            x = block(x, condition)
        return self.out(x)


class SemanticUpsampler(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.semantic_stride = config.semantic_stride
        self.semantic_proj = nn.Linear(config.semantic_dim, config.d_model)
        self.prompt_attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.memory_dropout,
            batch_first=True,
        )
        self.summary_proj = nn.Linear(config.d_model, config.d_model)
        self.language_proj = nn.Embedding(512, config.d_model)
        self.blocks = nn.ModuleList(
            [
                ConditionedTransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_model,
                    dropout=config.memory_dropout,
                )
                for _ in range(2)
            ]
        )

    def forward(
        self,
        semantic_states: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_summary: torch.Tensor,
        language_ids: torch.Tensor,
        target_length: int,
        residual_text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.semantic_proj(
            repeat_to_length(
                semantic_states,
                repeat_factor=self.semantic_stride,
                target_length=target_length,
            )
        )
        if residual_text is not None:
            x = x + align_sequence_length(residual_text, target_length)
        prompt_context, _ = self.prompt_attn(x, prompt_tokens, prompt_tokens)
        x = x + prompt_context
        condition = self.summary_proj(prompt_summary) + self.language_proj(language_ids)
        for block in self.blocks:
            x = block(x, condition)
        return x
