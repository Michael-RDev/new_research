from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aoede.model.modules import ConditionedTransformerBlock


def masked_mean(x: torch.Tensor, mask: torch.Tensor):
    weights = mask.to(x.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom


def pool_sequence(hidden: torch.Tensor, stride: int):
    if hidden.size(1) == 0:
        return hidden
    pad = (-hidden.size(1)) % stride
    if pad > 0:
        hidden = F.pad(hidden, (0, 0, 0, pad))
    batch_size, total_steps, width = hidden.shape
    hidden = hidden.view(batch_size, total_steps // stride, stride, width)
    return hidden.mean(dim=2)


class SpeakerMemoryEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        num_memory_tokens: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_memory_tokens = num_memory_tokens
        self.latent_proj = nn.Linear(latent_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.depthwise_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=5,
            padding=2,
            groups=hidden_size,
        )
        self.conv_proj = nn.Linear(hidden_size, hidden_size)
        self.memory_queries = nn.Parameter(
            torch.randn(num_memory_tokens, hidden_size) * (hidden_size**-0.5)
        )
        self.memory_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.null_memory = nn.Parameter(torch.randn(num_memory_tokens, hidden_size))
        self.null_summary = nn.Parameter(torch.randn(hidden_size))

    def null_context(self, batch_size: int, device: torch.device):
        null_memory = self.null_memory.to(device).unsqueeze(0).expand(batch_size, -1, -1)
        null_summary = self.null_summary.to(device).unsqueeze(0).expand(batch_size, -1)
        return null_memory, null_summary

    def forward(
        self,
        reference_latents: torch.Tensor,
        reference_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if reference_latents.dim() == 2:
            reference_latents = reference_latents.unsqueeze(0)
        batch_size, ref_steps, _ = reference_latents.shape
        if reference_mask is None:
            reference_mask = torch.ones(
                batch_size,
                ref_steps,
                dtype=torch.bool,
                device=reference_latents.device,
            )
        if reference_mask.dim() == 1:
            reference_mask = reference_mask.unsqueeze(0)
        valid_reference = reference_mask.any(dim=1)
        if ref_steps == 0:
            memory, summary = self.null_context(batch_size, reference_latents.device)
            return memory, summary, valid_reference

        hidden = self.latent_proj(reference_latents)
        hidden = self.norm(hidden)
        conv = self.depthwise_conv(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = hidden + self.conv_proj(conv)
        safe_mask = reference_mask.clone()
        if ref_steps > 0 and not valid_reference.all():
            safe_mask[~valid_reference, 0] = True

        memory_queries = self.memory_queries.unsqueeze(0).expand(batch_size, -1, -1)
        memory_tokens, _ = self.memory_attn(
            query=memory_queries,
            key=hidden,
            value=hidden,
            key_padding_mask=~safe_mask,
        )
        summary = self.summary_proj(masked_mean(hidden, safe_mask))

        if not valid_reference.all():
            memory_tokens = memory_tokens.clone()
            summary = summary.clone()
            null_memory, null_summary = self.null_context(batch_size, reference_latents.device)
            memory_tokens[~valid_reference] = null_memory[~valid_reference]
            summary[~valid_reference] = null_summary[~valid_reference]

        return memory_tokens, summary, valid_reference


class ContinuousProsodyPlanner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        planner_dim: int,
        stride: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.stride = stride
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.prosody_head = nn.Linear(hidden_size, planner_dim)

    def forward(
        self,
        frame_states: torch.Tensor,
        speaker_memory: torch.Tensor,
        prosody_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if frame_states.size(1) == 0:
            empty = frame_states.new_zeros(
                (frame_states.size(0), 0, self.prosody_head.out_features)
            )
            return empty, None

        plan_hidden = pool_sequence(frame_states, self.stride)
        attended, _ = self.cross_attn(
            self.query_norm(plan_hidden),
            self.memory_norm(speaker_memory),
            self.memory_norm(speaker_memory),
        )
        plan_hidden = plan_hidden + attended
        plan_hidden = plan_hidden + self.ffn(plan_hidden)
        plan_vectors = self.prosody_head(plan_hidden)

        planner_loss = None
        if prosody_targets is not None and prosody_targets.size(1) > 0:
            max_steps = min(plan_vectors.size(1), prosody_targets.size(1))
            predicted = plan_vectors[:, :max_steps]
            targets = prosody_targets[:, :max_steps]
            mask = targets.abs().sum(dim=-1) > 0
            if mask.any():
                loss = F.smooth_l1_loss(predicted, targets, reduction="none").mean(dim=-1)
                planner_loss = (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
            else:
                planner_loss = predicted.new_zeros(())

        return plan_vectors, planner_loss


class AtlasComposer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        planner_dim: int,
        decoder_layers: int,
        num_heads: int,
        composer_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.decoder_layers = decoder_layers
        self.summary_proj = nn.Linear(hidden_size, hidden_size)
        self.phrase_proj = nn.Linear(planner_dim, hidden_size)
        self.memory_query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.memory_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                ConditionedTransformerBlock(
                    hidden_size,
                    num_heads,
                    hidden_size,
                    dropout=dropout,
                )
                for _ in range(composer_layers)
            ]
        )
        self.speaker_gate = nn.Linear(hidden_size * 3, 1)
        self.decoder_condition_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, decoder_layers * hidden_size),
        )
        self.out_norm = nn.LayerNorm(hidden_size)

    def _expand_plan(self, plan_vectors: torch.Tensor, frame_count: int, stride: int):
        if frame_count == 0:
            return plan_vectors.new_zeros(plan_vectors.size(0), 0, self.hidden_size)
        if plan_vectors.size(1) == 0:
            return plan_vectors.new_zeros(plan_vectors.size(0), frame_count, self.hidden_size)
        phrase_states = self.phrase_proj(plan_vectors)
        expanded = torch.repeat_interleave(phrase_states, stride, dim=1)
        if expanded.size(1) > frame_count:
            return expanded[:, :frame_count]
        if expanded.size(1) < frame_count:
            pad = frame_count - expanded.size(1)
            expanded = F.pad(expanded, (0, 0, 0, pad))
        return expanded

    def forward(
        self,
        frame_states: torch.Tensor,
        speaker_memory: torch.Tensor,
        speaker_summary: torch.Tensor,
        plan_vectors: torch.Tensor,
        planner_stride: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, frame_count, _ = frame_states.shape
        if frame_count == 0:
            empty_gate = frame_states.new_zeros(batch_size, 0)
            empty_residuals = frame_states.new_zeros(
                batch_size,
                self.decoder_layers,
                self.hidden_size,
            )
            return frame_states, empty_gate, empty_residuals

        attended_memory, _ = self.memory_attn(
            self.memory_query_norm(frame_states),
            self.memory_norm(speaker_memory),
            self.memory_norm(speaker_memory),
        )
        utterance_context = self.summary_proj(speaker_summary).unsqueeze(1).expand(-1, frame_count, -1)
        phrase_context = self._expand_plan(plan_vectors, frame_count, planner_stride)
        speaker_gate = torch.sigmoid(
            self.speaker_gate(
                torch.cat([frame_states, attended_memory, utterance_context], dim=-1)
            )
        )
        composed = frame_states + utterance_context + phrase_context + speaker_gate * attended_memory
        for block in self.blocks:
            composed = block(composed, speaker_summary)
        composed = self.out_norm(composed)

        if plan_vectors.size(1) > 0:
            plan_summary = self.phrase_proj(plan_vectors.mean(dim=1))
        else:
            plan_summary = speaker_summary.new_zeros(batch_size, self.hidden_size)
        decoder_residuals = self.decoder_condition_proj(
            torch.cat([speaker_summary, plan_summary], dim=-1)
        ).view(batch_size, self.decoder_layers, self.hidden_size)
        return composed, speaker_gate.squeeze(-1), decoder_residuals
