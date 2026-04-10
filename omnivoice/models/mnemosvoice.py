#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MnemosVoice modules for speaker-memory and prosody-plan conditioning.

This file contains lightweight components that can be enabled inside the
existing OmniVoice model without changing the public inference API:

- ``SpeakerMemoryEncoder`` compresses prompt audio tokens into persistent
  speaker memory tokens.
- ``ProsodyPlanAdapter`` predicts a coarse acoustic plan over downsampled
  target positions.
- ``MemoryConditioner`` injects the speaker memory and plan tokens into the
  target hidden states before the audio-token heads.

The design goal is to improve zero-shot voice cloning without replacing the
baseline LLM-backed masked audio-token decoder.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool a sequence with a boolean mask."""
    weights = mask.to(x.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom


def pool_target_states(
    target_hidden: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """Average-pool a target sequence into coarse planning steps."""
    if target_hidden.size(1) == 0:
        return target_hidden
    pad = (-target_hidden.size(1)) % stride
    if pad > 0:
        target_hidden = F.pad(target_hidden, (0, 0, 0, pad))
    bsz, total_steps, dim = target_hidden.shape
    target_hidden = target_hidden.view(bsz, total_steps // stride, stride, dim)
    return target_hidden.mean(dim=2)


def pool_target_tokens(
    target_tokens: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """Create coarse token targets by keeping the first item in each block."""
    if target_tokens.size(1) == 0:
        return target_tokens
    pad = (-target_tokens.size(1)) % stride
    if pad > 0:
        pad_values = target_tokens[:, -1:].expand(-1, pad)
        target_tokens = torch.cat([target_tokens, pad_values], dim=1)
    return target_tokens[:, ::stride]


class SpeakerMemoryEncoder(nn.Module):
    """Compress prompt audio embeddings into a small speaker-memory bank."""

    def __init__(
        self,
        hidden_size: int,
        num_memory_tokens: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
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
        self.global_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.null_memory = nn.Parameter(torch.randn(num_memory_tokens, hidden_size))
        self.null_global = nn.Parameter(torch.randn(hidden_size))

    def forward(
        self,
        prompt_hidden: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return speaker tokens, pooled speaker vector, and valid-prompt mask."""
        bsz = prompt_hidden.size(0)
        valid_prompt = prompt_mask.any(dim=1)
        if prompt_hidden.size(1) == 0:
            return (
                self.null_memory.unsqueeze(0).expand(bsz, -1, -1),
                self.null_global.unsqueeze(0).expand(bsz, -1),
                valid_prompt,
            )

        hidden = self.norm(prompt_hidden)
        conv = self.depthwise_conv(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = hidden + self.conv_proj(conv)

        memory_queries = self.memory_queries.unsqueeze(0).expand(bsz, -1, -1)
        memory_tokens, _ = self.memory_attn(
            query=memory_queries,
            key=hidden,
            value=hidden,
            key_padding_mask=~prompt_mask,
        )
        pooled = self.global_proj(masked_mean(hidden, prompt_mask))

        if not valid_prompt.all():
            memory_tokens = memory_tokens.clone()
            pooled = pooled.clone()
            memory_tokens[~valid_prompt] = self.null_memory
            pooled[~valid_prompt] = self.null_global

        return memory_tokens, pooled, valid_prompt


class ProsodyPlanAdapter(nn.Module):
    """Predict and embed a coarse target-side acoustic plan."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
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
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.plan_head = nn.Linear(hidden_size, vocab_size)
        self.plan_embedding = nn.Embedding(vocab_size, hidden_size)
        self.null_plan = nn.Parameter(torch.randn(1, hidden_size))

    def forward(
        self,
        target_hidden: torch.Tensor,
        speaker_tokens: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Return plan memory, optional plan loss, and chosen coarse tokens."""
        if target_hidden.size(1) == 0:
            empty_tokens = target_hidden.new_zeros(
                (target_hidden.size(0), 0), dtype=torch.long
            )
            return (
                self.null_plan.expand(target_hidden.size(0), -1, -1),
                None,
                empty_tokens,
            )

        plan_hidden = pool_target_states(target_hidden, self.stride)
        plan_hidden = self.query_norm(plan_hidden)
        speaker_tokens = self.memory_norm(speaker_tokens)
        attended, _ = self.cross_attn(plan_hidden, speaker_tokens, speaker_tokens)
        plan_hidden = plan_hidden + attended
        plan_hidden = plan_hidden + self.ffn(plan_hidden)
        plan_logits = self.plan_head(plan_hidden)

        plan_loss = None
        if target_tokens is not None and target_tokens.size(1) > 0:
            coarse_targets = pool_target_tokens(target_tokens, self.stride)
            max_steps = min(coarse_targets.size(1), plan_logits.size(1))
            coarse_targets = coarse_targets[:, :max_steps]
            plan_logits = plan_logits[:, :max_steps]
            plan_loss = F.cross_entropy(
                plan_logits.reshape(-1, plan_logits.size(-1)),
                coarse_targets.reshape(-1),
            )
            plan_tokens = coarse_targets
        else:
            plan_tokens = plan_logits.argmax(dim=-1)

        plan_memory = plan_hidden[:, : plan_tokens.size(1)] + self.plan_embedding(
            plan_tokens
        )
        return plan_memory, plan_loss, plan_tokens


class MemoryConditioner(nn.Module):
    """Inject speaker memory and coarse plan tokens into target states."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(
        self,
        target_hidden: torch.Tensor,
        memory_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if target_hidden.size(1) == 0 or memory_tokens.size(1) == 0:
            return target_hidden

        attended, _ = self.cross_attn(
            self.query_norm(target_hidden),
            self.memory_norm(memory_tokens),
            self.memory_norm(memory_tokens),
        )
        gate = torch.sigmoid(
            self.gate(torch.cat([target_hidden, attended], dim=-1))
        )
        target_hidden = target_hidden + gate * attended
        target_hidden = target_hidden + self.ffn(target_hidden)
        return target_hidden
