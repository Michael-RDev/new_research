from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        half_dim = d_model // 2
        scale = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim) * scale)
        self.register_buffer("frequencies", frequencies)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor):
        angles = t.unsqueeze(-1) * self.frequencies.unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.proj(emb)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.affine = nn.Linear(cond_dim, d_model * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        gamma, beta = self.affine(condition).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class ConditionedTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = AdaptiveLayerNorm(d_model, cond_dim)
        self.norm2 = AdaptiveLayerNorm(d_model, cond_dim)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        attn_input = self.norm1(x, condition)
        residual, _ = self.attn(attn_input, attn_input, attn_input, key_padding_mask=key_padding_mask)
        x = x + residual
        x = x + self.ff(self.norm2(x, condition))
        return x


class DiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = AdaptiveLayerNorm(d_model, cond_dim)
        self.norm2 = AdaptiveLayerNorm(d_model, cond_dim)
        self.norm3 = AdaptiveLayerNorm(d_model, cond_dim)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, condition: torch.Tensor):
        h = self.norm1(x, condition)
        residual, _ = self.self_attn(h, h, h)
        x = x + residual
        h = self.norm2(x, condition)
        residual, _ = self.cross_attn(h, context, context)
        x = x + residual
        x = x + self.ff(self.norm3(x, condition))
        return x
