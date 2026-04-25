from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aoede.config import ModelConfig
from aoede.model.core import TextEncoder
from aoede.model.modules import DiTBlock, SinusoidalTimeEmbedding


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    error = (prediction - target).pow(2).mean(dim=-1)
    if mask is None:
        return error.mean()
    weights = mask.to(error.dtype)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=1)
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class SotaResidualFlowModel(nn.Module):
    """
    Residual flow refiner over a frozen pretrained teacher proposal.

    The model never has to synthesize speech from pure noise. It learns a
    velocity field that moves normalized teacher DAC latents toward normalized
    real DAC latents, conditioned on text, speaker identity, reference latents,
    and the teacher proposal itself.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.text_encoder = TextEncoder(config)
        self.teacher_in = nn.Linear(config.codec_latent_dim, config.d_model)
        self.noisy_in = nn.Linear(config.codec_latent_dim, config.d_model)
        self.reference_in = nn.Linear(config.codec_latent_dim, config.d_model)
        self.speaker_proj = nn.Linear(config.speaker_dim, config.d_model)
        self.language_proj = nn.Embedding(512, config.d_model)
        self.time_emb = SinusoidalTimeEmbedding(config.d_model)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(config.d_model, config.n_heads, config.d_model)
                for _ in range(config.n_decoder_layers)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.codec_latent_dim),
        )
        self.speaker_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.speaker_dim),
        )

    def _condition(
        self,
        token_ids: torch.Tensor,
        language_ids: torch.Tensor,
        speaker_embedding: torch.Tensor,
        reference_latents: torch.Tensor,
        reference_mask: Optional[torch.Tensor],
        t: torch.Tensor,
    ) -> torch.Tensor:
        text_summary = self.text_encoder(token_ids, language_ids).mean(dim=1)
        reference_summary = masked_mean(
            self.reference_in(reference_latents),
            reference_mask,
        )
        return (
            text_summary
            + reference_summary
            + self.speaker_proj(speaker_embedding)
            + self.language_proj(language_ids)
            + self.time_emb(t)
        )

    def forward(
        self,
        noisy_latents: torch.Tensor,
        teacher_latents: torch.Tensor,
        token_ids: torch.Tensor,
        language_ids: torch.Tensor,
        speaker_embedding: torch.Tensor,
        reference_latents: torch.Tensor,
        t: torch.Tensor,
        reference_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.noisy_in(noisy_latents) + self.teacher_in(teacher_latents)
        condition = self._condition(
            token_ids=token_ids,
            language_ids=language_ids,
            speaker_embedding=speaker_embedding,
            reference_latents=reference_latents,
            reference_mask=reference_mask,
            t=t,
        )
        teacher_states = self.teacher_in(teacher_latents)
        for block in self.blocks:
            x = block(x, teacher_states, condition)
        return self.out(x)

    def loss(
        self,
        token_ids: torch.Tensor,
        language_ids: torch.Tensor,
        speaker_embedding: torch.Tensor,
        target_latents: torch.Tensor,
        teacher_latents: torch.Tensor,
        reference_latents: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        reference_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = target_latents.shape[0]
        t = torch.rand(batch_size, device=target_latents.device)
        noise = torch.randn_like(target_latents)
        sigma = 0.05 * torch.sin(torch.pi * t).view(-1, 1, 1)
        xt = (
            (1.0 - t.view(-1, 1, 1)) * teacher_latents
            + t.view(-1, 1, 1) * target_latents
            + sigma * noise
        )
        target_velocity = target_latents - teacher_latents
        velocity = self.forward(
            noisy_latents=xt,
            teacher_latents=teacher_latents,
            token_ids=token_ids,
            language_ids=language_ids,
            speaker_embedding=speaker_embedding,
            reference_latents=reference_latents,
            reference_mask=reference_mask,
            t=t,
        )
        flow_loss = masked_mse(velocity, target_velocity, target_mask)

        refined = teacher_latents + self.forward(
            noisy_latents=teacher_latents,
            teacher_latents=teacher_latents,
            token_ids=token_ids,
            language_ids=language_ids,
            speaker_embedding=speaker_embedding,
            reference_latents=reference_latents,
            reference_mask=reference_mask,
            t=torch.ones(batch_size, device=target_latents.device),
        )
        recon_loss = masked_mse(refined, target_latents, target_mask)
        anchor_loss = masked_mse(refined, teacher_latents, target_mask)

        pooled_refined = masked_mean(self.teacher_in(refined), target_mask)
        speaker_prediction = F.normalize(self.speaker_head(pooled_refined), dim=-1)
        speaker_target = F.normalize(speaker_embedding, dim=-1)
        speaker_loss = 1.0 - F.cosine_similarity(
            speaker_prediction,
            speaker_target,
            dim=-1,
        ).mean()

        semantic_loss = masked_mse(
            refined.mean(dim=-1, keepdim=True),
            target_latents.mean(dim=-1, keepdim=True),
            target_mask,
        )
        total = (
            flow_loss
            + 0.5 * recon_loss
            + 0.1 * speaker_loss
            + 0.1 * semantic_loss
            + 0.01 * anchor_loss
        )
        return {
            "loss": total,
            "flow_loss": flow_loss,
            "recon_loss": recon_loss,
            "speaker_loss": speaker_loss,
            "semantic_loss": semantic_loss,
            "anchor_loss": anchor_loss,
        }

    @torch.no_grad()
    def refine(
        self,
        token_ids: torch.Tensor,
        language_ids: torch.Tensor,
        speaker_embedding: torch.Tensor,
        teacher_latents: torch.Tensor,
        reference_latents: torch.Tensor,
        reference_mask: Optional[torch.Tensor] = None,
        steps: int = 4,
    ) -> torch.Tensor:
        x = teacher_latents
        dt = 1.0 / max(steps, 1)
        for index in range(max(steps, 1)):
            t = torch.full(
                (teacher_latents.shape[0],),
                min((index + 1) * dt, 1.0),
                device=teacher_latents.device,
            )
            velocity = self.forward(
                noisy_latents=x,
                teacher_latents=teacher_latents,
                token_ids=token_ids,
                language_ids=language_ids,
                speaker_embedding=speaker_embedding,
                reference_latents=reference_latents,
                reference_mask=reference_mask,
                t=t,
            )
            x = x + dt * velocity
        return x
