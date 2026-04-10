from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenAudioCodec(nn.Module):
    """
    Deterministic frozen codec fallback.

    The interface mirrors a neural audio codec even when a pretrained external
    backend is unavailable. Audio is framed, projected into a latent space with a
    fixed analysis basis, then reconstructed with a matching synthesis basis.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        latent_dim: int = 128,
        frame_size: int = 640,
        hop_length: int = 320,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.frame_size = frame_size
        self.hop_length = hop_length

        basis = self._build_basis(frame_size, latent_dim)
        synthesis = torch.linalg.pinv(basis)
        self.register_buffer("analysis_basis", basis)
        self.register_buffer("synthesis_basis", synthesis)

        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _build_basis(frame_size: int, latent_dim: int):
        time = torch.linspace(0.0, 1.0, frame_size)
        rows = []
        for idx in range(latent_dim):
            frequency = 1.0 + idx // 2
            phase = 0.0 if idx % 2 == 0 else math.pi / 2.0
            rows.append(torch.sin(2.0 * math.pi * frequency * time + phase))
        basis = torch.stack(rows, dim=0)
        basis = F.normalize(basis, dim=1)
        return basis

    def _prepare_waveform(self, waveform: torch.Tensor):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        return waveform.float()

    def encode(self, waveform: torch.Tensor):
        waveform = self._prepare_waveform(waveform)
        pad = (self.frame_size - waveform.shape[-1] % self.hop_length) % self.hop_length
        if pad:
            waveform = F.pad(waveform, (0, pad + self.frame_size))
        else:
            waveform = F.pad(waveform, (0, self.frame_size))
        frames = waveform.unfold(-1, self.frame_size, self.hop_length)
        latents = torch.einsum("btf,df->btd", frames, self.analysis_basis)
        return torch.tanh(latents)

    def decode(self, latents: torch.Tensor):
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)
        frames = torch.einsum("btd,fd->btf", latents, self.synthesis_basis)
        total = (frames.shape[1] - 1) * self.hop_length + self.frame_size
        waveform = frames.new_zeros(frames.shape[0], total)
        window = torch.hann_window(self.frame_size, device=frames.device).clamp_min(1e-3)
        window_sum = frames.new_zeros(total)

        for index in range(frames.shape[1]):
            start = index * self.hop_length
            waveform[:, start : start + self.frame_size] += frames[:, index] * window
            window_sum[start : start + self.frame_size] += window

        waveform = waveform / window_sum.clamp_min(1e-3)
        waveform = waveform / waveform.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        return waveform

    def forward(self, waveform: torch.Tensor):
        return self.encode(waveform)
