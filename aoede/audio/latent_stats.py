from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch


@dataclass
class LatentStats:
    mean: torch.Tensor
    std: torch.Tensor
    count: int
    eps: float = 1e-5

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=latents.device, dtype=latents.dtype)
        std = self.std.to(device=latents.device, dtype=latents.dtype)
        return (latents - mean) / std.clamp_min(self.eps)

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=latents.device, dtype=latents.dtype)
        std = self.std.to(device=latents.device, dtype=latents.dtype)
        return latents * std.clamp_min(self.eps) + mean

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
            "count": int(self.count),
            "eps": float(self.eps),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LatentStats":
        return cls(
            mean=torch.tensor(payload["mean"], dtype=torch.float32),
            std=torch.tensor(payload["std"], dtype=torch.float32),
            count=int(payload.get("count", 0)),
            eps=float(payload.get("eps", 1e-5)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LatentStats":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class RunningLatentStats:
    def __init__(self, latent_dim: int):
        self.latent_dim = latent_dim
        self.count = 0
        self.sum = torch.zeros(latent_dim, dtype=torch.float64)
        self.sum_sq = torch.zeros(latent_dim, dtype=torch.float64)

    def update(self, latents: torch.Tensor) -> None:
        if latents.numel() == 0:
            return
        flattened = latents.detach().cpu().float().reshape(-1, latents.shape[-1])
        if flattened.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, got {flattened.shape[-1]}."
            )
        self.count += int(flattened.shape[0])
        self.sum += flattened.double().sum(dim=0)
        self.sum_sq += flattened.double().pow(2).sum(dim=0)

    def finalize(self) -> LatentStats:
        if self.count <= 0:
            raise RuntimeError("Cannot finalize latent stats with zero frames.")
        mean = self.sum / self.count
        var = (self.sum_sq / self.count) - mean.pow(2)
        std = var.clamp_min(1e-8).sqrt()
        return LatentStats(
            mean=mean.float(),
            std=std.float(),
            count=self.count,
        )


def compute_latent_stats(
    paths: Iterable[Path],
    latent_dim: int,
) -> LatentStats:
    running = RunningLatentStats(latent_dim=latent_dim)
    for path in paths:
        running.update(torch.load(path, map_location="cpu"))
    return running.finalize()


def align_latent_pair(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(left.shape[0], right.shape[0])
    if length <= 0:
        raise ValueError("Cannot align empty latent sequences.")
    return left[:length], right[:length]


def pad_latent_sequences(sequences: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(sequence.shape[0] for sequence in sequences)
    latent_dim = sequences[0].shape[-1]
    batch = torch.zeros(len(sequences), max_length, latent_dim, dtype=torch.float32)
    mask = torch.zeros(len(sequences), max_length, dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        length = sequence.shape[0]
        batch[index, :length] = sequence.float()
        mask[index, :length] = True
    return batch, mask
