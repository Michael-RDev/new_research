from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader

from aoede.config import AppConfig
from aoede.model.core import AoedeModel


class Trainer:
    def __init__(self, model: AoedeModel, config: AppConfig, device: Optional[str] = None):
        self.model = model
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.step = 0

    def _autocast(self):
        if self.config.training.mixed_precision and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def train_step(self, batch: Dict[str, Union[torch.Tensor, List[str]]]):
        self.model.train()
        tensor_batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            output = self.model(
                token_ids=tensor_batch["token_ids"],
                language_ids=tensor_batch["language_ids"],
                speaker_embedding=tensor_batch["speaker_ref"],
                target_latents=tensor_batch["codec_latents"],
                target_durations=tensor_batch["durations"],
                reference_latents=tensor_batch.get("reference_latents"),
                reference_mask=tensor_batch.get("reference_mask"),
                prosody_targets=tensor_batch.get("prosody_targets"),
                has_reference=tensor_batch.get("has_reference"),
            )
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
        self.optimizer.step()
        self.step += 1
        return {key: float(value.detach().cpu()) for key, value in output.items()}

    def run(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        max_steps: Optional[int] = None,
    ):
        max_steps = max_steps or self.config.training.max_steps
        while self.step < max_steps:
            for batch in train_loader:
                metrics = self.train_step(batch)
                if self.step % self.config.training.checkpoint_every == 0:
                    self.save_checkpoint(self.config.resolve(self.config.artifacts.checkpoints_dir) / f"step_{self.step:07d}.pt")
                if self.step >= max_steps:
                    break

    def save_checkpoint(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = int(checkpoint["step"])
