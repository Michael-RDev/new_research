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
        self.autocast_dtype: Optional[torch.dtype] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        if self.config.training.mixed_precision and self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                self.autocast_dtype = torch.bfloat16
            else:
                self.autocast_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()
        self.step = 0

    def _autocast(self):
        if self.autocast_dtype is not None:
            return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)
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
        non_finite = [
            key
            for key, value in output.items()
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
        ]
        if non_finite:
            raise FloatingPointError(
                f"Non-finite model outputs at step {self.step + 1}: {', '.join(non_finite)}"
            )

        loss = output["loss"]
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.training.grad_clip
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite gradient norm at step {self.step + 1}: {float(grad_norm.detach().cpu())}"
            )

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.step += 1
        metrics = {key: float(value.detach().cpu()) for key, value in output.items()}
        metrics["grad_norm"] = float(grad_norm.detach().cpu())
        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
        max_batches: int = 8,
    ):
        self.model.eval()
        aggregate: Dict[str, float] = {}
        batches = 0
        for batch in eval_loader:
            tensor_batch = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
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
            for key, value in output.items():
                aggregate[key] = aggregate.get(key, 0.0) + float(value.detach().cpu())
            batches += 1
            if batches >= max_batches:
                break
        if batches == 0:
            return {}
        return {
            f"eval_{key}": value / batches
            for key, value in sorted(aggregate.items())
        }

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
                if (
                    eval_loader is not None
                    and self.config.training.eval_every > 0
                    and self.step % self.config.training.eval_every == 0
                ):
                    self.evaluate(eval_loader)
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
