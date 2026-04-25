from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from aoede.audio.latent_stats import LatentStats
from aoede.config import AppConfig, ModelConfig, TrainingConfig
from aoede.data.sota_distill import (
    SotaDistillDataset,
    collate_sota_distill,
    load_sota_manifest,
)
from aoede.model.residualflow import SotaResidualFlowModel
from aoede.text.tokenizer import UnicodeTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    tqdm = None


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Aoede SOTA residual-flow refiner.")
    parser.add_argument("--train-manifest", type=Path, default=Path("artifacts/sota_distill/train.sota.jsonl"))
    parser.add_argument("--eval-manifest", type=Path, default=Path("artifacts/sota_distill/eval.sota.jsonl"))
    parser.add_argument("--latent-stats", type=Path, default=Path("artifacts/sota_distill/latent_stats.json"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments/sota_residualflow"))
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-text-layers", type=int, default=6)
    parser.add_argument("--n-decoder-layers", type=int, default=8)
    parser.add_argument("--codec-latent-dim", type=int, default=1024)
    parser.add_argument("--speaker-dim", type=int, default=192)
    parser.add_argument("--mixed-precision", action="store_true")
    return parser


def _emit(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message, flush=True)


def _build_config(args: argparse.Namespace, tokenizer: UnicodeTokenizer, output_root: Path) -> AppConfig:
    return AppConfig(
        project_root=output_root,
        model=ModelConfig(
            vocab_size=tokenizer.size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_text_layers=args.n_text_layers,
            n_decoder_layers=args.n_decoder_layers,
            codec_backend="dac",
            codec_model_type="24khz",
            codec_latent_dim=args.codec_latent_dim,
            codec_hop_length=320,
            speaker_dim=args.speaker_dim,
            speaker_encoder_backend="ecapa",
            speaker_encoder_source="speechbrain/spkrec-ecapa-voxceleb",
            architecture_variant="sota_residualflow",
            max_text_tokens=512,
            max_latent_frames=1600,
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            checkpoint_every=args.checkpoint_every,
            eval_every=args.eval_every,
            mixed_precision=args.mixed_precision or args.device.startswith("cuda"),
        ),
    )


class SotaTrainer:
    def __init__(
        self,
        model: SotaResidualFlowModel,
        config: AppConfig,
        device: str,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.autocast_dtype: Optional[torch.dtype] = None
        self.scaler = None
        if config.training.mixed_precision and self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                self.autocast_dtype = torch.bfloat16
            else:
                self.autocast_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()
        self.step = 0

    def _autocast(self):
        if self.autocast_dtype is None:
            from contextlib import nullcontext

            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def _to_device(self, batch: dict) -> dict:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def train_step(self, batch: dict) -> dict[str, float]:
        self.model.train()
        batch = self._to_device(batch)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            output = self.model.loss(
                token_ids=batch["token_ids"],
                language_ids=batch["language_ids"],
                speaker_embedding=batch["speaker_embedding"],
                target_latents=batch["target_latents"],
                teacher_latents=batch["teacher_latents"],
                reference_latents=batch["reference_latents"],
                target_mask=batch["target_mask"],
                reference_mask=batch["reference_mask"],
            )
        loss = output["loss"]
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.training.grad_clip,
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
    def evaluate(self, loader: DataLoader, max_batches: int = 8) -> dict[str, float]:
        self.model.eval()
        aggregate = defaultdict(float)
        count = 0
        for batch in loader:
            batch = self._to_device(batch)
            with self._autocast():
                output = self.model.loss(
                    token_ids=batch["token_ids"],
                    language_ids=batch["language_ids"],
                    speaker_embedding=batch["speaker_embedding"],
                    target_latents=batch["target_latents"],
                    teacher_latents=batch["teacher_latents"],
                    reference_latents=batch["reference_latents"],
                    target_mask=batch["target_mask"],
                    reference_mask=batch["reference_mask"],
                )
            for key, value in output.items():
                aggregate[f"eval_{key}"] += float(value.detach().cpu())
            count += 1
            if count >= max_batches:
                break
        if count == 0:
            return {}
        return {key: value / count for key, value in sorted(aggregate.items())}

    def save_checkpoint(
        self,
        path: Path,
        latent_stats: LatentStats,
        tokenizer_path: Path,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "latent_stats": latent_stats.to_dict(),
                "tokenizer_path": str(tokenizer_path),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = int(checkpoint["step"])


def main() -> None:
    args = get_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer_path = (repo_root / args.tokenizer_path).resolve()
    tokenizer = UnicodeTokenizer(tokenizer_path)
    latent_stats = LatentStats.load((repo_root / args.latent_stats).resolve())
    config = _build_config(args, tokenizer, output_root)
    config.ensure_directories()
    config.save(output_root / "train_config.json")

    train_entries = load_sota_manifest((repo_root / args.train_manifest).resolve())
    train_dataset = SotaDistillDataset(train_entries, tokenizer, latent_stats)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_sota_distill,
    )
    eval_loader = None
    eval_path = (repo_root / args.eval_manifest).resolve()
    if eval_path.exists():
        eval_entries = load_sota_manifest(eval_path)
        if eval_entries:
            eval_loader = DataLoader(
                SotaDistillDataset(eval_entries, tokenizer, latent_stats),
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_sota_distill,
            )

    model = SotaResidualFlowModel(config.model)
    trainer = SotaTrainer(model, config, args.device)
    if args.resume_from is not None:
        resume_path = args.resume_from
        if not resume_path.is_absolute():
            resume_path = repo_root / resume_path
        trainer.load_checkpoint(resume_path.resolve())
        _emit(f"resumed checkpoint={resume_path} step={trainer.step}")

    history = []
    progress = None
    if tqdm is not None:
        progress = tqdm(
            total=config.training.max_steps,
            initial=trainer.step,
            desc="sota-train",
            unit="step",
            dynamic_ncols=True,
        )
    try:
        while trainer.step < config.training.max_steps:
            for batch in train_loader:
                metrics = trainer.train_step(batch)
                history.append({"step": trainer.step, **metrics})
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(
                        loss=f"{metrics.get('loss', 0.0):.4f}",
                        flow=f"{metrics.get('flow_loss', 0.0):.4f}",
                        recon=f"{metrics.get('recon_loss', 0.0):.4f}",
                        speaker=f"{metrics.get('speaker_loss', 0.0):.4f}",
                    )
                if (
                    eval_loader is not None
                    and config.training.eval_every > 0
                    and trainer.step % config.training.eval_every == 0
                ):
                    eval_metrics = trainer.evaluate(eval_loader)
                    history[-1]["eval"] = eval_metrics
                    _emit("eval " + " ".join(f"{key}={value:.4f}" for key, value in eval_metrics.items()))
                if trainer.step % config.training.checkpoint_every == 0:
                    checkpoint = output_root / "artifacts" / "checkpoints" / f"step_{trainer.step:07d}.pt"
                    trainer.save_checkpoint(checkpoint, latent_stats, tokenizer_path)
                    last = output_root / "artifacts" / "checkpoints" / "checkpoint-last.pt"
                    last.write_bytes(checkpoint.read_bytes())
                    _emit(f"saved {checkpoint}")
                if trainer.step >= config.training.max_steps:
                    break
    finally:
        if progress is not None:
            progress.close()

    final_checkpoint = output_root / "artifacts" / "checkpoints" / "checkpoint-last.pt"
    trainer.save_checkpoint(final_checkpoint, latent_stats, tokenizer_path)
    summary = {
        "train_manifest": str((repo_root / args.train_manifest).resolve()),
        "eval_manifest": str(eval_path) if eval_loader is not None else None,
        "output_root": str(output_root),
        "final_checkpoint": str(final_checkpoint),
        "latent_stats": str((repo_root / args.latent_stats).resolve()),
        "tokenizer_path": str(tokenizer_path),
        "max_steps": config.training.max_steps,
        "batch_size": config.training.batch_size,
        "architecture_variant": config.model.architecture_variant,
        "final_metrics": history[-1] if history else {},
    }
    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
