from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aoede.audio.codec import build_audio_codec, codec_cache_key, normalize_codec_backend
from aoede.audio.speaker import build_speaker_encoder, speaker_cache_key
from aoede.config import AppConfig, ModelConfig, TrainingConfig
from aoede.data.dataset import ManifestDataset, collate_training_examples
from aoede.data.manifest import load_manifest, save_manifest
from aoede.languages import normalize_language
from aoede.loaders import initialize_aoede_from_omnivoice
from aoede.model.core import AoedeModel
from aoede.text.tokenizer import UnicodeTokenizer
from aoede.training.filtering import filter_trainable_entries
from aoede.training.trainer import Trainer

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    tqdm = None


DEFAULT_MAX_TEXT_TOKENS = 512
DEFAULT_MAX_LATENT_FRAMES = 1600
DEFAULT_FROZEN_CODEC_LATENT_DIM = 128
DEFAULT_FROZEN_CODEC_HOP_LENGTH = 320
DEFAULT_DAC_CODEC_LATENT_DIM = 1024
DEFAULT_DAC_CODEC_HOP_LENGTH = 320
DEFAULT_SAMPLE_RATE = 24_000


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train unified Aoede from manifest data.")
    parser.add_argument("--source-manifest", type=Path, default=Path("artifacts/manifests/train.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments/aoede_stage1"))
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume optimizer/model state from.",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shared-cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument(
        "--codec-backend",
        type=str,
        default="frozen",
        choices=["frozen", "fallback", "deterministic", "dac", "descript"],
        help="Audio codec target space. Use dac for real pretrained codec latents.",
    )
    parser.add_argument(
        "--codec-latent-dim",
        type=int,
        default=None,
        help="Codec latent width. Defaults to 1024 for DAC and 128 for the frozen fallback.",
    )
    parser.add_argument(
        "--codec-hop-length",
        type=int,
        default=None,
        help="Samples per latent frame. Defaults to 512 for DAC and 320 for the frozen fallback.",
    )
    parser.add_argument(
        "--codec-model-type",
        type=str,
        default="24khz",
        help="DAC model type passed to dac.utils.download when --codec-backend=dac.",
    )
    parser.add_argument(
        "--codec-model-path",
        type=str,
        default=None,
        help="Optional local DAC checkpoint path. If unset, DAC downloads the requested model type.",
    )
    parser.add_argument(
        "--codec-device",
        type=str,
        default=None,
        help="Device used for offline codec feature extraction. Defaults to --device.",
    )
    parser.add_argument(
        "--speaker-encoder",
        type=str,
        default="frozen",
        choices=["frozen", "fallback", "ecapa", "speechbrain"],
        help="Speaker embedding backend. Use ecapa for SpeechBrain ECAPA-TDNN.",
    )
    parser.add_argument(
        "--speaker-model-source",
        type=str,
        default="speechbrain/spkrec-ecapa-voxceleb",
        help="SpeechBrain speaker encoder source when --speaker-encoder=ecapa.",
    )
    parser.add_argument(
        "--architecture-variant",
        type=str,
        default="mosaicflow",
        choices=["baseline", "atlasflow", "mosaicflow", "sota_residualflow"],
    )
    parser.add_argument(
        "--init-from-omnivoice",
        type=str,
        default=None,
        help="Path or model ID for OmniVoice checkpoint to warm-start Aoede.",
    )
    return parser


def _round_robin_subset(entries, max_samples: int, seed: int):
    by_language = defaultdict(list)
    for item in entries:
        by_language[normalize_language(item.language_code)].append(item)

    rng = random.Random(seed)
    languages = sorted(by_language.keys())
    for language in languages:
        rng.shuffle(by_language[language])

    subset = []
    while len(subset) < max_samples:
        added = False
        for language in languages:
            pool = by_language[language]
            if not pool:
                continue
            subset.append(pool.pop())
            added = True
            if len(subset) >= max_samples:
                break
        if not added:
            break
    return subset


def _load_or_fit_tokenizer(
    repo_root: Path,
    entries,
    explicit: Path | None,
) -> UnicodeTokenizer:
    source = explicit
    if source is None:
        candidate = repo_root / "artifacts" / "tokenizer.json"
        if candidate.exists():
            source = candidate

    if source is not None and source.exists():
        return UnicodeTokenizer(source)

    tokenizer = UnicodeTokenizer()
    tokenizer.fit((entry.text for entry in entries), (entry.language_code for entry in entries))
    return tokenizer


def _normalize_entries(entries):
    normalized_entries = []
    for entry in entries:
        language_code = normalize_language(entry.language_code)
        if language_code == entry.language_code:
            normalized_entries.append(entry)
            continue
        normalized_entries.append(replace(entry, language_code=language_code))
    return normalized_entries


def _save_tokenizer(
    tokenizer: UnicodeTokenizer,
    output_root: Path,
) -> Path:
    target_tokenizer = output_root / "artifacts" / "tokenizer.json"
    target_tokenizer.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(target_tokenizer)
    return target_tokenizer


def _build_config(args: argparse.Namespace, output_root: Path, tokenizer: UnicodeTokenizer) -> AppConfig:
    codec_backend = normalize_codec_backend(args.codec_backend)
    codec_latent_dim = args.codec_latent_dim
    if codec_latent_dim is None:
        codec_latent_dim = (
            DEFAULT_DAC_CODEC_LATENT_DIM
            if codec_backend == "dac"
            else DEFAULT_FROZEN_CODEC_LATENT_DIM
        )
    codec_hop_length = args.codec_hop_length
    if codec_hop_length is None:
        codec_hop_length = (
            DEFAULT_DAC_CODEC_HOP_LENGTH
            if codec_backend == "dac"
            else DEFAULT_FROZEN_CODEC_HOP_LENGTH
        )

    config = AppConfig(
        project_root=output_root,
        model=ModelConfig(
            vocab_size=tokenizer.size,
            d_model=512,
            n_heads=8,
            n_text_layers=8,
            n_decoder_layers=8,
            semantic_layers=4,
            style_dim=64,
            speaker_dim=192,
            speaker_encoder_backend=args.speaker_encoder,
            speaker_encoder_source=args.speaker_model_source,
            codec_backend=codec_backend,
            codec_model_type=args.codec_model_type,
            codec_model_path=args.codec_model_path,
            codec_latent_dim=codec_latent_dim,
            codec_hop_length=codec_hop_length,
            semantic_dim=96,
            semantic_stride=4,
            prompt_token_count=8,
            max_text_tokens=DEFAULT_MAX_TEXT_TOKENS,
            max_latent_frames=DEFAULT_MAX_LATENT_FRAMES,
            duration_predictor_layers=4,
            architecture_variant=args.architecture_variant,
            speaker_memory_tokens=8,
            planner_stride=4,
            planner_dim=128,
            memory_conditioning_heads=6,
            composer_layers=2,
            memory_dropout=0.1,
            planner_loss_weight=0.1,
            memory_speaker_loss_weight=0.1,
            semantic_loss_weight=0.2,
            prompt_loss_weight=0.1,
            coverage_loss_weight=0.05,
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            grad_clip=1.0,
            max_steps=args.max_steps,
            mixed_precision=args.device.startswith("cuda"),
            checkpoint_every=args.checkpoint_every,
            log_every=10,
        ),
    )
    config.ensure_directories()
    return config


def _format_metrics(step: int, metrics: dict[str, float]) -> str:
    pieces = [f"step={step}"]
    label_map = {
        "loss": "loss",
        "flow_loss": "flow",
        "semantic_loss": "semantic",
        "duration_loss": "duration",
        "speaker_loss": "speaker",
        "prompt_loss": "prompt",
        "coverage_loss": "coverage",
        "planner_loss": "planner",
        "grad_norm": "grad",
    }
    for key in (
        "loss",
        "flow_loss",
        "semantic_loss",
        "duration_loss",
        "speaker_loss",
        "prompt_loss",
        "coverage_loss",
        "planner_loss",
        "grad_norm",
    ):
        if key in metrics:
            pieces.append(f"{label_map[key]}={metrics[key]:.4f}")
    return " ".join(pieces)


def _emit_progress_message(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message, flush=True)


def main() -> None:
    args = get_parser().parse_args()
    if args.architecture_variant == "sota_residualflow":
        raise SystemExit(
            "sota_residualflow uses the teacher-distillation pipeline. Run "
            "`python -m aoede.training.train_sota_residualflow` or "
            "`bash scripts/train_sota_residualflow.sh core`."
        )

    repo_root = Path(__file__).resolve().parents[2]
    source_manifest = (repo_root / args.source_manifest).resolve()
    output_root = (repo_root / args.output_root).resolve()
    resume_path = None
    if args.resume_from is not None:
        resume_path = args.resume_from
        if not resume_path.is_absolute():
            resume_path = repo_root / resume_path
        resume_path = resume_path.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_entries = _normalize_entries(load_manifest(source_manifest))
    tokenizer_path = (repo_root / args.tokenizer_path).resolve() if args.tokenizer_path else None
    tokenizer = _load_or_fit_tokenizer(repo_root, source_entries, tokenizer_path)
    config = _build_config(args, output_root, tokenizer)
    filtered_entries, filter_stats = filter_trainable_entries(
        source_entries,
        tokenizer,
        max_text_tokens=DEFAULT_MAX_TEXT_TOKENS,
        max_latent_frames=DEFAULT_MAX_LATENT_FRAMES,
        codec_hop_length=config.model.codec_hop_length,
        sample_rate=config.model.sample_rate,
        validate_audio_paths=True,
    )
    if args.max_samples > 0:
        filtered_entries = _round_robin_subset(
            filtered_entries,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    if not filtered_entries:
        raise RuntimeError("No trainable entries remain after filtering the source manifest.")

    print(
        "filtered_entries source={source} kept={kept} dropped_text={dropped_text} "
        "dropped_audio={dropped_audio} dropped_unreadable={dropped_unreadable} "
        "cleared_speaker_ref={cleared_speaker_ref}".format(
            source=filter_stats.source_entries,
            kept=len(filtered_entries),
            dropped_text=filter_stats.dropped_text_too_long,
            dropped_audio=filter_stats.dropped_audio_too_long,
            dropped_unreadable=filter_stats.dropped_audio_unreadable,
            cleared_speaker_ref=filter_stats.cleared_speaker_ref_unreadable,
        ),
        flush=True,
    )

    train_manifest = output_root / "train_manifest.jsonl"
    save_manifest(filtered_entries, train_manifest)
    resolved_tokenizer_path = _save_tokenizer(tokenizer, output_root)

    shared_cache_root = (repo_root / args.shared_cache_dir).resolve()
    cache_dir = shared_cache_root / (
        codec_cache_key(config.model)
        + "_"
        + speaker_cache_key(
            config.model.speaker_encoder_backend,
            config.model.speaker_dim,
            config.model.speaker_encoder_source,
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    codec_device = args.codec_device or args.device
    codec = build_audio_codec(config.model, device=codec_device)
    if normalize_codec_backend(config.model.codec_backend) == "dac":
        _emit_progress_message(
            "initializing DAC audio codec "
            f"model_type={config.model.codec_model_type} "
            f"latent_dim={config.model.codec_latent_dim} "
            f"hop_length={config.model.codec_hop_length} "
            f"device={codec_device}"
        )
        codec.validate()

    speaker_encoder = build_speaker_encoder(
        backend=config.model.speaker_encoder_backend,
        embedding_dim=config.model.speaker_dim,
        device=args.device,
        source=config.model.speaker_encoder_source,
    )

    config_path = output_root / "train_config.json"
    config.save(config_path)

    dataset = ManifestDataset(
        filtered_entries,
        tokenizer=tokenizer,
        codec=codec,
        speaker_encoder=speaker_encoder,
        cache_dir=cache_dir,
        planner_stride=config.model.planner_stride,
        planner_dim=config.model.planner_dim,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_training_examples,
    )
    eval_loader = None
    eval_manifest = repo_root / "artifacts" / "manifests" / "eval.jsonl"
    if eval_manifest.exists():
        eval_entries = _normalize_entries(load_manifest(eval_manifest))
        eval_entries, _ = filter_trainable_entries(
            eval_entries,
            tokenizer,
            max_text_tokens=DEFAULT_MAX_TEXT_TOKENS,
            max_latent_frames=DEFAULT_MAX_LATENT_FRAMES,
            codec_hop_length=config.model.codec_hop_length,
            sample_rate=config.model.sample_rate,
            validate_audio_paths=True,
        )
        if eval_entries:
            eval_dataset = ManifestDataset(
                eval_entries,
                tokenizer=tokenizer,
                codec=codec,
                speaker_encoder=speaker_encoder,
                cache_dir=cache_dir,
                planner_stride=config.model.planner_stride,
                planner_dim=config.model.planner_dim,
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_training_examples,
            )

    model = AoedeModel(config.model)
    transfer_report = None
    if resume_path is None and args.init_from_omnivoice:
        if args.architecture_variant == "mosaicflow":
            print(
                "Skipping OmniVoice warm-start because mosaicflow does not share the legacy Aoede parameter layout.",
                flush=True,
            )
        else:
            transfer_report = initialize_aoede_from_omnivoice(
                model,
                args.init_from_omnivoice,
            )

    trainer = Trainer(model, config, device=args.device)
    initial_step = 0
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        trainer.load_checkpoint(resume_path)
        initial_step = trainer.step
        _emit_progress_message(f"resumed checkpoint={resume_path} step={trainer.step}")

    steps = 0
    history = []
    steps = trainer.step
    train_progress = None
    if tqdm is not None:
        train_progress = tqdm(
            total=args.max_steps,
            initial=trainer.step,
            desc="train",
            unit="step",
            dynamic_ncols=True,
        )
    try:
        while steps < args.max_steps:
            for batch in loader:
                metrics = trainer.train_step(batch)
                steps = trainer.step
                history.append({"step": steps, **metrics})
                if train_progress is not None:
                    train_progress.update(1)
                    train_progress.set_postfix(
                        loss=f"{metrics.get('loss', 0.0):.4f}",
                        flow=f"{metrics.get('flow_loss', 0.0):.4f}",
                        semantic=f"{metrics.get('semantic_loss', 0.0):.4f}",
                        speaker=f"{metrics.get('speaker_loss', 0.0):.4f}",
                    )
                elif steps % max(config.training.log_every, 1) == 0:
                    print(_format_metrics(steps, metrics), flush=True)

                if (
                    eval_loader is not None
                    and config.training.eval_every > 0
                    and steps % config.training.eval_every == 0
                ):
                    eval_metrics = trainer.evaluate(eval_loader)
                    history[-1]["eval"] = eval_metrics
                    _emit_progress_message(
                        "eval "
                        + " ".join(
                            f"{key}={value:.4f}"
                            for key, value in sorted(eval_metrics.items())
                        )
                    )
                if steps % args.checkpoint_every == 0:
                    checkpoint_path = output_root / "artifacts" / "checkpoints" / f"step_{steps:07d}.pt"
                    trainer.save_checkpoint(checkpoint_path)
                    (output_root / "artifacts" / "checkpoints" / "checkpoint-last.pt").write_bytes(
                        checkpoint_path.read_bytes()
                    )
                    _emit_progress_message(f"saved {checkpoint_path}")
                if steps >= args.max_steps:
                    break
    finally:
        if train_progress is not None:
            train_progress.close()

    final_checkpoint = output_root / "artifacts" / "checkpoints" / "checkpoint-last.pt"
    trainer.save_checkpoint(final_checkpoint)

    language_counts = defaultdict(int)
    reference_count = 0
    for entry in filtered_entries:
        language_counts[normalize_language(entry.language_code)] += 1
        if entry.speaker_ref:
            reference_count += 1

    summary = {
        "source_manifest": str(source_manifest),
        "train_manifest": str(train_manifest),
        "output_root": str(output_root),
        "final_checkpoint": str(final_checkpoint),
        "cache_dir": str(cache_dir),
        "device": args.device,
        "source_entry_count": len(source_entries),
        "initial_step": initial_step,
        "max_samples": len(filtered_entries),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "architecture_variant": args.architecture_variant,
        "codec_backend": config.model.codec_backend,
        "codec_model_type": config.model.codec_model_type,
        "codec_model_path": config.model.codec_model_path,
        "codec_latent_dim": config.model.codec_latent_dim,
        "codec_hop_length": config.model.codec_hop_length,
        "speaker_encoder_backend": config.model.speaker_encoder_backend,
        "speaker_encoder_source": config.model.speaker_encoder_source,
        "resume_from": str(resume_path) if resume_path else None,
        "languages": dict(sorted(language_counts.items())),
        "samples_with_speaker_ref": reference_count,
        "eval_manifest": str(eval_manifest) if eval_loader is not None else None,
        "tokenizer_path": str(resolved_tokenizer_path),
        "filter_stats": filter_stats.to_dict(),
        "omnivoice_init": args.init_from_omnivoice,
        "transfer_report": (
            {
                "source": transfer_report.source,
                "transferred": transfer_report.transferred,
                "skipped": transfer_report.skipped,
                "warnings": transfer_report.warnings,
            }
            if transfer_report
            else None
        ),
        "final_metrics": history[-1] if history else {},
    }
    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
