from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aoede.audio.codec import FrozenAudioCodec
from aoede.audio.speaker import FrozenSpeakerEncoder
from aoede.config import AppConfig, ModelConfig, TrainingConfig
from aoede.data.dataset import ManifestDataset, collate_training_examples
from aoede.data.manifest import load_manifest, save_manifest
from aoede.languages import normalize_language
from aoede.loaders import initialize_aoede_from_omnivoice
from aoede.model.core import AoedeModel
from aoede.text.tokenizer import UnicodeTokenizer
from aoede.training.filtering import filter_trainable_entries
from aoede.training.trainer import Trainer


DEFAULT_MAX_TEXT_TOKENS = 512
DEFAULT_MAX_LATENT_FRAMES = 1600
DEFAULT_CODEC_HOP_LENGTH = 320
DEFAULT_SAMPLE_RATE = 24_000


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train unified Aoede from manifest data.")
    parser.add_argument("--source-manifest", type=Path, default=Path("artifacts/manifests/train.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments/aoede_stage1"))
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shared-cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument(
        "--architecture-variant",
        type=str,
        default="atlasflow",
        choices=["baseline", "atlasflow"],
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
    config = AppConfig(
        project_root=output_root,
        model=ModelConfig(
            vocab_size=tokenizer.size,
            d_model=384,
            n_heads=6,
            n_text_layers=10,
            n_decoder_layers=10,
            style_dim=32,
            speaker_dim=192,
            codec_latent_dim=128,
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


def main() -> None:
    args = get_parser().parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    source_manifest = (repo_root / args.source_manifest).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_entries = _normalize_entries(load_manifest(source_manifest))
    tokenizer_path = (repo_root / args.tokenizer_path).resolve() if args.tokenizer_path else None
    tokenizer = _load_or_fit_tokenizer(repo_root, source_entries, tokenizer_path)
    filtered_entries, filter_stats = filter_trainable_entries(
        source_entries,
        tokenizer,
        max_text_tokens=DEFAULT_MAX_TEXT_TOKENS,
        max_latent_frames=DEFAULT_MAX_LATENT_FRAMES,
        codec_hop_length=DEFAULT_CODEC_HOP_LENGTH,
        sample_rate=DEFAULT_SAMPLE_RATE,
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
        "filtered_entries source={source} kept={kept} dropped_text={dropped_text} dropped_audio={dropped_audio}".format(
            source=filter_stats.source_entries,
            kept=len(filtered_entries),
            dropped_text=filter_stats.dropped_text_too_long,
            dropped_audio=filter_stats.dropped_audio_too_long,
        ),
        flush=True,
    )

    train_manifest = output_root / "train_manifest.jsonl"
    save_manifest(filtered_entries, train_manifest)
    resolved_tokenizer_path = _save_tokenizer(tokenizer, output_root)

    config = _build_config(args, output_root, tokenizer)
    config_path = output_root / "train_config.json"
    config.save(config_path)

    shared_cache_root = (repo_root / args.shared_cache_dir).resolve()
    cache_dir = shared_cache_root / f"codec{config.model.codec_latent_dim}_spk{config.model.speaker_dim}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    codec = FrozenAudioCodec(
        sample_rate=config.model.sample_rate,
        latent_dim=config.model.codec_latent_dim,
        frame_size=config.model.codec_frame_size,
        hop_length=config.model.codec_hop_length,
    )

    dataset = ManifestDataset(
        filtered_entries,
        tokenizer=tokenizer,
        codec=codec,
        speaker_encoder=FrozenSpeakerEncoder(embedding_dim=config.model.speaker_dim),
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

    model = AoedeModel(config.model)
    transfer_report = None
    if args.init_from_omnivoice:
        transfer_report = initialize_aoede_from_omnivoice(model, args.init_from_omnivoice)

    trainer = Trainer(model, config, device=args.device)

    steps = 0
    history = []
    while steps < args.max_steps:
        for batch in loader:
            metrics = trainer.train_step(batch)
            steps += 1
            history.append({"step": steps, **metrics})
            print(
                "step={step} loss={loss:.4f} flow={flow_loss:.4f} duration={duration_loss:.4f} "
                "speaker={speaker_loss:.4f} planner={planner_loss:.4f} grad={grad_norm:.4f}".format(
                    **history[-1]
                ),
                flush=True,
            )
            if steps % args.checkpoint_every == 0:
                checkpoint_path = output_root / "artifacts" / "checkpoints" / f"step_{steps:07d}.pt"
                trainer.save_checkpoint(checkpoint_path)
                (output_root / "artifacts" / "checkpoints" / "checkpoint-last.pt").write_bytes(
                    checkpoint_path.read_bytes()
                )
                print(f"saved {checkpoint_path}", flush=True)
            if steps >= args.max_steps:
                break

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
        "max_samples": len(filtered_entries),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "architecture_variant": args.architecture_variant,
        "languages": dict(sorted(language_counts.items())),
        "samples_with_speaker_ref": reference_count,
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
