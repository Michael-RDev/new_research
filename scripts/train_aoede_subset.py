from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aoede.audio.codec import FrozenAudioCodec
from aoede.audio.speaker import FrozenSpeakerEncoder
from aoede.config import AppConfig, ModelConfig, TrainingConfig
from aoede.data.dataset import ManifestDataset, collate_training_examples
from aoede.data.manifest import load_manifest, save_manifest
from aoede.model.core import AoedeModel
from aoede.text.tokenizer import UnicodeTokenizer
from aoede.training.filtering import filter_trainable_entries
from aoede.training.trainer import Trainer


DEFAULT_MAX_TEXT_TOKENS = 384
DEFAULT_MAX_LATENT_FRAMES = 1600
DEFAULT_CODEC_HOP_LENGTH = 320
DEFAULT_SAMPLE_RATE = 24_000


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a small Aoede model on a deterministic subset of a manifest."
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("artifacts/manifests/train.jsonl"),
        help="Source JSONL manifest with local audio paths.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/quickruns/aoede_quick50"),
        help="Directory where the subset manifest, config, logs, and checkpoints are written.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum number of training samples to keep.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=12,
        help="Number of optimizer steps for the quick run.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for the quick run.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=6,
        help="Save an intermediate checkpoint every N steps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to use. Defaults to cpu for portability on this machine.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic subset selection and training.",
    )
    parser.add_argument(
        "--shared-cache-dir",
        type=Path,
        default=Path("artifacts/cache"),
        help="Feature cache root directory. A model-shape-specific subdirectory is created under it.",
    )
    parser.add_argument(
        "--architecture-variant",
        type=str,
        default="atlasflow",
        choices=["baseline", "atlasflow"],
        help="Aoede architecture variant to train.",
    )
    return parser


def _round_robin_subset(entries, max_samples: int, seed: int):
    by_language = defaultdict(list)
    for entries_index in entries:
        by_language[entries_index.language_code].append(entries_index)

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


def _copy_or_fit_tokenizer(repo_root: Path, output_root: Path, entries) -> UnicodeTokenizer:
    source_tokenizer = repo_root / "artifacts" / "tokenizer.json"
    if source_tokenizer.exists():
        tokenizer = UnicodeTokenizer(source_tokenizer)
    else:
        tokenizer = UnicodeTokenizer()
        tokenizer.fit((entry.text for entry in entries), (entry.language_code for entry in entries))
    target_tokenizer = output_root / "artifacts" / "tokenizer.json"
    target_tokenizer.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(target_tokenizer)
    return tokenizer


def _load_or_fit_source_tokenizer(repo_root: Path, entries) -> UnicodeTokenizer:
    source_tokenizer = repo_root / "artifacts" / "tokenizer.json"
    if source_tokenizer.exists():
        return UnicodeTokenizer(source_tokenizer)

    tokenizer = UnicodeTokenizer()
    tokenizer.fit((entry.text for entry in entries), (entry.language_code for entry in entries))
    return tokenizer


def _build_config(args: argparse.Namespace, output_root: Path, tokenizer: UnicodeTokenizer) -> AppConfig:
    config = AppConfig(
        project_root=output_root,
        model=ModelConfig(
            vocab_size=tokenizer.size,
            d_model=128,
            n_heads=4,
            n_text_layers=4,
            n_decoder_layers=4,
            style_dim=32,
            speaker_dim=192,
            codec_latent_dim=128,
            max_text_tokens=DEFAULT_MAX_TEXT_TOKENS,
            max_latent_frames=DEFAULT_MAX_LATENT_FRAMES,
            duration_predictor_layers=2,
            architecture_variant=args.architecture_variant,
            speaker_memory_tokens=8,
            planner_stride=4,
            planner_dim=64,
            memory_conditioning_heads=4,
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
            mixed_precision=False,
            checkpoint_every=args.checkpoint_every,
            log_every=1,
        ),
    )
    config.ensure_directories()
    return config


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_manifest = (repo_root / args.source_manifest).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_entries = load_manifest(source_manifest)
    source_tokenizer = _load_or_fit_source_tokenizer(repo_root, source_entries)
    filtered_entries, filter_stats = filter_trainable_entries(
        source_entries,
        source_tokenizer,
        max_text_tokens=DEFAULT_MAX_TEXT_TOKENS,
        max_latent_frames=DEFAULT_MAX_LATENT_FRAMES,
        codec_hop_length=DEFAULT_CODEC_HOP_LENGTH,
        sample_rate=DEFAULT_SAMPLE_RATE,
    )
    subset_entries = _round_robin_subset(
        filtered_entries,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if not subset_entries:
        raise RuntimeError("No trainable entries remain after filtering the source manifest.")
    print(
        "filtered_entries source={source} kept={kept} dropped_text={dropped_text} dropped_audio={dropped_audio}".format(
            source=filter_stats.source_entries,
            kept=len(subset_entries),
            dropped_text=filter_stats.dropped_text_too_long,
            dropped_audio=filter_stats.dropped_audio_too_long,
        ),
        flush=True,
    )
    subset_manifest = output_root / "subset_manifest.jsonl"
    save_manifest(subset_entries, subset_manifest)

    tokenizer = _copy_or_fit_tokenizer(repo_root, output_root, subset_entries)
    config = _build_config(args, output_root, tokenizer)
    config_path = output_root / "train_config.json"
    config.save(config_path)
    shared_cache_root = (repo_root / args.shared_cache_dir).resolve()
    cache_dir = shared_cache_root / (
        f"codec{config.model.codec_latent_dim}_spk{config.model.speaker_dim}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    codec = FrozenAudioCodec(
        sample_rate=config.model.sample_rate,
        latent_dim=config.model.codec_latent_dim,
        frame_size=config.model.codec_frame_size,
        hop_length=config.model.codec_hop_length,
    )
    dataset = ManifestDataset(
        subset_entries,
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
                print(f"saved {checkpoint_path}", flush=True)
            if steps >= args.max_steps:
                break

    final_checkpoint = output_root / "artifacts" / "checkpoints" / "quick_final.pt"
    trainer.save_checkpoint(final_checkpoint)

    language_counts = defaultdict(int)
    reference_count = 0
    for entry in subset_entries:
        language_counts[entry.language_code] += 1
        if entry.speaker_ref:
            reference_count += 1

    summary = {
        "source_manifest": str(source_manifest),
        "subset_manifest": str(subset_manifest),
        "output_root": str(output_root),
        "final_checkpoint": str(final_checkpoint),
        "cache_dir": str(cache_dir),
        "device": args.device,
        "source_entry_count": len(source_entries),
        "max_samples": len(subset_entries),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "architecture_variant": args.architecture_variant,
        "languages": dict(sorted(language_counts.items())),
        "samples_with_speaker_ref": reference_count,
        "filter_stats": filter_stats.to_dict(),
        "final_metrics": history[-1] if history else {},
    }
    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
