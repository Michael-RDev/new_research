from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import torch

from aoede.audio.codec import build_audio_codec
from aoede.audio.io import load_audio_file, save_audio_bytes
from aoede.audio.latent_stats import RunningLatentStats
from aoede.audio.speaker import build_speaker_encoder, speaker_cache_key
from aoede.config import ModelConfig
from aoede.data.manifest import ManifestEntry, load_manifest
from aoede.data.sota_distill import SotaDistillEntry, save_sota_manifest
from aoede.languages import normalize_language
from aoede.providers import get_provider, provider_cache_key
from aoede.text.tokenizer import UnicodeTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    tqdm = None


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage teacher-distillation caches for Aoede SOTA residual flow.",
    )
    parser.add_argument("--train-manifest", type=Path, default=Path("artifacts/manifests/train.jsonl"))
    parser.add_argument("--eval-manifest", type=Path, default=Path("artifacts/manifests/eval.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/sota_distill"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--provider", default="voxcpm2")
    parser.add_argument("--provider-model-id", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--codec-backend", default="dac")
    parser.add_argument("--codec-model-type", default="24khz")
    parser.add_argument("--codec-hop-length", type=int, default=320)
    parser.add_argument("--codec-latent-dim", type=int, default=1024)
    parser.add_argument("--speaker-encoder", default="ecapa")
    parser.add_argument("--speaker-model-source", default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--max-train-examples", type=int, default=1024)
    parser.add_argument("--max-eval-examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref-text-field", default="")
    return parser


def _load_tokenizer(path: Path, entries: list[ManifestEntry]) -> UnicodeTokenizer:
    if path.exists():
        return UnicodeTokenizer(path)
    tokenizer = UnicodeTokenizer()
    tokenizer.fit((entry.text for entry in entries), (entry.language_code for entry in entries))
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(path)
    return tokenizer


def _normalize_entries(entries: list[ManifestEntry]) -> list[ManifestEntry]:
    normalized = []
    for entry in entries:
        language_code = normalize_language(entry.language_code)
        normalized.append(
            entry if language_code == entry.language_code else replace(entry, language_code=language_code)
        )
    return normalized


def _subset(entries: list[ManifestEntry], max_examples: int, seed: int) -> list[ManifestEntry]:
    if max_examples <= 0 or len(entries) <= max_examples:
        return entries
    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    return shuffled[:max_examples]


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _encode_audio(codec, audio_path: str, sample_rate: int = 24000) -> torch.Tensor:
    audio, _ = load_audio_file(audio_path, target_sample_rate=sample_rate)
    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    with torch.no_grad():
        return codec.encode(waveform)[0].detach().cpu()


def _stage_split(
    split_name: str,
    entries: list[ManifestEntry],
    output_root: Path,
    provider,
    provider_name: str,
    codec,
    speaker_encoder,
    latent_stats: RunningLatentStats,
    sample_rate: int,
    update_latent_stats: bool = False,
) -> list[SotaDistillEntry]:
    split_root = output_root / split_name
    latents_dir = split_root / "latents"
    speakers_dir = split_root / "speakers"
    teacher_audio_dir = split_root / "teacher_audio"
    for path in (latents_dir, speakers_dir, teacher_audio_dir):
        path.mkdir(parents=True, exist_ok=True)

    iterator = entries
    if tqdm is not None:
        iterator = tqdm(entries, desc=f"sota-stage-{split_name}", unit="item")

    staged = []
    skipped = []
    for entry in iterator:
        item_id = _safe_id(entry.item_id)
        real_latents_path = latents_dir / f"{item_id}.real.pt"
        teacher_latents_path = latents_dir / f"{item_id}.teacher.pt"
        reference_latents_path = latents_dir / f"{item_id}.reference.pt"
        speaker_embedding_path = speakers_dir / f"{item_id}.speaker.pt"
        teacher_audio_path = teacher_audio_dir / f"{item_id}.wav"
        reference_audio = entry.speaker_ref or entry.audio_path

        try:
            if not real_latents_path.exists():
                torch.save(_encode_audio(codec, entry.audio_path, sample_rate), real_latents_path)
            real_latents = torch.load(real_latents_path, map_location="cpu")

            if not reference_latents_path.exists():
                torch.save(_encode_audio(codec, reference_audio, sample_rate), reference_latents_path)

            if not speaker_embedding_path.exists():
                audio, sr = load_audio_file(reference_audio, target_sample_rate=sample_rate)
                speaker_embedding = torch.from_numpy(
                    speaker_encoder.encode(audio, sample_rate=sr)
                ).float()
                torch.save(speaker_embedding, speaker_embedding_path)

            if not teacher_audio_path.exists():
                result = provider.synthesize(
                    text=entry.text,
                    reference_audio=reference_audio,
                    language=entry.language_code,
                    prompt_text=None,
                )
                teacher_audio_path.write_bytes(
                    save_audio_bytes(result.audio, sample_rate=result.sample_rate)
                )
            if not teacher_latents_path.exists():
                torch.save(_encode_audio(codec, str(teacher_audio_path), sample_rate), teacher_latents_path)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            skipped.append(
                {
                    "item_id": entry.item_id,
                    "audio_path": entry.audio_path,
                    "speaker_ref": entry.speaker_ref,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if tqdm is not None:
                tqdm.write(f"[sota-stage] skip split={split_name} item={entry.item_id}: {type(exc).__name__}: {exc}")
            else:
                print(f"[sota-stage] skip split={split_name} item={entry.item_id}: {type(exc).__name__}: {exc}", flush=True)
            for partial_path in (
                real_latents_path,
                teacher_latents_path,
                reference_latents_path,
                speaker_embedding_path,
                teacher_audio_path,
            ):
                if partial_path.exists() and partial_path.stat().st_size == 0:
                    partial_path.unlink()
            continue

        if update_latent_stats:
            latent_stats.update(real_latents)

        staged.append(
            SotaDistillEntry(
                item_id=entry.item_id,
                text=entry.text,
                language_code=entry.language_code,
                audio_path=entry.audio_path,
                speaker_ref=entry.speaker_ref,
                real_latents_path=str(real_latents_path),
                teacher_latents_path=str(teacher_latents_path),
                reference_latents_path=str(reference_latents_path),
                speaker_embedding_path=str(speaker_embedding_path),
                teacher_audio_path=str(teacher_audio_path),
                provider=provider_name,
            )
        )
    skip_path = split_root / "skipped.jsonl"
    if skipped:
        with skip_path.open("w", encoding="utf-8") as handle:
            for row in skipped:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return staged


def main() -> None:
    args = get_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_entries = _normalize_entries(load_manifest((repo_root / args.train_manifest).resolve()))
    eval_entries = _normalize_entries(load_manifest((repo_root / args.eval_manifest).resolve()))
    train_entries = _subset(train_entries, args.max_train_examples, args.seed)
    eval_entries = _subset(eval_entries, args.max_eval_examples, args.seed + 1)
    tokenizer = _load_tokenizer(
        (repo_root / args.tokenizer_path).resolve(),
        train_entries + eval_entries,
    )

    model_config = ModelConfig(
        vocab_size=tokenizer.size,
        codec_backend=args.codec_backend,
        codec_model_type=args.codec_model_type,
        codec_latent_dim=args.codec_latent_dim,
        codec_hop_length=args.codec_hop_length,
        speaker_dim=192,
        architecture_variant="sota_residualflow",
    )
    codec = build_audio_codec(model_config, device=args.device)
    if hasattr(codec, "validate"):
        codec.validate()
    speaker_encoder = build_speaker_encoder(
        backend=args.speaker_encoder,
        embedding_dim=model_config.speaker_dim,
        device=args.device,
        source=args.speaker_model_source,
    )
    provider = get_provider(args.provider, device=args.device, model_id=args.provider_model_id)

    latent_stats = RunningLatentStats(latent_dim=args.codec_latent_dim)
    train_staged = _stage_split(
        "train",
        train_entries,
        output_root,
        provider,
        args.provider,
        codec,
        speaker_encoder,
        latent_stats,
        sample_rate=model_config.sample_rate,
        update_latent_stats=True,
    )
    eval_staged = _stage_split(
        "eval",
        eval_entries,
        output_root,
        provider,
        args.provider,
        codec,
        speaker_encoder,
        latent_stats,
        sample_rate=model_config.sample_rate,
        update_latent_stats=False,
    )

    stats = latent_stats.finalize()
    stats_path = output_root / "latent_stats.json"
    stats.save(stats_path)
    train_manifest = output_root / "train.sota.jsonl"
    eval_manifest = output_root / "eval.sota.jsonl"
    save_sota_manifest(train_staged, train_manifest)
    save_sota_manifest(eval_staged, eval_manifest)
    summary = {
        "provider": args.provider,
        "provider_model_id": args.provider_model_id,
        "codec_backend": args.codec_backend,
        "codec_model_type": args.codec_model_type,
        "codec_hop_length": args.codec_hop_length,
        "codec_latent_dim": args.codec_latent_dim,
        "speaker_encoder": args.speaker_encoder,
        "speaker_model_source": args.speaker_model_source,
        "speaker_cache_key": speaker_cache_key(args.speaker_encoder, model_config.speaker_dim, args.speaker_model_source),
        "provider_cache_key": provider_cache_key(args.provider, args.provider_model_id),
        "train_entries": len(train_staged),
        "eval_entries": len(eval_staged),
        "train_manifest": str(train_manifest),
        "eval_manifest": str(eval_manifest),
        "latent_stats": str(stats_path),
        "latent_stats_count": stats.count,
    }
    summary_path = output_root / "stage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
