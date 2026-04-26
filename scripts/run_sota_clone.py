from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aoede.audio.codec import build_audio_codec
from aoede.audio.io import load_audio_file, resample_audio, save_audio_bytes
from aoede.audio.latent_stats import LatentStats
from aoede.audio.speaker import build_speaker_encoder
from aoede.config import AppConfig, ArtifactsConfig, ModelConfig, ServiceConfig, TrainingConfig
from aoede.languages import language_index, normalize_language
from aoede.model.residualflow import SotaResidualFlowModel
from aoede.providers import ProviderResult, get_provider
from aoede.text.tokenizer import UnicodeTokenizer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Aoede SOTA hybrid voice cloning.")
    parser.add_argument("--provider", default="auto", choices=["auto", "voxcpm2", "qwen3", "aoede-refiner", "passthrough"])
    parser.add_argument("--teacher-provider", default="voxcpm2")
    parser.add_argument("--teacher-model-id", default=None)
    parser.add_argument(
        "--teacher-audio",
        default=None,
        help="Use an existing teacher WAV instead of running a pretrained teacher provider.",
    )
    parser.add_argument("--model", default=None, help="SOTA residual-flow checkpoint.")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Override tokenizer JSON path for checkpoints trained on another machine.",
    )
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--refine-steps", type=int, default=4)
    parser.add_argument("--speaker-encoder", default="ecapa")
    parser.add_argument("--speaker-model-source", default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--speaker-margin", type=float, default=0.03)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sota_clone"))
    parser.add_argument("--output-file", type=Path, default=None)
    return parser


def _app_config_from_dict(payload: dict) -> AppConfig:
    return AppConfig(
        project_root=Path(payload.get("project_root", ".")),
        model=ModelConfig(**payload.get("model", {})),
        training=TrainingConfig(**payload.get("training", {})),
        service=ServiceConfig(**payload.get("service", {})),
        artifacts=ArtifactsConfig(
            **{
                key: Path(value)
                for key, value in payload.get("artifacts", {}).items()
            }
        ),
    )


def _load_refiner(checkpoint_path: str, device: str, tokenizer_path_override: str | None = None):
    checkpoint = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location=device)
    config = _app_config_from_dict(checkpoint["config"])
    model = SotaResidualFlowModel(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    stats = LatentStats.from_dict(checkpoint["latent_stats"])
    if tokenizer_path_override:
        tokenizer_path = Path(tokenizer_path_override).expanduser()
    else:
        tokenizer_path = Path(checkpoint.get("tokenizer_path", "artifacts/tokenizer.json"))
    if not tokenizer_path.is_absolute():
        tokenizer_path = Path.cwd() / tokenizer_path
    if not tokenizer_path.exists() and tokenizer_path.is_absolute():
        local_tokenizer_path = Path.cwd() / "artifacts" / "tokenizer.json"
        if local_tokenizer_path.exists():
            tokenizer_path = local_tokenizer_path
    tokenizer = UnicodeTokenizer(tokenizer_path)
    return model, config, stats, tokenizer


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = (float(np.linalg.norm(left)) * float(np.linalg.norm(right))) or 1.0
    return float(np.dot(left, right) / denom)


def _sane_audio(audio: np.ndarray) -> bool:
    if audio.size < 1024:
        return False
    if not np.isfinite(audio).all():
        return False
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    return rms > 1e-4 and peak > 1e-3


def _run_teacher(args) -> ProviderResult:
    if args.teacher_audio:
        audio, sample_rate = load_audio_file(args.teacher_audio)
        return ProviderResult(
            audio=audio,
            sample_rate=sample_rate,
            provider="teacher-audio",
            metadata={
                "source": str(Path(args.teacher_audio).expanduser().resolve()),
                "language": args.language,
            },
        )
    provider = get_provider(
        args.teacher_provider if args.provider in {"auto", "aoede-refiner"} else args.provider,
        device=args.device,
        model_id=args.teacher_model_id,
    )
    return provider.synthesize(
        text=args.text,
        reference_audio=args.ref_audio,
        language=args.language,
        prompt_text=args.ref_text,
    )


def _run_refiner(args, teacher_result: ProviderResult) -> ProviderResult:
    if not args.model:
        raise ValueError("--model is required for --provider aoede-refiner or --provider auto refinement.")
    model, config, stats, tokenizer = _load_refiner(args.model, args.device, args.tokenizer_path)
    codec = build_audio_codec(config.model, device=args.device)
    speaker_encoder = build_speaker_encoder(
        backend=args.speaker_encoder,
        embedding_dim=config.model.speaker_dim,
        device=args.device,
        source=args.speaker_model_source,
    )

    teacher_audio = resample_audio(
        teacher_result.audio,
        teacher_result.sample_rate,
        config.model.sample_rate,
    )
    teacher_wave = torch.from_numpy(teacher_audio).float().unsqueeze(0).to(args.device)
    with torch.no_grad():
        teacher_latents = codec.encode(teacher_wave)

    reference_audio, reference_sr = load_audio_file(
        args.ref_audio,
        target_sample_rate=config.model.sample_rate,
    )
    reference_wave = torch.from_numpy(reference_audio).float().unsqueeze(0).to(args.device)
    with torch.no_grad():
        reference_latents = codec.encode(reference_wave)
    speaker_embedding = torch.from_numpy(
        speaker_encoder.encode(reference_audio, sample_rate=reference_sr)
    ).float().unsqueeze(0).to(args.device)

    language_code = normalize_language(args.language)
    token_ids = tokenizer.encode(args.text, language_code, add_new_tokens=False)
    token_tensor = torch.tensor([token_ids], dtype=torch.long, device=args.device)
    language_tensor = torch.tensor(
        [language_index(language_code)],
        dtype=torch.long,
        device=args.device,
    )
    reference_mask = torch.ones(
        reference_latents.shape[:2],
        dtype=torch.bool,
        device=args.device,
    )
    with torch.no_grad():
        normalized_teacher = stats.normalize(teacher_latents)
        normalized_reference = stats.normalize(reference_latents)
        refined = model.refine(
            token_ids=token_tensor,
            language_ids=language_tensor,
            speaker_embedding=speaker_embedding,
            teacher_latents=normalized_teacher,
            reference_latents=normalized_reference,
            reference_mask=reference_mask,
            steps=args.refine_steps,
        )
        audio = codec.decode(stats.denormalize(refined))[0].detach().cpu().numpy()
    return ProviderResult(
        audio=audio.astype(np.float32),
        sample_rate=config.model.sample_rate,
        provider="aoede-refiner",
        metadata={
            "teacher_provider": teacher_result.provider,
            "refine_steps": args.refine_steps,
        },
    )


def _choose_auto(args, teacher: ProviderResult, refined: ProviderResult) -> ProviderResult:
    if not _sane_audio(refined.audio):
        teacher.metadata["auto_reason"] = "refined_audio_failed_sanity_gate"
        return teacher

    speaker_encoder = build_speaker_encoder(
        backend=args.speaker_encoder,
        embedding_dim=192,
        device=args.device,
        source=args.speaker_model_source,
    )
    ref_audio, ref_sr = load_audio_file(args.ref_audio, target_sample_rate=24000)
    ref_embedding = speaker_encoder.encode(ref_audio, sample_rate=ref_sr)
    teacher_embedding = speaker_encoder.encode(teacher.audio, sample_rate=teacher.sample_rate)
    refined_embedding = speaker_encoder.encode(refined.audio, sample_rate=refined.sample_rate)
    teacher_sim = _cosine(ref_embedding, teacher_embedding)
    refined_sim = _cosine(ref_embedding, refined_embedding)
    refined.metadata["teacher_similarity"] = teacher_sim
    refined.metadata["refined_similarity"] = refined_sim
    if refined_sim + args.speaker_margin >= teacher_sim:
        refined.metadata["auto_reason"] = "refined_passed_similarity_gate"
        return refined
    teacher.metadata["teacher_similarity"] = teacher_sim
    teacher.metadata["refined_similarity"] = refined_sim
    teacher.metadata["auto_reason"] = "teacher_similarity_was_better"
    return teacher


def main() -> None:
    args = get_parser().parse_args()
    teacher = _run_teacher(args)
    if args.provider == "voxcpm2" or args.provider == "qwen3" or args.provider == "passthrough":
        result = teacher
    else:
        try:
            refined = _run_refiner(args, teacher)
            result = refined if args.provider == "aoede-refiner" else _choose_auto(args, teacher, refined)
        except Exception as exc:
            if args.provider == "aoede-refiner":
                raise
            teacher.metadata["auto_reason"] = f"refiner_failed: {type(exc).__name__}: {exc}"
            result = teacher

    if args.output_file is not None:
        output_path = args.output_file.expanduser().resolve()
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (args.output_dir / f"sota_clone_{result.provider}_{stamp}.wav").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(save_audio_bytes(result.audio, sample_rate=result.sample_rate))
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "provider": result.provider,
                "sample_rate": result.sample_rate,
                "metadata": result.metadata,
                "output": str(output_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
