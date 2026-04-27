from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from aoede.audio.codec import build_audio_codec
from aoede.audio.io import load_audio_file, resample_audio, save_audio_bytes
from aoede.audio.speaker import build_speaker_encoder
from aoede.eval.common import peak_rss_bytes
from aoede.languages import language_index, normalize_language
from aoede.providers import ProviderResult, get_provider
from scripts.run_sota_clone import _cosine, _load_refiner, _sane_audio


DEFAULT_SCALES = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0)

RUNTIME_FIELDS = [
    "id",
    "latency_s",
    "output_duration_s",
    "real_time_factor",
    "rss_bytes",
]

DECISION_FIELDS = [
    "id",
    "text",
    "language",
    "selected_scale",
    "accepted",
    "selected_reason",
    "teacher_similarity",
    "selected_similarity",
    "selected_delta",
    "speaker_margin",
    "teacher_wav",
    "selected_wav",
    "dac_roundtrip_wav",
    "full_residual_wav",
    "candidate_deltas_json",
    "candidate_similarities_json",
    "candidate_sanity_json",
    "latency_s",
    "output_duration_s",
    "rss_bytes",
]

CANDIDATE_FIELDS = [
    "id",
    "text",
    "language",
    "scale",
    "teacher_similarity",
    "candidate_similarity",
    "delta",
    "audio_sane",
    "accepted",
    "candidate_duration_s",
    "candidate_rms",
    "candidate_peak",
    "candidate_wav",
]


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    text: str
    ref_audio: Path
    ref_text: str | None
    language: str


@dataclass(frozen=True)
class CandidateAudio:
    scale: float
    audio: np.ndarray
    sample_rate: int
    similarity: float
    delta: float
    audio_sane: bool
    wav_path: str
    duration_s: float
    rms: float
    peak: float


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer Aoede DCRL lattice outputs over an OmniVoice-style benchmark JSONL."
    )
    parser.add_argument("--model", required=True, help="Aoede SOTA residual-flow checkpoint.")
    parser.add_argument("--test_list", required=True, help="OmniVoice-style benchmark JSONL.")
    parser.add_argument("--res_dir", required=True, help="Directory for selected benchmark wavs.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--tokenizer-path", default=os.environ.get("DCRL_TOKENIZER_PATH"))
    parser.add_argument("--device", default=os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--teacher-provider", default=os.environ.get("AOEDE_TEACHER", "voxcpm2"))
    parser.add_argument("--teacher-model-id", default=os.environ.get("AOEDE_TEACHER_MODEL_ID") or None)
    parser.add_argument("--speaker-encoder", default=os.environ.get("AOEDE_SPEAKER_ENCODER", "ecapa"))
    parser.add_argument("--speaker-model-source", default=os.environ.get("AOEDE_SPEAKER_MODEL_SOURCE", "speechbrain/spkrec-ecapa-voxceleb"))
    parser.add_argument("--speaker-margin", type=float, default=float(os.environ.get("DCRL_SPEAKER_MARGIN", "0.0")))
    parser.add_argument("--scales", nargs="+", type=float, default=_env_scales())
    parser.add_argument("--refine-steps", type=int, default=int(os.environ.get("DCRL_REFINE_STEPS", "4")))
    parser.add_argument("--min-duration-s", type=float, default=float(os.environ.get("DCRL_MIN_DURATION_S", "0.25")))
    parser.add_argument("--save-all-candidates", action="store_true", default=os.environ.get("DCRL_SAVE_ALL_CANDIDATES", "0") == "1")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--decision-csv-path", default=None)
    parser.add_argument("--candidate-csv-path", default=None)
    parser.add_argument("--runtime_csv_path", default=None)
    parser.add_argument("--runtime_summary_path", default=None)

    # Compatibility flags supplied by OmniVoice benchmark_compare. They are
    # accepted here so this module can be swapped in as candidate-infer-module.
    parser.add_argument("--num_step", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--t_shift", type=float, default=None)
    parser.add_argument("--nj_per_gpu", type=int, default=1)
    parser.add_argument("--audio_chunk_duration", type=float, default=15.0)
    parser.add_argument("--audio_chunk_threshold", type=float, default=30.0)
    parser.add_argument("--batch_duration", type=float, default=1000.0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--preprocess_prompt", default="False")
    parser.add_argument("--postprocess_output", default="False")
    parser.add_argument("--denoise", default="True")
    parser.add_argument("--lang_id", default=None)
    return parser


def _env_scales() -> list[float]:
    text = os.environ.get("DCRL_SCALES", "")
    if not text.strip():
        return list(DEFAULT_SCALES)
    return [float(part) for part in re.split(r"[,\s]+", text.strip()) if part]


def _safe_id(value: object) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe.strip("._") or "item"


def _scale_label(scale: float) -> str:
    return f"{scale:.4f}".rstrip("0").rstrip(".").replace("-", "neg").replace(".", "p")


def _resolve_ref_audio(ref_audio: str, test_list: Path) -> Path:
    path = Path(ref_audio).expanduser()
    if path.is_absolute() and path.exists():
        return path
    candidates = [
        (test_list.parent / path).resolve(),
        (Path.cwd() / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_test_list(path: Path, lang_override: str | None = None) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = _safe_id(row.get("id") or row.get("item_id") or row.get("utt_id") or f"item_{line_number:05d}")
            text = str(row.get("text") or row.get("target_text") or row.get("transcript") or "").strip()
            ref_audio = row.get("ref_audio") or row.get("reference_audio") or row.get("prompt_wav") or row.get("audio_path")
            if not text or not ref_audio:
                raise ValueError(f"{path}:{line_number}: DCRL inference requires text and ref_audio.")
            language = normalize_language(row.get("language_id") or row.get("language") or row.get("language_name") or lang_override or "en")
            ref_text = row.get("ref_text") or row.get("prompt_text") or row.get("reference_text")
            items.append(
                BenchmarkItem(
                    item_id=item_id,
                    text=text,
                    ref_audio=_resolve_ref_audio(str(ref_audio), path),
                    ref_text=str(ref_text) if ref_text else None,
                    language=language,
                )
            )
    return items


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_runtime_summary(path: Path, rows: list[dict[str, object]], checkpoint_path: Path) -> None:
    durations = [float(row["output_duration_s"]) for row in rows]
    latencies = [float(row["latency_s"]) for row in rows]
    rtfs = [float(row["real_time_factor"]) for row in rows]
    summary = {
        "num_samples": len(rows),
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "mean_output_duration_s": sum(durations) / len(durations) if durations else 0.0,
        "mean_real_time_factor": sum(rtfs) / len(rtfs) if rtfs else 0.0,
        "max_rss_bytes": max((int(row["rss_bytes"]) for row in rows), default=0),
        "checkpoint": str(checkpoint_path),
        "inference_module": "aoede.eval.infer_dcrl_lattice",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _audio_stats(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    if audio.size == 0:
        return {"duration_s": 0.0, "rms": 0.0, "peak": 0.0}
    finite = np.asarray(audio[np.isfinite(audio)], dtype=np.float32)
    if finite.size == 0:
        return {"duration_s": len(audio) / float(sample_rate), "rms": math.nan, "peak": math.nan}
    return {
        "duration_s": len(audio) / float(sample_rate),
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
        "peak": float(np.max(np.abs(finite))),
    }


def _is_audio_sane(audio: np.ndarray, sample_rate: int, min_duration_s: float) -> bool:
    return bool(_sane_audio(audio) and _audio_stats(audio, sample_rate)["duration_s"] >= min_duration_s)


def _save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(save_audio_bytes(audio, sample_rate=sample_rate))
    return str(path)


def _encode_audio(codec, audio: np.ndarray, source_sr: int, target_sr: int, device: str) -> torch.Tensor:
    resampled = resample_audio(audio, source_sr, target_sr)
    waveform = torch.from_numpy(resampled).float().unsqueeze(0).to(device)
    return codec.encode(waveform)


def _run_teacher(item: BenchmarkItem, provider, device: str) -> ProviderResult:
    return provider.synthesize(
        text=item.text,
        reference_audio=str(item.ref_audio),
        language=item.language,
        prompt_text=item.ref_text,
    )


def _build_candidates(
    *,
    item: BenchmarkItem,
    args: argparse.Namespace,
    model,
    config,
    stats,
    tokenizer,
    codec,
    speaker_encoder,
    teacher: ProviderResult,
    teacher_similarity: float,
    ref_embedding_np: np.ndarray,
    diagnostic_dir: Path,
) -> list[CandidateAudio]:
    with torch.inference_mode():
        teacher_latents = _encode_audio(
            codec,
            teacher.audio,
            teacher.sample_rate,
            config.model.sample_rate,
            args.device,
        )
        ref_audio_codec, ref_sr_codec = load_audio_file(
            item.ref_audio,
            target_sample_rate=config.model.sample_rate,
        )
        reference_latents = _encode_audio(
            codec,
            ref_audio_codec,
            ref_sr_codec,
            config.model.sample_rate,
            args.device,
        )
        speaker_embedding = torch.from_numpy(ref_embedding_np).float().unsqueeze(0).to(args.device)
        token_ids = tokenizer.encode(item.text, item.language, add_new_tokens=False)
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=args.device)
        language_tensor = torch.tensor([language_index(item.language)], dtype=torch.long, device=args.device)
        reference_mask = torch.ones(reference_latents.shape[:2], dtype=torch.bool, device=args.device)
        normalized_teacher = stats.normalize(teacher_latents)
        normalized_reference = stats.normalize(reference_latents)
        full_refined = model.refine(
            token_ids=token_tensor,
            language_ids=language_tensor,
            speaker_embedding=speaker_embedding,
            teacher_latents=normalized_teacher,
            reference_latents=normalized_reference,
            reference_mask=reference_mask,
            steps=args.refine_steps,
        )
        residual = full_refined - normalized_teacher

    candidates: list[CandidateAudio] = []
    save_key_scales = {0.0, 1.0}
    for scale in args.scales:
        with torch.inference_mode():
            scaled = normalized_teacher + float(scale) * residual
            audio = (
                codec.decode(stats.denormalize(scaled))[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        sane = _is_audio_sane(audio, config.model.sample_rate, args.min_duration_s)
        similarity = math.nan
        if sane:
            embedding = speaker_encoder.encode(audio, sample_rate=config.model.sample_rate)
            similarity = _cosine(ref_embedding_np, embedding)
        delta = similarity - teacher_similarity if math.isfinite(similarity) else math.nan
        stats_row = _audio_stats(audio, config.model.sample_rate)
        wav_path = ""
        if args.save_all_candidates or float(scale) in save_key_scales:
            wav_path = _save_wav(
                diagnostic_dir / f"alpha_{_scale_label(float(scale))}.wav",
                audio,
                config.model.sample_rate,
            )
        candidates.append(
            CandidateAudio(
                scale=float(scale),
                audio=audio,
                sample_rate=config.model.sample_rate,
                similarity=similarity,
                delta=delta,
                audio_sane=sane,
                wav_path=wav_path,
                duration_s=stats_row["duration_s"],
                rms=stats_row["rms"],
                peak=stats_row["peak"],
            )
        )
    return candidates


def _select_candidate(
    candidates: list[CandidateAudio],
    speaker_margin: float,
) -> CandidateAudio | None:
    accepted = [
        candidate
        for candidate in candidates
        if candidate.audio_sane
        and math.isfinite(candidate.delta)
        and candidate.delta >= -speaker_margin
        and candidate.delta > 0.0
    ]
    if not accepted:
        return None
    return max(accepted, key=lambda candidate: candidate.delta)


def _json_map(candidates: list[CandidateAudio], attr: str) -> str:
    payload = {str(candidate.scale): getattr(candidate, attr) for candidate in candidates}
    return json.dumps(payload, sort_keys=True)


def _run_item(
    *,
    item: BenchmarkItem,
    args: argparse.Namespace,
    provider,
    model,
    config,
    stats,
    tokenizer,
    codec,
    speaker_encoder,
    res_dir: Path,
    diagnostics_root: Path,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    start = time.perf_counter()
    diagnostic_dir = diagnostics_root / item.item_id
    teacher = _run_teacher(item, provider, args.device)
    teacher_wav = _save_wav(diagnostic_dir / "teacher.wav", teacher.audio, teacher.sample_rate)

    ref_audio_speaker, ref_sr_speaker = load_audio_file(item.ref_audio, target_sample_rate=24000)
    ref_embedding_np = speaker_encoder.encode(ref_audio_speaker, sample_rate=ref_sr_speaker)
    teacher_embedding_np = speaker_encoder.encode(teacher.audio, sample_rate=teacher.sample_rate)
    teacher_similarity = _cosine(ref_embedding_np, teacher_embedding_np)

    candidates = _build_candidates(
        item=item,
        args=args,
        model=model,
        config=config,
        stats=stats,
        tokenizer=tokenizer,
        codec=codec,
        speaker_encoder=speaker_encoder,
        teacher=teacher,
        teacher_similarity=teacher_similarity,
        ref_embedding_np=ref_embedding_np,
        diagnostic_dir=diagnostic_dir,
    )

    selected = _select_candidate(candidates, args.speaker_margin)
    accepted = selected is not None
    if selected is None:
        selected_audio = teacher.audio
        selected_sample_rate = teacher.sample_rate
        selected_scale = "teacher"
        selected_similarity = teacher_similarity
        selected_delta = 0.0
        selected_reason = "fallback_teacher_no_positive_verified_candidate"
    else:
        selected_audio = selected.audio
        selected_sample_rate = selected.sample_rate
        selected_scale = selected.scale
        selected_similarity = selected.similarity
        selected_delta = selected.delta
        selected_reason = "accepted_best_positive_verified_candidate"

    selected_root_wav = _save_wav(res_dir / f"{item.item_id}.wav", selected_audio, selected_sample_rate)
    selected_diag_wav = _save_wav(diagnostic_dir / "selected.wav", selected_audio, selected_sample_rate)
    latency_s = time.perf_counter() - start
    output_duration_s = len(selected_audio) / float(selected_sample_rate)

    scale_zero = next((candidate.wav_path for candidate in candidates if candidate.scale == 0.0), "")
    scale_one = next((candidate.wav_path for candidate in candidates if candidate.scale == 1.0), "")
    decision = {
        "id": item.item_id,
        "text": item.text,
        "language": item.language,
        "selected_scale": selected_scale,
        "accepted": accepted,
        "selected_reason": selected_reason,
        "teacher_similarity": teacher_similarity,
        "selected_similarity": selected_similarity,
        "selected_delta": selected_delta,
        "speaker_margin": args.speaker_margin,
        "teacher_wav": teacher_wav,
        "selected_wav": selected_diag_wav,
        "dac_roundtrip_wav": scale_zero,
        "full_residual_wav": scale_one,
        "candidate_deltas_json": _json_map(candidates, "delta"),
        "candidate_similarities_json": _json_map(candidates, "similarity"),
        "candidate_sanity_json": _json_map(candidates, "audio_sane"),
        "latency_s": round(latency_s, 4),
        "output_duration_s": round(output_duration_s, 4),
        "rss_bytes": peak_rss_bytes(),
    }
    candidate_rows = [
        {
            "id": item.item_id,
            "text": item.text,
            "language": item.language,
            "scale": candidate.scale,
            "teacher_similarity": teacher_similarity,
            "candidate_similarity": candidate.similarity,
            "delta": candidate.delta,
            "audio_sane": candidate.audio_sane,
            "accepted": bool(
                candidate.audio_sane
                and math.isfinite(candidate.delta)
                and candidate.delta >= -args.speaker_margin
                and candidate.delta > 0.0
            ),
            "candidate_duration_s": candidate.duration_s,
            "candidate_rms": candidate.rms,
            "candidate_peak": candidate.peak,
            "candidate_wav": candidate.wav_path,
        }
        for candidate in candidates
    ]
    runtime = {
        "id": item.item_id,
        "latency_s": round(latency_s, 4),
        "output_duration_s": round(output_duration_s, 4),
        "real_time_factor": round(latency_s / max(output_duration_s, 1e-6), 4),
        "rss_bytes": peak_rss_bytes(),
    }
    return decision, candidate_rows, runtime


def main() -> None:
    args = get_parser().parse_args()
    test_list = Path(args.test_list).expanduser().resolve()
    res_dir = Path(args.res_dir).expanduser().resolve()
    res_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_root = res_dir / "_dcrl"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    items = read_test_list(test_list, args.lang_id)
    if not items:
        raise RuntimeError(f"No benchmark items found in {test_list}.")

    model, config, stats, tokenizer = _load_refiner(args.model, args.device, args.tokenizer_path)
    codec = build_audio_codec(config.model, device=args.device)
    speaker_encoder = build_speaker_encoder(
        backend=args.speaker_encoder,
        embedding_dim=config.model.speaker_dim,
        device=args.device,
        source=args.speaker_model_source,
    )
    provider = get_provider(args.teacher_provider, device=args.device, model_id=args.teacher_model_id)

    decision_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for item in items:
        try:
            decision, candidates, runtime = _run_item(
                item=item,
                args=args,
                provider=provider,
                model=model,
                config=config,
                stats=stats,
                tokenizer=tokenizer,
                codec=codec,
                speaker_encoder=speaker_encoder,
                res_dir=res_dir,
                diagnostics_root=diagnostics_root,
            )
            decision_rows.append(decision)
            candidate_rows.extend(candidates)
            runtime_rows.append(runtime)
        except Exception as exc:
            error = {
                "id": item.item_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            errors.append(error)
            print(f"[dcrl-infer] skip id={item.item_id}: {error['error_type']}: {error['error']}")
            if args.fail_fast:
                raise

    decision_csv = Path(args.decision_csv_path).expanduser().resolve() if args.decision_csv_path else diagnostics_root / "dcrl_decisions.csv"
    candidate_csv = Path(args.candidate_csv_path).expanduser().resolve() if args.candidate_csv_path else diagnostics_root / "dcrl_candidates.csv"
    _write_csv(decision_csv, decision_rows, DECISION_FIELDS)
    _write_csv(candidate_csv, candidate_rows, CANDIDATE_FIELDS)
    with (diagnostics_root / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error) + "\n")

    if args.runtime_csv_path:
        _write_csv(Path(args.runtime_csv_path), runtime_rows, RUNTIME_FIELDS)
    if args.runtime_summary_path:
        _write_runtime_summary(Path(args.runtime_summary_path), runtime_rows, Path(args.model).expanduser().resolve())

    summary = {
        "items": len(items),
        "completed": len(decision_rows),
        "errors": len(errors),
        "accepted": sum(1 for row in decision_rows if row["accepted"] is True),
        "acceptance_rate": (
            sum(1 for row in decision_rows if row["accepted"] is True) / len(decision_rows)
            if decision_rows
            else 0.0
        ),
        "decision_csv": str(decision_csv),
        "candidate_csv": str(candidate_csv),
    }
    (diagnostics_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
