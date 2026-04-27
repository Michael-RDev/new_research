#!/usr/bin/env python3
"""Sweep residual strength for Aoede's teacher-relative latent refiner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aoede.audio.codec import build_audio_codec
from aoede.audio.io import load_audio_file, resample_audio, save_audio_bytes
from aoede.audio.speaker import build_speaker_encoder
from aoede.languages import language_index, normalize_language
from aoede.providers import get_provider
from scripts.run_sota_clone import _cosine, _load_refiner, _sane_audio

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


RAW_FIELDS = [
    "run_name",
    "item_id",
    "text",
    "language",
    "scale",
    "teacher_similarity",
    "refined_similarity",
    "delta",
    "audio_sane",
    "strict_accept",
    "margin_accept",
    "teacher_wav",
    "candidate_wav",
    "ref_audio",
    "teacher_provider",
    "teacher_sample_rate",
    "candidate_sample_rate",
    "candidate_duration_s",
    "candidate_rms",
    "candidate_peak",
    "refine_steps",
]

SUMMARY_FIELDS = [
    "run_name",
    "scale",
    "samples",
    "sane_samples",
    "positive_cases",
    "strict_acceptance_rate",
    "margin_acceptance_rate",
    "mean_delta",
    "median_delta",
    "max_delta",
    "min_delta",
    "mean_teacher_similarity",
    "mean_refined_similarity",
]

BEST_FIELDS = [
    "run_name",
    "item_id",
    "text",
    "language",
    "best_scale",
    "teacher_similarity",
    "best_similarity",
    "best_delta",
    "audio_sane",
    "positive_case",
    "candidate_wav",
]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run z_alpha = z_teacher + alpha * (z_refined_full - z_teacher)."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--refine-steps", type=int, default=4)
    parser.add_argument("--speaker-margin", type=float, default=0.0)
    parser.add_argument("--scales", nargs="+", type=float, required=True)
    parser.add_argument("--output-root", default="paper/results/residual_scale_sweep")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--teacher-provider", default="voxcpm2")
    parser.add_argument("--teacher-model-id", default=None)
    parser.add_argument("--speaker-encoder", default="ecapa")
    parser.add_argument("--speaker-model-source", default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--min-duration-s", type=float, default=0.25)
    parser.add_argument("--no-save-wavs", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "item"


def _scale_label(scale: float) -> str:
    return f"{scale:.4f}".rstrip("0").rstrip(".").replace("-", "neg").replace(".", "p")


def _iter_manifest(path: Path, limit: int | None):
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        item_id = str(payload.get("id") or payload.get("item_id") or f"item_{line_number:04d}")
        text = str(payload.get("text") or "").strip()
        ref_audio = payload.get("ref_audio") or payload.get("reference_audio") or payload.get("audio_path")
        language = str(payload.get("language") or payload.get("language_code") or "en").strip()
        if not text or not ref_audio:
            raise ValueError(f"{path}:{line_number}: rows require text and ref_audio")
        yield {
            "id": item_id,
            "text": text,
            "ref_audio": str(_resolve(ref_audio)),
            "ref_text": str(payload.get("ref_text") or payload.get("reference_text") or ""),
            "language": language,
        }
        count += 1
        if limit is not None and count >= limit:
            return


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
    stats = _audio_stats(audio, sample_rate)
    return bool(_sane_audio(audio) and stats["duration_s"] >= min_duration_s)


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _build_summary_by_scale(run_name: str, raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        grouped[float(row["scale"])].append(row)

    rows: list[dict[str, object]] = []
    for scale in sorted(grouped):
        scale_rows = grouped[scale]
        deltas = [float(row["delta"]) for row in scale_rows if row.get("audio_sane")]
        teacher_scores = [float(row["teacher_similarity"]) for row in scale_rows]
        refined_scores = [
            float(row["refined_similarity"])
            for row in scale_rows
            if row.get("audio_sane") and math.isfinite(float(row["refined_similarity"]))
        ]
        samples = len(scale_rows)
        sane_samples = sum(1 for row in scale_rows if row.get("audio_sane"))
        positives = sum(1 for row in scale_rows if row.get("strict_accept"))
        margin_accepts = sum(1 for row in scale_rows if row.get("margin_accept"))
        rows.append(
            {
                "run_name": run_name,
                "scale": scale,
                "samples": samples,
                "sane_samples": sane_samples,
                "positive_cases": positives,
                "strict_acceptance_rate": positives / samples if samples else 0.0,
                "margin_acceptance_rate": margin_accepts / samples if samples else 0.0,
                "mean_delta": _mean(deltas),
                "median_delta": float(median(deltas)) if deltas else math.nan,
                "max_delta": max(deltas) if deltas else math.nan,
                "min_delta": min(deltas) if deltas else math.nan,
                "mean_teacher_similarity": _mean(teacher_scores),
                "mean_refined_similarity": _mean(refined_scores),
            }
        )
    return rows


def _build_best_by_item(run_name: str, raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        grouped[str(row["item_id"])].append(row)

    rows: list[dict[str, object]] = []
    for item_id, item_rows in grouped.items():
        sane_rows = [row for row in item_rows if row.get("audio_sane")]
        candidates = sane_rows or item_rows
        best = max(candidates, key=lambda row: float(row["delta"]))
        rows.append(
            {
                "run_name": run_name,
                "item_id": item_id,
                "text": best["text"],
                "language": best["language"],
                "best_scale": best["scale"],
                "teacher_similarity": best["teacher_similarity"],
                "best_similarity": best["refined_similarity"],
                "best_delta": best["delta"],
                "audio_sane": best["audio_sane"],
                "positive_case": bool(best["audio_sane"] and float(best["delta"]) > 0.0),
                "candidate_wav": best["candidate_wav"],
            }
        )
    return sorted(rows, key=lambda row: float(row["best_delta"]), reverse=True)


def _save_wav(path: Path, audio: np.ndarray, sample_rate: int, disabled: bool) -> str:
    if disabled:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(save_audio_bytes(audio, sample_rate=sample_rate))
    return str(path)


def _encode_audio(codec, audio: np.ndarray, source_sr: int, sample_rate: int, device: str) -> torch.Tensor:
    resampled = resample_audio(audio, source_sr, sample_rate)
    waveform = torch.from_numpy(resampled).float().unsqueeze(0).to(device)
    return codec.encode(waveform)


def _run_one_item(args, item, run_dir, provider, model, config, stats, tokenizer, codec, speaker_encoder):
    item_id = _safe_id(item["id"])
    item_wav_dir = run_dir / "wavs" / item_id
    language_code = normalize_language(item["language"])
    ref_audio_path = Path(item["ref_audio"])
    if not ref_audio_path.exists():
        raise FileNotFoundError(ref_audio_path)

    teacher = provider.synthesize(
        text=item["text"],
        reference_audio=str(ref_audio_path),
        language=language_code,
        prompt_text=item.get("ref_text") or None,
    )
    teacher_wav = _save_wav(item_wav_dir / "teacher.wav", teacher.audio, teacher.sample_rate, args.no_save_wavs)

    ref_audio_speaker, ref_sr_speaker = load_audio_file(ref_audio_path, target_sample_rate=24000)
    ref_embedding_np = speaker_encoder.encode(ref_audio_speaker, sample_rate=ref_sr_speaker)
    teacher_embedding_np = speaker_encoder.encode(teacher.audio, sample_rate=teacher.sample_rate)
    teacher_similarity = _cosine(ref_embedding_np, teacher_embedding_np)

    with torch.inference_mode():
        teacher_latents = _encode_audio(codec, teacher.audio, teacher.sample_rate, config.model.sample_rate, args.device)
        ref_audio_codec, ref_sr_codec = load_audio_file(ref_audio_path, target_sample_rate=config.model.sample_rate)
        reference_latents = _encode_audio(codec, ref_audio_codec, ref_sr_codec, config.model.sample_rate, args.device)
        speaker_embedding = torch.from_numpy(ref_embedding_np).float().unsqueeze(0).to(args.device)
        token_ids = tokenizer.encode(item["text"], language_code, add_new_tokens=False)
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=args.device)
        language_tensor = torch.tensor([language_index(language_code)], dtype=torch.long, device=args.device)
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

    rows: list[dict[str, object]] = []
    residual = full_refined - normalized_teacher
    for scale in args.scales:
        with torch.inference_mode():
            scaled = normalized_teacher + float(scale) * residual
            audio = codec.decode(stats.denormalize(scaled))[0].detach().cpu().numpy().astype(np.float32)
        candidate_stats = _audio_stats(audio, config.model.sample_rate)
        audio_sane = _is_audio_sane(audio, config.model.sample_rate, args.min_duration_s)
        refined_similarity = math.nan
        if audio_sane:
            refined_embedding_np = speaker_encoder.encode(audio, sample_rate=config.model.sample_rate)
            refined_similarity = _cosine(ref_embedding_np, refined_embedding_np)
        delta = refined_similarity - teacher_similarity if math.isfinite(refined_similarity) else math.nan
        candidate_wav = _save_wav(
            item_wav_dir / f"alpha_{_scale_label(float(scale))}.wav",
            audio,
            config.model.sample_rate,
            args.no_save_wavs,
        )
        rows.append(
            {
                "run_name": args.run_name,
                "item_id": item_id,
                "text": item["text"],
                "language": language_code,
                "scale": float(scale),
                "teacher_similarity": teacher_similarity,
                "refined_similarity": refined_similarity,
                "delta": delta,
                "audio_sane": audio_sane,
                "strict_accept": bool(audio_sane and delta > 0.0),
                "margin_accept": bool(audio_sane and refined_similarity + args.speaker_margin >= teacher_similarity),
                "teacher_wav": teacher_wav,
                "candidate_wav": candidate_wav,
                "ref_audio": str(ref_audio_path),
                "teacher_provider": teacher.provider,
                "teacher_sample_rate": teacher.sample_rate,
                "candidate_sample_rate": config.model.sample_rate,
                "candidate_duration_s": candidate_stats["duration_s"],
                "candidate_rms": candidate_stats["rms"],
                "candidate_peak": candidate_stats["peak"],
                "refine_steps": args.refine_steps,
            }
        )
    return rows


def main() -> None:
    args = get_parser().parse_args()
    manifest = _resolve(args.manifest)
    checkpoint = _resolve(args.checkpoint)
    tokenizer_path = _resolve(args.tokenizer_path)
    run_dir = _resolve(args.output_root) / _safe_id(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    items = list(_iter_manifest(manifest, args.limit))
    if not items:
        raise RuntimeError(f"No sweep items found in {manifest}.")

    model, config, stats, tokenizer = _load_refiner(str(checkpoint), args.device, str(tokenizer_path))
    codec = build_audio_codec(config.model, device=args.device)
    speaker_encoder = build_speaker_encoder(
        backend=args.speaker_encoder,
        embedding_dim=config.model.speaker_dim,
        device=args.device,
        source=args.speaker_model_source,
    )
    provider = get_provider(args.teacher_provider, device=args.device, model_id=args.teacher_model_id)

    raw_rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    start_time = time.time()
    iterator = tqdm(items, desc=f"sweep:{args.run_name}") if tqdm else items
    for item in iterator:
        try:
            raw_rows.extend(
                _run_one_item(args, item, run_dir, provider, model, config, stats, tokenizer, codec, speaker_encoder)
            )
        except Exception as exc:
            error = {"item_id": item.get("id", ""), "error_type": type(exc).__name__, "error": str(exc)}
            errors.append(error)
            print(f"[residual-scale-sweep] skip item={error['item_id']}: {error['error_type']}: {error['error']}")
            if args.fail_fast:
                raise

    summary_rows = _build_summary_by_scale(args.run_name, raw_rows)
    best_rows = _build_best_by_item(args.run_name, raw_rows)
    _write_csv(run_dir / "raw.csv", raw_rows, RAW_FIELDS)
    _write_csv(run_dir / "summary_by_scale.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(run_dir / "best_by_item.csv", best_rows, BEST_FIELDS)
    with (run_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")

    positive_best = [row for row in best_rows if row["positive_case"]]
    best_overall = best_rows[0] if best_rows else None
    best_fixed = max(
        summary_rows,
        key=lambda row: float(row["mean_delta"]) if math.isfinite(float(row["mean_delta"])) else -1e9,
        default=None,
    )
    summary = {
        "run_name": args.run_name,
        "manifest": str(manifest),
        "checkpoint": str(checkpoint),
        "tokenizer_path": str(tokenizer_path),
        "device": args.device,
        "teacher_provider": args.teacher_provider,
        "speaker_encoder": args.speaker_encoder,
        "speaker_margin": args.speaker_margin,
        "refine_steps": args.refine_steps,
        "scales": args.scales,
        "items_requested": len(items),
        "items_completed": len(best_rows),
        "candidate_rows": len(raw_rows),
        "errors": len(errors),
        "oracle_positive_cases": len(positive_best),
        "oracle_positive_rate": len(positive_best) / len(best_rows) if best_rows else 0.0,
        "best_delta": _finite_or_none(best_overall["best_delta"]) if best_overall else None,
        "best_item_id": best_overall["item_id"] if best_overall else None,
        "best_scale": _finite_or_none(best_overall["best_scale"]) if best_overall else None,
        "best_fixed_scale": _finite_or_none(best_fixed["scale"]) if best_fixed else None,
        "best_fixed_mean_delta": _finite_or_none(best_fixed["mean_delta"]) if best_fixed else None,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - start_time, 3),
        "outputs": {
            "raw": str(run_dir / "raw.csv"),
            "summary_by_scale": str(run_dir / "summary_by_scale.csv"),
            "best_by_item": str(run_dir / "best_by_item.csv"),
            "errors": str(run_dir / "errors.jsonl"),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
