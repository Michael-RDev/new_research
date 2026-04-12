from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

from aoede.audio.io import save_audio_bytes
from aoede.eval.common import LoadedAoedeModel, normalize_language, peak_rss_bytes


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer an Aoede checkpoint over an OmniVoice-style benchmark JSONL."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_list", type=str, required=True)
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_step", type=int, default=18)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--t_shift", type=float, default=0.1)
    parser.add_argument("--nj_per_gpu", type=int, default=1)
    parser.add_argument("--audio_chunk_duration", type=float, default=15.0)
    parser.add_argument("--audio_chunk_threshold", type=float, default=30.0)
    parser.add_argument("--batch_duration", type=float, default=1000.0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--preprocess_prompt", type=str, default="False")
    parser.add_argument("--postprocess_output", type=str, default="False")
    parser.add_argument("--layer_penalty_factor", type=float, default=5.0)
    parser.add_argument("--position_temperature", type=float, default=5.0)
    parser.add_argument("--class_temperature", type=float, default=0.0)
    parser.add_argument("--denoise", type=str, default="True")
    parser.add_argument("--lang_id", type=str, default=None)
    parser.add_argument("--runtime_csv_path", type=str, default=None)
    parser.add_argument("--runtime_summary_path", type=str, default=None)
    return parser


def read_test_list(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_runtime_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "latency_s",
                "output_duration_s",
                "real_time_factor",
                "rss_bytes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_runtime_summary(path: Path, rows: list[dict], checkpoint_path: Path) -> None:
    summary = {
        "num_samples": len(rows),
        "mean_latency_s": _mean(rows, "latency_s"),
        "mean_output_duration_s": _mean(rows, "output_duration_s"),
        "mean_real_time_factor": _mean(rows, "real_time_factor"),
        "max_rss_bytes": max((int(row["rss_bytes"]) for row in rows), default=0),
        "checkpoint": str(checkpoint_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _mean(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def main() -> None:
    args = get_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model = LoadedAoedeModel.load(
        args.model,
        project_root=args.project_root,
        device=args.device,
    )
    entries = read_test_list(Path(args.test_list))
    res_dir = Path(args.res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    condition_cache: dict[str, object] = {}
    rows: list[dict] = []

    for warmup_index in range(args.warmup):
        if not entries:
            break
        entry = entries[warmup_index % len(entries)]
        ref_audio = entry.get("ref_audio")
        if ref_audio:
            key = str(Path(ref_audio).expanduser().resolve())
            condition = condition_cache.get(key)
            if condition is None:
                condition = model.prepare_voice_condition(key)
                condition_cache[key] = condition
            model.synthesize(
                text=entry["text"],
                language_code=entry.get("language_id") or args.lang_id or "en",
                condition=condition,
                sampling_steps=args.num_step,
            )

    for entry in entries:
        item_id = str(entry["id"])
        ref_audio = entry.get("ref_audio")
        if not ref_audio:
            raise ValueError(
                f"Benchmark entry {item_id} does not include ref_audio; "
                "Aoede benchmark inference currently expects voice-cloning manifests."
            )

        cache_key = str(Path(ref_audio).expanduser().resolve())
        condition = condition_cache.get(cache_key)
        if condition is None:
            condition = model.prepare_voice_condition(cache_key)
            condition_cache[cache_key] = condition

        language_code = normalize_language(
            entry.get("language_id") or entry.get("language_name") or args.lang_id
        )

        start = time.perf_counter()
        audio = model.synthesize(
            text=entry["text"],
            language_code=language_code,
            condition=condition,
            sampling_steps=args.num_step,
        )
        latency_s = time.perf_counter() - start

        output_path = res_dir / f"{item_id}.wav"
        output_path.write_bytes(
            save_audio_bytes(audio, sample_rate=model.config.model.sample_rate)
        )
        output_duration_s = len(audio) / float(model.config.model.sample_rate)
        rows.append(
            {
                "id": item_id,
                "latency_s": round(latency_s, 4),
                "output_duration_s": round(output_duration_s, 4),
                "real_time_factor": round(
                    latency_s / max(output_duration_s, 1e-6),
                    4,
                ),
                "rss_bytes": peak_rss_bytes(),
            }
        )

    if args.runtime_csv_path:
        write_runtime_csv(Path(args.runtime_csv_path), rows)
    if args.runtime_summary_path:
        write_runtime_summary(
            Path(args.runtime_summary_path),
            rows,
            model.checkpoint_path,
        )


if __name__ == "__main__":
    main()
