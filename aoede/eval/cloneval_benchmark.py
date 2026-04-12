from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from aoede.audio.io import load_audio_file, save_audio_bytes
from aoede.eval.common import LoadedAoedeModel, normalize_language, peak_rss_bytes


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CloneEval with an Aoede checkpoint.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_list", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--cloneval_repo", type=str, default="../cloneval")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_step", type=int, default=18)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--t_shift", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--evaluate_emotion_transfer", action="store_true")
    return parser


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_runtime_report(path: Path, rows: list[dict], summary: dict) -> None:
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
    path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = get_parser().parse_args()
    sys.path.insert(0, str(Path(args.cloneval_repo).expanduser().resolve()))
    from cloneval.cloneval import ClonEval

    model = LoadedAoedeModel.load(
        args.model,
        project_root=args.project_root,
        device=args.device_map or args.device,
    )
    entries = read_jsonl(Path(args.test_list))
    if args.limit > 0:
        entries = entries[: args.limit]

    work_dir = Path(args.work_dir)
    ref_dir = work_dir / "reference_audio"
    gen_dir = work_dir / "generated_audio"
    metrics_dir = work_dir / "metrics"
    ref_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    condition_cache: dict[str, object] = {}
    rows: list[dict] = []
    for entry in entries:
        item_id = str(entry["id"])
        ref_audio = entry["ref_audio"]
        cache_key = str(Path(ref_audio).expanduser().resolve())

        condition = condition_cache.get(cache_key)
        if condition is None:
            condition = model.prepare_voice_condition(cache_key)
            condition_cache[cache_key] = condition

        reference_audio, _ = load_audio_file(
            cache_key,
            target_sample_rate=model.config.model.sample_rate,
        )
        (ref_dir / f"{item_id}.wav").write_bytes(
            save_audio_bytes(reference_audio, sample_rate=model.config.model.sample_rate)
        )

        start = time.perf_counter()
        generated = model.synthesize(
            text=entry["text"],
            language_code=normalize_language(
                entry.get("language_id") or entry.get("language_name")
            ),
            condition=condition,
            sampling_steps=args.num_step,
        )
        latency_s = time.perf_counter() - start
        (gen_dir / f"{item_id}.wav").write_bytes(
            save_audio_bytes(generated, sample_rate=model.config.model.sample_rate)
        )

        output_duration_s = len(generated) / float(model.config.model.sample_rate)
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

    cloneval = ClonEval()
    cloneval.evaluate(
        original_dir=str(ref_dir),
        cloned_dir=str(gen_dir),
        evaluate_emotion_transfer=args.evaluate_emotion_transfer,
        output_dir=str(metrics_dir),
    )

    summary = {
        "num_samples": len(rows),
        "mean_latency_s": sum(row["latency_s"] for row in rows) / max(len(rows), 1),
        "mean_output_duration_s": sum(row["output_duration_s"] for row in rows)
        / max(len(rows), 1),
        "mean_real_time_factor": sum(row["real_time_factor"] for row in rows)
        / max(len(rows), 1),
        "max_rss_bytes": max((row["rss_bytes"] for row in rows), default=0),
        "reference_dir": str(ref_dir),
        "generated_dir": str(gen_dir),
        "metrics_dir": str(metrics_dir),
        "checkpoint": str(model.checkpoint_path),
    }
    save_runtime_report(metrics_dir / "runtime_metrics.csv", rows, summary)


if __name__ == "__main__":
    main()
