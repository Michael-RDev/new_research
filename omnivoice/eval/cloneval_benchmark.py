#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run OmniVoice-compatible models against the official ClonEval benchmark.

This script bridges the OmniVoice JSONL test-list format and the ClonEval
directory-based evaluation API. It:

1. Generates cloned speech with ``OmniVoice.generate``.
2. Materializes reference audio into an ``original_dir`` using the same file ids.
3. Runs the official ClonEval evaluator.
4. Saves speed and memory summaries next to the metric CSVs.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import psutil
import torch
import torchaudio

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.audio import load_audio


def get_parser():
    parser = argparse.ArgumentParser(description="Run Cloneval with OmniVoice")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_list", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument(
        "--cloneval_repo",
        type=str,
        default="../cloneval",
        help="Path to the cloned official ClonEval repository",
    )
    parser.add_argument("--num_step", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--t_shift", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device_map", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--evaluate_emotion_transfer", action="store_true")
    return parser


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ensure_wav(path: Path, wav: torch.Tensor, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav.cpu(), sample_rate)


def save_speed_report(path: Path, rows, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
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
    with open(path.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    args = get_parser().parse_args()
    sys.path.insert(0, str(Path(args.cloneval_repo).resolve()))
    from cloneval.cloneval import ClonEval

    dtype = getattr(torch, args.dtype)
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=args.device_map,
        dtype=dtype,
    )
    process = psutil.Process(os.getpid())

    entries = read_jsonl(args.test_list)
    if args.limit > 0:
        entries = entries[: args.limit]

    work_dir = Path(args.work_dir)
    ref_dir = work_dir / "reference_audio"
    gen_dir = work_dir / "generated_audio"
    metrics_dir = work_dir / "metrics"
    ref_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in entries:
        item_id = entry["id"]
        ref_audio = entry["ref_audio"]
        ref_text = entry.get("ref_text")
        language = entry.get("language_name") or entry.get("language_id")

        wav = load_audio(ref_audio, model.sampling_rate)
        ensure_wav(ref_dir / f"{item_id}.wav", wav, model.sampling_rate)

        start = time.perf_counter()
        generated = model.generate(
            text=entry["text"],
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            num_step=args.num_step,
            guidance_scale=args.guidance_scale,
            t_shift=args.t_shift,
        )[0]
        latency_s = time.perf_counter() - start
        ensure_wav(gen_dir / f"{item_id}.wav", generated, model.sampling_rate)

        output_duration_s = generated.size(-1) / model.sampling_rate
        rows.append(
            {
                "id": item_id,
                "latency_s": round(latency_s, 4),
                "output_duration_s": round(output_duration_s, 4),
                "real_time_factor": round(
                    latency_s / max(output_duration_s, 1e-6),
                    4,
                ),
                "rss_bytes": process.memory_info().rss,
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
        "mean_latency_s": sum(r["latency_s"] for r in rows) / max(len(rows), 1),
        "mean_output_duration_s": sum(r["output_duration_s"] for r in rows)
        / max(len(rows), 1),
        "mean_real_time_factor": sum(r["real_time_factor"] for r in rows)
        / max(len(rows), 1),
        "max_rss_bytes": max((r["rss_bytes"] for r in rows), default=0),
        "reference_dir": str(ref_dir),
        "generated_dir": str(gen_dir),
        "metrics_dir": str(metrics_dir),
    }
    save_speed_report(metrics_dir / "runtime_metrics.csv", rows, summary)


if __name__ == "__main__":
    main()
