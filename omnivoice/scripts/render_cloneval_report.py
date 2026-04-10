#!/usr/bin/env python3

"""Render baseline / proposed / ablation CloneEval results into markdown."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_aggregated(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _read_runtime_summary(path: Path):
    summary_path = path.with_suffix(".summary.json")
    if not summary_path.exists():
        return {}
    import json

    return json.loads(summary_path.read_text(encoding="utf-8"))


def _mean_metric(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        values.append(float(value))
    if not values:
        return "n/a"
    return f"{sum(values) / len(values):.4f}"


def _row_for(label: str, metrics_dir: Path):
    rows = _read_aggregated(metrics_dir / "aggregated_results.csv")
    runtime = _read_runtime_summary(metrics_dir / "runtime_metrics.csv")
    return {
        "name": label,
        "speaker_similarity": _mean_metric(rows, "speaker_similarity"),
        "intelligibility": _mean_metric(rows, "intelligibility"),
        "naturalness": _mean_metric(rows, "naturalness"),
        "emotion_transfer": _mean_metric(rows, "emotion_transfer"),
        "mean_latency_s": f"{runtime.get('mean_latency_s', 'n/a')}",
        "mean_rtf": f"{runtime.get('mean_real_time_factor', 'n/a')}",
    }


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--mnemosvoice_dir", type=str, required=True)
    parser.add_argument(
        "--ablation",
        action="append",
        default=[],
        help="Additional report rows in NAME=METRICS_DIR format.",
    )
    parser.add_argument("--output", type=str, required=True)
    return parser


def main():
    args = get_parser().parse_args()
    rows = [
        _row_for("OmniVoice baseline", Path(args.baseline_dir)),
        _row_for("MnemosVoice", Path(args.mnemosvoice_dir)),
    ]
    for item in args.ablation:
        label, metrics_dir = item.split("=", 1)
        rows.append(_row_for(label, Path(metrics_dir)))

    output_lines = [
        "| Model | Speaker Similarity | Intelligibility | Naturalness | Emotion Transfer | Mean Latency (s) | Mean RTF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        output_lines.append(
            "| {name} | {speaker_similarity} | {intelligibility} | {naturalness} | {emotion_transfer} | {mean_latency_s} | {mean_rtf} |".format(
                **row
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

