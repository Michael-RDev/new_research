#!/usr/bin/env python3
"""Compute practical speech metric summaries from residual-scale sweep rows.

This is intentionally conservative.  ECAPA deltas and audio sanity statistics
come from the sweep CSV.  WER is reported only when an explicit ASR output file
is provided; otherwise the script writes a clear missing-backend/status row so
the paper cannot accidentally imply WER was measured.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


SCALE_FIELDS = [
    "scale",
    "samples",
    "sane_rate",
    "mean_delta",
    "positive_rate",
    "regression_rate",
    "mean_duration_s",
    "mean_rms",
    "mean_peak",
]

STATUS_FIELDS = ["metric", "status", "details"]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output-root", default="paper/results/speech_metrics")
    parser.add_argument("--table-root", default="paper/tables")
    parser.add_argument("--wer-csv", default="", help="Optional CSV with item_id,system,wer columns.")
    return parser


def _resolve(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _float(value: object, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_raw(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        parsed.append(
            {
                **row,
                "scale": _float(row.get("scale")),
                "delta": _float(row.get("delta")),
                "audio_sane": _bool(row.get("audio_sane")),
                "candidate_duration_s": _float(row.get("candidate_duration_s")),
                "candidate_rms": _float(row.get("candidate_rms")),
                "candidate_peak": _float(row.get("candidate_peak")),
            }
        )
    return parsed


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize_scales(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["scale"])].append(row)
    out = []
    for scale in sorted(grouped):
        scale_rows = grouped[scale]
        sane = [row for row in scale_rows if bool(row["audio_sane"])]
        deltas = [float(row["delta"]) for row in sane]
        out.append(
            {
                "scale": scale,
                "samples": len(scale_rows),
                "sane_rate": len(sane) / len(scale_rows) if scale_rows else 0.0,
                "mean_delta": mean(deltas) if deltas else 0.0,
                "positive_rate": sum(delta > 0.0 for delta in deltas) / len(scale_rows) if scale_rows else 0.0,
                "regression_rate": sum(delta < 0.0 for delta in deltas) / len(scale_rows) if scale_rows else 0.0,
                "mean_duration_s": mean(float(row["candidate_duration_s"]) for row in sane) if sane else 0.0,
                "mean_rms": mean(float(row["candidate_rms"]) for row in sane) if sane else 0.0,
                "mean_peak": mean(float(row["candidate_peak"]) for row in sane) if sane else 0.0,
            }
        )
    return out


def main() -> None:
    args = get_parser().parse_args()
    raw_path = _resolve(args.raw)
    output_root = _resolve(args.output_root)
    table_root = _resolve(args.table_root)
    rows = _read_raw(raw_path)
    scale_rows = _summarize_scales(rows)
    status_rows = [
        {
            "metric": "ecapa_speaker_similarity",
            "status": "observed",
            "details": "Read from residual-scale sweep raw.csv.",
        },
        {
            "metric": "audio_sanity",
            "status": "observed",
            "details": "Finite, non-silent, duration, RMS, and peak summaries read from sweep raw.csv.",
        },
    ]
    if args.wer_csv:
        wer_path = _resolve(args.wer_csv)
        status = "observed" if wer_path.exists() else "missing_file"
        detail = str(wer_path)
    else:
        status = "missing_asr_backend"
        detail = "No ASR/WER CSV was supplied; paper must not claim WER evidence for this run."
    status_rows.append({"metric": "wer", "status": status, "details": detail})
    status_rows.append(
        {
            "metric": "utmos_dnsmos_human_preference",
            "status": "not_measured",
            "details": "Reserved for follow-up; not used in current claims.",
        }
    )

    output_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "speech_scale_metrics.csv", scale_rows, SCALE_FIELDS)
    _write_csv(output_root / "speech_metric_status.csv", status_rows, STATUS_FIELDS)
    _write_csv(table_root / "speech_scale_metrics.csv", scale_rows, SCALE_FIELDS)
    _write_csv(table_root / "speech_metric_status.csv", status_rows, STATUS_FIELDS)
    (output_root / "summary.json").write_text(
        json.dumps({"raw": str(raw_path), "rows": len(rows), "status": status_rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"scale_metrics": str(table_root / "speech_scale_metrics.csv"), "metric_status": str(table_root / "speech_metric_status.csv")}, indent=2))


if __name__ == "__main__":
    main()
