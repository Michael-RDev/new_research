#!/usr/bin/env python3
"""Compute practical speech metric summaries for VRA/DCRL speech evidence.

This is intentionally conservative.  ECAPA deltas and audio sanity statistics
come from the sweep CSV.  WER is reported only when an explicit ASR output file
or benchmark comparison output is provided; otherwise the script writes a clear
missing-backend/status row so the paper cannot accidentally imply WER was
measured.
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

BENCHMARK_FIELDS = [
    "benchmark",
    "label",
    "wer_selected",
    "sim_selected",
    "utmos_selected",
    "mean_real_time_factor",
    "max_rss_bytes",
    "num_samples",
]

REGRESSION_CI_FIELDS = [
    "source",
    "estimate",
    "ci_low",
    "ci_high",
    "samples",
    "details",
]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="", help="Residual-scale sweep raw.csv.")
    parser.add_argument(
        "--benchmark-root",
        default="",
        help="Optional benchmark_compare output root containing comparison/benchmark_summary.csv.",
    )
    parser.add_argument("--output-root", default="paper/results/speech_metrics")
    parser.add_argument("--table-root", default="paper/tables")
    parser.add_argument("--wer-csv", default="", help="Optional CSV with item_id,system,wer columns.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
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


def _read_csv_if_exists(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    import random

    estimate = mean(values)
    if len(values) == 1:
        return estimate, estimate, estimate
    rng = random.Random(seed)
    draws = []
    n = len(values)
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        draws.append(mean(draw))
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    return estimate, low, high


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


def _find_benchmark_summary(benchmark_root: Path) -> Path | None:
    direct = benchmark_root / "comparison" / "benchmark_summary.csv"
    if direct.exists():
        return direct
    matches = sorted(benchmark_root.glob("**/comparison/benchmark_summary.csv"))
    return matches[0] if matches else None


def _benchmark_rows(benchmark_root: Path) -> list[dict[str, object]]:
    summary_path = _find_benchmark_summary(benchmark_root)
    if summary_path is None:
        return []
    summary_json_path = summary_path.with_suffix(".json")
    sample_counts: dict[tuple[str, str], object] = {}
    if summary_json_path.exists():
        try:
            payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
            for benchmark, systems in payload.get("standard_benchmarks", {}).items():
                for label, summary in systems.items():
                    if isinstance(summary, dict):
                        sample_counts[(str(benchmark), str(label))] = summary.get("num_samples", "")
        except Exception:
            sample_counts = {}
    rows = _read_csv_if_exists(summary_path)
    normalized = []
    for row in rows:
        benchmark = row.get("benchmark", "")
        label = row.get("label", "")
        normalized.append(
            {
                "benchmark": benchmark,
                "label": label,
                "wer_selected": row.get("wer_selected", ""),
                "sim_selected": row.get("sim_selected", ""),
                "utmos_selected": row.get("utmos_selected", ""),
                "mean_real_time_factor": row.get("mean_real_time_factor", ""),
                "max_rss_bytes": row.get("max_rss_bytes", ""),
                "num_samples": row.get("num_samples", "") or sample_counts.get((str(benchmark), str(label)), ""),
            }
        )
    return normalized


def _decision_rows(benchmark_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(benchmark_root.glob("**/dcrl_decisions.csv")):
        for row in _read_csv_if_exists(path):
            row["source_path"] = str(path)
            row["accepted_bool"] = _bool(row.get("accepted"))
            row["selected_delta_float"] = _float(row.get("selected_delta"), 0.0)
            rows.append(row)
    return rows


def _regression_ci_rows(
    decisions: list[dict[str, object]],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    if not decisions:
        return []
    accepted = [1.0 if row["accepted_bool"] else 0.0 for row in decisions]
    selected_deltas = [float(row["selected_delta_float"]) for row in decisions]
    regressions = [
        1.0
        if row["accepted_bool"] and float(row["selected_delta_float"]) < 0.0
        else 0.0
        for row in decisions
    ]
    rows = []
    for name, values, detail in [
        ("dcrl_acceptance_rate", accepted, "Fraction of benchmark items where the lattice selected a residual candidate."),
        ("dcrl_mean_selected_delta", selected_deltas, "Mean teacher-relative verifier delta after fallback."),
        ("dcrl_accepted_regression_rate", regressions, "Accepted residuals with negative measured verifier delta."),
    ]:
        estimate, low, high = _bootstrap_ci(values, bootstrap_samples, seed)
        rows.append(
            {
                "source": name,
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "samples": len(values),
                "details": detail,
            }
        )
    return rows


def main() -> None:
    args = get_parser().parse_args()
    output_root = _resolve(args.output_root)
    table_root = _resolve(args.table_root)
    raw_path = _resolve(args.raw) if args.raw else None
    benchmark_root = _resolve(args.benchmark_root) if args.benchmark_root else None
    if raw_path is None and benchmark_root is None:
        raise SystemExit("Provide --raw and/or --benchmark-root.")

    raw_rows = _read_raw(raw_path) if raw_path is not None else []
    scale_rows = _summarize_scales(raw_rows) if raw_rows else []
    benchmark_rows = _benchmark_rows(benchmark_root) if benchmark_root is not None else []
    decisions = _decision_rows(benchmark_root) if benchmark_root is not None else []
    ci_rows = _regression_ci_rows(decisions, args.bootstrap_samples, args.seed)

    status_rows = [
        {
            "metric": "ecapa_speaker_similarity",
            "status": "observed" if raw_rows or decisions or benchmark_rows else "not_observed",
            "details": "Read from residual-scale sweep rows, DCRL decisions, or benchmark SIM logs.",
        },
        {
            "metric": "audio_sanity",
            "status": "observed" if raw_rows or decisions else "not_observed",
            "details": "Finite, non-silent, duration, RMS, and peak summaries read where available.",
        },
    ]
    if args.wer_csv:
        wer_path = _resolve(args.wer_csv)
        status = "observed" if wer_path.exists() else "missing_file"
        detail = str(wer_path)
    elif any(str(row.get("wer_selected", "")).strip() for row in benchmark_rows):
        status = "observed"
        detail = "Read from OmniVoice benchmark_compare summary."
    else:
        status = "missing_asr_backend"
        detail = "No ASR/WER CSV was supplied; paper must not claim WER evidence for this run."
    status_rows.append({"metric": "wer", "status": status, "details": detail})
    if any(str(row.get("utmos_selected", "")).strip() for row in benchmark_rows):
        mos_status = "observed"
        mos_detail = "Read from OmniVoice benchmark_compare UTMOS summary."
    else:
        mos_status = "not_measured"
        mos_detail = "Reserved for benchmark runs with UTMOS/DNSMOS/human preference."
    status_rows.append(
        {
            "metric": "utmos_dnsmos_human_preference",
            "status": mos_status,
            "details": mos_detail,
        }
    )

    output_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    if scale_rows:
        _write_csv(output_root / "speech_scale_metrics.csv", scale_rows, SCALE_FIELDS)
        _write_csv(table_root / "speech_scale_metrics.csv", scale_rows, SCALE_FIELDS)
    if benchmark_rows:
        _write_csv(output_root / "speech_benchmark_metrics.csv", benchmark_rows, BENCHMARK_FIELDS)
        _write_csv(table_root / "speech_benchmark_metrics.csv", benchmark_rows, BENCHMARK_FIELDS)
    if ci_rows:
        _write_csv(output_root / "speech_regression_ci.csv", ci_rows, REGRESSION_CI_FIELDS)
        _write_csv(table_root / "speech_regression_ci.csv", ci_rows, REGRESSION_CI_FIELDS)
    _write_csv(output_root / "speech_metric_status.csv", status_rows, STATUS_FIELDS)
    _write_csv(table_root / "speech_metric_status.csv", status_rows, STATUS_FIELDS)
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "raw": str(raw_path) if raw_path else None,
                "benchmark_root": str(benchmark_root) if benchmark_root else None,
                "raw_rows": len(raw_rows),
                "benchmark_rows": len(benchmark_rows),
                "decision_rows": len(decisions),
                "status": status_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "scale_metrics": str(table_root / "speech_scale_metrics.csv") if scale_rows else None,
                "benchmark_metrics": str(table_root / "speech_benchmark_metrics.csv") if benchmark_rows else None,
                "regression_ci": str(table_root / "speech_regression_ci.csv") if ci_rows else None,
                "metric_status": str(table_root / "speech_metric_status.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
