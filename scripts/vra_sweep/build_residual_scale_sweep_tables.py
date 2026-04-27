#!/usr/bin/env python3
"""Build CSV summaries for VRA residual-scale sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
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
    "status",
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
    "status",
]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate residual-scale sweep paper tables.")
    parser.add_argument("--input-root", default="paper/results/residual_scale_sweep")
    parser.add_argument("--primary-run", default="core_eval_32")
    parser.add_argument("--personal-run", default="me_voice")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--table-root", default="paper/tables")
    return parser


def _resolve(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _pending_summary(run_name: str) -> dict[str, object]:
    return {
        "run_name": run_name,
        "scale": 0.0,
        "samples": 0,
        "sane_samples": 0,
        "positive_cases": 0,
        "strict_acceptance_rate": 0.0,
        "margin_acceptance_rate": 0.0,
        "mean_delta": 0.0,
        "median_delta": 0.0,
        "max_delta": 0.0,
        "min_delta": 0.0,
        "mean_teacher_similarity": 0.0,
        "mean_refined_similarity": 0.0,
        "status": "pending_run",
    }


def _pending_best(run_name: str) -> dict[str, object]:
    return {
        "run_name": run_name,
        "item_id": "pending",
        "text": "run residual-scale sweep",
        "language": "",
        "best_scale": 0.0,
        "teacher_similarity": 0.0,
        "best_similarity": 0.0,
        "best_delta": 0.0,
        "audio_sane": False,
        "positive_case": False,
        "candidate_wav": "",
        "status": "pending_run",
    }


def _load_run(input_root: Path, run_name: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_dir = input_root / run_name
    summary_rows = _read_csv(run_dir / "summary_by_scale.csv")
    best_rows = _read_csv(run_dir / "best_by_item.csv")
    for row in summary_rows:
        row["run_name"] = row.get("run_name") or run_name
        row["status"] = "observed"
    for row in best_rows:
        row["run_name"] = row.get("run_name") or run_name
        row["status"] = "observed"
    if not summary_rows:
        summary_rows = [_pending_summary(run_name)]
    if not best_rows:
        best_rows = [_pending_best(run_name)]
    return summary_rows, best_rows


def main() -> None:
    args = get_parser().parse_args()
    input_root = _resolve(args.input_root)
    table_root = _resolve(args.table_root)
    runs = [args.primary_run, args.personal_run]
    summary_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    for run_name in runs:
        run_summary, run_best = _load_run(input_root, run_name)
        summary_rows.extend(run_summary)
        best_rows.extend(run_best)
    best_rows = sorted(
        best_rows,
        key=lambda row: _float(row.get("best_delta"), default=-1e9),
        reverse=True,
    )[: args.top_k]
    summary_path = table_root / "residual_scale_sweep_summary.csv"
    best_path = table_root / "residual_scale_sweep_best_cases.csv"
    _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    _write_csv(best_path, best_rows, BEST_FIELDS)
    print(json.dumps({"summary_table": str(summary_path), "best_cases_table": str(best_path)}, indent=2))


if __name__ == "__main__":
    main()
