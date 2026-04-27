#!/usr/bin/env python3
"""Fit a small cross-validated residual-scale selector from sweep rows.

This script is intentionally lightweight: it uses only NumPy and the raw sweep
CSV.  The selector predicts teacher-relative verifier delta for each candidate
from deployable candidate metadata, then selects the highest predicted nonzero
scale if the predicted improvement is positive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


CV_FIELDS = [
    "item_id",
    "selected_scale",
    "predicted_delta",
    "actual_delta",
    "accepted",
    "positive_case",
    "regression",
]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output-root", default="paper/results/vra_selector")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--include-verifier-score",
        action="store_true",
        help="Include refined_similarity and delta as features. Useful as an upper-bound diagnostic, not as a strict pre-verifier selector.",
    )
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


def _read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        parsed.append(
            {
                **row,
                "scale": _float(row.get("scale")),
                "teacher_similarity": _float(row.get("teacher_similarity")),
                "refined_similarity": _float(row.get("refined_similarity")),
                "delta": _float(row.get("delta")),
                "audio_sane": _bool(row.get("audio_sane")),
                "candidate_duration_s": _float(row.get("candidate_duration_s"), 0.0),
                "candidate_rms": _float(row.get("candidate_rms"), 0.0),
                "candidate_peak": _float(row.get("candidate_peak"), 0.0),
            }
        )
    return parsed


def _features(row: dict[str, object], include_verifier_score: bool) -> list[float]:
    scale = float(row["scale"])
    values = [
        1.0,
        scale,
        scale * scale,
        math.log1p(max(0.0, float(row["candidate_duration_s"]))),
        float(row["candidate_rms"]),
        float(row["candidate_peak"]),
        float(row["teacher_similarity"]),
    ]
    if include_verifier_score:
        values.extend([float(row["refined_similarity"]), float(row["delta"])])
    return values


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    xtx = x.T @ x
    penalty = np.eye(xtx.shape[0]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(xtx + penalty, x.T @ y)


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = get_parser().parse_args()
    raw_path = _resolve(args.raw)
    rows = [row for row in _read_rows(raw_path) if bool(row["audio_sane"])]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["item_id"])].append(row)

    cv_rows: list[dict[str, object]] = []
    item_ids = sorted(grouped)
    for heldout in item_ids:
        train = [row for item_id in item_ids if item_id != heldout for row in grouped[item_id]]
        test = [row for row in grouped[heldout] if float(row["scale"]) > 0.0]
        if not train or not test:
            continue
        x = np.asarray([_features(row, args.include_verifier_score) for row in train], dtype=np.float64)
        y = np.asarray([float(row["delta"]) for row in train], dtype=np.float64)
        weights = _fit_ridge(x, y, args.ridge)
        predictions = []
        for row in test:
            pred = float(np.dot(np.asarray(_features(row, args.include_verifier_score)), weights))
            predictions.append((pred, row))
        predicted_delta, selected = max(predictions, key=lambda item: item[0])
        accepted = predicted_delta > 0.0
        actual_delta = float(selected["delta"]) if accepted else 0.0
        cv_rows.append(
            {
                "item_id": heldout,
                "selected_scale": selected["scale"] if accepted else "teacher",
                "predicted_delta": predicted_delta,
                "actual_delta": actual_delta,
                "accepted": accepted,
                "positive_case": actual_delta > 0.0,
                "regression": actual_delta < 0.0,
            }
        )

    n = len(cv_rows)
    accepted = sum(1 for row in cv_rows if row["accepted"])
    positives = sum(1 for row in cv_rows if row["positive_case"])
    regressions = sum(1 for row in cv_rows if row["regression"])
    deltas = [float(row["actual_delta"]) for row in cv_rows]
    summary = {
        "raw": str(raw_path),
        "samples": n,
        "accepted": accepted,
        "acceptance_rate": accepted / n if n else 0.0,
        "positive_cases": positives,
        "positive_rate": positives / n if n else 0.0,
        "regressions": regressions,
        "regression_rate": regressions / n if n else 0.0,
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
        "include_verifier_score": args.include_verifier_score,
        "ridge": args.ridge,
    }

    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "selector_cv.csv", cv_rows, CV_FIELDS)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
