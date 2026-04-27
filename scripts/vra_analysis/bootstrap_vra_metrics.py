#!/usr/bin/env python3
"""Build policy metrics and bootstrap confidence intervals for VRA sweeps.

The input is the raw residual-scale sweep CSV written by
``scripts/vra_sweep/run_residual_scale_sweep.py``.  The script converts the
per-candidate rows into deployment-policy rows:

* teacher-only fallback,
* DAC roundtrip / alpha=0,
* fixed residual scales,
* full residual / alpha=1,
* nonzero oracle residual scale,
* DCRL verified lattice, and
* leave-one-item-out learned fixed-scale selector.

All metrics are teacher-relative.  A policy's selected delta is the verifier
utility change over the teacher for each item; fallback has delta zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


POLICY_FIELDS = [
    "policy",
    "samples",
    "accepted",
    "acceptance_rate",
    "positive_cases",
    "positive_rate",
    "regressions",
    "regression_rate",
    "mean_delta",
    "median_delta",
    "best_delta",
    "worst_delta",
    "mean_selected_scale",
    "metric_family",
    "status",
]

CI_FIELDS = [
    "policy",
    "metric",
    "estimate",
    "ci_low",
    "ci_high",
    "bootstrap_samples",
]

SELECTED_FIELDS = [
    "policy",
    "item_id",
    "selected_scale",
    "selected_delta",
    "accepted",
    "positive_case",
    "regression",
    "audio_sane",
    "teacher_similarity",
    "selected_similarity",
    "candidate_wav",
]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Raw residual-scale sweep CSV.")
    parser.add_argument("--output-root", default="paper/results/vra_policy")
    parser.add_argument("--table-root", default="paper/tables")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-regression-rate", type=float, default=0.0)
    parser.add_argument("--metric-family", default="ecapa_speaker_similarity")
    return parser


def _resolve(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: object, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _read_raw(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                **row,
                "scale": _float(row.get("scale")),
                "teacher_similarity": _float(row.get("teacher_similarity")),
                "refined_similarity": _float(row.get("refined_similarity")),
                "delta": _float(row.get("delta")),
                "audio_sane": _bool(row.get("audio_sane")),
                "strict_accept": _bool(row.get("strict_accept")),
                "margin_accept": _bool(row.get("margin_accept")),
                "candidate_duration_s": _float(row.get("candidate_duration_s")),
                "candidate_rms": _float(row.get("candidate_rms")),
                "candidate_peak": _float(row.get("candidate_peak")),
            }
        )
    return parsed


def _group_by_item(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["item_id"])].append(row)
    for item_rows in grouped.values():
        item_rows.sort(key=lambda row: float(row["scale"]))
    return dict(grouped)


def _find_scale(item_rows: list[dict[str, object]], scale: float) -> dict[str, object] | None:
    for row in item_rows:
        if math.isclose(float(row["scale"]), scale, abs_tol=1e-9):
            return row
    return None


def _fallback(policy: str, item_id: str, teacher_similarity: float = math.nan) -> dict[str, object]:
    return {
        "policy": policy,
        "item_id": item_id,
        "selected_scale": "teacher",
        "selected_delta": 0.0,
        "accepted": False,
        "positive_case": False,
        "regression": False,
        "audio_sane": True,
        "teacher_similarity": teacher_similarity,
        "selected_similarity": teacher_similarity,
        "candidate_wav": "",
    }


def _select_row(policy: str, item_id: str, row: dict[str, object], accepted: bool = True) -> dict[str, object]:
    delta = float(row["delta"])
    sane = bool(row["audio_sane"])
    return {
        "policy": policy,
        "item_id": item_id,
        "selected_scale": row["scale"],
        "selected_delta": delta,
        "accepted": bool(accepted),
        "positive_case": sane and delta > 0.0,
        "regression": sane and delta < 0.0,
        "audio_sane": sane,
        "teacher_similarity": row["teacher_similarity"],
        "selected_similarity": row["refined_similarity"],
        "candidate_wav": row.get("candidate_wav", ""),
    }


def _fixed_policy(grouped: dict[str, list[dict[str, object]]], scale: float) -> list[dict[str, object]]:
    policy = "dac_roundtrip_alpha0" if math.isclose(scale, 0.0) else f"fixed_alpha_{scale:g}"
    if math.isclose(scale, 1.0):
        policy = "full_residual_alpha1"
    selected = []
    for item_id, rows in grouped.items():
        row = _find_scale(rows, scale)
        if row is None:
            teacher = float(rows[0]["teacher_similarity"]) if rows else math.nan
            selected.append(_fallback(policy, item_id, teacher))
        else:
            selected.append(_select_row(policy, item_id, row, accepted=True))
    return selected


def _oracle_nonzero(grouped: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    selected = []
    for item_id, rows in grouped.items():
        teacher = float(rows[0]["teacher_similarity"]) if rows else math.nan
        candidates = [row for row in rows if float(row["scale"]) > 0.0 and bool(row["audio_sane"])]
        if not candidates:
            selected.append(_fallback("oracle_nonzero_lattice", item_id, teacher))
            continue
        best = max(candidates, key=lambda row: float(row["delta"]))
        if float(best["delta"]) > 0.0:
            selected.append(_select_row("oracle_nonzero_lattice", item_id, best, accepted=True))
        else:
            selected.append(_fallback("oracle_nonzero_lattice", item_id, teacher))
    return selected


def _dcrl_verified(grouped: dict[str, list[dict[str, object]]], threshold: float = 0.0) -> list[dict[str, object]]:
    policy = "dcrl_verified_lattice" if threshold == 0.0 else f"dcrl_calibrated_lattice_tau_{threshold:g}"
    selected = []
    for item_id, rows in grouped.items():
        teacher = float(rows[0]["teacher_similarity"]) if rows else math.nan
        candidates = [
            row
            for row in rows
            if float(row["scale"]) > 0.0
            and bool(row["audio_sane"])
            and float(row["delta"]) >= threshold
        ]
        if not candidates:
            selected.append(_fallback(policy, item_id, teacher))
        else:
            selected.append(_select_row(policy, item_id, max(candidates, key=lambda row: float(row["delta"])), True))
    return selected


def _loo_learned_fixed(grouped: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    item_ids = list(grouped)
    selected = []
    for heldout in item_ids:
        train_rows = [row for item_id in item_ids if item_id != heldout for row in grouped[item_id]]
        by_scale: dict[float, list[float]] = defaultdict(list)
        for row in train_rows:
            scale = float(row["scale"])
            if scale > 0.0 and bool(row["audio_sane"]):
                by_scale[scale].append(float(row["delta"]))
        scale_scores = {scale: mean(values) for scale, values in by_scale.items() if values}
        teacher = float(grouped[heldout][0]["teacher_similarity"]) if grouped[heldout] else math.nan
        if not scale_scores:
            selected.append(_fallback("learned_fixed_scale_loo", heldout, teacher))
            continue
        best_scale, best_train_score = max(scale_scores.items(), key=lambda item: item[1])
        if best_train_score <= 0.0:
            selected.append(_fallback("learned_fixed_scale_loo", heldout, teacher))
            continue
        row = _find_scale(grouped[heldout], best_scale)
        if row is None:
            selected.append(_fallback("learned_fixed_scale_loo", heldout, teacher))
        else:
            selected.append(_select_row("learned_fixed_scale_loo", heldout, row, accepted=True))
    return selected


def _summarize(policy: str, rows: list[dict[str, object]], metric_family: str) -> dict[str, object]:
    deltas = [float(row["selected_delta"]) for row in rows]
    accepted = [row for row in rows if _bool(row["accepted"])]
    positives = [row for row in rows if _bool(row["positive_case"])]
    regressions = [row for row in rows if _bool(row["regression"])]
    scales = [
        float(row["selected_scale"])
        for row in rows
        if str(row["selected_scale"]) != "teacher" and math.isfinite(_float(row["selected_scale"]))
    ]
    n = len(rows)
    return {
        "policy": policy,
        "samples": n,
        "accepted": len(accepted),
        "acceptance_rate": len(accepted) / n if n else 0.0,
        "positive_cases": len(positives),
        "positive_rate": len(positives) / n if n else 0.0,
        "regressions": len(regressions),
        "regression_rate": len(regressions) / n if n else 0.0,
        "mean_delta": mean(deltas) if deltas else 0.0,
        "median_delta": median(deltas) if deltas else 0.0,
        "best_delta": max(deltas) if deltas else 0.0,
        "worst_delta": min(deltas) if deltas else 0.0,
        "mean_selected_scale": mean(scales) if scales else 0.0,
        "metric_family": metric_family,
        "status": "observed",
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_ci(
    policy: str,
    rows: list[dict[str, object]],
    samples: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    n = len(rows)
    if n == 0:
        return []
    metrics = {
        "mean_delta": lambda sample: mean(float(row["selected_delta"]) for row in sample),
        "positive_rate": lambda sample: sum(_bool(row["positive_case"]) for row in sample) / len(sample),
        "regression_rate": lambda sample: sum(_bool(row["regression"]) for row in sample) / len(sample),
        "acceptance_rate": lambda sample: sum(_bool(row["accepted"]) for row in sample) / len(sample),
    }
    out = []
    for metric, fn in metrics.items():
        estimate = fn(rows)
        draws = []
        for _ in range(samples):
            sample = [rows[rng.randrange(n)] for _ in range(n)]
            draws.append(float(fn(sample)))
        out.append(
            {
                "policy": policy,
                "metric": metric,
                "estimate": estimate,
                "ci_low": _percentile(draws, 0.025),
                "ci_high": _percentile(draws, 0.975),
                "bootstrap_samples": samples,
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = get_parser().parse_args()
    raw_path = _resolve(args.raw)
    output_root = _resolve(args.output_root)
    table_root = _resolve(args.table_root)
    rows = _read_raw(raw_path)
    grouped = _group_by_item(rows)
    scales = sorted({float(row["scale"]) for row in rows})

    policy_rows: dict[str, list[dict[str, object]]] = {
        "teacher_only": [
            _fallback("teacher_only", item_id, float(item_rows[0]["teacher_similarity"]))
            for item_id, item_rows in grouped.items()
        ],
        "oracle_nonzero_lattice": _oracle_nonzero(grouped),
        "dcrl_verified_lattice": _dcrl_verified(grouped, threshold=0.0),
        "learned_fixed_scale_loo": _loo_learned_fixed(grouped),
    }
    for scale in scales:
        selected = _fixed_policy(grouped, scale)
        policy_rows[str(selected[0]["policy"])] = selected

    ordered_policies = [
        "teacher_only",
        "dac_roundtrip_alpha0",
        "fixed_alpha_0.01",
        "fixed_alpha_0.02",
        "fixed_alpha_0.05",
        "fixed_alpha_0.1",
        "fixed_alpha_0.2",
        "fixed_alpha_0.4",
        "fixed_alpha_0.7",
        "full_residual_alpha1",
        "learned_fixed_scale_loo",
        "dcrl_verified_lattice",
        "oracle_nonzero_lattice",
    ]
    ordered_policies.extend(policy for policy in policy_rows if policy not in ordered_policies)

    summaries = [
        _summarize(policy, policy_rows[policy], args.metric_family)
        for policy in ordered_policies
        if policy in policy_rows
    ]
    rng = random.Random(args.seed)
    cis = [
        ci
        for policy in ordered_policies
        if policy in policy_rows
        for ci in _bootstrap_ci(policy, policy_rows[policy], args.bootstrap_samples, rng)
    ]
    selected_rows = [
        row
        for policy in ordered_policies
        if policy in policy_rows
        for row in policy_rows[policy]
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "policy_selected.csv", selected_rows, SELECTED_FIELDS)
    _write_csv(output_root / "vra_policy_metrics.csv", summaries, POLICY_FIELDS)
    _write_csv(output_root / "vra_bootstrap_ci.csv", cis, CI_FIELDS)
    _write_csv(table_root / "vra_policy_metrics.csv", summaries, POLICY_FIELDS)
    _write_csv(table_root / "vra_bootstrap_ci.csv", cis, CI_FIELDS)
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "raw": str(raw_path),
                "items": len(grouped),
                "candidate_rows": len(rows),
                "policies": ordered_policies,
                "bootstrap_samples": args.bootstrap_samples,
                "metric_family": args.metric_family,
                "outputs": {
                    "policy_selected": str(output_root / "policy_selected.csv"),
                    "policy_metrics": str(output_root / "vra_policy_metrics.csv"),
                    "bootstrap_ci": str(output_root / "vra_bootstrap_ci.csv"),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"policy_metrics": str(table_root / "vra_policy_metrics.csv"), "bootstrap_ci": str(table_root / "vra_bootstrap_ci.csv")}, indent=2))


if __name__ == "__main__":
    main()
