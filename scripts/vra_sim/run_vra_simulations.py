#!/usr/bin/env python3
"""Simulate when verified residual lattices beat unsafe residual deployment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


SUMMARY_FIELDS = [
    "setting",
    "policy",
    "examples",
    "accepted",
    "acceptance_rate",
    "mean_delta",
    "positive_rate",
    "regression_rate",
    "worst_delta",
]

PHASE_FIELDS = [
    "verifier_noise",
    "helpfulness",
    "policy",
    "mean_delta",
    "regression_rate",
    "acceptance_rate",
]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--examples", type=int, default=10000)
    parser.add_argument("--output-root", default="paper/results/vra_sim")
    parser.add_argument("--table-root", default="paper/tables")
    parser.add_argument("--target-regression-rate", type=float, default=0.05)
    return parser


def _resolve(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _simulate(
    rng: np.random.Generator,
    examples: int,
    verifier_noise: float,
    helpfulness: float,
    target_regression_rate: float,
) -> dict[str, dict[str, float]]:
    scales = np.asarray([0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00], dtype=np.float64)
    latent_help = rng.normal(helpfulness, 0.10, size=(examples, 1))
    drift = rng.lognormal(mean=-0.2, sigma=0.45, size=(examples, 1))
    hetero = rng.normal(0.0, 0.015, size=(examples, len(scales)))
    true_delta = latent_help * scales - drift * (scales**2) + hetero
    observed_delta = true_delta + rng.normal(0.0, verifier_noise, size=true_delta.shape)

    calib_n = max(10, examples // 3)
    calib_n = min(calib_n, max(1, examples - 1))
    calib_true = true_delta[:calib_n]
    calib_obs = observed_delta[:calib_n]
    test_true = true_delta[calib_n:]
    test_obs = observed_delta[calib_n:]

    thresholds = np.quantile(calib_obs.reshape(-1), np.linspace(0.0, 0.99, 100))
    best_tau = 0.0
    for tau in thresholds:
        accepted = calib_obs >= tau
        if not accepted.any():
            best_tau = float(tau)
            break
        regressions = (calib_true < 0.0) & accepted
        if regressions.sum() / accepted.sum() <= target_regression_rate:
            best_tau = float(tau)
            break

    def summarize(policy: str, selected_delta: np.ndarray, accepted: np.ndarray) -> dict[str, float]:
        n = selected_delta.size
        return {
            "policy": policy,
            "examples": float(n),
            "accepted": float(accepted.sum()),
            "acceptance_rate": float(accepted.mean()),
            "mean_delta": float(selected_delta.mean()),
            "positive_rate": float((selected_delta > 0.0).mean()),
            "regression_rate": float((selected_delta < 0.0).mean()),
            "worst_delta": float(selected_delta.min()),
        }

    out: dict[str, dict[str, float]] = {}
    n_test = test_true.shape[0]
    out["teacher_only"] = summarize("teacher_only", np.zeros(n_test), np.zeros(n_test, dtype=bool))
    fixed_idx = int(np.where(np.isclose(scales, 0.10))[0][0])
    out["fixed_alpha_0.1"] = summarize("fixed_alpha_0.1", test_true[:, fixed_idx], np.ones(n_test, dtype=bool))
    out["full_residual_alpha1"] = summarize("full_residual_alpha1", test_true[:, -1], np.ones(n_test, dtype=bool))

    oracle_idx = np.argmax(test_true, axis=1)
    oracle_delta = test_true[np.arange(n_test), oracle_idx]
    oracle_selected = np.where(oracle_delta > 0.0, oracle_delta, 0.0)
    out["oracle_lattice"] = summarize("oracle_lattice", oracle_selected, oracle_delta > 0.0)

    dcrl_idx = np.argmax(test_obs, axis=1)
    dcrl_obs = test_obs[np.arange(n_test), dcrl_idx]
    dcrl_true = test_true[np.arange(n_test), dcrl_idx]
    dcrl_selected = np.where(dcrl_obs > 0.0, dcrl_true, 0.0)
    out["uncalibrated_dcrl"] = summarize("uncalibrated_dcrl", dcrl_selected, dcrl_obs > 0.0)

    cal_selected = np.where(dcrl_obs >= best_tau, dcrl_true, 0.0)
    out["calibrated_dcrl"] = summarize("calibrated_dcrl", cal_selected, dcrl_obs >= best_tau)
    out["calibrated_dcrl"]["tau"] = best_tau
    return out


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = get_parser().parse_args()
    output_root = _resolve(args.output_root)
    table_root = _resolve(args.table_root)
    output_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    base = _simulate(
        rng,
        examples=args.examples,
        verifier_noise=0.03,
        helpfulness=0.16,
        target_regression_rate=args.target_regression_rate,
    )
    summary_rows = [{"setting": "base", **values} for values in base.values()]

    phase_rows: list[dict[str, object]] = []
    for verifier_noise in [0.0, 0.01, 0.03, 0.06, 0.10]:
        for helpfulness in [0.04, 0.08, 0.12, 0.16, 0.24]:
            local_rng = np.random.default_rng(args.seed + int(verifier_noise * 1000) + int(helpfulness * 100))
            result = _simulate(
                local_rng,
                examples=max(1000, args.examples // 2),
                verifier_noise=verifier_noise,
                helpfulness=helpfulness,
                target_regression_rate=args.target_regression_rate,
            )
            for policy in ["full_residual_alpha1", "uncalibrated_dcrl", "calibrated_dcrl", "oracle_lattice"]:
                phase_rows.append(
                    {
                        "verifier_noise": verifier_noise,
                        "helpfulness": helpfulness,
                        "policy": policy,
                        "mean_delta": result[policy]["mean_delta"],
                        "regression_rate": result[policy]["regression_rate"],
                        "acceptance_rate": result[policy]["acceptance_rate"],
                    }
                )

    _write_csv(output_root / "simulation_summary.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(output_root / "phase_diagram.csv", phase_rows, PHASE_FIELDS)
    _write_csv(table_root / "simulation_summary.csv", summary_rows, SUMMARY_FIELDS)
    (output_root / "summary.json").write_text(
        json.dumps({"examples": args.examples, "seed": args.seed, "target_regression_rate": args.target_regression_rate, "base": base}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"simulation_summary": str(table_root / "simulation_summary.csv"), "phase_diagram": str(output_root / "phase_diagram.csv")}, indent=2))


if __name__ == "__main__":
    main()
