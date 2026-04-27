#!/usr/bin/env python3
"""Evaluate code VRA candidates from a JSONL file.

Each JSONL row should contain:

```
{
  "task_id": "example",
  "prompt": "...",
  "function_name": "solve",
  "teacher_code": "def solve(...): ...",
  "candidate_code": "def solve(...): ...",
  "public_tests": [{"args": [...], "expected": ...}],
  "hidden_tests": [{"args": [...], "expected": ...}]
}
```

For multiple residual candidates, use ``candidate_codes`` as a list of either
strings or ``{"name": ..., "code": ...}`` objects. Public tests are the
deployable verifier; hidden tests are evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


RAW_FIELDS = [
    "task_id",
    "policy",
    "candidate",
    "accepted",
    "public_passed",
    "public_total",
    "hidden_passed",
    "hidden_total",
    "hidden_delta",
    "patch_chars",
]

SUMMARY_FIELDS = [
    "policy",
    "tasks",
    "accepted",
    "acceptance_rate",
    "public_pass_rate",
    "hidden_pass_rate",
    "mean_hidden_delta",
    "hidden_regressions",
    "hidden_regression_rate",
    "mean_patch_chars",
]


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


@dataclass(frozen=True)
class Candidate:
    name: str
    code: str


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    function_name: str
    teacher_code: str
    candidates: tuple[Candidate, ...]
    public_tests: tuple[tuple[tuple[Any, ...], Any], ...]
    hidden_tests: tuple[tuple[tuple[Any, ...], Any], ...]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Candidate JSONL.")
    parser.add_argument("--output-root", default="paper/results/code_vra_jsonl")
    parser.add_argument("--table-root", default="paper/tables")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _resolve(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _normalize_tests(raw: object) -> tuple[tuple[tuple[Any, ...], Any], ...]:
    tests = []
    for item in raw or []:
        if isinstance(item, dict):
            args = item.get("args", [])
            expected = item.get("expected")
        else:
            args, expected = item
        if not isinstance(args, (list, tuple)):
            args = [args]
        tests.append((tuple(args), expected))
    return tuple(tests)


def _normalize_candidates(row: dict[str, Any]) -> tuple[Candidate, ...]:
    raw_candidates = row.get("candidate_codes")
    if raw_candidates is None:
        raw_candidates = row.get("candidates")
    if raw_candidates is None:
        raw_candidates = [row.get("candidate_code", "")]
    candidates = []
    for index, candidate in enumerate(raw_candidates):
        if isinstance(candidate, dict):
            name = str(candidate.get("name") or f"candidate_{index}")
            code = str(candidate.get("code") or candidate.get("candidate_code") or "")
        else:
            name = f"candidate_{index}"
            code = str(candidate)
        if code.strip():
            candidates.append(Candidate(name=name, code=code))
    return tuple(candidates)


def _read_tasks(path: Path, limit: int) -> list[Task]:
    tasks = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            task = Task(
                task_id=str(row.get("task_id") or row.get("id") or f"task_{line_number:05d}"),
                prompt=str(row.get("prompt") or ""),
                function_name=str(row.get("function_name") or "solve"),
                teacher_code=str(row.get("teacher_code") or ""),
                candidates=_normalize_candidates(row),
                public_tests=_normalize_tests(row.get("public_tests")),
                hidden_tests=_normalize_tests(row.get("hidden_tests")),
            )
            if not task.teacher_code.strip():
                raise ValueError(f"{path}:{line_number}: teacher_code is required")
            if not task.candidates:
                raise ValueError(f"{path}:{line_number}: at least one candidate_code is required")
            tasks.append(task)
            if limit and len(tasks) >= limit:
                break
    return tasks


def _load_function(code: str, function_name: str) -> Callable[..., Any]:
    namespace: dict[str, Any] = {}
    exec(code, {"__builtins__": SAFE_BUILTINS}, namespace)
    fn = namespace[function_name]
    if not callable(fn):
        raise TypeError(f"{function_name} is not callable")
    return fn


def _score(code: str, task: Task, tests: tuple[tuple[tuple[Any, ...], Any], ...]) -> tuple[int, int]:
    if not tests:
        return 0, 0
    try:
        fn = _load_function(code, task.function_name)
    except Exception:
        return 0, len(tests)
    passed = 0
    for args, expected in tests:
        try:
            actual = fn(*args)
        except Exception:
            continue
        if actual == expected:
            passed += 1
    return passed, len(tests)


def _row(task: Task, policy: str, candidate_name: str, code: str, accepted: bool, teacher_hidden: int) -> dict[str, object]:
    public_passed, public_total = _score(code, task, task.public_tests)
    hidden_passed, hidden_total = _score(code, task, task.hidden_tests)
    return {
        "task_id": task.task_id,
        "policy": policy,
        "candidate": candidate_name,
        "accepted": accepted,
        "public_passed": public_passed,
        "public_total": public_total,
        "hidden_passed": hidden_passed,
        "hidden_total": hidden_total,
        "hidden_delta": hidden_passed - teacher_hidden,
        "patch_chars": max(0, len(code) - len(task.teacher_code)),
    }


def _choose_verified(task: Task) -> Candidate | None:
    for candidate in task.candidates:
        passed, total = _score(candidate.code, task, task.public_tests)
        if total > 0 and passed == total:
            return candidate
    return None


def _choose_oracle(task: Task) -> Candidate | None:
    teacher_hidden, _ = _score(task.teacher_code, task, task.hidden_tests)
    best = max(
        task.candidates,
        key=lambda candidate: _score(candidate.code, task, task.hidden_tests)[0] - teacher_hidden,
    )
    best_hidden, _ = _score(best.code, task, task.hidden_tests)
    return best if best_hidden >= teacher_hidden else None


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for policy in sorted({str(row["policy"]) for row in rows}):
        policy_rows = [row for row in rows if row["policy"] == policy]
        tasks = len(policy_rows)
        public_passed = sum(int(row["public_passed"]) for row in policy_rows)
        public_total = sum(int(row["public_total"]) for row in policy_rows)
        hidden_passed = sum(int(row["hidden_passed"]) for row in policy_rows)
        hidden_total = sum(int(row["hidden_total"]) for row in policy_rows)
        accepted = sum(1 for row in policy_rows if row["accepted"] is True or str(row["accepted"]).lower() == "true")
        regressions = sum(1 for row in policy_rows if int(row["hidden_delta"]) < 0)
        out.append(
            {
                "policy": policy,
                "tasks": tasks,
                "accepted": accepted,
                "acceptance_rate": accepted / tasks if tasks else 0.0,
                "public_pass_rate": public_passed / public_total if public_total else 0.0,
                "hidden_pass_rate": hidden_passed / hidden_total if hidden_total else 0.0,
                "mean_hidden_delta": sum(int(row["hidden_delta"]) for row in policy_rows) / tasks if tasks else 0.0,
                "hidden_regressions": regressions,
                "hidden_regression_rate": regressions / tasks if tasks else 0.0,
                "mean_patch_chars": sum(int(row["patch_chars"]) for row in policy_rows) / tasks if tasks else 0.0,
            }
        )
    order = {
        "teacher_only": 0,
        "random_residual": 1,
        "no_gate_first_patch": 2,
        "verified_patch": 3,
        "oracle_patch": 4,
    }
    return sorted(out, key=lambda row: order.get(str(row["policy"]), 99))


def main() -> None:
    args = get_parser().parse_args()
    rng = random.Random(args.seed)
    tasks = _read_tasks(_resolve(args.input), args.limit)
    raw_rows: list[dict[str, object]] = []
    for task in tasks:
        teacher_hidden, _ = _score(task.teacher_code, task, task.hidden_tests)
        raw_rows.append(_row(task, "teacher_only", "teacher", task.teacher_code, False, teacher_hidden))
        random_candidate = rng.choice(task.candidates)
        raw_rows.append(_row(task, "random_residual", random_candidate.name, random_candidate.code, True, teacher_hidden))
        first = task.candidates[0]
        raw_rows.append(_row(task, "no_gate_first_patch", first.name, first.code, True, teacher_hidden))
        verified = _choose_verified(task)
        if verified is None:
            raw_rows.append(_row(task, "verified_patch", "teacher_fallback", task.teacher_code, False, teacher_hidden))
        else:
            raw_rows.append(_row(task, "verified_patch", verified.name, verified.code, True, teacher_hidden))
        oracle = _choose_oracle(task)
        if oracle is None:
            raw_rows.append(_row(task, "oracle_patch", "teacher_fallback", task.teacher_code, False, teacher_hidden))
        else:
            raw_rows.append(_row(task, "oracle_patch", oracle.name, oracle.code, True, teacher_hidden))

    summary_rows = _summarize(raw_rows)
    output_root = _resolve(args.output_root)
    table_root = _resolve(args.table_root)
    output_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "raw.csv", raw_rows, RAW_FIELDS)
    _write_csv(output_root / "summary.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(table_root / "code_vra_jsonl_results.csv", summary_rows, SUMMARY_FIELDS)
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "tasks": len(tasks),
                "policies": [row["policy"] for row in summary_rows],
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(table_root / "code_vra_jsonl_results.csv"), "tasks": len(tasks)}, indent=2))


if __name__ == "__main__":
    main()
