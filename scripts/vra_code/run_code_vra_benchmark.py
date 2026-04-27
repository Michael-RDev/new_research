#!/usr/bin/env python3
"""Run a small deterministic code-domain VRA benchmark.

The benchmark is deliberately local and dependency-free.  Each task contains a
teacher implementation plus residual patch candidates.  Public tests are the
deployment verifier; hidden tests are the held-out evaluation target.  This gives
the paper a second domain where VRA is not speech-specific.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Candidate:
    name: str
    code: str


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    function_name: str
    teacher: str
    candidates: tuple[Candidate, ...]
    public_tests: tuple[tuple[tuple[Any, ...], Any], ...]
    hidden_tests: tuple[tuple[tuple[Any, ...], Any], ...]


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


TASKS: tuple[Task, ...] = (
    Task(
        task_id="dedupe_preserve_order",
        prompt="Return a list with duplicate values removed while preserving first occurrence order.",
        function_name="solve",
        teacher="def solve(xs):\n    return sorted(set(xs))\n",
        candidates=(
            Candidate("overfit_public_numbers", "def solve(xs):\n    return [1, 2, 3] if xs == [1, 2, 1, 3] else sorted(set(xs))\n"),
            Candidate("order_preserving", "def solve(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out\n"),
            Candidate("identity", "def solve(xs):\n    return xs\n"),
        ),
        public_tests=((((1, 2, 1, 3),), [1, 2, 3]),),
        hidden_tests=((((3, 1, 3, 2),), [3, 1, 2]), ((("a", "b", "a"),), ["a", "b"])),
    ),
    Task(
        task_id="is_prime",
        prompt="Return True iff n is a prime integer.",
        function_name="solve",
        teacher="def solve(n):\n    if n < 1:\n        return False\n    for k in range(2, n):\n        if n % k == 0:\n            return False\n    return True\n",
        candidates=(
            Candidate("fix_one_only", "def solve(n):\n    if n == 1:\n        return False\n    if n < 1:\n        return False\n    for k in range(2, n):\n        if n % k == 0:\n            return False\n    return True\n"),
            Candidate("sqrt_prime", "def solve(n):\n    if n < 2:\n        return False\n    k = 2\n    while k * k <= n:\n        if n % k == 0:\n            return False\n        k += 1\n    return True\n"),
            Candidate("all_odd", "def solve(n):\n    return n == 2 or n % 2 == 1\n"),
        ),
        public_tests=(((1,), False), ((2,), True), ((9,), False)),
        hidden_tests=(((0,), False), ((17,), True), ((21,), False), ((97,), True)),
    ),
    Task(
        task_id="clamp",
        prompt="Clamp x into the inclusive range [lo, hi].",
        function_name="solve",
        teacher="def solve(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n",
        candidates=(
            Candidate("swap_bounds", "def solve(x, lo, hi):\n    if lo > hi:\n        lo, hi = hi, lo\n    return max(lo, min(hi, x))\n"),
            Candidate("always_low", "def solve(x, lo, hi):\n    return lo\n"),
            Candidate("exclusive_bug", "def solve(x, lo, hi):\n    return min(max(x, lo + 1), hi - 1)\n"),
        ),
        public_tests=(((5, 0, 10), 5), ((-3, 0, 10), 0), ((12, 0, 10), 10)),
        hidden_tests=(((10, 0, 10), 10), ((0, 0, 10), 0), ((7, 10, 0), 7)),
    ),
    Task(
        task_id="balanced_parentheses",
        prompt="Return True iff parentheses are balanced; ignore non-parenthesis characters.",
        function_name="solve",
        teacher="def solve(s):\n    return s.count('(') == s.count(')')\n",
        candidates=(
            Candidate("stack", "def solve(s):\n    depth = 0\n    for ch in s:\n        if ch == '(':\n            depth += 1\n        elif ch == ')':\n            depth -= 1\n            if depth < 0:\n                return False\n    return depth == 0\n"),
            Candidate("starts_ends", "def solve(s):\n    return s.startswith('(') and s.endswith(')')\n"),
            Candidate("teacher_copy", "def solve(s):\n    return s.count('(') == s.count(')')\n"),
        ),
        public_tests=((("(())",), True), (("(()",), False), (("())(",), False)),
        hidden_tests=((("a(b)c",), True), ((")(" ,), False), (("(()())",), True)),
    ),
    Task(
        task_id="flatten_once",
        prompt="Flatten one level of lists and tuples.",
        function_name="solve",
        teacher="def solve(xs):\n    out = []\n    for x in xs:\n        if isinstance(x, list):\n            out.extend(x)\n        else:\n            out.append(x)\n    return out\n",
        candidates=(
            Candidate("list_tuple", "def solve(xs):\n    out = []\n    for x in xs:\n        if isinstance(x, (list, tuple)):\n            out.extend(x)\n        else:\n            out.append(x)\n    return out\n"),
            Candidate("recursive", "def solve(xs):\n    out = []\n    for x in xs:\n        if isinstance(x, (list, tuple)):\n            out.extend(solve(x))\n        else:\n            out.append(x)\n    return out\n"),
            Candidate("first_only", "def solve(xs):\n    return list(xs[0]) if xs else []\n"),
        ),
        public_tests=(((([1, 2], 3),), [1, 2, 3]),),
        hidden_tests=(((((1, 2), 3),), [1, 2, 3]), ((([1, [2]], 3),), [1, [2], 3])),
    ),
    Task(
        task_id="median",
        prompt="Return the median; for even length use the arithmetic mean of the two middle values.",
        function_name="solve",
        teacher="def solve(xs):\n    ys = sorted(xs)\n    return ys[len(ys)//2]\n",
        candidates=(
            Candidate("even_average", "def solve(xs):\n    ys = sorted(xs)\n    n = len(ys)\n    mid = n // 2\n    if n % 2:\n        return ys[mid]\n    return (ys[mid - 1] + ys[mid]) / 2\n"),
            Candidate("lower_median", "def solve(xs):\n    ys = sorted(xs)\n    return ys[(len(ys)-1)//2]\n"),
            Candidate("mean", "def solve(xs):\n    return sum(xs) / len(xs)\n"),
        ),
        public_tests=((((3, 1, 2),), 2), (((1, 2, 3, 4),), 2.5)),
        hidden_tests=((((10, 1, 2, 8),), 5.0), (((5,),), 5), (((-1, 1),), 0.0)),
    ),
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="paper/results/code_vra")
    parser.add_argument("--table-root", default="paper/tables")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _resolve(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _load_function(code: str, function_name: str) -> Callable[..., Any]:
    namespace: dict[str, Any] = {}
    exec(code, {"__builtins__": {"abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "enumerate": enumerate, "float": float, "int": int, "isinstance": isinstance, "len": len, "list": list, "max": max, "min": min, "range": range, "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple}}, namespace)
    fn = namespace[function_name]
    if not callable(fn):
        raise TypeError(f"{function_name} is not callable")
    return fn


def _score(code: str, task: Task, tests: tuple[tuple[tuple[Any, ...], Any], ...]) -> tuple[int, int]:
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
        "patch_chars": max(0, len(code) - len(task.teacher)),
    }


def _choose_verified(task: Task) -> Candidate | None:
    for candidate in task.candidates:
        passed, total = _score(candidate.code, task, task.public_tests)
        if passed == total:
            return candidate
    return None


def _choose_oracle(task: Task) -> Candidate | None:
    teacher_hidden, _ = _score(task.teacher, task, task.hidden_tests)
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
    policies = sorted({str(row["policy"]) for row in rows})
    out = []
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        tasks = len(policy_rows)
        public_passed = sum(int(row["public_passed"]) for row in policy_rows)
        public_total = sum(int(row["public_total"]) for row in policy_rows)
        hidden_passed = sum(int(row["hidden_passed"]) for row in policy_rows)
        hidden_total = sum(int(row["hidden_total"]) for row in policy_rows)
        accepted = sum(1 for row in policy_rows if str(row["accepted"]).lower() == "true" or row["accepted"] is True)
        hidden_regressions = sum(1 for row in policy_rows if int(row["hidden_delta"]) < 0)
        out.append(
            {
                "policy": policy,
                "tasks": tasks,
                "accepted": accepted,
                "acceptance_rate": accepted / tasks if tasks else 0.0,
                "public_pass_rate": public_passed / public_total if public_total else 0.0,
                "hidden_pass_rate": hidden_passed / hidden_total if hidden_total else 0.0,
                "mean_hidden_delta": sum(int(row["hidden_delta"]) for row in policy_rows) / tasks if tasks else 0.0,
                "hidden_regressions": hidden_regressions,
                "hidden_regression_rate": hidden_regressions / tasks if tasks else 0.0,
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
    tasks = list(TASKS[: args.limit] if args.limit else TASKS)
    raw_rows: list[dict[str, object]] = []
    for task in tasks:
        teacher_hidden, _ = _score(task.teacher, task, task.hidden_tests)
        raw_rows.append(_row(task, "teacher_only", "teacher", task.teacher, False, teacher_hidden))
        random_candidate = rng.choice(task.candidates)
        raw_rows.append(_row(task, "random_residual", random_candidate.name, random_candidate.code, True, teacher_hidden))
        first = task.candidates[0]
        raw_rows.append(_row(task, "no_gate_first_patch", first.name, first.code, True, teacher_hidden))
        verified = _choose_verified(task)
        if verified is None:
            raw_rows.append(_row(task, "verified_patch", "teacher_fallback", task.teacher, False, teacher_hidden))
        else:
            raw_rows.append(_row(task, "verified_patch", verified.name, verified.code, True, teacher_hidden))
        oracle = _choose_oracle(task)
        if oracle is None:
            raw_rows.append(_row(task, "oracle_patch", "teacher_fallback", task.teacher, False, teacher_hidden))
        else:
            raw_rows.append(_row(task, "oracle_patch", oracle.name, oracle.code, True, teacher_hidden))

    summary_rows = _summarize(raw_rows)
    output_root = _resolve(args.output_root)
    table_root = _resolve(args.table_root)
    output_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "raw.csv", raw_rows, RAW_FIELDS)
    _write_csv(output_root / "summary.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(table_root / "code_vra_results.csv", summary_rows, SUMMARY_FIELDS)
    (output_root / "summary.json").write_text(
        json.dumps({"tasks": len(tasks), "policies": [row["policy"] for row in summary_rows], "summary": summary_rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(table_root / "code_vra_results.csv"), "tasks": len(tasks)}, indent=2))


if __name__ == "__main__":
    main()
