#!/usr/bin/env python3

"""Small helpers for reading .env files without extra dependencies."""

from __future__ import annotations

import os
from pathlib import Path


def parse_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    env_path = Path(path)
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values


def load_env_file(path: str | os.PathLike[str], override: bool = False) -> dict[str, str]:
    values = parse_env_file(path)
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values
