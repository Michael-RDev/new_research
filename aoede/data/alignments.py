from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence


def proportional_durations(token_count: int, total_frames: int):
    token_count = max(token_count, 1)
    total_frames = max(total_frames, token_count)
    base = total_frames // token_count
    remainder = total_frames % token_count
    durations = [base + (1 if index < remainder else 0) for index in range(token_count)]
    durations = [max(1, item) for item in durations]
    correction = total_frames - sum(durations)
    durations[-1] += correction
    return durations


def save_alignment(path: Path, durations: Sequence[int], method: str = "proportional"):
    path.write_text(json.dumps({"method": method, "durations": list(map(int, durations))}, indent=2))


def load_alignment(path: Optional[Path]):
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text())
    return [int(value) for value in payload.get("durations", [])]
