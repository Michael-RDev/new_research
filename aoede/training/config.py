from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from aoede.config import AppConfig, default_config


@dataclass
class ExperimentConfig:
    app: AppConfig = field(default_factory=default_config)
    manifest_path: Path = Path("artifacts/manifests/train.jsonl")
    eval_manifest_path: Path = Path("artifacts/manifests/eval.jsonl")

    def ensure(self):
        self.app.ensure_directories()
