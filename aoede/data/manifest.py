from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


CORPUS_PLAN = {
    "production": {
        "mls": ["en", "de", "es", "fr", "it", "nl", "pl", "pt"],
        "fleurs": ["en", "es", "hi", "zh"],
        "peoples_speech": ["en"],
    },
    "voice_cloning": {
        "waxalnlp": ["wo"],
        "emilia_dataset": ["de", "en", "fr", "ja", "ko", "zh"],
        "emilia_nv": ["zh"],
        "parler_mls_eng_10k": ["en"],
    },
    "evaluation": {"fleurs": ["en", "es", "hi", "zh"]},
}


@dataclass
class ManifestEntry:
    item_id: str
    audio_path: str
    text: str
    language_code: str
    split: str = "train"
    dataset_name: str = "custom"
    duration_s: float = 0.0
    speaker_ref: Optional[str] = None
    alignment_path: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str):
        return cls(**json.loads(payload))


def load_manifest(path: Path):
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entries.append(ManifestEntry.from_json(line))
    return entries


def save_manifest(entries: Iterable[ManifestEntry], path: Path):
    payload = "\n".join(entry.to_json() for entry in entries)
    path.write_text(payload)


def export_plan(path: Path):
    path.write_text(json.dumps(CORPUS_PLAN, indent=2))
