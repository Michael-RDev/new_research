from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset

from aoede.audio.latent_stats import LatentStats, align_latent_pair, pad_latent_sequences
from aoede.languages import language_index, normalize_language
from aoede.text.tokenizer import UnicodeTokenizer


@dataclass
class SotaDistillEntry:
    item_id: str
    text: str
    language_code: str
    audio_path: str
    speaker_ref: Optional[str]
    real_latents_path: str
    teacher_latents_path: str
    reference_latents_path: str
    speaker_embedding_path: str
    teacher_audio_path: Optional[str] = None
    provider: str = "voxcpm2"

    @classmethod
    def from_json(cls, line: str) -> "SotaDistillEntry":
        return cls(**json.loads(line))

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def save_sota_manifest(entries: Sequence[SotaDistillEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.to_json() + "\n")


def load_sota_manifest(path: Path) -> list[SotaDistillEntry]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            SotaDistillEntry.from_json(line)
            for line in handle
            if line.strip()
        ]


class SotaDistillDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[SotaDistillEntry],
        tokenizer: UnicodeTokenizer,
        latent_stats: LatentStats,
    ):
        self.entries = list(entries)
        self.tokenizer = tokenizer
        self.latent_stats = latent_stats

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        language_code = normalize_language(entry.language_code)
        real = torch.load(entry.real_latents_path, map_location="cpu").float()
        teacher = torch.load(entry.teacher_latents_path, map_location="cpu").float()
        real, teacher = align_latent_pair(real, teacher)
        reference = torch.load(entry.reference_latents_path, map_location="cpu").float()
        speaker = torch.load(entry.speaker_embedding_path, map_location="cpu").float()
        return {
            "text": entry.text,
            "language_code": language_code,
            "token_ids": torch.tensor(
                self.tokenizer.encode(entry.text, language_code),
                dtype=torch.long,
            ),
            "language_id": torch.tensor(language_index(language_code), dtype=torch.long),
            "target_latents": self.latent_stats.normalize(real),
            "teacher_latents": self.latent_stats.normalize(teacher),
            "reference_latents": self.latent_stats.normalize(reference),
            "speaker_embedding": speaker,
            "provider": entry.provider,
        }


def collate_sota_distill(examples: Sequence[dict]) -> dict:
    token_max = max(example["token_ids"].shape[0] for example in examples)
    token_ids = torch.zeros(len(examples), token_max, dtype=torch.long)
    for index, example in enumerate(examples):
        token_ids[index, : example["token_ids"].shape[0]] = example["token_ids"]

    target_latents, target_mask = pad_latent_sequences(
        [example["target_latents"] for example in examples]
    )
    teacher_latents, _ = pad_latent_sequences(
        [example["teacher_latents"] for example in examples]
    )
    reference_latents, reference_mask = pad_latent_sequences(
        [example["reference_latents"] for example in examples]
    )
    return {
        "texts": [example["text"] for example in examples],
        "language_codes": [example["language_code"] for example in examples],
        "providers": [example["provider"] for example in examples],
        "token_ids": token_ids,
        "language_ids": torch.stack([example["language_id"] for example in examples]),
        "target_latents": target_latents,
        "teacher_latents": teacher_latents,
        "target_mask": target_mask,
        "reference_latents": reference_latents,
        "reference_mask": reference_mask,
        "speaker_embedding": torch.stack(
            [example["speaker_embedding"] for example in examples],
            dim=0,
        ),
    }
