from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from aoede.audio.codec import FrozenAudioCodec
from aoede.audio.io import load_audio_file
from aoede.audio.speaker import FrozenSpeakerEncoder
from aoede.data.alignments import load_alignment, proportional_durations
from aoede.data.manifest import ManifestEntry
from aoede.languages import production_languages
from aoede.text.tokenizer import UnicodeTokenizer


LANGUAGE_INDEX = {
    spec.code: index for index, spec in enumerate(production_languages(), start=1)
}


@dataclass
class TrainingExample:
    text: str
    language_code: str
    token_ids: torch.Tensor
    waveform: torch.Tensor
    codec_latents: torch.Tensor
    durations: torch.Tensor
    speaker_ref: torch.Tensor


class ManifestDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[ManifestEntry],
        tokenizer: UnicodeTokenizer,
        codec: FrozenAudioCodec,
        speaker_encoder: Optional[FrozenSpeakerEncoder] = None,
        sample_rate: int = 24000,
        cache_dir: Optional[Path] = None,
    ):
        self.entries = list(entries)
        self.tokenizer = tokenizer
        self.codec = codec
        self.speaker_encoder = speaker_encoder or FrozenSpeakerEncoder()
        self.sample_rate = sample_rate
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.entries)

    def _latent_cache_path(self, item_id: str):
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{item_id}.pt"

    def _speaker_cache_path(self, item_id: str):
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{item_id}.speaker.pt"

    def __getitem__(self, index: int):
        entry = self.entries[index]
        waveform_np, _ = load_audio_file(
            entry.audio_path, target_sample_rate=self.sample_rate
        )
        waveform = torch.from_numpy(waveform_np).float()
        token_ids = torch.tensor(
            self.tokenizer.encode(entry.text, entry.language_code), dtype=torch.long
        )

        latent_cache = self._latent_cache_path(entry.item_id)
        if latent_cache and latent_cache.exists():
            latents = torch.load(latent_cache, map_location="cpu")
        else:
            with torch.no_grad():
                latents = self.codec.encode(waveform.unsqueeze(0))[0]
            if latent_cache:
                torch.save(latents, latent_cache)

        speaker_cache = self._speaker_cache_path(entry.item_id)
        if speaker_cache and speaker_cache.exists():
            speaker_embedding = torch.load(speaker_cache, map_location="cpu")
        else:
            speaker_embedding = torch.from_numpy(
                self.speaker_encoder.encode(waveform_np, sample_rate=self.sample_rate)
            ).float()
            if speaker_cache:
                torch.save(speaker_embedding, speaker_cache)

        alignment = load_alignment(
            Path(entry.alignment_path) if entry.alignment_path else None
        )
        durations = torch.tensor(
            alignment
            if alignment
            else proportional_durations(len(token_ids), latents.shape[0]),
            dtype=torch.long,
        )

        return TrainingExample(
            text=entry.text,
            language_code=entry.language_code,
            token_ids=token_ids,
            waveform=waveform,
            codec_latents=latents,
            durations=durations,
            speaker_ref=speaker_embedding,
        )


def collate_training_examples(examples: Sequence[TrainingExample]):
    token_lengths = [len(example.token_ids) for example in examples]
    latent_lengths = [example.codec_latents.shape[0] for example in examples]
    waveform_lengths = [example.waveform.shape[0] for example in examples]
    token_max = max(token_lengths)
    latent_max = max(latent_lengths)
    waveform_max = max(waveform_lengths)
    latent_dim = examples[0].codec_latents.shape[-1]

    token_ids = torch.zeros(len(examples), token_max, dtype=torch.long)
    durations = torch.ones(len(examples), token_max, dtype=torch.long)
    latents = torch.zeros(len(examples), latent_max, latent_dim, dtype=torch.float32)
    waveforms = torch.zeros(len(examples), waveform_max, dtype=torch.float32)
    speaker_refs = torch.stack([example.speaker_ref for example in examples], dim=0)
    language_ids = torch.tensor(
        [LANGUAGE_INDEX.get(example.language_code, 0) for example in examples],
        dtype=torch.long,
    )

    for batch_index, example in enumerate(examples):
        length = len(example.token_ids)
        latent_length = example.codec_latents.shape[0]
        waveform_length = example.waveform.shape[0]
        token_ids[batch_index, :length] = example.token_ids
        durations[batch_index, :length] = example.durations
        latents[batch_index, :latent_length] = example.codec_latents
        waveforms[batch_index, :waveform_length] = example.waveform

    return {
        "texts": [example.text for example in examples],
        "language_codes": [example.language_code for example in examples],
        "token_ids": token_ids,
        "language_ids": language_ids,
        "durations": durations,
        "codec_latents": latents,
        "waveforms": waveforms,
        "speaker_ref": speaker_refs,
    }
