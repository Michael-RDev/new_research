from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from aoede.audio.codec import FrozenAudioCodec
from aoede.audio.io import load_audio_file
from aoede.audio.speaker import FrozenSpeakerEncoder
from aoede.data.alignments import load_alignment, proportional_durations
from aoede.data.manifest import ManifestEntry
from aoede.languages import language_index, normalize_language
from aoede.text.tokenizer import UnicodeTokenizer


@dataclass
class TrainingExample:
    text: str
    language_code: str
    token_ids: torch.Tensor
    waveform: torch.Tensor
    codec_latents: torch.Tensor
    reference_latents: torch.Tensor
    reference_mask: torch.Tensor
    prosody_targets: torch.Tensor
    durations: torch.Tensor
    speaker_ref: torch.Tensor
    has_reference: bool


class ManifestDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[ManifestEntry],
        tokenizer: UnicodeTokenizer,
        codec: FrozenAudioCodec,
        speaker_encoder: Optional[FrozenSpeakerEncoder] = None,
        sample_rate: int = 24000,
        cache_dir: Optional[Path] = None,
        planner_stride: int = 4,
        planner_dim: int = 128,
    ):
        self.entries = list(entries)
        self.tokenizer = tokenizer
        self.codec = codec
        self.speaker_encoder = speaker_encoder or FrozenSpeakerEncoder()
        self.sample_rate = sample_rate
        self.cache_dir = cache_dir
        self.planner_stride = planner_stride
        self.planner_dim = planner_dim
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

    def _cache_key(self, raw_value: str):
        stem = Path(raw_value).stem or "audio"
        digest = hashlib.sha1(raw_value.encode("utf-8")).hexdigest()[:16]
        return f"{stem}-{digest}"

    def _load_audio_features(self, audio_path: str, cache_key: str):
        waveform_np, _ = load_audio_file(audio_path, target_sample_rate=self.sample_rate)
        waveform = torch.from_numpy(waveform_np).float()

        latent_cache = self._latent_cache_path(cache_key)
        if latent_cache and latent_cache.exists():
            latents = torch.load(latent_cache, map_location="cpu")
        else:
            with torch.no_grad():
                latents = self.codec.encode(waveform.unsqueeze(0))[0]
            if latent_cache:
                torch.save(latents, latent_cache)

        speaker_cache = self._speaker_cache_path(cache_key)
        if speaker_cache and speaker_cache.exists():
            speaker_embedding = torch.load(speaker_cache, map_location="cpu")
        else:
            speaker_embedding = torch.from_numpy(
                self.speaker_encoder.encode(waveform_np, sample_rate=self.sample_rate)
            ).float()
            if speaker_cache:
                torch.save(speaker_embedding, speaker_cache)

        return waveform, latents, speaker_embedding

    def _build_prosody_targets(self, latents: torch.Tensor):
        if latents.shape[0] == 0:
            return latents.new_zeros(0, self.planner_dim)
        pad = (-latents.shape[0]) % self.planner_stride
        if pad > 0:
            latents = torch.cat([latents, latents[-1:].expand(pad, -1)], dim=0)
        pooled = latents.view(-1, self.planner_stride, latents.shape[-1]).mean(dim=1)
        energy = (
            latents.abs()
            .view(-1, self.planner_stride, latents.shape[-1])
            .mean(dim=(1, 2), keepdim=False)
            .unsqueeze(-1)
        )
        targets = torch.cat([pooled, energy], dim=-1)
        if targets.shape[-1] > self.planner_dim:
            return targets[:, : self.planner_dim]
        if targets.shape[-1] < self.planner_dim:
            pad_width = self.planner_dim - targets.shape[-1]
            return torch.nn.functional.pad(targets, (0, pad_width))
        return targets

    def __getitem__(self, index: int):
        entry = self.entries[index]
        language_code = normalize_language(entry.language_code)
        waveform, latents, speaker_embedding = self._load_audio_features(
            entry.audio_path,
            entry.item_id,
        )
        token_ids = torch.tensor(
            self.tokenizer.encode(entry.text, language_code), dtype=torch.long
        )

        alignment = load_alignment(
            Path(entry.alignment_path) if entry.alignment_path else None
        )
        durations = torch.tensor(
            alignment
            if alignment
            else proportional_durations(len(token_ids), latents.shape[0]),
            dtype=torch.long,
        )
        if entry.speaker_ref:
            reference_key = self._cache_key(entry.speaker_ref)
            _, reference_latents, reference_speaker = self._load_audio_features(
                entry.speaker_ref,
                reference_key,
            )
            has_reference = True
            speaker_embedding = reference_speaker
        else:
            reference_frames = max(1, latents.shape[0] // 4)
            reference_latents = latents[:reference_frames].clone()
            has_reference = False
        reference_mask = torch.ones(reference_latents.shape[0], dtype=torch.bool)
        prosody_targets = self._build_prosody_targets(latents)

        return TrainingExample(
            text=entry.text,
            language_code=language_code,
            token_ids=token_ids,
            waveform=waveform,
            codec_latents=latents,
            reference_latents=reference_latents,
            reference_mask=reference_mask,
            prosody_targets=prosody_targets,
            durations=durations,
            speaker_ref=speaker_embedding,
            has_reference=has_reference,
        )


def collate_training_examples(examples: Sequence[TrainingExample]):
    token_lengths = [len(example.token_ids) for example in examples]
    latent_lengths = [example.codec_latents.shape[0] for example in examples]
    waveform_lengths = [example.waveform.shape[0] for example in examples]
    reference_lengths = [example.reference_latents.shape[0] for example in examples]
    prosody_lengths = [example.prosody_targets.shape[0] for example in examples]
    token_max = max(token_lengths)
    latent_max = max(latent_lengths)
    waveform_max = max(waveform_lengths)
    reference_max = max(reference_lengths)
    prosody_max = max(prosody_lengths)
    latent_dim = examples[0].codec_latents.shape[-1]
    planner_dim = examples[0].prosody_targets.shape[-1]

    token_ids = torch.zeros(len(examples), token_max, dtype=torch.long)
    durations = torch.ones(len(examples), token_max, dtype=torch.long)
    latents = torch.zeros(len(examples), latent_max, latent_dim, dtype=torch.float32)
    waveforms = torch.zeros(len(examples), waveform_max, dtype=torch.float32)
    reference_latents = torch.zeros(len(examples), reference_max, latent_dim, dtype=torch.float32)
    reference_mask = torch.zeros(len(examples), reference_max, dtype=torch.bool)
    prosody_targets = torch.zeros(len(examples), prosody_max, planner_dim, dtype=torch.float32)
    has_reference = torch.tensor([example.has_reference for example in examples], dtype=torch.bool)
    speaker_refs = torch.stack([example.speaker_ref for example in examples], dim=0)
    language_ids = torch.tensor(
        [language_index(example.language_code) for example in examples],
        dtype=torch.long,
    )

    for batch_index, example in enumerate(examples):
        length = len(example.token_ids)
        latent_length = example.codec_latents.shape[0]
        waveform_length = example.waveform.shape[0]
        reference_length = example.reference_latents.shape[0]
        prosody_length = example.prosody_targets.shape[0]
        token_ids[batch_index, :length] = example.token_ids
        durations[batch_index, :length] = example.durations
        latents[batch_index, :latent_length] = example.codec_latents
        waveforms[batch_index, :waveform_length] = example.waveform
        reference_latents[batch_index, :reference_length] = example.reference_latents
        reference_mask[batch_index, :reference_length] = example.reference_mask
        prosody_targets[batch_index, :prosody_length] = example.prosody_targets

    return {
        "texts": [example.text for example in examples],
        "language_codes": [example.language_code for example in examples],
        "token_ids": token_ids,
        "language_ids": language_ids,
        "durations": durations,
        "codec_latents": latents,
        "reference_latents": reference_latents,
        "reference_mask": reference_mask,
        "prosody_targets": prosody_targets,
        "waveforms": waveforms,
        "speaker_ref": speaker_refs,
        "has_reference": has_reference,
    }
