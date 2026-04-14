from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from aoede.data.manifest import ManifestEntry
from aoede.text.tokenizer import UnicodeTokenizer


@dataclass(frozen=True)
class TrainingFilterStats:
    source_entries: int
    kept_entries: int
    dropped_text_too_long: int
    dropped_audio_too_long: int
    max_text_tokens_allowed: int
    max_audio_duration_s_allowed: float
    max_token_length_kept: int
    max_duration_s_kept: float

    def to_dict(self):
        return asdict(self)


def max_supported_duration_s(
    max_latent_frames: int,
    codec_hop_length: int,
    sample_rate: int,
) -> float:
    return float(max_latent_frames * codec_hop_length) / float(sample_rate)


def filter_trainable_entries(
    entries: Sequence[ManifestEntry],
    tokenizer: UnicodeTokenizer,
    *,
    max_text_tokens: int,
    max_latent_frames: int,
    codec_hop_length: int,
    sample_rate: int,
) -> tuple[list[ManifestEntry], TrainingFilterStats]:
    max_audio_duration_s = max_supported_duration_s(
        max_latent_frames=max_latent_frames,
        codec_hop_length=codec_hop_length,
        sample_rate=sample_rate,
    )
    filtered: list[ManifestEntry] = []
    dropped_text_too_long = 0
    dropped_audio_too_long = 0
    max_token_length_kept = 0
    max_duration_s_kept = 0.0

    for entry in entries:
        token_length = len(tokenizer.encode(entry.text, entry.language_code))
        if token_length > max_text_tokens:
            dropped_text_too_long += 1
            continue

        duration_s = float(entry.duration_s or 0.0)
        if duration_s > 0 and duration_s > max_audio_duration_s:
            dropped_audio_too_long += 1
            continue

        filtered.append(entry)
        max_token_length_kept = max(max_token_length_kept, token_length)
        max_duration_s_kept = max(max_duration_s_kept, duration_s)

    stats = TrainingFilterStats(
        source_entries=len(entries),
        kept_entries=len(filtered),
        dropped_text_too_long=dropped_text_too_long,
        dropped_audio_too_long=dropped_audio_too_long,
        max_text_tokens_allowed=max_text_tokens,
        max_audio_duration_s_allowed=max_audio_duration_s,
        max_token_length_kept=max_token_length_kept,
        max_duration_s_kept=max_duration_s_kept,
    )
    return filtered, stats
