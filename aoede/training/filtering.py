from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Sequence

from aoede.audio.io import probe_audio_file
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
    dropped_audio_unreadable: int = 0
    cleared_speaker_ref_unreadable: int = 0

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
    validate_audio_paths: bool = False,
) -> tuple[list[ManifestEntry], TrainingFilterStats]:
    max_audio_duration_s = max_supported_duration_s(
        max_latent_frames=max_latent_frames,
        codec_hop_length=codec_hop_length,
        sample_rate=sample_rate,
    )
    filtered: list[ManifestEntry] = []
    dropped_text_too_long = 0
    dropped_audio_too_long = 0
    dropped_audio_unreadable = 0
    cleared_speaker_ref_unreadable = 0
    max_token_length_kept = 0
    max_duration_s_kept = 0.0
    audio_readability_cache: dict[str, bool] = {}

    for entry in entries:
        token_length = len(tokenizer.encode(entry.text, entry.language_code))
        if token_length > max_text_tokens:
            dropped_text_too_long += 1
            continue

        duration_s = float(entry.duration_s or 0.0)
        if duration_s > 0 and duration_s > max_audio_duration_s:
            dropped_audio_too_long += 1
            continue

        candidate = entry
        if validate_audio_paths:
            audio_ok = audio_readability_cache.get(entry.audio_path)
            if audio_ok is None:
                try:
                    probe_audio_file(entry.audio_path)
                    audio_ok = True
                except Exception:
                    audio_ok = False
                audio_readability_cache[entry.audio_path] = audio_ok
            if not audio_ok:
                dropped_audio_unreadable += 1
                continue

            if entry.speaker_ref:
                speaker_ok = audio_readability_cache.get(entry.speaker_ref)
                if speaker_ok is None:
                    try:
                        probe_audio_file(entry.speaker_ref)
                        speaker_ok = True
                    except Exception:
                        speaker_ok = False
                    audio_readability_cache[entry.speaker_ref] = speaker_ok
                if not speaker_ok:
                    candidate = replace(entry, speaker_ref=None)
                    cleared_speaker_ref_unreadable += 1

        filtered.append(candidate)
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
        dropped_audio_unreadable=dropped_audio_unreadable,
        cleared_speaker_ref_unreadable=cleared_speaker_ref_unreadable,
    )
    return filtered, stats
