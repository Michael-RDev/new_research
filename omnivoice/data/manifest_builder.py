#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Build OmniVoice-ready JSONL manifests from streaming Hugging Face corpora.

The primary goal of this module is to make the MnemosVoice research pipeline
repeatable across local machines and Runpod pods:

1. enumerate a staged set of corpora,
2. materialize paired audio/text samples to local files,
3. write OmniVoice JSONL manifests,
4. emit sidecar summaries that later scripts can convert into token shards and
   multilingual ``data_config.json`` files.

Stage 1 is fully automated around the Hugging Face datasets used in the
research plan:

- ``amphion/Emilia-Dataset``
- ``MLCommons/peoples_speech``
- ``google/WaxalNLP``
- ``facebook/multilingual_librispeech``

Stage 2 includes Common Voice plus placeholders for local corpora such as
LibriTTS-R, VCTK, MUSAN, and RIRS_NOISES so the later expansion run can share
the same data-config format.
"""

from __future__ import annotations

import io
import json
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import soundfile as sf

from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID

TRANSCRIPT_KEYS = (
    "text",
    "transcription",
    "transcript",
    "normalized_text",
    "sentence",
    "utterance",
)
LANGUAGE_KEYS = ("language", "lang", "language_id", "locale")
SPEAKER_KEYS = ("speaker_id", "speaker", "client_id", "voice_id")
ID_KEYS = ("id", "utt_id", "audio_id", "sample_id", "key")
AUDIO_BYTES_KEYS = ("flac", "wav", "mp3", "ogg", "opus", "m4a")


@dataclass(frozen=True)
class CorpusRecipe:
    name: str
    stage: str
    kind: str
    dataset_id: Optional[str] = None
    config_names: tuple[str, ...] = ()
    split_name: str = "train"
    role: str = "train"
    output_stem: str = ""
    language_id: Optional[str] = None
    requires_hf_token: bool = False
    min_duration_s: float = 1.5
    max_duration_s: float = 20.0
    max_hours_total: Optional[float] = None
    max_hours_per_language: Optional[float] = None
    max_records: Optional[int] = None
    hash_dev_ratio: float = 0.0
    min_dev_examples_per_language: int = 0
    repeat_hint: int = 1
    notes: str = ""


@dataclass
class ManifestRecord:
    id: str
    audio_path: str
    text: str
    language_id: str
    dataset_name: str
    source_id: str
    speaker_id: Optional[str]
    duration_s: float
    config_name: Optional[str] = None


@dataclass
class ManifestSummary:
    dataset_name: str
    split_role: str
    language_id: str
    output_stem: str
    manifest_path: str
    num_samples: int = 0
    total_duration_s: float = 0.0
    repeat_hint: int = 1
    source_configs: list[str] = field(default_factory=list)
    notes: str = ""


STAGE1_EMILIA_LANGUAGES = ("de", "en", "fr", "ja", "ko", "zh")
STAGE1_MLS_LANGUAGES = {
    "german": "de",
    "dutch": "nl",
    "spanish": "es",
    "french": "fr",
    "italian": "it",
    "polish": "pl",
    "portuguese": "pt",
}

WAXAL_LANGUAGE_ID_MAP = {
    "ach": None,
    "aka": None,
    "amh": "am",
    "dag": "dag",
    "dga": None,
    "ewe": None,
    "fat": None,
    "ful": "ff",
    "hau": "ha",
    "ibo": "ig",
    "kik": None,
    "kpo": None,
    "lin": "ln",
    "lug": None,
    "luo": "luo",
    "mas": None,
    "mlg": None,
    "nyn": None,
    "orm": "om",
    "sid": None,
    "sna": "sn",
    "sog": None,
    "swa": "sw",
    "tir": "ti",
    "twi": "tw",
    "wal": None,
    "yor": "yo",
}

WAXAL_GROUPED_SUBSETS = {
    "am": ("amh_asr",),
    "dag": ("dag_asr",),
    "ff": ("ful_asr", "ful_tts"),
    "ha": ("hau_tts",),
    "ig": ("ibo_tts",),
    "ln": ("lin_asr",),
    "luo": ("luo_tts",),
    "om": ("orm_asr",),
    "sn": ("sna_asr",),
    "sw": ("swa_tts",),
    "ti": ("tir_asr",),
    "tw": ("twi_tts",),
    "yo": ("yor_tts",),
}

STAGE2_COMMON_VOICE_LANGUAGES = {
    "en": "en",
    "de": "de",
    "es": "es",
    "fr": "fr",
    "it": "it",
    "pt": "pt",
    "ar": "ar",
    "hi": "hi",
    "id": "id",
    "ru": "ru",
    "sw": "sw",
    "vi": "vi",
    "zh-CN": "zh",
}

STAGE2_LOCAL_CORPORA = {
    "libritts_r": "Provide local JSONL manifests extracted from LibriTTS-R.",
    "vctk": "Provide local JSONL manifests extracted from VCTK.",
    "musan": "Provide local JSONL manifests for noise-only prompt augmentation.",
    "rirs_noises": "Provide local JSONL manifests for room impulse responses.",
}


def stage_recipes(stage: str, profile: str) -> list[CorpusRecipe]:
    stage = stage.lower()
    profile = profile.lower()
    if stage not in {"stage1", "stage2", "all"}:
        raise ValueError(f"Unsupported stage: {stage}")
    if profile not in {"smoke", "core"}:
        raise ValueError(f"Unsupported profile: {profile}")

    recipes: list[CorpusRecipe] = []
    if stage in {"stage1", "all"}:
        recipes.extend(_stage1_recipes(profile))
    if stage in {"stage2", "all"}:
        recipes.extend(_stage2_recipes(profile))
    return recipes


def _stage1_recipes(profile: str) -> list[CorpusRecipe]:
    smoke = profile == "smoke"
    waxal_records = 32 if smoke else None
    peoples_records = 48 if smoke else None
    emilia_records = 64 if smoke else None
    mls_records = 32 if smoke else None

    recipes = [
        CorpusRecipe(
            name="emilia",
            stage="stage1",
            kind="hf_emilia",
            dataset_id="amphion/Emilia-Dataset",
            split_name="train",
            role="mixed",
            output_stem="emilia",
            requires_hf_token=True,
            max_hours_total=12.0 if smoke else 600.0,
            max_hours_per_language=2.0 if smoke else 100.0,
            max_records=emilia_records,
            hash_dev_ratio=0.01,
            min_dev_examples_per_language=4 if smoke else 32,
            notes="Stage-1 core multilingual Emilia stream.",
        ),
        CorpusRecipe(
            name="peoples_speech",
            stage="stage1",
            kind="hf_audio",
            dataset_id="MLCommons/peoples_speech",
            config_names=("clean",),
            split_name="train",
            role="mixed",
            output_stem="peoples_speech_clean",
            language_id="en",
            max_hours_total=2.0 if smoke else 150.0,
            max_records=peoples_records,
            hash_dev_ratio=0.01,
            min_dev_examples_per_language=8 if smoke else 64,
            notes="People's Speech clean split only.",
        ),
        CorpusRecipe(
            name="peoples_speech",
            stage="stage1",
            kind="hf_audio",
            dataset_id="MLCommons/peoples_speech",
            config_names=("clean_sa",),
            split_name="train",
            role="mixed",
            output_stem="peoples_speech_clean_sa",
            language_id="en",
            max_hours_total=2.0 if smoke else 150.0,
            max_records=peoples_records,
            hash_dev_ratio=0.01,
            min_dev_examples_per_language=8 if smoke else 64,
            notes="People's Speech CC-BY-SA clean split only.",
        ),
    ]

    for config_name, language_id in STAGE1_MLS_LANGUAGES.items():
        recipes.append(
            CorpusRecipe(
                name="mls",
                stage="stage1",
                kind="hf_audio",
                dataset_id="facebook/multilingual_librispeech",
                config_names=(config_name,),
                split_name="train",
                role="train",
                output_stem=f"mls_{language_id}_train",
                language_id=language_id,
                max_hours_total=1.0 if smoke else 50.0,
                max_records=mls_records,
                repeat_hint=2,
                notes=f"MLS {config_name} train split.",
            )
        )
        recipes.append(
            CorpusRecipe(
                name="mls",
                stage="stage1",
                kind="hf_audio",
                dataset_id="facebook/multilingual_librispeech",
                config_names=(config_name,),
                split_name="dev",
                role="dev",
                output_stem=f"mls_{language_id}_dev",
                language_id=language_id,
                max_records=8 if smoke else None,
                repeat_hint=1,
                notes=f"MLS {config_name} dev split.",
            )
        )

    for language_id, subsets in WAXAL_GROUPED_SUBSETS.items():
        recipes.append(
            CorpusRecipe(
                name="waxal",
                stage="stage1",
                kind="hf_audio",
                dataset_id="google/WaxalNLP",
                config_names=subsets,
                split_name="train",
                role="train",
                output_stem=f"waxal_{language_id}_train",
                language_id=language_id,
                max_hours_total=0.5 if smoke else 25.0,
                max_records=waxal_records,
                repeat_hint=4,
                notes=f"Waxal supervised train subsets for {language_id}.",
            )
        )
        recipes.append(
            CorpusRecipe(
                name="waxal",
                stage="stage1",
                kind="hf_audio",
                dataset_id="google/WaxalNLP",
                config_names=subsets,
                split_name="validation",
                role="dev",
                output_stem=f"waxal_{language_id}_dev",
                language_id=language_id,
                max_records=8 if smoke else None,
                repeat_hint=1,
                notes=f"Waxal supervised validation subsets for {language_id}.",
            )
        )

    return recipes


def _stage2_recipes(profile: str) -> list[CorpusRecipe]:
    smoke = profile == "smoke"
    recipes: list[CorpusRecipe] = []
    for locale, language_id in STAGE2_COMMON_VOICE_LANGUAGES.items():
        recipes.append(
            CorpusRecipe(
                name="common_voice",
                stage="stage2",
                kind="hf_audio",
                dataset_id="mozilla-foundation/common_voice_17_0",
                config_names=(locale,),
                split_name="train",
                role="train",
                output_stem=f"common_voice_{language_id}_train",
                language_id=language_id,
                max_hours_total=0.5 if smoke else 25.0,
                max_records=16 if smoke else None,
                repeat_hint=1,
                notes=f"Common Voice 17.0 train split for {locale}.",
            )
        )
        recipes.append(
            CorpusRecipe(
                name="common_voice",
                stage="stage2",
                kind="hf_audio",
                dataset_id="mozilla-foundation/common_voice_17_0",
                config_names=(locale,),
                split_name="validation",
                role="dev",
                output_stem=f"common_voice_{language_id}_dev",
                language_id=language_id,
                max_records=4 if smoke else None,
                repeat_hint=1,
                notes=f"Common Voice 17.0 validation split for {locale}.",
            )
        )

    for corpus_name, notes in STAGE2_LOCAL_CORPORA.items():
        recipes.append(
            CorpusRecipe(
                name=corpus_name,
                stage="stage2",
                kind="placeholder_local",
                output_stem=corpus_name,
                notes=notes,
            )
        )

    return recipes


def list_supported_waxal_codes() -> dict[str, Optional[str]]:
    return dict(WAXAL_LANGUAGE_ID_MAP)


def _normalise_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def _safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._")
    return value or "sample"


def _stable_bucket(value: str, modulo: int = 10_000) -> int:
    digest = sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def _candidate(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _coerce_json_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    raw = sample.get("json")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        return json.loads(raw)
    return {}


def _resolve_language_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    lowered = candidate.lower()
    if lowered in LANG_IDS:
        return lowered
    if candidate in LANG_IDS:
        return candidate
    if lowered in LANG_NAME_TO_ID:
        return LANG_NAME_TO_ID[lowered]
    if lowered in WAXAL_LANGUAGE_ID_MAP and WAXAL_LANGUAGE_ID_MAP[lowered]:
        return WAXAL_LANGUAGE_ID_MAP[lowered]
    if "-" in lowered:
        prefix = lowered.split("-", 1)[0]
        if prefix in LANG_IDS:
            return prefix
    return None


def _extract_duration_hint_seconds(sample: dict[str, Any], meta: dict[str, Any]) -> Optional[float]:
    for mapping in (sample, meta):
        duration_ms = mapping.get("duration_ms")
        if duration_ms not in (None, ""):
            return float(duration_ms) / 1000.0
        duration_s = mapping.get("duration_s") or mapping.get("audio_duration")
        if duration_s not in (None, ""):
            return float(duration_s)
    audio = sample.get("audio")
    if isinstance(audio, dict) and audio.get("array") is not None:
        sampling_rate = audio.get("sampling_rate") or 24_000
        return float(len(audio["array"])) / float(sampling_rate)
    return None


def _extract_text(sample: dict[str, Any], meta: dict[str, Any]) -> str:
    text = _candidate(sample, TRANSCRIPT_KEYS)
    if text is None:
        text = _candidate(meta, TRANSCRIPT_KEYS)
    return _normalise_text(str(text or ""))


def _extract_speaker_id(sample: dict[str, Any], meta: dict[str, Any]) -> Optional[str]:
    speaker = _candidate(sample, SPEAKER_KEYS)
    if speaker is None:
        speaker = _candidate(meta, SPEAKER_KEYS)
    return str(speaker) if speaker not in (None, "") else None


def _extract_source_id(sample: dict[str, Any], meta: dict[str, Any], fallback_prefix: str) -> str:
    source_id = _candidate(sample, ID_KEYS)
    if source_id is None:
        source_id = _candidate(meta, ID_KEYS)
    if source_id is None:
        source_id = sample.get("__key__")
    if source_id is None:
        source_id = f"{fallback_prefix}-{_stable_bucket(json.dumps(meta, sort_keys=True)[:256])}"
    return str(source_id)


def _materialise_audio(sample: dict[str, Any], destination_base: Path) -> Path:
    audio = sample.get("audio")
    if isinstance(audio, dict):
        path = audio.get("path")
        if path:
            src = Path(path)
            if src.exists():
                out_path = destination_base.with_suffix(src.suffix or ".wav")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if src.resolve() != out_path.resolve():
                    shutil.copy2(src, out_path)
                return out_path

        audio_bytes = audio.get("bytes")
        if audio_bytes:
            ext = Path(path).suffix if path else ".wav"
            out_path = destination_base.with_suffix(ext or ".wav")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(audio_bytes)
            return out_path

        audio_array = audio.get("array")
        if audio_array is not None:
            out_path = destination_base.with_suffix(".wav")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sampling_rate = audio.get("sampling_rate") or 24_000
            sf.write(
                out_path,
                np.asarray(audio_array, dtype=np.float32),
                int(sampling_rate),
            )
            return out_path

    for ext in AUDIO_BYTES_KEYS:
        if ext in sample and sample[ext]:
            out_path = destination_base.with_suffix(f".{ext}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = sample[ext]
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            out_path.write_bytes(payload)
            return out_path

    raise ValueError("sample did not contain usable audio content")


def _audio_duration_seconds(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


class ManifestWriter:
    def __init__(self, manifest_root: Path):
        self.manifest_root = manifest_root
        self.manifest_root.mkdir(parents=True, exist_ok=True)
        self._handles: dict[Path, Any] = {}
        self._stats: dict[Path, ManifestSummary] = {}

    def write(self, output_stem: str, record: ManifestRecord, repeat_hint: int, source_configs: Iterable[str], notes: str, role: str):
        manifest_path = self.manifest_root / f"{output_stem}.jsonl"
        if manifest_path not in self._handles:
            self._handles[manifest_path] = open(manifest_path, "a", encoding="utf-8")
            self._stats[manifest_path] = ManifestSummary(
                dataset_name=record.dataset_name,
                split_role=role,
                language_id=record.language_id,
                output_stem=output_stem,
                manifest_path=str(manifest_path),
                repeat_hint=repeat_hint,
                source_configs=sorted(set(source_configs)),
                notes=notes,
            )

        handle = self._handles[manifest_path]
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        stats = self._stats[manifest_path]
        stats.num_samples += 1
        stats.total_duration_s += record.duration_s

    def close(self):
        for handle in self._handles.values():
            handle.close()
        for manifest_path, summary in self._stats.items():
            summary_path = manifest_path.with_suffix(".summary.json")
            summary_path.write_text(
                json.dumps(asdict(summary), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def build_hf_manifests(
    recipes: Iterable[CorpusRecipe],
    manifest_root: str,
    audio_root: str,
    hf_token: Optional[str] = None,
    skip_existing: bool = False,
) -> list[Path]:
    writer = ManifestWriter(Path(manifest_root))
    audio_root_path = Path(audio_root)
    built_paths: list[Path] = []

    try:
        for recipe in recipes:
            if recipe.kind == "placeholder_local":
                built_paths.extend(_write_local_placeholder(recipe, Path(manifest_root)))
                continue
            if recipe.kind == "hf_emilia":
                _build_emilia_recipe(
                    recipe,
                    writer=writer,
                    audio_root=audio_root_path,
                    hf_token=hf_token,
                    skip_existing=skip_existing,
                )
            elif recipe.kind == "hf_audio":
                _build_fixed_hf_recipe(
                    recipe,
                    writer=writer,
                    audio_root=audio_root_path,
                    hf_token=hf_token,
                    skip_existing=skip_existing,
                )
            else:
                raise ValueError(f"Unsupported recipe kind: {recipe.kind}")
    finally:
        writer.close()

    for summary_path in Path(manifest_root).glob("*.summary.json"):
        built_paths.append(summary_path.with_suffix("").with_suffix(".jsonl"))
    return sorted(set(built_paths))


def _iter_hf_rows(
    dataset_id: str,
    config_name: Optional[str],
    split_name: str,
    hf_token: Optional[str],
    streaming: bool = True,
):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"streaming": streaming}
    if hf_token:
        kwargs["token"] = hf_token

    try:
        ds = load_dataset(dataset_id, config_name, split=split_name, **kwargs)
    except Exception:
        ds = load_dataset(dataset_id, config_name, **kwargs)
        if isinstance(ds, dict):
            if split_name in ds:
                ds = ds[split_name]
            else:
                first_split = next(iter(ds))
                ds = ds[first_split]
    return ds


def _build_fixed_hf_recipe(
    recipe: CorpusRecipe,
    writer: ManifestWriter,
    audio_root: Path,
    hf_token: Optional[str],
    skip_existing: bool,
):
    seen_source_ids: set[str] = set()
    total_hours = 0.0
    record_count = 0
    train_records = 0
    dev_records = 0

    for config_name in recipe.config_names or (None,):
        dataset = _iter_hf_rows(
            dataset_id=recipe.dataset_id,
            config_name=config_name,
            split_name=recipe.split_name,
            hf_token=hf_token if recipe.requires_hf_token else hf_token,
        )
        for sample in dataset:
            meta = _coerce_json_metadata(sample)
            text = _extract_text(sample, meta)
            if not text:
                continue

            language_id = _resolve_language_id(
                recipe.language_id or _candidate(sample, LANGUAGE_KEYS) or _candidate(meta, LANGUAGE_KEYS)
            )
            if not language_id:
                continue

            source_id = _extract_source_id(
                sample,
                meta,
                fallback_prefix=recipe.output_stem or recipe.name,
            )
            dedupe_key = f"{recipe.name}:{source_id}"
            if dedupe_key in seen_source_ids:
                continue

            split_role = recipe.role
            if recipe.role == "mixed":
                if dev_records < recipe.min_dev_examples_per_language:
                    split_role = "dev"
                elif _stable_bucket(source_id) < int(recipe.hash_dev_ratio * 10_000):
                    split_role = "dev"
                else:
                    split_role = "train"

            output_stem = recipe.output_stem
            if recipe.role == "mixed":
                output_stem = f"{recipe.output_stem}_{split_role}"

            relative_dir = audio_root / output_stem
            base_name = _safe_name(source_id)
            try:
                audio_path = _materialise_audio(
                    sample,
                    destination_base=relative_dir / base_name,
                )
            except Exception:
                continue

            duration_s = _extract_duration_hint_seconds(sample, meta) or _audio_duration_seconds(audio_path)
            if duration_s < recipe.min_duration_s or duration_s > recipe.max_duration_s:
                audio_path.unlink(missing_ok=True)
                continue

            if recipe.max_hours_total is not None and split_role == "train":
                if total_hours + (duration_s / 3600.0) > recipe.max_hours_total:
                    audio_path.unlink(missing_ok=True)
                    break

            if recipe.max_records is not None:
                if split_role == "train" and train_records >= recipe.max_records:
                    audio_path.unlink(missing_ok=True)
                    break
                if split_role == "dev" and dev_records >= max(4, recipe.max_records // 4):
                    audio_path.unlink(missing_ok=True)
                    continue

            if skip_existing and (writer.manifest_root / f"{output_stem}.jsonl").exists():
                audio_path.unlink(missing_ok=True)
                continue

            record = ManifestRecord(
                id=f"{output_stem}_{base_name}",
                audio_path=str(audio_path),
                text=text,
                language_id=language_id,
                dataset_name=recipe.name,
                source_id=source_id,
                speaker_id=_extract_speaker_id(sample, meta),
                duration_s=duration_s,
                config_name=config_name,
            )
            writer.write(
                output_stem=output_stem,
                record=record,
                repeat_hint=recipe.repeat_hint,
                source_configs=recipe.config_names,
                notes=recipe.notes,
                role=split_role,
            )
            seen_source_ids.add(dedupe_key)
            if split_role == "train":
                total_hours += duration_s / 3600.0
                train_records += 1
            else:
                dev_records += 1
            record_count += 1


def _build_emilia_recipe(
    recipe: CorpusRecipe,
    writer: ManifestWriter,
    audio_root: Path,
    hf_token: Optional[str],
    skip_existing: bool,
):
    dataset = _iter_hf_rows(
        dataset_id=recipe.dataset_id,
        config_name=None,
        split_name=recipe.split_name,
        hf_token=hf_token,
    )

    train_hours_total = 0.0
    train_hours_by_lang: dict[str, float] = defaultdict(float)
    train_records_by_lang: dict[str, int] = defaultdict(int)
    dev_records_by_lang: dict[str, int] = defaultdict(int)
    seen_source_ids: set[str] = set()

    for sample in dataset:
        meta = _coerce_json_metadata(sample)
        text = _extract_text(sample, meta)
        if not text:
            continue

        language_id = _resolve_language_id(
            _candidate(sample, LANGUAGE_KEYS) or _candidate(meta, LANGUAGE_KEYS)
        )
        if language_id not in STAGE1_EMILIA_LANGUAGES:
            continue

        source_id = _extract_source_id(sample, meta, fallback_prefix="emilia")
        dedupe_key = f"emilia:{source_id}"
        if dedupe_key in seen_source_ids:
            continue

        role = "train"
        if dev_records_by_lang[language_id] < recipe.min_dev_examples_per_language:
            role = "dev"
        elif _stable_bucket(source_id) < int(recipe.hash_dev_ratio * 10_000):
            role = "dev"

        relative_dir = audio_root / f"emilia_{language_id}_{role}"
        base_name = _safe_name(source_id)
        try:
            audio_path = _materialise_audio(sample, destination_base=relative_dir / base_name)
        except Exception:
            continue

        duration_s = _extract_duration_hint_seconds(sample, meta) or _audio_duration_seconds(audio_path)
        if duration_s < recipe.min_duration_s or duration_s > recipe.max_duration_s:
            audio_path.unlink(missing_ok=True)
            continue

        if role == "train":
            if recipe.max_hours_total is not None and train_hours_total + (duration_s / 3600.0) > recipe.max_hours_total:
                audio_path.unlink(missing_ok=True)
                continue
            if recipe.max_hours_per_language is not None and (
                train_hours_by_lang[language_id] + (duration_s / 3600.0) > recipe.max_hours_per_language
            ):
                audio_path.unlink(missing_ok=True)
                continue
            if recipe.max_records is not None and train_records_by_lang[language_id] >= recipe.max_records:
                audio_path.unlink(missing_ok=True)
                continue
        elif recipe.max_records is not None and dev_records_by_lang[language_id] >= max(4, recipe.max_records // 4):
            audio_path.unlink(missing_ok=True)
            continue

        output_stem = f"emilia_{language_id}_{role}"
        if skip_existing and (writer.manifest_root / f"{output_stem}.jsonl").exists():
            continue

        record = ManifestRecord(
            id=f"{output_stem}_{base_name}",
            audio_path=str(audio_path),
            text=text,
            language_id=language_id,
            dataset_name="emilia",
            source_id=source_id,
            speaker_id=_extract_speaker_id(sample, meta),
            duration_s=duration_s,
            config_name=None,
        )
        writer.write(
            output_stem=output_stem,
            record=record,
            repeat_hint=recipe.repeat_hint,
            source_configs=["train"],
            notes=recipe.notes,
            role=role,
        )
        seen_source_ids.add(dedupe_key)
        if role == "train":
            train_hours_total += duration_s / 3600.0
            train_hours_by_lang[language_id] += duration_s / 3600.0
            train_records_by_lang[language_id] += 1
        else:
            dev_records_by_lang[language_id] += 1


def _write_local_placeholder(recipe: CorpusRecipe, manifest_root: Path) -> list[Path]:
    placeholder_dir = manifest_root / "stage2_local_templates"
    placeholder_dir.mkdir(parents=True, exist_ok=True)
    readme_path = placeholder_dir / f"{recipe.name}.README.md"
    example_jsonl = placeholder_dir / f"{recipe.name}_train.example.jsonl"
    readme_path.write_text(
        "\n".join(
            [
                f"# {recipe.name}",
                "",
                recipe.notes,
                "",
                "Expected JSONL row format:",
                '  {"id":"utt_0001","audio_path":"/abs/path.wav","text":"...","language_id":"en","speaker_id":"spk_01"}',
                "",
                "Place your real manifests under a stable path on /workspace and pass them",
                "to the tokenization and data-config stages.",
            ]
        ),
        encoding="utf-8",
    )
    example_jsonl.write_text(
        json.dumps(
            {
                "id": f"{recipe.name}_example_0001",
                "audio_path": f"/workspace/data/local/{recipe.name}/sample.wav",
                "text": "Replace this line with the transcript for a real sample.",
                "language_id": "en",
                "speaker_id": "speaker-0001",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return [example_jsonl]


def write_manifest_plan(recipes: Iterable[CorpusRecipe], output_path: str):
    payload = [asdict(recipe) for recipe in recipes]
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
