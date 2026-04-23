from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from aoede.audio.io import load_audio_bytes, resample_audio, save_audio_bytes
from aoede.config import default_config
from aoede.data.manifest import ManifestEntry, save_manifest
from aoede.languages import canonical_language
from aoede.text.tokenizer import UnicodeTokenizer


COMMON_LANGUAGE_ALIASES = {
    "cmn_hans_cn": "zh",
    "cmn_hant_tw": "zh",
    "de_de": "de",
    "en_gb": "en",
    "en_us": "en",
    "es_419": "es",
    "es_es": "es",
    "fr_fr": "fr",
    "hi_in": "hi",
    "it_it": "it",
    "ja_jp": "ja",
    "ko_kr": "ko",
    "nl_nl": "nl",
    "pt_br": "pt",
    "pt_pt": "pt",
    "ru_ru": "ru",
    "sw_ke": "sw",
    "sw_tz": "sw",
    "tr_tr": "tr",
    "vi_vn": "vi",
    "wo_sn": "wo",
    "wol": "wo",
    "zh_cn": "zh",
}

RAW_AUDIO_BYTE_FIELDS = ("flac", "wav", "mp3", "ogg", "opus", "m4a")
ATLASFLOW_MAX_AUDIO_DURATION_S = (1600 * 320) / 24_000


@dataclass(frozen=True)
class HFDatasetSpec:
    source_id: str
    dataset_path: str
    description: str
    default_config: Optional[str] = None
    default_language_code: Optional[str] = None
    split_aliases: dict[str, str] = field(default_factory=dict)
    config_language_map: dict[str, str] = field(default_factory=dict)
    audio_fields: tuple[str, ...] = ("audio",)
    text_fields: tuple[str, ...] = (
        "text",
        "normalized_text",
        "transcription",
        "transcript",
        "sentence",
        "raw_transcription",
    )
    speaker_fields: tuple[str, ...] = ("speaker_id", "speaker", "client_id")
    locale_fields: tuple[str, ...] = ("locale", "language", "lang", "lang_id")
    gender_fields: tuple[str, ...] = ("gender",)
    id_fields: tuple[str, ...] = ("id", "utterance_id", "utt_id")
    duration_fields: tuple[str, ...] = (
        "duration_s",
        "duration",
        "audio_duration",
        "duration_ms",
    )
    trust_remote_code: bool = False
    gated: bool = False
    non_commercial: bool = False
    notes: str = ""


@dataclass(frozen=True)
class HFIngestRequest:
    source_id: str
    split: str = "train"
    config_name: Optional[str] = None
    language_code: Optional[str] = None
    max_examples: Optional[int] = None
    min_duration_s: float = 0.5
    max_duration_s: Optional[float] = 30.0
    streaming: bool = True


@dataclass
class PreparedSource:
    request: HFIngestRequest
    spec: HFDatasetSpec
    manifest_path: Path
    entries: list[ManifestEntry]
    skipped_rows: int = 0

    def to_dict(self):
        return {
            "source_id": self.request.source_id,
            "dataset_path": self.spec.dataset_path,
            "config_name": self.request.config_name or self.spec.default_config,
            "split": self.request.split,
            "language_code": self.request.language_code,
            "streaming": self.request.streaming,
            "entry_count": len(self.entries),
            "skipped_rows": self.skipped_rows,
            "manifest_path": str(self.manifest_path),
            "gated": self.spec.gated,
            "non_commercial": self.spec.non_commercial,
            "notes": self.spec.notes,
        }


HF_DATASET_SPECS: dict[str, HFDatasetSpec] = {
    "waxalnlp": HFDatasetSpec(
        source_id="waxalnlp",
        dataset_path="galsenai/WaxalNLP",
        description="Small Wolof multi-speaker speech/text dataset with explicit speaker IDs.",
        default_config="default",
        default_language_code="wo",
        notes="Maps locale 'wol' to Aoede language code 'wo'.",
    ),
    "peoples_speech": HFDatasetSpec(
        source_id="peoples_speech",
        dataset_path="MLCommons/peoples_speech",
        description="Large English speech corpus; best for text-audio scale rather than explicit cloning.",
        default_config="clean",
        default_language_code="en",
        notes="Subset configs include clean, clean_sa, dirty, dirty_sa, microset, validation, and test.",
    ),
    "emilia_nv": HFDatasetSpec(
        source_id="emilia_nv",
        dataset_path="amphion/Emilia-NV",
        description="Mandarin NVSpeech with inline paralinguistic annotations for expressive TTS.",
        default_language_code="zh",
        gated=True,
        non_commercial=True,
        notes="Requires accepting Hugging Face access terms and is research/non-commercial only.",
    ),
    "emilia_dataset": HFDatasetSpec(
        source_id="emilia_dataset",
        dataset_path="amphion/Emilia-Dataset",
        description="Multilingual Emilia / Emilia-YODAS WebDataset stream for staged TTS training.",
        gated=True,
        notes=(
            "Requires reviewing the dataset card access terms; licensing varies across included "
            "Emilia and Emilia-YODAS subsets."
        ),
    ),
    "fleurs": HFDatasetSpec(
        source_id="fleurs",
        dataset_path="google/fleurs",
        description="Wide multilingual read speech corpus with train/validation/test splits.",
        default_config="en_us",
        config_language_map={
            "cmn_hans_cn": "zh",
            "de_de": "de",
            "en_us": "en",
            "es_419": "es",
            "fr_fr": "fr",
            "hi_in": "hi",
            "wo_sn": "wo",
        },
        trust_remote_code=True,
        notes="Good multilingual coverage; speaker IDs are not exposed, so Aoede uses same-utterance reference fallback.",
    ),
    "mls": HFDatasetSpec(
        source_id="mls",
        dataset_path="facebook/multilingual_librispeech",
        description="Multi-speaker multilingual audiobook corpus with speaker IDs.",
        default_config="spanish",
        split_aliases={"validation": "dev"},
        config_language_map={
            "dutch": "nl",
            "english": "en",
            "french": "fr",
            "german": "de",
            "italian": "it",
            "polish": "pl",
            "portuguese": "pt",
            "spanish": "es",
        },
        notes=(
            "Config names inferred from the official HF dataset tree and MLS paper naming. "
            "The current builder exposes dutch, french, german, italian, polish, portuguese, and spanish."
        ),
    ),
    "parler_mls_eng_10k": HFDatasetSpec(
        source_id="parler_mls_eng_10k",
        dataset_path="parler-tts/mls_eng_10k",
        description="Smaller English MLS-derived speaker-labeled TTS subset for quick bootstrap runs.",
        default_language_code="en",
        split_aliases={"validation": "dev"},
        notes="Useful for smoke-scale AtlasFlow cloning experiments before larger MLS runs.",
    ),
}


def supported_hf_datasets():
    return HF_DATASET_SPECS


def atlasflow_default_requests(
    max_train_examples: Optional[int] = 20_000,
    max_eval_examples: Optional[int] = 512,
    max_duration_s: Optional[float] = ATLASFLOW_MAX_AUDIO_DURATION_S,
    include_gated: bool = False,
):
    requests = [
        HFIngestRequest(
            "waxalnlp",
            split="train",
            max_examples=min_or_none(max_train_examples, 834),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "waxalnlp",
            split="validation",
            max_examples=min_or_none(max_eval_examples, 97),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "peoples_speech",
            split="train",
            config_name="clean",
            max_examples=normalize_limit(max_train_examples),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "peoples_speech",
            split="validation",
            config_name="clean",
            max_examples=normalize_limit(max_eval_examples),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "fleurs",
            split="train",
            config_name="en_us",
            max_examples=min_or_none(max_train_examples, 2_500),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "fleurs",
            split="validation",
            config_name="en_us",
            max_examples=max_eval_examples,
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "fleurs",
            split="train",
            config_name="es_419",
            max_examples=min_or_none(max_train_examples, 2_500),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "fleurs",
            split="validation",
            config_name="es_419",
            max_examples=max_eval_examples,
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "fleurs",
            split="train",
            config_name="hi_in",
            max_examples=min_or_none(max_train_examples, 2_500),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "fleurs",
            split="validation",
            config_name="hi_in",
            max_examples=max_eval_examples,
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "fleurs",
            split="train",
            config_name="cmn_hans_cn",
            max_examples=min_or_none(max_train_examples, 2_500),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "fleurs",
            split="validation",
            config_name="cmn_hans_cn",
            max_examples=max_eval_examples,
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "mls",
            split="train",
            config_name="english",
            max_examples=normalize_limit(max_train_examples),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "mls",
            split="validation",
            config_name="english",
            max_examples=normalize_limit(max_eval_examples),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "mls",
            split="train",
            config_name="spanish",
            max_examples=normalize_limit(max_train_examples),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "mls",
            split="validation",
            config_name="spanish",
            max_examples=normalize_limit(max_eval_examples),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "mls",
            split="train",
            config_name="french",
            max_examples=normalize_limit(max_train_examples),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "mls",
            split="validation",
            config_name="french",
            max_examples=normalize_limit(max_eval_examples),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "mls",
            split="train",
            config_name="german",
            max_examples=normalize_limit(max_train_examples),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "mls",
            split="validation",
            config_name="german",
            max_examples=normalize_limit(max_eval_examples),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "parler_mls_eng_10k",
            split="train",
            max_examples=min_or_none(max_train_examples, 10_000),
            max_duration_s=max_duration_s,
        ),
        HFIngestRequest(
            "parler_mls_eng_10k",
            split="validation",
            max_examples=max_eval_examples,
            max_duration_s=max_duration_s,
        ),
    ]
    if include_gated:
        requests.extend(
            [
                HFIngestRequest(
                    "emilia_nv",
                    split="train",
                    max_examples=normalize_limit(max_train_examples),
                    max_duration_s=max_duration_s,
                ),
                HFIngestRequest(
                    "emilia_dataset",
                    split="train",
                    max_examples=normalize_limit(max_train_examples),
                    max_duration_s=max_duration_s,
                ),
            ]
        )
    return requests


def min_or_none(value: Optional[int], upper_bound: int):
    if value is None or value <= 0:
        return None
    return min(value, upper_bound)


def normalize_limit(value: Optional[int]):
    if value is None or value <= 0:
        return None
    return value


def _pick_value(row: dict[str, Any], field_names: Sequence[str]):
    for field_name in field_names:
        value = row.get(field_name)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _pick_named_value(row: dict[str, Any], field_names: Sequence[str]):
    for field_name in field_names:
        value = row.get(field_name)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return field_name, value
    return None, None


def _pick_value_with_meta(
    row: dict[str, Any], meta: dict[str, Any], field_names: Sequence[str]
):
    value = _pick_value(row, field_names)
    if value is not None:
        return value
    return _pick_value(meta, field_names)


def _pick_named_value_with_meta(
    row: dict[str, Any], meta: dict[str, Any], field_names: Sequence[str]
):
    field_name, value = _pick_named_value(row, field_names)
    if value is not None:
        return field_name, value
    return _pick_named_value(meta, field_names)


def _coerce_json_metadata(row: dict[str, Any]):
    raw = row.get("json")
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
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _pick_audio_value(row: dict[str, Any], spec: HFDatasetSpec):
    audio_field, audio_value = _pick_named_value(row, spec.audio_fields)
    if audio_value is not None:
        return audio_field, audio_value
    return _pick_named_value(row, RAW_AUDIO_BYTE_FIELDS)


def _normalize_language_code(
    spec: HFDatasetSpec,
    row: dict[str, Any],
    meta: dict[str, Any],
    request: HFIngestRequest,
):
    if request.language_code:
        return canonical_language(request.language_code) or request.language_code
    raw_code = _pick_value_with_meta(row, meta, spec.locale_fields)
    if isinstance(raw_code, str):
        normalized = canonical_language(raw_code)
        if normalized is not None:
            return normalized
        mapped = spec.config_language_map.get(raw_code.lower())
        if mapped is not None:
            return canonical_language(mapped) or mapped
    config_name = request.config_name or spec.default_config
    if config_name:
        mapped = spec.config_language_map.get(config_name.lower(), config_name)
        normalized = canonical_language(mapped)
        if normalized is not None:
            return normalized
    fallback = request.language_code or spec.default_language_code or "und"
    return canonical_language(fallback) or fallback


def _coerce_audio_array(audio_value: Any):
    if not isinstance(audio_value, dict):
        return None, None
    array = audio_value.get("array")
    sample_rate = audio_value.get("sampling_rate")
    if array is None and sample_rate is None:
        audio_bytes = audio_value.get("bytes")
        if audio_bytes:
            waveform, decoded_rate = load_audio_bytes(audio_bytes, target_sample_rate=24_000)
            return waveform, int(decoded_rate)
    if array is None or sample_rate is None:
        return None, None
    waveform = np.asarray(array, dtype=np.float32)
    if waveform.ndim == 2:
        if waveform.shape[0] <= 8 and waveform.shape[0] < waveform.shape[1]:
            waveform = waveform.mean(axis=0)
        else:
            waveform = waveform.mean(axis=-1)
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)
    return waveform.astype(np.float32), int(sample_rate)


def _duration_seconds(
    row: dict[str, Any],
    meta: dict[str, Any],
    waveform: np.ndarray,
    sample_rate: int,
    spec: HFDatasetSpec,
):
    duration_field, raw_duration = _pick_named_value_with_meta(
        row, meta, spec.duration_fields
    )
    if raw_duration is not None:
        duration = float(raw_duration)
        if duration_field == "duration_ms":
            # Some corpora store milliseconds.
            duration = duration / 1_000.0
        return duration
    return float(len(waveform)) / float(sample_rate)


def _materialize_audio(
    audio_value: Any,
    output_path: Path,
    target_sample_rate: int,
):
    if isinstance(audio_value, (bytes, bytearray)):
        waveform, _ = load_audio_bytes(bytes(audio_value), target_sample_rate=target_sample_rate)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(save_audio_bytes(waveform, sample_rate=target_sample_rate))
        return waveform
    waveform, source_rate = _coerce_audio_array(audio_value)
    if waveform is None or source_rate is None:
        raise ValueError("Audio field must provide an 'array' and 'sampling_rate'.")
    waveform = resample_audio(waveform, source_rate, target_sample_rate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(save_audio_bytes(waveform, sample_rate=target_sample_rate))
    return waveform


def _speaker_key(spec: HFDatasetSpec, row: dict[str, Any], meta: dict[str, Any]):
    value = _pick_value_with_meta(row, meta, spec.speaker_fields)
    if value is None:
        return None
    return str(value)


def _row_id(
    spec: HFDatasetSpec, row: dict[str, Any], meta: dict[str, Any], fallback_index: int
):
    raw_value = _pick_value_with_meta(row, meta, spec.id_fields)
    if raw_value is None:
        raw_value = f"{fallback_index:08d}"
    return str(raw_value)


def _source_slug(request: HFIngestRequest):
    parts = [request.source_id]
    if request.config_name:
        parts.append(request.config_name)
    parts.append(request.split)
    return "--".join(part.replace("/", "_") for part in parts)


def _iter_dataset_rows(request: HFIngestRequest, spec: HFDatasetSpec):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face importer requires the optional 'training' dependencies."
        ) from exc

    split_name = spec.split_aliases.get(request.split, request.split)
    load_kwargs: dict[str, Any] = {"split": split_name, "streaming": request.streaming}
    if spec.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    config_name = request.config_name or spec.default_config
    if config_name:
        dataset = load_dataset(spec.dataset_path, config_name, **load_kwargs)
    else:
        dataset = load_dataset(spec.dataset_path, **load_kwargs)
    return dataset


def materialize_rows_to_manifest(
    rows: Iterable[dict[str, Any]],
    request: HFIngestRequest,
    spec: HFDatasetSpec,
    audio_root: Path,
    manifest_path: Path,
    sample_rate: int = 24_000,
):
    source_slug = _source_slug(request)
    entries: list[ManifestEntry] = []
    speaker_groups: dict[str, list[int]] = {}
    skipped_rows = 0

    for row_index, row in enumerate(rows):
        meta = _coerce_json_metadata(row)
        _, audio_value = _pick_audio_value(row, spec)
        text_value = _pick_value_with_meta(row, meta, spec.text_fields)
        if audio_value is None or text_value is None:
            skipped_rows += 1
            continue

        row_id = _row_id(spec, row, meta, row_index)
        digest = hashlib.sha1(f"{source_slug}:{row_id}".encode("utf-8")).hexdigest()[
            :12
        ]
        item_id = f"{source_slug}-{digest}"
        output_path = audio_root / request.source_id / source_slug / f"{item_id}.wav"

        try:
            waveform = _materialize_audio(
                audio_value, output_path, target_sample_rate=sample_rate
            )
        except Exception:
            skipped_rows += 1
            continue

        duration_s = _duration_seconds(row, meta, waveform, sample_rate, spec)
        if duration_s < request.min_duration_s:
            skipped_rows += 1
            output_path.unlink(missing_ok=True)
            continue
        if request.max_duration_s is not None and duration_s > request.max_duration_s:
            skipped_rows += 1
            output_path.unlink(missing_ok=True)
            continue

        language_code = _normalize_language_code(spec, row, meta, request)
        speaker_key = _speaker_key(spec, row, meta)
        metadata = {
            "hf_dataset": spec.dataset_path,
            "hf_source_id": request.source_id,
            "hf_split": request.split,
            "hf_row_id": row_id,
        }
        config_name = request.config_name or spec.default_config
        if config_name:
            metadata["hf_config"] = config_name
        gender_value = _pick_value_with_meta(row, meta, spec.gender_fields)
        if gender_value is not None:
            metadata["gender"] = str(gender_value)
        if speaker_key is not None:
            metadata["speaker_key"] = speaker_key

        entries.append(
            ManifestEntry(
                item_id=item_id,
                audio_path=str(output_path),
                text=str(text_value),
                language_code=language_code,
                split=request.split,
                dataset_name=request.source_id,
                duration_s=float(duration_s),
                speaker_ref=None,
                metadata=metadata,
            )
        )
        if speaker_key is not None:
            speaker_groups.setdefault(speaker_key, []).append(len(entries) - 1)
        if request.max_examples is not None and len(entries) >= request.max_examples:
            break

    for index_list in speaker_groups.values():
        if len(index_list) < 2:
            continue
        for offset, entry_index in enumerate(index_list):
            ref_index = index_list[(offset + 1) % len(index_list)]
            entries[entry_index].speaker_ref = entries[ref_index].audio_path

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    save_manifest(entries, manifest_path)
    return PreparedSource(
        request=request,
        spec=spec,
        manifest_path=manifest_path,
        entries=entries,
        skipped_rows=skipped_rows,
    )


def prepare_hf_source(
    request: HFIngestRequest,
    manifest_dir: Path,
    audio_root: Path,
    sample_rate: int = 24_000,
):
    if request.source_id not in HF_DATASET_SPECS:
        supported = ", ".join(sorted(HF_DATASET_SPECS))
        raise KeyError(
            f"Unsupported Hugging Face source '{request.source_id}'. Supported: {supported}"
        )
    spec = HF_DATASET_SPECS[request.source_id]
    rows = _iter_dataset_rows(request, spec)
    manifest_path = manifest_dir / f"{_source_slug(request)}.jsonl"
    return materialize_rows_to_manifest(
        rows=rows,
        request=request,
        spec=spec,
        audio_root=audio_root,
        manifest_path=manifest_path,
        sample_rate=sample_rate,
    )


def combine_prepared_sources(
    prepared_sources: Sequence[PreparedSource],
    manifest_path: Path,
):
    entries: list[ManifestEntry] = []
    for prepared in prepared_sources:
        entries.extend(prepared.entries)
    save_manifest(entries, manifest_path)
    return entries


def fit_and_save_tokenizer(
    train_entries: Sequence[ManifestEntry],
    tokenizer_path: Path,
):
    tokenizer = UnicodeTokenizer()
    tokenizer.fit(
        [entry.text for entry in train_entries],
        [entry.language_code for entry in train_entries],
    )
    tokenizer.save(tokenizer_path)
    return tokenizer


def prepare_atlasflow_training_assets(
    project_root: Path,
    requests: Optional[Sequence[HFIngestRequest]] = None,
    sample_rate: int = 24_000,
    train_manifest_name: str = "train.jsonl",
    eval_manifest_name: str = "eval.jsonl",
):
    config = default_config(project_root)
    manifest_dir = config.resolve(config.artifacts.manifest_dir)
    audio_root = config.resolve(config.artifacts.datasets_dir) / "hf_audio"
    requests = list(requests or atlasflow_default_requests())
    prepared: list[PreparedSource] = []
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    total = len(requests)
    for index, request in enumerate(requests, start=1):
        print(
            f"[{index}/{total}] START source={request.source_id} config={request.config_name} split={request.split}",
            flush=True,
        )
        spec = HF_DATASET_SPECS[request.source_id]
        try:
            prepared_source = prepare_hf_source(
                request=request,
                manifest_dir=manifest_dir,
                audio_root=audio_root,
                sample_rate=sample_rate,
            )
        except Exception as exc:
            failure = {
                "source_id": request.source_id,
                "dataset_path": spec.dataset_path,
                "config_name": request.config_name or spec.default_config,
                "split": request.split,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(
                f"[{index}/{total}] FAIL {failure['error_type']}: {failure['error']}",
                flush=True,
            )
            continue

        prepared.append(prepared_source)
        if not prepared_source.entries:
            warning = {
                "source_id": request.source_id,
                "dataset_path": spec.dataset_path,
                "config_name": request.config_name or spec.default_config,
                "split": request.split,
                "warning_type": "empty_source",
                "warning": "No entries were materialized for this source.",
                "skipped_rows": prepared_source.skipped_rows,
                "manifest_path": str(prepared_source.manifest_path),
            }
            warnings.append(warning)
            print(
                f"[{index}/{total}] WARN empty_source skipped={prepared_source.skipped_rows} manifest={prepared_source.manifest_path}",
                flush=True,
            )
        else:
            print(
                f"[{index}/{total}] DONE entries={len(prepared_source.entries)} skipped={prepared_source.skipped_rows} manifest={prepared_source.manifest_path}",
                flush=True,
            )

    train_sources = [item for item in prepared if item.request.split == "train"]
    eval_sources = [item for item in prepared if item.request.split != "train"]

    train_entries = combine_prepared_sources(
        train_sources, manifest_dir / train_manifest_name
    )
    eval_entries = combine_prepared_sources(
        eval_sources, manifest_dir / eval_manifest_name
    )
    tokenizer = fit_and_save_tokenizer(
        train_entries=train_entries,
        tokenizer_path=config.resolve(config.artifacts.tokenizer_path),
    )

    summary = {
        "prepared_sources": [item.to_dict() for item in prepared],
        "warnings": warnings,
        "failures": failures,
        "train_manifest": str(manifest_dir / train_manifest_name),
        "eval_manifest": str(manifest_dir / eval_manifest_name),
        "train_entries": len(train_entries),
        "eval_entries": len(eval_entries),
        "tokenizer_path": str(config.resolve(config.artifacts.tokenizer_path)),
        "tokenizer_size": tokenizer.size,
    }
    summary_path = manifest_dir / "atlasflow_hf_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def parse_request(value: str):
    parts = value.split(":")
    source_id = parts[0]
    config_name = parts[1] if len(parts) > 1 and parts[1] else None
    split = parts[2] if len(parts) > 2 and parts[2] else "train"
    return HFIngestRequest(source_id=source_id, config_name=config_name, split=split)


def _resolve_env_file(env_file: str | os.PathLike[str] | None, project_root: Path) -> Optional[Path]:
    if not env_file:
        return None
    candidate = Path(env_file).expanduser()
    if candidate.is_absolute():
        return candidate
    project_candidate = (project_root / candidate).resolve()
    if project_candidate.exists():
        return project_candidate
    if candidate.exists():
        return candidate.resolve()
    return project_candidate


def _load_env_file(path: Optional[Path], override: bool = False) -> dict[str, str]:
    if path is None or not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        values[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return values


def _cli(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Prepare Aoede manifests and tokenizer assets from Hugging Face datasets.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Workspace root where Aoede artifacts live.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Optional .env file to load before talking to Hugging Face.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source request in the form 'source_id[:config_name[:split]]'. Repeatable.",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=20_000,
        help="Default cap used by the built-in AtlasFlow mix for each train source.",
    )
    parser.add_argument(
        "--max-eval-examples",
        type=int,
        default=512,
        help="Default cap used by the built-in AtlasFlow mix for each eval source.",
    )
    parser.add_argument(
        "--include-gated",
        action="store_true",
        help="Include gated Emilia datasets in the default staging mix.",
    )
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    _load_env_file(_resolve_env_file(args.env_file, project_root))

    if args.source:
        requests = [parse_request(raw) for raw in args.source]
    else:
        requests = atlasflow_default_requests(
            max_train_examples=args.max_train_examples,
            max_eval_examples=args.max_eval_examples,
            include_gated=args.include_gated,
        )
    summary = prepare_atlasflow_training_assets(
        project_root=project_root,
        requests=requests,
    )
    print(json.dumps(summary, indent=2))


def main():
    _cli()


if __name__ == "__main__":
    main()
