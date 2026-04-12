from pathlib import Path

import numpy as np

from aoede.audio.io import save_audio_bytes
from aoede.data.huggingface import (
    HFIngestRequest,
    PreparedSource,
    atlasflow_default_requests,
    fit_and_save_tokenizer,
    materialize_rows_to_manifest,
    prepare_atlasflow_training_assets,
    supported_hf_datasets,
)


def _audio_row(text: str, speaker_id: str, locale: str = "wol", duration_s: float = 1.0):
    sample_rate = 16_000
    num_samples = int(sample_rate * duration_s)
    t = np.linspace(0.0, duration_s, num=num_samples, endpoint=False)
    waveform = 0.1 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    return {
        "id": f"{speaker_id}-{text[:6]}",
        "speaker_id": speaker_id,
        "locale": locale,
        "gender": "female",
        "audio": {"array": waveform, "sampling_rate": sample_rate},
        "text": text,
    }


def _audio_bytes(duration_s: float = 1.0, sample_rate: int = 16_000):
    num_samples = int(sample_rate * duration_s)
    t = np.linspace(0.0, duration_s, num=num_samples, endpoint=False)
    waveform = 0.1 * np.sin(2.0 * np.pi * 330.0 * t).astype(np.float32)
    return save_audio_bytes(waveform, sample_rate=sample_rate)


def test_supported_registry_includes_requested_sources():
    registry = supported_hf_datasets()
    assert "waxalnlp" in registry
    assert registry["peoples_speech"].dataset_path == "MLCommons/peoples_speech"
    assert registry["emilia_nv"].gated is True
    assert registry["emilia_dataset"].dataset_path == "amphion/Emilia-Dataset"
    assert registry["emilia_dataset"].gated is True
    assert registry["mls"].split_aliases["validation"] == "dev"


def test_materialize_rows_builds_speaker_refs_and_language_alias(tmp_path: Path):
    request = HFIngestRequest(source_id="waxalnlp", split="train", max_examples=3)
    spec = supported_hf_datasets()["waxalnlp"]
    rows = [
        _audio_row("Nanga def", "speaker-a"),
        _audio_row("Mangi fi rekk", "speaker-a"),
        _audio_row("Jerejef", "speaker-b"),
    ]

    prepared = materialize_rows_to_manifest(
        rows=rows,
        request=request,
        spec=spec,
        audio_root=tmp_path / "audio",
        manifest_path=tmp_path / "train.jsonl",
        sample_rate=24_000,
    )

    assert len(prepared.entries) == 3
    assert prepared.entries[0].language_code == "wo"
    assert prepared.entries[0].speaker_ref == prepared.entries[1].audio_path
    assert prepared.entries[2].speaker_ref is None
    assert Path(prepared.entries[0].audio_path).exists()


def test_fit_and_save_tokenizer_handles_new_language_tokens(tmp_path: Path):
    request = HFIngestRequest(source_id="waxalnlp")
    spec = supported_hf_datasets()["waxalnlp"]
    prepared = materialize_rows_to_manifest(
        rows=[_audio_row("Lii mooy Waxal", "speaker-a")],
        request=request,
        spec=spec,
        audio_root=tmp_path / "audio",
        manifest_path=tmp_path / "train.jsonl",
    )

    tokenizer = fit_and_save_tokenizer(prepared.entries, tmp_path / "tokenizer.json")
    assert "<lang:wo>" in tokenizer.token_to_id
    assert (tmp_path / "tokenizer.json").exists()


def test_materialize_emilia_dataset_rows_from_json_metadata_and_mp3_bytes(tmp_path: Path):
    request = HFIngestRequest(source_id="emilia_dataset", split="train", max_examples=2)
    spec = supported_hf_datasets()["emilia_dataset"]
    rows = [
        {
            "mp3": _audio_bytes(duration_s=1.2),
            "json": {
                "id": "emilia-en-001",
                "text": "hello from emilia",
                "language": "en",
                "speaker": "speaker-emilia",
                "duration": 1.2,
            },
        },
        {
            "mp3": _audio_bytes(duration_s=1.1),
            "json": {
                "id": "emilia-en-002",
                "text": "second emilia sample",
                "language": "en",
                "speaker": "speaker-emilia",
                "duration": 1.1,
            },
        },
    ]

    prepared = materialize_rows_to_manifest(
        rows=rows,
        request=request,
        spec=spec,
        audio_root=tmp_path / "audio",
        manifest_path=tmp_path / "train.jsonl",
        sample_rate=24_000,
    )

    assert len(prepared.entries) == 2
    assert prepared.entries[0].language_code == "en"
    assert prepared.entries[0].text == "hello from emilia"
    assert prepared.entries[0].metadata["hf_row_id"] == "emilia-en-001"
    assert prepared.entries[0].metadata["speaker_key"] == "speaker-emilia"
    assert prepared.entries[0].speaker_ref == prepared.entries[1].audio_path
    assert Path(prepared.entries[0].audio_path).exists()


def test_prepare_training_assets_writes_combined_manifests_and_summary(tmp_path: Path, monkeypatch):
    def fake_prepare_hf_source(request, manifest_dir, audio_root, sample_rate=24_000):
        spec = supported_hf_datasets()[request.source_id]
        entries = materialize_rows_to_manifest(
            rows=[_audio_row(f"{request.source_id}-{request.split}", f"{request.source_id}-speaker", locale="en_us")],
            request=request,
            spec=spec,
            audio_root=audio_root,
            manifest_path=manifest_dir / f"{request.source_id}-{request.split}.jsonl",
            sample_rate=sample_rate,
        ).entries
        return PreparedSource(
            request=request,
            spec=spec,
            manifest_path=manifest_dir / f"{request.source_id}-{request.split}.jsonl",
            entries=entries,
            skipped_rows=0,
        )

    monkeypatch.setattr("aoede.data.huggingface.prepare_hf_source", fake_prepare_hf_source)
    requests = [
        HFIngestRequest(source_id="waxalnlp", split="train"),
        HFIngestRequest(source_id="peoples_speech", split="validation", config_name="clean"),
    ]
    summary = prepare_atlasflow_training_assets(project_root=tmp_path, requests=requests)

    assert summary["train_entries"] == 1
    assert summary["eval_entries"] == 1
    assert (tmp_path / "artifacts" / "manifests" / "train.jsonl").exists()
    assert (tmp_path / "artifacts" / "manifests" / "eval.jsonl").exists()
    assert (tmp_path / "artifacts" / "manifests" / "atlasflow_hf_summary.json").exists()
    assert (tmp_path / "artifacts" / "tokenizer.json").exists()


def test_default_atlasflow_requests_cover_named_datasets():
    requests = atlasflow_default_requests(max_train_examples=100, max_eval_examples=10)
    keys = {request.source_id for request in requests}
    assert {"waxalnlp", "peoples_speech", "emilia_nv", "emilia_dataset"}.issubset(keys)
