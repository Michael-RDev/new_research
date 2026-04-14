import os
import runpy
import sys
import types
import warnings
from pathlib import Path

import numpy as np

from aoede.audio.io import save_audio_bytes
from aoede.data.manifest import save_manifest
from aoede.data.huggingface import (
    ATLASFLOW_MAX_AUDIO_DURATION_S,
    _cli,
    _iter_dataset_rows,
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


def test_prepare_training_assets_continues_with_failures_and_empty_sources(tmp_path: Path, monkeypatch):
    def fake_prepare_hf_source(request, manifest_dir, audio_root, sample_rate=24_000):
        spec = supported_hf_datasets()[request.source_id]
        manifest_path = manifest_dir / f"{request.source_id}-{request.split}.jsonl"
        if request.source_id == "emilia_nv":
            save_manifest([], manifest_path)
            return PreparedSource(
                request=request,
                spec=spec,
                manifest_path=manifest_path,
                entries=[],
                skipped_rows=17,
            )
        if request.source_id == "fleurs":
            raise ValueError("trust_remote_code missing")

        entries = materialize_rows_to_manifest(
            rows=[_audio_row(f"{request.source_id}-{request.split}", f"{request.source_id}-speaker", locale="en_us")],
            request=request,
            spec=spec,
            audio_root=audio_root,
            manifest_path=manifest_path,
            sample_rate=sample_rate,
        ).entries
        return PreparedSource(
            request=request,
            spec=spec,
            manifest_path=manifest_path,
            entries=entries,
            skipped_rows=0,
        )

    monkeypatch.setattr("aoede.data.huggingface.prepare_hf_source", fake_prepare_hf_source)
    requests = [
        HFIngestRequest(source_id="waxalnlp", split="train"),
        HFIngestRequest(source_id="peoples_speech", split="validation", config_name="clean"),
        HFIngestRequest(source_id="emilia_nv", split="train"),
        HFIngestRequest(source_id="fleurs", split="train", config_name="en_us"),
    ]
    summary = prepare_atlasflow_training_assets(project_root=tmp_path, requests=requests)

    assert summary["train_entries"] == 1
    assert summary["eval_entries"] == 1
    assert len(summary["warnings"]) == 1
    assert summary["warnings"][0]["source_id"] == "emilia_nv"
    assert len(summary["failures"]) == 1
    assert summary["failures"][0]["source_id"] == "fleurs"
    assert (tmp_path / "artifacts" / "manifests" / "train.jsonl").exists()
    assert (tmp_path / "artifacts" / "manifests" / "eval.jsonl").exists()
    assert (tmp_path / "artifacts" / "manifests" / "atlasflow_hf_summary.json").exists()
    assert (tmp_path / "artifacts" / "tokenizer.json").exists()


def test_default_atlasflow_requests_cover_named_datasets():
    requests = atlasflow_default_requests(max_train_examples=100, max_eval_examples=10)
    keys = {request.source_id for request in requests}
    assert {"waxalnlp", "peoples_speech", "emilia_nv", "emilia_dataset"}.issubset(keys)


def test_default_atlasflow_requests_drop_mls_english_and_cap_duration():
    requests = atlasflow_default_requests(max_train_examples=100, max_eval_examples=10)
    assert all(
        not (request.source_id == "mls" and request.config_name == "english")
        for request in requests
    )
    assert all(request.max_duration_s == ATLASFLOW_MAX_AUDIO_DURATION_S for request in requests)


def test_iter_dataset_rows_enables_trust_remote_code_for_fleurs(monkeypatch):
    calls = []

    def fake_load_dataset(path, *args, **kwargs):
        calls.append((path, args, kwargs))
        return []

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    request = HFIngestRequest(source_id="fleurs", split="train", config_name="en_us")
    spec = supported_hf_datasets()["fleurs"]
    list(_iter_dataset_rows(request, spec))

    assert calls[0][0] == "google/fleurs"
    assert calls[0][2]["trust_remote_code"] is True


def test_cli_loads_hf_token_from_project_root_env(tmp_path: Path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("HF_TOKEN=hf_test_token\n", encoding="utf-8")

    previous_hf_token = os.environ.pop("HF_TOKEN", None)

    def fake_prepare_training_assets(project_root, requests):
        assert project_root == tmp_path.resolve()
        assert os.environ.get("HF_TOKEN") == "hf_test_token"
        assert len(requests) == 1
        return {"train_entries": 0, "eval_entries": 0}

    monkeypatch.setattr(
        "aoede.data.huggingface.prepare_atlasflow_training_assets",
        fake_prepare_training_assets,
    )

    try:
        _cli(["--project-root", str(tmp_path), "--source", "waxalnlp"])
    finally:
        if previous_hf_token is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = previous_hf_token


def test_module_main_executes_cli(tmp_path: Path, monkeypatch):
    def fake_prepare_training_assets(project_root, requests):
        assert project_root == tmp_path.resolve()
        assert len(requests) == 1
        return {"train_entries": 0, "eval_entries": 0}

    monkeypatch.setattr(
        "aoede.data.huggingface.prepare_atlasflow_training_assets",
        fake_prepare_training_assets,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aoede.data.huggingface",
            "--project-root",
            str(tmp_path),
            "--source",
            "waxalnlp",
        ],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module("aoede.data.huggingface", run_name="__main__")
