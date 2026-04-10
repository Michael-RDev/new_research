import json
from pathlib import Path

import numpy as np

from omnivoice.data import manifest_builder as mb


def _audio_row(sample_id, text, language, duration_s=2.0, speaker_id="spk-1"):
    sr = 16_000
    num_samples = int(sr * duration_s)
    return {
        "id": sample_id,
        "text": text,
        "language": language,
        "speaker_id": speaker_id,
        "audio": {
            "array": np.zeros(num_samples, dtype=np.float32),
            "sampling_rate": sr,
        },
    }


def test_stage1_recipes_cover_core_datasets():
    recipes = mb.stage_recipes("stage1", "smoke")
    names = {recipe.name for recipe in recipes}
    assert {"emilia", "peoples_speech", "mls", "waxal"} <= names


def test_waxal_language_map_is_complete_and_deterministic():
    mapping = mb.list_supported_waxal_codes()
    assert mapping["amh"] == "am"
    assert mapping["yor"] == "yo"
    assert mapping["ach"] is None
    assert mapping["swa"] == "sw"


def test_fixed_hf_recipe_materializes_manifest_and_summary(tmp_path, monkeypatch):
    recipe = mb.CorpusRecipe(
        name="peoples_speech",
        stage="stage1",
        kind="hf_audio",
        dataset_id="MLCommons/peoples_speech",
        config_names=("clean",),
        split_name="train",
        role="train",
        output_stem="peoples_speech_clean_train",
        language_id="en",
        max_records=4,
    )

    monkeypatch.setattr(
        mb,
        "_iter_hf_rows",
        lambda dataset_id, config_name, split_name, hf_token, streaming=True: [
            _audio_row("utt-001", "hello world", "en"),
            _audio_row("utt-001", "duplicate id", "en"),
            _audio_row("utt-002", "second sample", "en"),
        ],
    )

    manifest_root = tmp_path / "manifests"
    audio_root = tmp_path / "audio"
    built = mb.build_hf_manifests([recipe], str(manifest_root), str(audio_root))

    manifest_path = manifest_root / "peoples_speech_clean_train.jsonl"
    summary_path = manifest_root / "peoples_speech_clean_train.summary.json"
    assert manifest_path in built
    assert manifest_path.exists()
    assert summary_path.exists()

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["language_id"] == "en"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["num_samples"] == 2
    assert summary["dataset_name"] == "peoples_speech"


def test_mixed_fixed_recipe_creates_train_and_dev_outputs(tmp_path, monkeypatch):
    recipe = mb.CorpusRecipe(
        name="peoples_speech",
        stage="stage1",
        kind="hf_audio",
        dataset_id="MLCommons/peoples_speech",
        config_names=("clean",),
        split_name="train",
        role="mixed",
        output_stem="peoples_speech_clean",
        language_id="en",
        max_records=8,
        hash_dev_ratio=0.0,
        min_dev_examples_per_language=1,
    )

    monkeypatch.setattr(
        mb,
        "_iter_hf_rows",
        lambda dataset_id, config_name, split_name, hf_token, streaming=True: [
            _audio_row("utt-101", "first sample", "en"),
            _audio_row("utt-102", "second sample", "en"),
        ],
    )

    manifest_root = tmp_path / "manifests"
    audio_root = tmp_path / "audio"
    mb.build_hf_manifests([recipe], str(manifest_root), str(audio_root))

    assert (manifest_root / "peoples_speech_clean_dev.jsonl").exists()
    assert (manifest_root / "peoples_speech_clean_train.jsonl").exists()


def test_emilia_recipe_creates_train_and_dev_manifests(tmp_path, monkeypatch):
    recipe = mb.CorpusRecipe(
        name="emilia",
        stage="stage1",
        kind="hf_emilia",
        dataset_id="amphion/Emilia-Dataset",
        split_name="train",
        role="mixed",
        output_stem="emilia",
        max_hours_total=10.0,
        max_hours_per_language=10.0,
        max_records=8,
        hash_dev_ratio=0.0,
        min_dev_examples_per_language=1,
    )

    monkeypatch.setattr(
        mb,
        "_iter_hf_rows",
        lambda dataset_id, config_name, split_name, hf_token, streaming=True: [
            {"json": {"id": "de-001", "text": "guten tag", "language": "de"}, "audio": {"array": np.zeros(32000), "sampling_rate": 16000}},
            {"json": {"id": "de-002", "text": "zweite probe", "language": "de"}, "audio": {"array": np.zeros(32000), "sampling_rate": 16000}},
            {"json": {"id": "en-001", "text": "hello there", "language": "en"}, "audio": {"array": np.zeros(32000), "sampling_rate": 16000}},
        ],
    )

    manifest_root = tmp_path / "manifests"
    audio_root = tmp_path / "audio"
    mb.build_hf_manifests([recipe], str(manifest_root), str(audio_root))

    assert (manifest_root / "emilia_de_dev.jsonl").exists()
    assert (manifest_root / "emilia_de_train.jsonl").exists()
    assert (manifest_root / "emilia_en_dev.jsonl").exists()
    assert not (manifest_root / "emilia_en_train.jsonl").exists()
