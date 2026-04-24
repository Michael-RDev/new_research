from pathlib import Path
import json

from aoede.config import default_config
from aoede.languages import production_languages, resolve_language
from aoede.profiles import VoiceProfileStore
from aoede.schemas import StyleControls, VoiceProfile
from aoede.text.tokenizer import UnicodeTokenizer


def test_language_registry_has_16_production_languages():
    production = production_languages()
    assert len(production) == 16
    assert resolve_language("en").name == "English"


def test_tokenizer_round_trip_and_language_token():
    tokenizer = UnicodeTokenizer()
    ids = tokenizer.encode("Hello", "en")
    assert ids[0] == tokenizer.token_to_id["<bos>"]
    assert tokenizer.decode(ids) == "hello"


def test_tokenizer_can_freeze_vocab_for_inference():
    tokenizer = UnicodeTokenizer()
    tokenizer.fit(["hello"], ["en"])
    original_size = tokenizer.size

    ids = tokenizer.encode("hello z", "en", add_new_tokens=False)

    assert tokenizer.size == original_size
    assert tokenizer.unk_id in ids
    assert max(ids) < original_size


def test_voice_profile_store_round_trip(tmp_path: Path):
    store = VoiceProfileStore(tmp_path / "voices")
    profile = VoiceProfile(
        voice_id="unit-test",
        speaker_embedding=[0.1] * 192,
        style_latent=[0.2] * 32,
        controls=StyleControls(),
        language_priors={"en": 1.0},
    )
    store.save(profile)
    reloaded = store.load("unit-test")
    assert reloaded is not None
    assert reloaded.voice_id == "unit-test"
    assert len(store.list()) == 1
    assert reloaded.speaker_memory is None
    assert reloaded.speaker_summary is None


def test_voice_profile_store_round_trip_with_atlasflow_memory(tmp_path: Path):
    store = VoiceProfileStore(tmp_path / "voices")
    profile = VoiceProfile(
        voice_id="atlas-test",
        speaker_embedding=[0.1] * 192,
        style_latent=[0.2] * 32,
        speaker_memory=[[0.3] * 8, [0.4] * 8],
        speaker_summary=[0.5] * 8,
        controls=StyleControls(),
        language_priors={"en": 1.0},
    )
    store.save(profile)
    reloaded = store.load("atlas-test")
    assert reloaded is not None
    assert reloaded.speaker_memory == profile.speaker_memory
    assert reloaded.speaker_summary == profile.speaker_summary


def test_voice_profile_backwards_compatible_without_atlasflow_fields():
    legacy_payload = json.dumps(
        {
            "voice_id": "legacy",
            "speaker_embedding": [0.1] * 192,
            "style_latent": [0.2] * 32,
            "language_priors": {"en": 1.0},
            "metadata": {},
            "controls": {},
        }
    )
    profile = VoiceProfile.model_validate_json(legacy_payload)
    assert profile.voice_id == "legacy"
    assert profile.speaker_memory is None
    assert profile.speaker_summary is None


def test_default_config_creates_artifact_directories(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = default_config(tmp_path)
    assert config.resolve(config.artifacts.voices_dir).exists()
    assert config.resolve(config.artifacts.checkpoints_dir).exists()
    assert config.resolve(config.artifacts.datasets_dir).exists()
    assert config.model.architecture_variant == "mosaicflow"
