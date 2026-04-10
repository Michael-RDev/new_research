from pathlib import Path

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


def test_default_config_creates_artifact_directories(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = default_config(tmp_path)
    assert config.resolve(config.artifacts.voices_dir).exists()
    assert config.resolve(config.artifacts.checkpoints_dir).exists()
