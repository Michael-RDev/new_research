from aoede.data.manifest import ManifestEntry
from aoede.text.tokenizer import UnicodeTokenizer
from aoede.training.filtering import filter_trainable_entries, max_supported_duration_s


def test_filter_trainable_entries_drops_text_and_audio_over_limits():
    tokenizer = UnicodeTokenizer()
    tokenizer.fit(
        ["short text", "x" * 80, "audio ok"],
        ["en", "en", "en"],
    )
    entries = [
        ManifestEntry(item_id="keep", audio_path="keep.wav", text="short text", language_code="en", duration_s=1.0),
        ManifestEntry(item_id="text", audio_path="text.wav", text="x" * 80, language_code="en", duration_s=1.0),
        ManifestEntry(item_id="audio", audio_path="audio.wav", text="audio ok", language_code="en", duration_s=4.0),
    ]

    filtered, stats = filter_trainable_entries(
        entries,
        tokenizer,
        max_text_tokens=20,
        max_latent_frames=100,
        codec_hop_length=320,
        sample_rate=24_000,
    )

    assert [entry.item_id for entry in filtered] == ["keep"]
    assert stats.source_entries == 3
    assert stats.kept_entries == 1
    assert stats.dropped_text_too_long == 1
    assert stats.dropped_audio_too_long == 1


def test_max_supported_duration_s_matches_model_frame_budget():
    assert max_supported_duration_s(1600, 320, 24_000) == (1600 * 320) / 24_000
