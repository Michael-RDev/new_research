from pathlib import Path

import numpy as np

from aoede.audio.io import load_audio_file, save_audio_bytes


class _BrokenSoundFile:
    def read(self, *args, **kwargs):
        raise RuntimeError("synthetic soundfile failure")


def test_load_audio_file_falls_back_to_wave_when_soundfile_read_fails(
    tmp_path: Path,
    monkeypatch,
):
    audio_path = tmp_path / "sample.wav"
    waveform = np.linspace(-0.5, 0.5, num=240, dtype=np.float32)
    audio_path.write_bytes(save_audio_bytes(waveform, sample_rate=24_000))

    monkeypatch.setattr("aoede.audio.io.sf", _BrokenSoundFile())

    decoded, sample_rate = load_audio_file(audio_path, target_sample_rate=24_000)

    assert sample_rate == 24_000
    assert decoded.dtype == np.float32
    assert len(decoded) == len(waveform)
    assert np.isclose(float(np.max(np.abs(decoded))), 1.0, atol=1e-4)
