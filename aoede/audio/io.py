from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Tuple, Union

import numpy as np


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int):
    if source_sr == target_sr:
        return audio.astype(np.float32)
    duration = len(audio) / float(source_sr)
    target_len = max(1, int(round(duration * target_sr)))
    source_positions = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _read_wave_stream(handle: io.BufferedIOBase):
    with wave.open(handle, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        raw = wav_file.readframes(num_frames)

    if sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), sample_rate


def load_audio_bytes(data: bytes, target_sample_rate: int = 24000):
    audio, source_sr = _read_wave_stream(io.BytesIO(data))
    audio = resample_audio(audio, source_sr, target_sample_rate)
    peak = float(np.max(np.abs(audio))) if len(audio) else 1.0
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32), target_sample_rate


def load_audio_file(path: Union[Path, str], target_sample_rate: int = 24000):
    with Path(path).open("rb") as handle:
        return load_audio_bytes(handle.read(), target_sample_rate=target_sample_rate)


def save_audio_bytes(audio: np.ndarray, sample_rate: int = 24000):
    clipped = np.clip(audio, -0.999, 0.999)
    pcm = (clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()
