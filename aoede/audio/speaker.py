from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from aoede.audio.io import resample_audio


@dataclass
class FrozenSpeakerEncoder:
    embedding_dim: int = 192
    target_sample_rate: int = 16000

    def encode(self, waveform: np.ndarray, sample_rate: int = 24000):
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
        waveform = waveform.astype(np.float32)
        waveform = resample_audio(waveform, sample_rate, self.target_sample_rate)
        if len(waveform) < 512:
            waveform = np.pad(waveform, (0, 512 - len(waveform)))

        frame_length = 512
        hop = 160
        frame_count = 1 + max(0, (len(waveform) - frame_length) // hop)
        frames = np.stack(
            [waveform[start : start + frame_length] for start in range(0, frame_count * hop, hop)],
            axis=0,
        )
        window = np.hanning(frame_length).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(frames * window[None, :], axis=-1)).astype(np.float32)
        log_spec = np.log1p(spectrum)

        feature_stack = np.concatenate(
            [
                log_spec.mean(axis=0),
                log_spec.std(axis=0),
                np.array(
                    [
                        waveform.mean(),
                        waveform.std(),
                        np.max(np.abs(waveform)),
                        np.percentile(np.abs(waveform), 95),
                    ],
                    dtype=np.float32,
                ),
            ]
        )

        seed = int(hashlib.sha256(b"aoede-speaker-fallback").hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        projection = rng.standard_normal((feature_stack.shape[0], self.embedding_dim)).astype(np.float32)
        embedding = feature_stack @ projection
        norm = float(np.linalg.norm(embedding)) or 1.0
        return (embedding / norm).astype(np.float32)
