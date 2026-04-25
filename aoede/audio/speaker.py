from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

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


@dataclass
class SpeechBrainEcapaSpeakerEncoder:
    embedding_dim: int = 192
    target_sample_rate: int = 16000
    source: str = "speechbrain/spkrec-ecapa-voxceleb"
    savedir: Optional[str] = None
    device: Optional[str] = None

    def __post_init__(self):
        self._classifier = None

    def _load_classifier(self):
        if self._classifier is not None:
            return self._classifier
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "AOEDE_SPEAKER_ENCODER=ecapa requires speechbrain. Install with "
                "`python -m pip install -e '.[audio,training,dev,codec,sota]'`."
            ) from exc

        savedir = self.savedir
        if savedir is None:
            safe_source = self.source.replace("/", "--")
            savedir = str(Path("pretrained_models") / safe_source)
        run_opts = {"device": self.device} if self.device else None
        kwargs = {"source": self.source, "savedir": savedir}
        if run_opts is not None:
            kwargs["run_opts"] = run_opts
        self._classifier = EncoderClassifier.from_hparams(**kwargs)
        return self._classifier

    def encode(self, waveform: np.ndarray, sample_rate: int = 24000):
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
        waveform = waveform.astype(np.float32)
        waveform = resample_audio(waveform, sample_rate, self.target_sample_rate)
        signal = torch.from_numpy(waveform).float().unsqueeze(0)
        classifier = self._load_classifier()
        with torch.no_grad():
            embeddings = classifier.encode_batch(signal)
        embedding = embeddings.detach().cpu().float().reshape(-1).numpy()
        if embedding.shape[0] != self.embedding_dim:
            if embedding.shape[0] > self.embedding_dim:
                embedding = embedding[: self.embedding_dim]
            else:
                embedding = np.pad(
                    embedding,
                    (0, self.embedding_dim - embedding.shape[0]),
                    mode="constant",
                )
        norm = float(np.linalg.norm(embedding)) or 1.0
        return (embedding / norm).astype(np.float32)


def normalize_speaker_encoder_backend(name: str) -> str:
    normalized = name.strip().lower()
    aliases = {
        "fallback": "frozen",
        "deterministic": "frozen",
        "speechbrain": "ecapa",
        "speechbrain-ecapa": "ecapa",
        "ecapa-tdnn": "ecapa",
    }
    return aliases.get(normalized, normalized)


def build_speaker_encoder(
    backend: str = "frozen",
    embedding_dim: int = 192,
    device: Optional[str] = None,
    source: str = "speechbrain/spkrec-ecapa-voxceleb",
    savedir: Optional[str] = None,
):
    backend = normalize_speaker_encoder_backend(backend)
    if backend == "frozen":
        return FrozenSpeakerEncoder(embedding_dim=embedding_dim)
    if backend == "ecapa":
        return SpeechBrainEcapaSpeakerEncoder(
            embedding_dim=embedding_dim,
            source=source,
            savedir=savedir,
            device=device,
        )
    raise ValueError(f"Unsupported speaker encoder backend: {backend}")


def speaker_cache_key(
    backend: str,
    embedding_dim: int = 192,
    source: str = "speechbrain/spkrec-ecapa-voxceleb",
) -> str:
    backend = normalize_speaker_encoder_backend(backend)
    safe_source = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in source
    )
    return f"{backend}_{safe_source}_spk{embedding_dim}"
