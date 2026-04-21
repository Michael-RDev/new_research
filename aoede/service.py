from __future__ import annotations

import base64
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Union

import numpy as np

from aoede.audio.io import load_audio_bytes, save_audio_bytes
from aoede.audio.speaker import FrozenSpeakerEncoder
from aoede.config import AppConfig, default_config
from aoede.languages import (
    experimental_languages,
    language_index,
    normalize_language,
    production_languages,
    resolve_language,
)
from aoede.profiles import VoiceProfileStore
from aoede.schemas import (
    HealthResponse,
    LanguageDescriptor,
    LanguageListResponse,
    StreamingEvent,
    StyleControls,
    SynthesisRequest,
    VoiceDesignRequest,
    VoiceEnrollmentResponse,
    VoiceListResponse,
    VoiceProfile,
)
from aoede.text.tokenizer import UnicodeTokenizer


def _hash_vector(name: str, dim: int):
    seed = sum(ord(char) * (index + 1) for index, char in enumerate(name)) % (2**32)
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vector) or 1.0
    return vector / norm


def _merge_controls(base: StyleControls, overlay: StyleControls):
    return StyleControls(
        pitch=overlay.pitch if overlay.pitch != 1.0 else base.pitch,
        pace=overlay.pace if overlay.pace != 1.0 else base.pace,
        energy=overlay.energy if overlay.energy != 1.0 else base.energy,
        brightness=overlay.brightness if overlay.brightness != 1.0 else base.brightness,
    )


class MockRuntime:
    def __init__(self, config: AppConfig):
        self.config = config
        self.sample_rate = config.model.sample_rate
        self.speaker_encoder = FrozenSpeakerEncoder(embedding_dim=config.model.speaker_dim)

    def health(self):
        return {"model_ready": True, "tokenizer_ready": True, "checkpoint": None}

    def enroll(self, audio_bytes: bytes, voice_id: Optional[str], metadata: Dict[str, str]):
        audio, sample_rate = load_audio_bytes(audio_bytes, target_sample_rate=self.sample_rate)
        speaker_embedding = self.speaker_encoder.encode(audio, sample_rate=sample_rate).tolist()
        style_anchor = np.concatenate(
            [
                np.array(
                    [
                        float(audio.mean()),
                        float(audio.std()),
                        float(np.max(np.abs(audio))),
                        float(np.percentile(np.abs(audio), 90)),
                    ],
                    dtype=np.float32,
                ),
                _hash_vector("mock-style", self.config.model.style_dim),
            ]
        )[: self.config.model.style_dim]
        profile = VoiceProfile(
            voice_id=voice_id or uuid.uuid4().hex,
            source="enrollment",
            preset="neutral",
            speaker_embedding=speaker_embedding,
            style_latent=style_anchor.tolist(),
            language_priors={spec.code: 1.0 for spec in production_languages()},
            metadata=metadata,
        )
        return profile

    def design(self, request: VoiceDesignRequest, base_profile: Optional[VoiceProfile]):
        speaker_embedding = (
            np.array(base_profile.speaker_embedding, dtype=np.float32)
            if base_profile is not None
            else _hash_vector(f"{request.preset}-speaker", self.config.model.speaker_dim)
        )
        style_latent = (
            np.array(base_profile.style_latent, dtype=np.float32)
            if base_profile is not None
            else _hash_vector(f"{request.preset}-style", self.config.model.style_dim)
        )
        controls = request.style_controls
        speaker_embedding = speaker_embedding * (1.0 + 0.05 * (controls.energy - 1.0))
        style_latent = style_latent.copy()
        style_latent[0] += controls.pitch - 1.0
        style_latent[1] += controls.pace - 1.0
        style_latent[2] += controls.energy - 1.0
        style_latent[3] += controls.brightness - 1.0

        return VoiceProfile(
            voice_id=request.voice_id or uuid.uuid4().hex,
            source="designed",
            preset=request.preset,
            speaker_embedding=(speaker_embedding / (np.linalg.norm(speaker_embedding) or 1.0)).tolist(),
            style_latent=style_latent.tolist(),
            language_priors=request.language_priors or {spec.code: 1.0 for spec in production_languages()},
            metadata=request.metadata,
            controls=request.style_controls,
        )

    def synthesize_array(self, request: SynthesisRequest, profile: VoiceProfile):
        controls = _merge_controls(profile.controls, request.style_controls)
        sample_rate = self.sample_rate
        text = request.text or " "
        speaker = np.asarray(profile.speaker_embedding, dtype=np.float32)
        style = np.asarray(profile.style_latent, dtype=np.float32)

        base_frequency = 140.0 + 35.0 * np.tanh(float(speaker[0]) * 2.0)
        base_frequency *= controls.pitch
        phoneme_seconds = max(0.035, 0.065 / controls.pace)
        segments = []

        for index, char in enumerate(text):
            samples = max(1, int(sample_rate * phoneme_seconds * (1.0 + (ord(char) % 5) * 0.03)))
            t = np.linspace(0.0, samples / sample_rate, num=samples, endpoint=False)
            frequency = base_frequency * (1.0 + 0.025 * ((ord(char) % 11) - 5))
            formant = 1.0 + 0.4 * controls.brightness + 0.15 * np.tanh(style[(index + 1) % len(style)])
            carrier = np.sin(2.0 * np.pi * frequency * t)
            harmonic = np.sin(2.0 * np.pi * frequency * 2.0 * t) * 0.35 * formant
            shimmer = np.sin(2.0 * np.pi * 6.0 * t + index * 0.3) * 0.05
            env = np.linspace(0.0, 1.0, num=samples)
            env = np.minimum(env, env[::-1]) * 2.0
            amplitude = 0.2 * controls.energy
            segments.append((carrier + harmonic + shimmer) * env * amplitude)

        pause = np.zeros(int(sample_rate * 0.08), dtype=np.float32)
        waveform = []
        for segment in segments:
            waveform.append(segment.astype(np.float32))
            waveform.append(pause)
        audio = np.concatenate(waveform) if waveform else pause
        audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.92
        return audio.astype(np.float32)


class TorchRuntime:
    def __init__(self, config: AppConfig):
        import torch

        from aoede.model.core import AoedeModel

        self.torch = torch
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = UnicodeTokenizer(config.resolve(config.artifacts.tokenizer_path))
        self.model = AoedeModel(config.model).to(self.device)
        self.model.eval()
        self.speaker_encoder = FrozenSpeakerEncoder(embedding_dim=config.model.speaker_dim)
        self.checkpoint = self._load_checkpoint()

    def _load_checkpoint(self):
        checkpoint_dir = self.config.resolve(self.config.artifacts.checkpoints_dir)
        checkpoints = sorted(checkpoint_dir.glob("*.pt"))
        if not checkpoints:
            return None
        latest = checkpoints[-1]
        state = self.torch.load(latest, map_location=self.device)
        model_state = state.get("model", state)
        self.model.load_state_dict(model_state, strict=False)
        return str(latest)

    def health(self):
        return {
            "model_ready": True,
            "tokenizer_ready": True,
            "checkpoint": self.checkpoint,
        }

    def enroll(self, audio_bytes: bytes, voice_id: Optional[str], metadata: Dict[str, str]):
        audio, sample_rate = load_audio_bytes(audio_bytes, target_sample_rate=self.config.model.sample_rate)
        speaker_embedding = self.speaker_encoder.encode(audio, sample_rate=sample_rate)
        waveform = self.torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        speaker_memory = None
        speaker_summary = None
        with self.torch.no_grad():
            latents = self.model.codec.encode(waveform)
            style_latent = self.model.infer_style(latents).cpu().numpy()[0]
            if self.model._atlasflow_enabled:
                reference_mask = self.torch.ones(
                    latents.shape[:2],
                    dtype=self.torch.bool,
                    device=latents.device,
                )
                memory, summary, _ = self.model.infer_reference_memory(latents, reference_mask)
                if memory is not None:
                    speaker_memory = memory[0].detach().cpu().tolist()
                if summary is not None:
                    speaker_summary = summary[0].detach().cpu().tolist()
        return VoiceProfile(
            voice_id=voice_id or uuid.uuid4().hex,
            source="enrollment",
            preset="neutral",
            speaker_embedding=speaker_embedding.tolist(),
            style_latent=style_latent.tolist(),
            speaker_memory=speaker_memory,
            speaker_summary=speaker_summary,
            language_priors={spec.code: 1.0 for spec in production_languages()},
            metadata=metadata,
        )

    def design(self, request: VoiceDesignRequest, base_profile: Optional[VoiceProfile]):
        base_speaker = (
            np.array(base_profile.speaker_embedding, dtype=np.float32)
            if base_profile is not None
            else _hash_vector(f"{request.preset}-speaker", self.config.model.speaker_dim)
        )
        base_style = (
            np.array(base_profile.style_latent, dtype=np.float32)
            if base_profile is not None
            else _hash_vector(f"{request.preset}-style", self.config.model.style_dim)
        )
        controls = request.style_controls
        styled = base_style.copy()
        styled[0] += controls.pitch - 1.0
        styled[1] += controls.pace - 1.0
        styled[2] += controls.energy - 1.0
        styled[3] += controls.brightness - 1.0
        speaker_memory = None
        speaker_summary = None
        if base_profile is not None:
            speaker_memory = base_profile.speaker_memory
            if base_profile.speaker_summary is not None:
                speaker_summary = base_profile.speaker_summary
            elif self.model._atlasflow_enabled:
                speaker_summary = _hash_vector(
                    f"{base_profile.voice_id}-summary",
                    self.config.model.d_model,
                ).tolist()
        elif self.model._atlasflow_enabled:
            speaker_summary = _hash_vector(
                f"{request.preset}-summary",
                self.config.model.d_model,
            ).tolist()
        return VoiceProfile(
            voice_id=request.voice_id or uuid.uuid4().hex,
            source="designed",
            preset=request.preset,
            speaker_embedding=(base_speaker / (np.linalg.norm(base_speaker) or 1.0)).tolist(),
            style_latent=styled.tolist(),
            speaker_memory=speaker_memory,
            speaker_summary=speaker_summary,
            language_priors=request.language_priors or {spec.code: 1.0 for spec in production_languages()},
            metadata=request.metadata,
            controls=request.style_controls,
        )

    def synthesize_array(self, request: SynthesisRequest, profile: VoiceProfile):
        token_ids = self.torch.tensor(
            [self.tokenizer.encode(request.text, normalize_language(request.language_code))],
            dtype=self.torch.long,
            device=self.device,
        )
        language_ids = self.torch.tensor(
            [language_index(request.language_code)],
            dtype=self.torch.long,
            device=self.device,
        )
        speaker_embedding = self.torch.tensor([profile.speaker_embedding], dtype=self.torch.float32, device=self.device)
        controls = _merge_controls(profile.controls, request.style_controls)
        style_latent = np.array(profile.style_latent, dtype=np.float32)
        style_latent[0] += controls.pitch - 1.0
        style_latent[1] += controls.pace - 1.0
        style_latent[2] += controls.energy - 1.0
        style_latent[3] += controls.brightness - 1.0
        style_tensor = self.torch.tensor([style_latent.tolist()], dtype=self.torch.float32, device=self.device)
        speaker_memory = None
        speaker_summary = None
        if self.model._atlasflow_enabled and profile.speaker_memory is not None:
            speaker_memory = self.torch.tensor(
                [profile.speaker_memory],
                dtype=self.torch.float32,
                device=self.device,
            )
        if self.model._atlasflow_enabled and profile.speaker_summary is not None:
            speaker_summary = self.torch.tensor(
                [profile.speaker_summary],
                dtype=self.torch.float32,
                device=self.device,
            )

        with self.torch.no_grad():
            waveform, _ = self.model.synthesize(
                token_ids=token_ids,
                language_ids=language_ids,
                speaker_embedding=speaker_embedding,
                style_latent=style_tensor,
                sampling_steps=request.sampling_steps,
                speaker_memory=speaker_memory,
                speaker_summary=speaker_summary,
            )
        return waveform[0].detach().cpu().numpy().astype(np.float32)

class AoedeService:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or default_config(Path.cwd())
        self.config.ensure_directories()
        self.profile_store = VoiceProfileStore(self.config.resolve(self.config.artifacts.voices_dir))
        self.tokenizer = UnicodeTokenizer(self.config.resolve(self.config.artifacts.tokenizer_path))
        if not self.config.resolve(self.config.artifacts.tokenizer_path).exists():
            self.tokenizer.save(self.config.resolve(self.config.artifacts.tokenizer_path))
        self.runtime_name = self._select_runtime()
        self.runtime = self._build_runtime()

    def _select_runtime(self):
        requested = os.environ.get("AOEDE_RUNTIME", self.config.service.runtime)
        if os.environ.get("AOEDE_DISABLE_TORCH") == "1":
            return "mock"
        if requested in {"mock", "torch"}:
            return requested
        return "torch" if ".venv_arm64" in sys.executable else "mock"

    def _build_runtime(self):
        if self.runtime_name == "torch":
            return TorchRuntime(self.config)
        return MockRuntime(self.config)

    def list_languages(self):
        return LanguageListResponse(
            production=[LanguageDescriptor(**spec.to_dict()) for spec in production_languages()],
            experimental=[LanguageDescriptor(**spec.to_dict()) for spec in experimental_languages()],
        )

    def health(self):
        details = self.runtime.health()
        return HealthResponse(
            status="ready",
            runtime=self.runtime_name,
            model_ready=bool(details["model_ready"]),
            tokenizer_ready=bool(details["tokenizer_ready"]),
            voices=len(self.profile_store.list()),
            checkpoint=details.get("checkpoint"),
        )

    def list_voices(self):
        return VoiceListResponse(voices=self.profile_store.list())

    def enroll(self, audio_bytes: bytes, voice_id: Optional[str] = None, metadata: Optional[Dict[str, str]] = None):
        profile = self.runtime.enroll(audio_bytes, voice_id, metadata or {})
        self.profile_store.save(profile)
        return VoiceEnrollmentResponse(
            voice_id=profile.voice_id,
            embedding_dim=len(profile.speaker_embedding),
            style_dim=len(profile.style_latent),
            saved_to=str(self.profile_store._path_for(profile.voice_id)),
            preview=[round(value, 4) for value in profile.speaker_embedding[:8]],
        )

    def design_voice(self, request: VoiceDesignRequest):
        base_profile = self.profile_store.load(request.source_voice_id) if request.source_voice_id else None
        profile = self.runtime.design(request, base_profile)
        self.profile_store.save(profile)
        return profile

    def resolve_profile(self, request: SynthesisRequest):
        profile = self.profile_store.resolve(request.voice_id, request.inline_profile)
        if profile is None:
            raise FileNotFoundError("Voice profile not found")
        return profile

    def synthesize(self, request: SynthesisRequest):
        profile = self.resolve_profile(request)
        audio = self.runtime.synthesize_array(request, profile)
        duration = len(audio) / float(self.config.model.sample_rate)
        return save_audio_bytes(audio, sample_rate=self.config.model.sample_rate), duration

    def stream_synthesis(self, request: SynthesisRequest):
        yield StreamingEvent(type="progress", stage="resolve_profile", progress=0.15)
        profile = self.resolve_profile(request)
        yield StreamingEvent(type="progress", stage="render", progress=0.5)
        audio = self.runtime.synthesize_array(request, profile)
        wav_bytes = save_audio_bytes(audio, sample_rate=self.config.model.sample_rate)
        chunk_size = self.config.service.stream_chunk_bytes
        chunks = [wav_bytes[index : index + chunk_size] for index in range(0, len(wav_bytes), chunk_size)]
        for index, chunk in enumerate(chunks, start=1):
            yield StreamingEvent(
                type="audio_chunk",
                stage="stream",
                progress=0.5 + 0.45 * (index / max(len(chunks), 1)),
                payload={"chunk_b64": base64.b64encode(chunk).decode("ascii"), "index": index},
            )
        yield StreamingEvent(
            type="done",
            stage="complete",
            progress=1.0,
            payload={
                "duration_s": round(len(audio) / float(self.config.model.sample_rate), 3),
                "sample_rate": self.config.model.sample_rate,
            },
        )


def build_service(config: Optional[AppConfig] = None):
    return AoedeService(config=config)
