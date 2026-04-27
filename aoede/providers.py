from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from aoede.audio.io import load_audio_file


@dataclass
class ProviderResult:
    audio: np.ndarray
    sample_rate: int
    provider: str
    metadata: dict


class VoiceCloneProvider:
    name = "base"

    def synthesize(
        self,
        text: str,
        reference_audio: str,
        language: str = "en",
        prompt_text: Optional[str] = None,
    ) -> ProviderResult:
        raise NotImplementedError


def _disable_teacher_compile_if_requested() -> None:
    if os.environ.get("AOEDE_DISABLE_TEACHER_COMPILE", "1") == "0":
        return
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHINDUCTOR_USE_CUDAGRAPHS", "0")
    try:
        import torch

        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.suppress_errors = True
        if hasattr(torch, "_inductor"):
            torch._inductor.config.triton.cudagraphs = False
    except Exception:
        return


class PassthroughProvider(VoiceCloneProvider):
    name = "passthrough"

    def synthesize(
        self,
        text: str,
        reference_audio: str,
        language: str = "en",
        prompt_text: Optional[str] = None,
    ) -> ProviderResult:
        audio, sample_rate = load_audio_file(reference_audio, target_sample_rate=24000)
        return ProviderResult(
            audio=audio,
            sample_rate=sample_rate,
            provider=self.name,
            metadata={"mode": "ground_truth_passthrough", "language": language},
        )


class VoxCpm2Provider(VoiceCloneProvider):
    name = "voxcpm2"

    def __init__(
        self,
        model_id: str = "openbmb/VoxCPM2",
        device: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ):
        self.model_id = model_id
        self.device = device
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        _disable_teacher_compile_if_requested()
        try:
            from voxcpm import VoxCPM
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "AOEDE_PROVIDER=voxcpm2 requires voxcpm. Install with "
                "`python -m pip install -e '.[audio,training,dev,codec,sota]'`."
            ) from exc
        kwargs = {"load_denoiser": False}
        if self.device:
            kwargs["device"] = self.device
        try:
            self._model = VoxCPM.from_pretrained(self.model_id, **kwargs)
        except TypeError:
            kwargs.pop("device", None)
            self._model = VoxCPM.from_pretrained(self.model_id, **kwargs)
        return self._model

    def synthesize(
        self,
        text: str,
        reference_audio: str,
        language: str = "en",
        prompt_text: Optional[str] = None,
    ) -> ProviderResult:
        model = self._load_model()
        kwargs = {
            "text": text,
            "reference_wav_path": str(reference_audio),
            "cfg_value": self.cfg_value,
            "inference_timesteps": self.inference_timesteps,
        }
        if prompt_text:
            kwargs["prompt_wav_path"] = str(reference_audio)
            kwargs["prompt_text"] = prompt_text
        wav = model.generate(**kwargs)
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        sample_rate = int(getattr(getattr(model, "tts_model", model), "sample_rate", 48000))
        return ProviderResult(
            audio=wav,
            sample_rate=sample_rate,
            provider=self.name,
            metadata={
                "model_id": self.model_id,
                "language": language,
                "cfg_value": self.cfg_value,
                "inference_timesteps": self.inference_timesteps,
                "prompt_text_used": bool(prompt_text),
            },
        )


class Qwen3Provider(VoiceCloneProvider):
    name = "qwen3"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device: Optional[str] = None,
        dtype: str = "bfloat16",
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "AOEDE_PROVIDER=qwen3 requires qwen-tts. Install with "
                "`python -m pip install -e '.[qwen]'`."
            ) from exc
        dtype = getattr(torch, self.dtype)
        self._model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=self.device or "auto",
            dtype=dtype,
        )
        return self._model

    def _language_name(self, language: str) -> str:
        mapping = {
            "zh": "Chinese",
            "cmn": "Chinese",
            "en": "English",
            "ja": "Japanese",
            "ko": "Korean",
            "de": "German",
            "fr": "French",
            "ru": "Russian",
            "pt": "Portuguese",
            "es": "Spanish",
            "it": "Italian",
        }
        normalized = language.strip().lower().replace("_", "-")
        base = normalized.split("-", maxsplit=1)[0]
        return mapping.get(base, language)

    def synthesize(
        self,
        text: str,
        reference_audio: str,
        language: str = "en",
        prompt_text: Optional[str] = None,
    ) -> ProviderResult:
        model = self._load_model()
        qwen_language = self._language_name(language)
        if hasattr(model, "create_voice_clone_prompt"):
            prompt = model.create_voice_clone_prompt(
                ref_audio=reference_audio,
                ref_text=prompt_text,
                x_vector_only_mode=not bool(prompt_text),
            )
            wavs, sample_rate = model.generate_voice_clone(
                text=text,
                language=qwen_language,
                voice_clone_prompt=prompt,
            )
            audio = wavs[0]
        else:
            wavs, sample_rate = model.generate_voice_clone(
                text=text,
                language=qwen_language,
                ref_audio=reference_audio,
                ref_text=prompt_text,
            )
            audio = wavs[0]
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        return ProviderResult(
            audio=np.asarray(audio, dtype=np.float32).reshape(-1),
            sample_rate=int(sample_rate),
            provider=self.name,
            metadata={
                "model_id": self.model_id,
                "language": language,
                "prompt_text_used": bool(prompt_text),
            },
        )


ProviderFactory = Callable[..., VoiceCloneProvider]


def get_provider(
    name: str,
    device: Optional[str] = None,
    model_id: Optional[str] = None,
) -> VoiceCloneProvider:
    normalized = name.strip().lower().replace("_", "-")
    if normalized in {"passthrough", "ground-truth", "groundtruth"}:
        return PassthroughProvider()
    if normalized in {"voxcpm", "voxcpm2"}:
        return VoxCpm2Provider(
            model_id=model_id or "openbmb/VoxCPM2",
            device=device,
        )
    if normalized in {"qwen", "qwen3", "qwen3-tts"}:
        return Qwen3Provider(
            model_id=model_id or "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device=device,
        )
    raise ValueError(f"Unsupported voice clone provider: {name}")


def provider_cache_key(name: str, model_id: Optional[str] = None) -> str:
    raw = f"{name}_{model_id or 'default'}"
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in raw)


def resolve_audio_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())
