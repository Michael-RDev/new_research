from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aoede.config import ModelConfig


class FrozenAudioCodec(nn.Module):
    """
    Deterministic frozen codec fallback.

    The interface mirrors a neural audio codec even when a pretrained external
    backend is unavailable. Audio is framed, projected into a latent space with a
    fixed analysis basis, then reconstructed with a matching synthesis basis.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        latent_dim: int = 128,
        frame_size: int = 640,
        hop_length: int = 320,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.frame_size = frame_size
        self.hop_length = hop_length

        basis = self._build_basis(frame_size, latent_dim)
        synthesis = torch.linalg.pinv(basis)
        self.register_buffer("analysis_basis", basis)
        self.register_buffer("synthesis_basis", synthesis)

        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _build_basis(frame_size: int, latent_dim: int):
        time = torch.linspace(0.0, 1.0, frame_size)
        rows = []
        for idx in range(latent_dim):
            frequency = 1.0 + idx // 2
            phase = 0.0 if idx % 2 == 0 else math.pi / 2.0
            rows.append(torch.sin(2.0 * math.pi * frequency * time + phase))
        basis = torch.stack(rows, dim=0)
        basis = F.normalize(basis, dim=1)
        return basis

    def _prepare_waveform(self, waveform: torch.Tensor):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        return waveform.float()

    def encode(self, waveform: torch.Tensor):
        waveform = self._prepare_waveform(waveform)
        pad = (self.frame_size - waveform.shape[-1] % self.hop_length) % self.hop_length
        if pad:
            waveform = F.pad(waveform, (0, pad + self.frame_size))
        else:
            waveform = F.pad(waveform, (0, self.frame_size))
        frames = waveform.unfold(-1, self.frame_size, self.hop_length)
        latents = torch.einsum("btf,df->btd", frames, self.analysis_basis)
        return torch.tanh(latents)

    def decode(self, latents: torch.Tensor):
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)
        frames = torch.einsum("btd,fd->btf", latents, self.synthesis_basis)
        total = (frames.shape[1] - 1) * self.hop_length + self.frame_size
        waveform = frames.new_zeros(frames.shape[0], total)
        window = torch.hann_window(self.frame_size, device=frames.device).clamp_min(
            1e-3
        )
        window_sum = frames.new_zeros(total)

        for index in range(frames.shape[1]):
            start = index * self.hop_length
            waveform[:, start : start + self.frame_size] += frames[:, index] * window
            window_sum[start : start + self.frame_size] += window

        waveform = waveform / window_sum.clamp_min(1e-3)
        waveform = waveform / waveform.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        return waveform

    def forward(self, waveform: torch.Tensor):
        return self.encode(waveform)


class DacAudioCodec(nn.Module):
    """
    Pretrained Descript Audio Codec backend.

    DAC gives Aoede a real codec-latent target and a real decoder/vocoder. The
    external DAC module is loaded lazily so constructing AoedeModel for training
    does not immediately download or register the frozen codec weights.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        latent_dim: int = 1024,
        hop_length: int = 512,
        model_type: str = "24khz",
        model_path: Optional[str] = None,
        device: Optional[str | torch.device] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.hop_length = hop_length
        self.model_type = model_type
        self.model_path = str(model_path) if model_path else None
        self._preferred_device = torch.device(device) if device is not None else None
        object.__setattr__(self, "_dac_model", None)
        object.__setattr__(self, "_dac_device", None)

    def _load_model(self, device: torch.device):
        model = object.__getattribute__(self, "_dac_model")
        current_device = object.__getattribute__(self, "_dac_device")
        if model is not None and current_device == device:
            return model

        if model is None:
            try:
                import dac
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "CODEC_BACKEND=dac requires descript-audio-codec. Install it with "
                    "`python -m pip install -e '.[audio,training,dev,codec]'` or "
                    "`python -m pip install descript-audio-codec`."
                ) from exc

            model_path = self.model_path
            if model_path is None:
                model_path = dac.utils.download(model_type=self.model_type)
            model = dac.DAC.load(model_path)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._validate_loaded_model(model)
            object.__setattr__(self, "_dac_model", model)

        model.to(device)
        object.__setattr__(self, "_dac_device", device)
        return model

    def _validate_loaded_model(self, model) -> None:
        actual_sample_rate = int(getattr(model, "sample_rate", self.sample_rate))
        if actual_sample_rate != self.sample_rate:
            raise ValueError(
                f"DAC model sample_rate={actual_sample_rate}, but Aoede is configured "
                f"for sample_rate={self.sample_rate}."
            )

        actual_latent_dim = int(getattr(model, "latent_dim", self.latent_dim))
        if actual_latent_dim != self.latent_dim:
            raise ValueError(
                f"DAC model latent_dim={actual_latent_dim}, but Aoede is configured "
                f"for codec_latent_dim={self.latent_dim}. Re-run training with "
                f"`--codec-latent-dim {actual_latent_dim}`."
            )

        actual_hop_length = int(getattr(model, "hop_length", self.hop_length))
        if actual_hop_length != self.hop_length:
            raise ValueError(
                f"DAC model hop_length={actual_hop_length}, but Aoede is configured "
                f"for codec_hop_length={self.hop_length}. Re-run training with "
                f"`--codec-hop-length {actual_hop_length}`."
            )

    def validate(self) -> None:
        device = self._preferred_device or torch.device("cpu")
        self._load_model(device)

    def _target_device(self, tensor: torch.Tensor) -> torch.device:
        if self._preferred_device is not None:
            return self._preferred_device
        return tensor.device

    def _prepare_waveform(self, waveform: torch.Tensor, device: torch.device):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.dim() != 3:
            raise ValueError(
                f"Expected waveform with shape [T], [B, T], or [B, C, T], got {tuple(waveform.shape)}."
            )
        if waveform.shape[1] != 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        return waveform.to(device=device, dtype=torch.float32)

    def encode(self, waveform: torch.Tensor):
        source_device = waveform.device
        target_device = self._target_device(waveform)
        model = self._load_model(target_device)
        audio = self._prepare_waveform(waveform, target_device)
        with torch.no_grad():
            audio = model.preprocess(audio, self.sample_rate)
            z, _, _, _, _ = model.encode(audio)
        if z.dim() != 3:
            raise ValueError(f"DAC encode returned unexpected shape: {tuple(z.shape)}.")
        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"DAC encode returned latent_dim={z.shape[1]}, but Aoede expected {self.latent_dim}."
            )
        return z.transpose(1, 2).contiguous().to(source_device)

    def decode(self, latents: torch.Tensor):
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)
        if latents.dim() != 3:
            raise ValueError(
                f"Expected latents with shape [T, D] or [B, T, D], got {tuple(latents.shape)}."
            )
        source_device = latents.device
        target_device = self._target_device(latents)
        model = self._load_model(target_device)
        z = (
            latents.to(device=target_device, dtype=torch.float32)
            .transpose(1, 2)
            .contiguous()
        )
        with torch.no_grad():
            audio = model.decode(z)
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        if audio.dim() != 2:
            raise ValueError(
                f"DAC decode returned unexpected shape: {tuple(audio.shape)}."
            )
        return audio.to(source_device)

    def forward(self, waveform: torch.Tensor):
        return self.encode(waveform)


def normalize_codec_backend(name: str) -> str:
    normalized = name.strip().lower()
    aliases = {
        "fallback": "frozen",
        "deterministic": "frozen",
        "frozen_audio_codec": "frozen",
        "descript": "dac",
        "descript-audio-codec": "dac",
    }
    return aliases.get(normalized, normalized)


def build_audio_codec(
    config: ModelConfig,
    device: Optional[str | torch.device] = None,
):
    backend = normalize_codec_backend(config.codec_backend)
    if backend == "frozen":
        return FrozenAudioCodec(
            sample_rate=config.sample_rate,
            latent_dim=config.codec_latent_dim,
            frame_size=config.codec_frame_size,
            hop_length=config.codec_hop_length,
        )
    if backend == "dac":
        return DacAudioCodec(
            sample_rate=config.sample_rate,
            latent_dim=config.codec_latent_dim,
            hop_length=config.codec_hop_length,
            model_type=config.codec_model_type,
            model_path=config.codec_model_path,
            device=device,
        )
    raise ValueError(f"Unsupported audio codec backend: {config.codec_backend}")


def codec_cache_key(config: ModelConfig) -> str:
    backend = normalize_codec_backend(config.codec_backend)
    model_id = config.codec_model_type
    if config.codec_model_path:
        model_id = Path(config.codec_model_path).stem
    safe_model_id = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in model_id
    )
    return (
        f"{backend}_{safe_model_id}_codec{config.codec_latent_dim}"
        f"_hop{config.codec_hop_length}_spk{config.speaker_dim}"
    )
