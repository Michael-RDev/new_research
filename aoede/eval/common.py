from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from aoede.audio.io import load_audio_file
from aoede.audio.speaker import FrozenSpeakerEncoder
from aoede.config import (
    AppConfig,
    ArtifactsConfig,
    ModelConfig,
    ServiceConfig,
    TrainingConfig,
)
from aoede.languages import (
    experimental_languages,
    language_index,
    normalize_language,
    production_languages,
)
from aoede.model.core import AoedeModel
from aoede.text.tokenizer import UnicodeTokenizer


def app_config_from_dict(payload: dict) -> AppConfig:
    return AppConfig(
        project_root=Path(payload.get("project_root", ".")),
        model=ModelConfig(**payload.get("model", {})),
        training=TrainingConfig(**payload.get("training", {})),
        service=ServiceConfig(**payload.get("service", {})),
        artifacts=ArtifactsConfig(
            **{
                key: Path(value)
                for key, value in payload.get("artifacts", {}).items()
            }
        ),
    )

def peak_rss_bytes() -> int:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports kilobytes.
        if usage < 10_000_000:
            return int(usage * 1024)
        return int(usage)
    except Exception:
        return 0


@dataclass
class VoiceCondition:
    speaker_embedding: np.ndarray
    style_latent: np.ndarray
    speaker_memory: Optional[np.ndarray]
    speaker_summary: Optional[np.ndarray]


@dataclass
class LoadedAoedeModel:
    config: AppConfig
    checkpoint_path: Path
    tokenizer: UnicodeTokenizer
    speaker_encoder: FrozenSpeakerEncoder
    model: AoedeModel
    device: object
    torch: object

    @classmethod
    def load(
        cls,
        model_path: str,
        project_root: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "LoadedAoedeModel":
        import torch

        resolved_model_path = Path(model_path).expanduser().resolve()
        checkpoint_path, guessed_root = _resolve_checkpoint_path(
            resolved_model_path,
            project_root,
        )
        resolved_device = torch.device(device or _default_device())
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)

        config_payload = checkpoint.get("config")
        if config_payload is not None:
            config = app_config_from_dict(config_payload)
        else:
            config = AppConfig(project_root=guessed_root)

        if project_root is not None:
            config.project_root = Path(project_root).expanduser().resolve()
        elif not config.project_root.is_absolute():
            config.project_root = guessed_root

        tokenizer_path = config.resolve(config.artifacts.tokenizer_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Aoede tokenizer not found at {tokenizer_path}. "
                "Run aoede-prepare-hf or point --project-root at the trained Aoede workspace."
            )

        tokenizer = UnicodeTokenizer(tokenizer_path)
        model = AoedeModel(config.model).to(resolved_device)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return cls(
            config=config,
            checkpoint_path=checkpoint_path,
            tokenizer=tokenizer,
            speaker_encoder=FrozenSpeakerEncoder(
                embedding_dim=config.model.speaker_dim
            ),
            model=model,
            device=resolved_device,
            torch=torch,
        )

    def prepare_voice_condition(self, audio_path: str) -> VoiceCondition:
        audio, sample_rate = load_audio_file(
            audio_path,
            target_sample_rate=self.config.model.sample_rate,
        )
        speaker_embedding = self.speaker_encoder.encode(audio, sample_rate=sample_rate)
        waveform = (
            self.torch.from_numpy(audio)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        with self.torch.no_grad():
            latents = self.model.codec.encode(waveform)
            reference_mask = self.torch.ones(
                latents.shape[:2],
                dtype=self.torch.bool,
                device=latents.device,
            )
            style_latent = (
                self.model.infer_style(latents, reference_mask)
                .detach()
                .cpu()
                .numpy()[0]
                .astype(np.float32)
            )

            speaker_memory = None
            speaker_summary = None
            if self.model._atlasflow_enabled:
                memory, summary, _ = self.model.infer_reference_memory(
                    latents,
                    reference_mask,
                )
                if memory is not None:
                    speaker_memory = memory[0].detach().cpu().numpy().astype(np.float32)
                if summary is not None:
                    speaker_summary = summary[0].detach().cpu().numpy().astype(np.float32)

        return VoiceCondition(
            speaker_embedding=speaker_embedding.astype(np.float32),
            style_latent=style_latent,
            speaker_memory=speaker_memory,
            speaker_summary=speaker_summary,
        )

    def synthesize(
        self,
        text: str,
        language_code: str,
        condition: VoiceCondition,
        sampling_steps: int,
    ) -> np.ndarray:
        normalized_language = normalize_language(language_code)
        token_ids = self.tokenizer.encode(text, normalized_language)
        max_token_id = max(token_ids, default=0)
        if max_token_id >= self.config.model.vocab_size:
            raise ValueError(
                f"Tokenizer emitted token id {max_token_id}, but Aoede vocab_size is "
                f"{self.config.model.vocab_size}. Rebuild the tokenizer for this checkpoint."
            )

        token_tensor = self.torch.tensor(
            [token_ids],
            dtype=self.torch.long,
            device=self.device,
        )
        language_tensor = self.torch.tensor(
            [language_index(normalized_language)],
            dtype=self.torch.long,
            device=self.device,
        )
        speaker_tensor = self.torch.tensor(
            [condition.speaker_embedding.tolist()],
            dtype=self.torch.float32,
            device=self.device,
        )
        style_tensor = self.torch.tensor(
            [condition.style_latent.tolist()],
            dtype=self.torch.float32,
            device=self.device,
        )

        speaker_memory = None
        speaker_summary = None
        if condition.speaker_memory is not None:
            speaker_memory = self.torch.tensor(
                [condition.speaker_memory.tolist()],
                dtype=self.torch.float32,
                device=self.device,
            )
        if condition.speaker_summary is not None:
            speaker_summary = self.torch.tensor(
                [condition.speaker_summary.tolist()],
                dtype=self.torch.float32,
                device=self.device,
            )

        with self.torch.no_grad():
            waveform, _ = self.model.synthesize(
                token_ids=token_tensor,
                language_ids=language_tensor,
                speaker_embedding=speaker_tensor,
                style_latent=style_tensor,
                sampling_steps=sampling_steps,
                speaker_memory=speaker_memory,
                speaker_summary=speaker_summary,
            )
        return waveform[0].detach().cpu().numpy().astype(np.float32)


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_checkpoint_path(
    model_path: Path,
    project_root: Optional[str],
) -> tuple[Path, Path]:
    guessed_root = Path(project_root).expanduser().resolve() if project_root else None

    if model_path.is_dir():
        candidates = sorted(model_path.glob("*.pt"))
        if not candidates:
            candidates = sorted((model_path / "artifacts" / "checkpoints").glob("*.pt"))
        if not candidates and model_path.name == "checkpoints":
            candidates = sorted(model_path.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"No Aoede checkpoints found under {model_path}. "
                "Point --model at a checkpoint .pt file or an Aoede project root."
            )
        checkpoint_path = candidates[-1]
        if guessed_root is None:
            if (model_path / "artifacts").is_dir():
                guessed_root = model_path
            elif model_path.name == "checkpoints":
                guessed_root = model_path.parent.parent
            else:
                guessed_root = checkpoint_path.parent.parent
    else:
        checkpoint_path = model_path
        if guessed_root is None:
            if checkpoint_path.parent.name == "checkpoints":
                guessed_root = checkpoint_path.parent.parent
            else:
                guessed_root = checkpoint_path.parent

    return checkpoint_path.resolve(), guessed_root.resolve()
