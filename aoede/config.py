from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    d_model: int = 384
    n_heads: int = 6
    n_text_layers: int = 12
    n_decoder_layers: int = 12
    semantic_layers: int = 4
    style_dim: int = 32
    speaker_dim: int = 192
    codec_latent_dim: int = 128
    codec_frame_size: int = 640
    codec_hop_length: int = 320
    sample_rate: int = 24000
    max_text_tokens: int = 512
    max_latent_frames: int = 1200
    duration_predictor_layers: int = 4
    architecture_variant: str = "mosaicflow"
    semantic_dim: int = 96
    semantic_stride: int = 4
    prompt_token_count: int = 8
    speaker_memory_tokens: int = 8
    planner_stride: int = 4
    planner_dim: int = 128
    memory_conditioning_heads: int = 6
    composer_layers: int = 2
    memory_dropout: float = 0.1
    planner_loss_weight: float = 0.1
    memory_speaker_loss_weight: float = 0.1
    semantic_loss_weight: float = 0.2
    prompt_loss_weight: float = 0.1
    coverage_loss_weight: float = 0.05


@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 200_000
    mixed_precision: bool = True
    speaker_loss_weight: float = 0.1
    duration_loss_weight: float = 0.1
    style_loss_weight: float = 0.01
    checkpoint_every: int = 1000
    eval_every: int = 500
    log_every: int = 50


@dataclass
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    runtime: str = "auto"
    stream_chunk_bytes: int = 16_384
    default_sampling_steps: int = 18


@dataclass
class ArtifactsConfig:
    root_dir: Path = Path("artifacts")
    voices_dir: Path = Path("artifacts/voices")
    checkpoints_dir: Path = Path("artifacts/checkpoints")
    tokenizer_path: Path = Path("artifacts/tokenizer.json")
    manifest_dir: Path = Path("artifacts/manifests")
    cache_dir: Path = Path("artifacts/cache")
    datasets_dir: Path = Path("artifacts/datasets")

    def ensure(self, base_dir: Path):
        for rel_path in (
            self.root_dir,
            self.voices_dir,
            self.checkpoints_dir,
            self.manifest_dir,
            self.cache_dir,
            self.datasets_dir,
        ):
            (base_dir / rel_path).mkdir(parents=True, exist_ok=True)
        tokenizer_parent = (base_dir / self.tokenizer_path).parent
        tokenizer_parent.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    project_root: Path = Path(".")
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)

    def ensure_directories(self):
        self.artifacts.ensure(self.project_root)

    def resolve(self, path: Path):
        return self.project_root / path

    def to_dict(self):
        raw = asdict(self)

        def _convert(value: Any):
            if isinstance(value, dict):
                return {key: _convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [_convert(item) for item in value]
            if isinstance(value, Path):
                return str(value)
            return value

        return _convert(raw)

    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(path.read_text())
        return cls(
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


def default_config(project_root: Optional[Path] = None):
    config = AppConfig(project_root=project_root or Path("."))
    config.ensure_directories()
    return config
