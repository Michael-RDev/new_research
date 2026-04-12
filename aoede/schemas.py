from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


class StyleControls(BaseModel):
    pitch: float = Field(default=1.0, ge=0.5, le=1.5)
    pace: float = Field(default=1.0, ge=0.5, le=1.5)
    energy: float = Field(default=1.0, ge=0.5, le=1.5)
    brightness: float = Field(default=1.0, ge=0.5, le=1.5)


class VoiceProfile(BaseModel):
    voice_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    source: str = "enrollment"
    preset: str = "neutral"
    speaker_embedding: List[float]
    style_latent: List[float]
    speaker_memory: Optional[List[List[float]]] = None
    speaker_summary: Optional[List[float]] = None
    language_priors: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, str] = Field(default_factory=dict)
    controls: StyleControls = Field(default_factory=StyleControls)


class VoiceEnrollmentResponse(BaseModel):
    voice_id: str
    embedding_dim: int
    style_dim: int
    saved_to: str
    preview: List[float]


class VoiceListResponse(BaseModel):
    voices: List[VoiceProfile]


class VoiceDesignRequest(BaseModel):
    preset: str = "neutral"
    source_voice_id: Optional[str] = None
    voice_id: Optional[str] = None
    language_priors: Dict[str, float] = Field(default_factory=dict)
    style_controls: StyleControls = Field(default_factory=StyleControls)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SynthesisRequest(BaseModel):
    text: str
    language_code: str
    voice_id: Optional[str] = None
    inline_profile: Optional[VoiceProfile] = None
    style_controls: StyleControls = Field(default_factory=StyleControls)
    stream: bool = False
    sampling_steps: int = Field(default=18, ge=4, le=64)

    @model_validator(mode="after")
    def validate_voice_reference(self):
        if not self.voice_id and not self.inline_profile:
            raise ValueError("Provide either voice_id or inline_profile")
        return self


class LanguageDescriptor(BaseModel):
    code: str
    name: str
    family: str
    script: str
    production: bool


class LanguageListResponse(BaseModel):
    production: List[LanguageDescriptor]
    experimental: List[LanguageDescriptor]


class HealthResponse(BaseModel):
    status: str
    runtime: str
    model_ready: bool
    tokenizer_ready: bool
    voices: int
    checkpoint: Optional[str] = None


class StreamingEvent(BaseModel):
    type: str
    stage: str
    progress: float
    payload: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class TrainingExample:
    text: str
    language_code: str
    token_ids: Any
    waveform: Any
    codec_latents: Any
    durations: Any
    speaker_ref: Any
