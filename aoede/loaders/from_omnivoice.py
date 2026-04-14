from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class OmniVoiceTransferReport:
    source: str
    transferred: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class OmniVoiceState:
    state_dict: dict[str, torch.Tensor]
    token_embeddings: Optional[torch.Tensor] = None
    audio_embeddings: Optional[torch.Tensor] = None
    audio_heads: Optional[torch.Tensor] = None


def _extract_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict) and value:
                return value
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return payload
    raise ValueError("Could not locate tensor state_dict in checkpoint payload")


def _find_first_tensor(
    state_dict: dict[str, torch.Tensor],
    patterns: list[str],
) -> Optional[torch.Tensor]:
    for key, tensor in state_dict.items():
        if all(pattern in key for pattern in patterns):
            return tensor
    return None


def _find_by_keys(
    state_dict: dict[str, torch.Tensor],
    keys: list[str],
) -> Optional[torch.Tensor]:
    for key in keys:
        if key in state_dict:
            return state_dict[key]
    return None


def _load_state_from_path(path: Path) -> OmniVoiceState:
    if path.is_file():
        payload = torch.load(path, map_location="cpu")
        state_dict = _extract_state_dict(payload)
        return OmniVoiceState(state_dict=state_dict)

    try:
        from omnivoice.models.omnivoice import OmniVoice

        model = OmniVoice.from_pretrained(
            str(path),
            train=True,
            load_asr=False,
            dtype=torch.float32,
        )
        return OmniVoiceState(
            state_dict=model.state_dict(),
            token_embeddings=model.llm.get_input_embeddings().weight.detach().cpu(),
            audio_embeddings=model.audio_embeddings.weight.detach().cpu(),
            audio_heads=model.audio_heads.weight.detach().cpu(),
        )
    except Exception:
        pass

    for filename in ("pytorch_model.bin", "model.bin", "model.pt"):
        candidate = path / filename
        if candidate.exists():
            payload = torch.load(candidate, map_location="cpu")
            state_dict = _extract_state_dict(payload)
            return OmniVoiceState(state_dict=state_dict)

    raise FileNotFoundError(f"No loadable OmniVoice checkpoint found at {path}")


def load_omnivoice_state(source: str) -> OmniVoiceState:
    source_path = Path(source).expanduser()
    if source_path.exists():
        state = _load_state_from_path(source_path.resolve())
    else:
        from omnivoice.models.omnivoice import OmniVoice

        model = OmniVoice.from_pretrained(
            source,
            train=True,
            load_asr=False,
            dtype=torch.float32,
        )
        state = OmniVoiceState(
            state_dict=model.state_dict(),
            token_embeddings=model.llm.get_input_embeddings().weight.detach().cpu(),
            audio_embeddings=model.audio_embeddings.weight.detach().cpu(),
            audio_heads=model.audio_heads.weight.detach().cpu(),
        )

    if state.token_embeddings is None:
        state.token_embeddings = _find_by_keys(
            state.state_dict,
            keys=[
                "llm.embed_tokens.weight",
                "llm.model.embed_tokens.weight",
                "model.embed_tokens.weight",
                "embed_tokens.weight",
            ],
        )
    if state.token_embeddings is None:
        state.token_embeddings = _find_first_tensor(
            state.state_dict,
            patterns=["embed_tokens", "weight"],
        )
    if state.audio_embeddings is None:
        state.audio_embeddings = _find_by_keys(
            state.state_dict,
            keys=["audio_embeddings.weight"],
        )
    if state.audio_embeddings is None:
        state.audio_embeddings = _find_first_tensor(
            state.state_dict,
            patterns=["audio_embeddings", "weight"],
        )
    if state.audio_heads is None:
        state.audio_heads = _find_by_keys(
            state.state_dict,
            keys=["audio_heads.weight"],
        )
    if state.audio_heads is None:
        state.audio_heads = _find_first_tensor(
            state.state_dict,
            patterns=["audio_heads", "weight"],
        )

    return state


def _copy_overlap(
    dst: torch.Tensor,
    src: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if dst.ndim != src.ndim:
        raise ValueError(f"Rank mismatch: dst rank {dst.ndim}, src rank {src.ndim}")
    slices = tuple(
        slice(0, min(dst_dim, src_dim)) for dst_dim, src_dim in zip(dst.shape, src.shape)
    )
    updated = dst.clone()
    updated[slices] = src[slices].to(updated.dtype)
    copied_shape = tuple(s.stop for s in slices)
    return updated, copied_shape


def initialize_aoede_from_omnivoice(
    model: torch.nn.Module,
    source: str,
) -> OmniVoiceTransferReport:
    state = load_omnivoice_state(source)
    report = OmniVoiceTransferReport(source=source)

    token_embeddings = state.token_embeddings
    if token_embeddings is None:
        report.warnings.append("Token embedding tensor was not found in OmniVoice checkpoint")
    else:
        try:
            dst = model.text_encoder.token_embedding.weight.data
            updated, copied = _copy_overlap(dst, token_embeddings)
            model.text_encoder.token_embedding.weight.data.copy_(updated)
            report.transferred.append(f"text_encoder.token_embedding.weight overlap={copied}")

            emb_std = float(token_embeddings.float().std().item())
            if emb_std > 0:
                with torch.no_grad():
                    model.text_encoder.position_embedding.mul_(0.0).normal_(std=emb_std * 0.5)
            report.transferred.append("text_encoder.position_embedding rescaled")
        except Exception as exc:
            report.skipped.append(f"token embedding transfer failed: {exc}")

    audio_embeddings = state.audio_embeddings
    audio_heads = state.audio_heads
    if audio_embeddings is None and audio_heads is None:
        report.warnings.append("No OmniVoice acoustic tensors found for decoder warm-start")
    else:
        with torch.no_grad():
            if audio_embeddings is not None:
                audio_std = float(audio_embeddings.float().std().item())
                audio_mean = float(audio_embeddings.float().mean().item())
                if audio_std > 0:
                    model.decoder.latents_in.weight.normal_(mean=audio_mean, std=audio_std)
                    model.decoder.latents_in.bias.zero_()
                    report.transferred.append(
                        "decoder.latents_in initialized from OmniVoice audio_embeddings stats"
                    )
            if audio_heads is not None:
                head_std = float(audio_heads.float().std().item())
                head_mean = float(audio_heads.float().mean().item())
                final_linear = model.decoder.out[1]
                if head_std > 0:
                    final_linear.weight.normal_(mean=head_mean, std=head_std)
                    if final_linear.bias is not None:
                        final_linear.bias.zero_()
                    report.transferred.append(
                        "decoder.out initialized from OmniVoice audio_heads stats"
                    )

    return report
