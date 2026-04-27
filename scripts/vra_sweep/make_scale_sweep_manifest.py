#!/usr/bin/env python3
"""Build a compact manifest for VRA residual-scale sweep experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert an Aoede eval manifest into sweep JSONL.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prefer-speaker-ref", action="store_true")
    parser.add_argument("--allow-missing-audio", action="store_true")
    return parser


def _resolve(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _metadata_text(payload: dict) -> str:
    metadata = payload.get("metadata") or {}
    for key in ("speaker_ref_text", "reference_text", "prompt_text", "transcript", "text"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _iter_jsonl(path: Path):
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield line_number, json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc


def main() -> None:
    args = get_parser().parse_args()
    source = _resolve(args.source)
    output = _resolve(args.output)
    if source is None or output is None:
        raise ValueError("--source and --output are required")
    if not source.exists():
        raise FileNotFoundError(source)

    rows: list[dict[str, str]] = []
    skipped = 0
    skipped_missing_audio = 0
    for _, payload in _iter_jsonl(source):
        if args.limit is not None and len(rows) >= args.limit:
            break
        item_id = str(payload.get("item_id") or payload.get("id") or f"item_{len(rows):04d}")
        text = str(payload.get("text") or "").strip()
        language = str(payload.get("language") or payload.get("language_code") or "en").strip()

        preferred_audio = _resolve(payload.get("speaker_ref")) if args.prefer_speaker_ref else None
        if preferred_audio is None:
            preferred_audio = _resolve(payload.get("audio_path"))
        if preferred_audio is None:
            skipped += 1
            continue

        fallback_audio = _resolve(payload.get("audio_path"))
        if not preferred_audio.exists() and fallback_audio is not None and fallback_audio.exists():
            preferred_audio = fallback_audio

        if not text or not language:
            skipped += 1
            continue
        if not preferred_audio.exists() and not args.allow_missing_audio:
            skipped += 1
            skipped_missing_audio += 1
            continue

        rows.append(
            {
                "id": item_id,
                "text": text,
                "ref_audio": str(preferred_audio),
                "ref_text": _metadata_text(payload),
                "language": language,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "source": str(source),
                "output": str(output),
                "written": len(rows),
                "skipped": skipped,
                "skipped_missing_audio": skipped_missing_audio,
                "prefer_speaker_ref": bool(args.prefer_speaker_ref),
                "limit": args.limit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
