from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aoede.audio.io import save_audio_bytes
from aoede.eval.common import LoadedAoedeModel


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an Aoede checkpoint as a zero-shot voice cloning model.",
    )
    parser.add_argument("--model", required=True, help="Checkpoint .pt or experiment root.")
    parser.add_argument("--ref-audio", required=True, help="Reference voice audio path.")
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--language", default="en", help="Target language code.")
    parser.add_argument("--project-root", default=None, help="Optional experiment root override.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or mps when available.")
    parser.add_argument("--num-step", type=int, default=18, help="Flow sampling steps.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/aoede_clone"),
        help="Directory for generated WAV output.",
    )
    parser.add_argument("--output-file", type=Path, default=None, help="Optional WAV path.")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    model = LoadedAoedeModel.load(
        args.model,
        project_root=args.project_root,
        device=args.device,
    )
    condition = model.prepare_voice_condition(args.ref_audio)
    audio = model.synthesize(
        text=args.text,
        language_code=args.language,
        condition=condition,
        sampling_steps=args.num_step,
    )

    if args.output_file is not None:
        output_path = args.output_file.expanduser().resolve()
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (args.output_dir / f"aoede_clone_{stamp}.wav").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        save_audio_bytes(audio, sample_rate=model.config.model.sample_rate)
    )
    print(output_path)


if __name__ == "__main__":
    main()
