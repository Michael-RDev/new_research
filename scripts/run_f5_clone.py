from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = REPO_ROOT / ".venv_arm64"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "f5_clone"


def _resolve_python(venv_path: Path) -> Path:
    python_path = venv_path / "bin" / "python"
    if not python_path.exists():
        raise FileNotFoundError(
            f"Python not found in virtualenv: {python_path}\n"
            "Expected an ARM venv at .venv_arm64."
        )
    return python_path


def _resolve_f5_cli(venv_path: Path) -> Path:
    cli_path = venv_path / "bin" / "f5-tts_infer-cli"
    if not cli_path.exists():
        raise FileNotFoundError(
            f"F5-TTS CLI not found: {cli_path}\n"
            "Install it first with:\n"
            f"  {venv_path / 'bin' / 'pip'} install f5-tts"
        )
    return cli_path


def _ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError(
            "ffmpeg is required to normalize reference audio before cloning, "
            "but it was not found on PATH."
        )
    return ffmpeg


def _normalize_audio(reference_audio: Path, ffmpeg: str, temp_dir: Path) -> Path:
    normalized = temp_dir / "reference.wav"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(reference_audio),
        "-ac",
        "1",
        "-ar",
        "24000",
        str(normalized),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return normalized


def _write_toml(
    *,
    model_name: str,
    reference_audio: Path,
    reference_text: str,
    generated_text: str,
    output_dir: Path,
    remove_silence: bool,
    speed: float,
    nfe_step: int,
    temp_dir: Path,
) -> Path:
    config_path = temp_dir / "f5_clone.toml"
    config_text = textwrap.dedent(
        f"""\
        model = "{model_name}"
        ref_audio = "{reference_audio}"
        ref_text = {reference_text!r}
        gen_text = {generated_text!r}
        gen_file = ""
        remove_silence = {"true" if remove_silence else "false"}
        output_dir = "{output_dir}"
        speed = {speed}
        nfe_step = {nfe_step}
        """
    )
    config_path.write_text(config_text)
    return config_path


def _pick_newest_output(output_dir: Path, started_at: float) -> Path | None:
    candidates = [
        path
        for path in output_dir.glob("*.wav")
        if path.is_file() and path.stat().st_mtime >= started_at - 1.0
    ]
    if not candidates:
        candidates = [path for path in output_dir.glob("*.wav") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clone a voice with the official F5-TTS CLI using a reference audio file and typed text."
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        default=REPO_ROOT / "me_voice.mp3",
        help="Reference audio file for voice cloning.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Text to synthesize in the reference voice.",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Exact transcript of the reference audio. Strongly recommended for quality.",
    )
    parser.add_argument(
        "--auto-ref-text",
        action="store_true",
        help=(
            "Allow F5-TTS to transcribe the reference audio automatically. "
            "This is less reliable than supplying the exact transcript."
        ),
    )
    parser.add_argument(
        "--model",
        default="F5TTS_v1_Base",
        help="F5-TTS model name passed through to the official CLI.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated audio should be written.",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=DEFAULT_VENV,
        help="Virtualenv that contains the F5-TTS package.",
    )
    parser.add_argument(
        "--remove-silence",
        action="store_true",
        help="Ask F5-TTS to trim silence in the generated file.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.85,
        help="Generation speed passed to F5-TTS. Lower values usually sound less rushed.",
    )
    parser.add_argument(
        "--nfe-step",
        type=int,
        default=32,
        help="Diffusion sampling steps passed to F5-TTS.",
    )
    args = parser.parse_args()

    reference_audio = args.ref_audio.expanduser().resolve()
    if not reference_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

    if args.ref_text is None and not args.auto_ref_text:
        raise SystemExit(
            "Ref text is required by default because auto-transcription often hurts "
            "voice-cloning quality.\n"
            "Either pass --ref-text \"...exact words from the reference clip...\" "
            "or explicitly opt into --auto-ref-text."
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _resolve_python(args.venv.expanduser().resolve())
    f5_cli = _resolve_f5_cli(args.venv.expanduser().resolve())
    ffmpeg = _ensure_ffmpeg()

    started_at = time.time()
    with tempfile.TemporaryDirectory(prefix="f5_clone_") as temp_name:
        temp_dir = Path(temp_name)
        normalized_audio = _normalize_audio(reference_audio, ffmpeg, temp_dir)
        config_path = _write_toml(
            model_name=args.model,
            reference_audio=normalized_audio,
            reference_text=args.ref_text or "",
            generated_text=args.text,
            output_dir=output_dir,
            remove_silence=args.remove_silence,
            speed=args.speed,
            nfe_step=args.nfe_step,
            temp_dir=temp_dir,
        )
        subprocess.run([str(f5_cli), "-c", str(config_path)], check=True)

    newest = _pick_newest_output(output_dir, started_at)
    if newest is None:
        print(
            f"F5-TTS finished, but no WAV file was found in {output_dir}.",
            file=sys.stderr,
        )
        return 1

    print(newest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
