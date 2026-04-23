# F5 Voice Cloning

This repo now includes a wrapper around the official `f5-tts_infer-cli` so you can do real reference-audio voice cloning once the package is installed in `.venv_arm64`.

## Why this path

- It is a real pretrained voice-cloning model, not Aoede's mock runtime.
- The official F5-TTS project documents Apple Silicon installation and CLI inference.
- The wrapper normalizes reference audio with `ffmpeg`, writes a temporary F5 config, runs the official CLI, and prints the generated WAV path.

## One-time install

```bash
/Users/michael/Desktop/new_research/.venv_arm64/bin/pip install f5-tts
```

## Basic run

F5-TTS works best when you provide the exact words spoken in the reference clip.
The wrapper now requires `--ref-text` by default for that reason.

```bash
/Users/michael/Desktop/new_research/.venv_arm64/bin/python \
  /Users/michael/Desktop/new_research/scripts/run_f5_clone.py \
  --ref-audio /Users/michael/Desktop/new_research/me_voice.mp3 \
  --ref-text "The exact words spoken in the reference clip." \
  --text "This is a real voice cloning test."
```

For a slightly less rushed delivery, you can slow the model down a bit:

```bash
/Users/michael/Desktop/new_research/.venv_arm64/bin/python \
  /Users/michael/Desktop/new_research/scripts/run_f5_clone.py \
  --ref-audio /Users/michael/Desktop/new_research/me_voice.mp3 \
  --ref-text "The exact words spoken in the reference clip." \
  --text "Now say this new sentence in my voice." \
  --speed 0.8
```

If you really want automatic transcription of the reference clip, you can opt in explicitly:

```bash
/Users/michael/Desktop/new_research/.venv_arm64/bin/python \
  /Users/michael/Desktop/new_research/scripts/run_f5_clone.py \
  --ref-audio /Users/michael/Desktop/new_research/me_voice.mp3 \
  --auto-ref-text \
  --text "This is a fallback test."
```

## Output

- Generated files are written to `/Users/michael/Desktop/new_research/artifacts/f5_clone`.
- The wrapper prints the newest generated WAV path on success.

## Notes

- Auto-transcribing the reference clip is convenient, but it is much less reliable than supplying the exact transcript yourself.
- A clean 4-8 second reference clip with one speaker and minimal pauses usually works better than a longer rambling clip.
- The wrapper uses `ffmpeg` to convert the reference clip to mono 24 kHz WAV before inference.
- If `f5-tts` is not installed, the wrapper will exit with a clear error telling you which install command to run.
