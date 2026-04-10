# Aoede v1

Aoede is a staged multilingual TTS baseline that replaces the original
mel-diffusion prototype with a clearer package layout, typed APIs, a codec-latent
model scaffold, and a React + FastAPI demo surface.

## What is implemented

- A real Python package under `aoede/`
- Typed voice profile, synthesis, and training interfaces
- Production language registry for 16 launch languages
- Unicode tokenizer with language tokens and a SentencePiece-compatible API
- Frozen speaker encoder and audio codec wrappers with deterministic fallbacks
- Torch model scaffold for a speaker-conditioned latent flow-matching TTS model
- Manifest-driven data pipeline and trainer scaffold
- FastAPI service with voice enrollment, voice design, synthesis, health, and websocket streaming
- React demo scaffold with Enroll, Design, and Synthesize views

## Runtime notes

- The API intentionally supports two runtimes:
  - `mock`: pure NumPy fallback for machines without a working Torch runtime
  - `torch`: PyTorch model runtime for the training/inference scaffold
- On this machine the system Python has FastAPI but not a usable Torch build, so
  the API defaults to `mock` unless `AOEDE_RUNTIME=torch` is set while
  running under the local `.venv_arm64`.

## Layout

- `aoede/`: backend package
- `apps/web/`: React demo
- `legacy/assets_prototype/`: preserved original prototype
- `tests/`: API and unit coverage plus a Torch smoke script

## Quick start

### API

```bash
python -m aoede.api.app
```

### Torch smoke test

```bash
/Users/michael/Desktop/new_research/.venv_arm64/bin/python tests/model_smoke.py
```

### Frontend

```bash
cd apps/web
npm install
npm run dev
```
