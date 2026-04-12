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

### Prepare Hugging Face training data

Aoede now includes a manifest builder for Hugging Face speech corpora. It can
materialize local WAV files under `artifacts/datasets/hf_audio/`, write merged
Aoede manifests to `artifacts/manifests/train.jsonl` and
`artifacts/manifests/eval.jsonl`, and fit a tokenizer from the resulting text.

```bash
aoede-prepare-hf
```

The built-in AtlasFlow mix includes:

- `galsenai/WaxalNLP`
- `MLCommons/peoples_speech`
- `amphion/Emilia-NV`
- `amphion/Emilia-Dataset`
- `google/fleurs`
- `facebook/multilingual_librispeech`
- `parler-tts/mls_eng_10k`

Notes:

- `amphion/Emilia-NV` requires accepting Hugging Face access terms and is
  research/non-commercial only.
- `amphion/Emilia-Dataset` is the multilingual Emilia / Emilia-YODAS
  WebDataset stream used for staged training. Review its dataset card access
  terms carefully because licensing varies across included subsets.
- Some corpora expose explicit `speaker_id` values and Aoede will link a
  `speaker_ref` clip automatically. Others fall back to Aoede's same-utterance
  reference slice for AtlasFlow training.
- You can override the built-in mix with repeated
  `--source source_id[:config_name[:split]]` arguments, including
  `--source emilia_dataset`.

### Frontend

```bash
cd apps/web
npm install
npm run dev
```
