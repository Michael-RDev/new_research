# Aoede

Aoede is a multilingual text-to-speech and zero-shot voice-cloning research scaffold.

The current default research architecture is `mosaicflow`, a prompt-conditioned model
that combines text/language encoding, reference-audio prompt factorization, semantic
planning, and acoustic flow matching over codec latents.

## RunPod Quick Start

```bash
cd /workspace/new_research
bash scripts/runpod/bootstrap_workspace.sh
bash scripts/stage_and_train_mosaicflow.sh full
```

`stage_and_train_mosaicflow.sh` defaults to `CODEC_BACKEND=dac`, which stages
training targets with the pretrained 24 kHz Descript Audio Codec instead of the
old deterministic fallback codec. Use `CODEC_BACKEND=frozen` only for offline
smoke tests or debugging; checkpoints trained with the fallback codec will not
produce realistic speech.
