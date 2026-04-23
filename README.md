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

