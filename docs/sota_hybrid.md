# Aoede SOTA Hybrid Voice-Cloning Workflow

Aoede's SOTA path is a hybrid system rather than a from-scratch TTS model. The
default runtime uses VoxCPM2 for human-quality cloning, DAC for codec latents,
SpeechBrain ECAPA-TDNN for speaker identity, and `sota_residualflow` as Aoede's
trainable residual refiner over teacher latents.

## Research Basis

- VoxCPM2 is the default teacher/runtime because its official project describes
  a permissive Apache-2.0, 2B parameter, 30-language, 48 kHz voice-cloning model:
  <https://github.com/OpenBMB/VoxCPM>.
- Qwen3-TTS remains optional because its official project provides Apache-2.0
  0.6B and 1.7B voice-clone models and a `qwen-tts` Python package:
  <https://github.com/QwenLM/Qwen3-TTS>.
- DAC is the codec target because its official implementation exposes
  MIT-licensed 16 kHz, 24 kHz, and 44.1 kHz encode/decode models:
  <https://github.com/descriptinc/descript-audio-codec>.
- SpeechBrain ECAPA-TDNN is the speaker backend because the model card provides
  an Apache-2.0 pretrained speaker-embedding model:
  <https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb>.
- F5-TTS is intentionally comparison/demo-only because its pretrained weights
  are documented as CC-BY-NC:
  <https://github.com/SWivid/F5-TTS>.

## Architecture

The training target is not raw waveform generation from scratch. Staging creates
paired latent supervision:

```text
reference audio + text
  -> VoxCPM2 teacher clone
  -> DAC teacher latents z_teacher

real dataset audio
  -> DAC real latents z_real

reference audio
  -> DAC reference latents
  -> SpeechBrain ECAPA speaker embedding
```

`sota_residualflow` then learns a residual velocity field between teacher and
real latents:

```text
z_t = (1 - t) * z_teacher + t * z_real + sigma(t) * eps
v*  = z_real - z_teacher
model(z_t, text, language, speaker, reference, teacher, t) -> v*
```

At inference time, `scripts/run_sota_clone.py --provider auto` first generates a
human-sounding VoxCPM2 teacher clone. If an Aoede checkpoint is provided, Aoede
tries to refine the teacher latents and decodes them through DAC. Auto mode
falls back to teacher audio unless the refined audio passes basic sanity and
speaker-similarity gates.

## RunPod Commands

Use these from a fresh RunPod shell:

```bash
apt update && apt install -y tmux
tmux new -s aoede-sota
```

Inside tmux:

```bash
cd /workspace

if [ ! -d /workspace/new_research/.git ]; then
  git clone https://github.com/Michael-RDev/new_research.git
fi

cd /workspace/new_research
git pull origin main

export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export DEVICE=cuda
export AOEDE_PROVIDER=voxcpm2
export AOEDE_TEACHER=voxcpm2
export AOEDE_SPEAKER_ENCODER=ecapa
export CODEC_BACKEND=dac
export CODEC_MODEL_TYPE=24khz
export CODEC_HOP_LENGTH=320
export CODEC_DEVICE=cuda

bash scripts/runpod/bootstrap_workspace.sh
bash scripts/bootstrap_sota_models.sh
mkdir -p artifacts/experiments/sota_residualflow_core
bash scripts/stage_sota_distill.sh core 2>&1 | tee artifacts/experiments/sota_residualflow_core/stage.log
bash scripts/train_sota_residualflow.sh core 2>&1 | tee artifacts/experiments/sota_residualflow_core/train.log
```

Detach tmux with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t aoede-sota
```

## Generation

After training:

```bash
cd /workspace/new_research
source .venv/bin/activate

python scripts/run_sota_clone.py \
  --provider auto \
  --model artifacts/experiments/sota_residualflow_core/artifacts/checkpoints/checkpoint-last.pt \
  --ref-audio me_voice.mp3 \
  --ref-text "EXACT TRANSCRIPT OF me_voice.mp3" \
  --text "This is the new Aoede SOTA hybrid voice cloning test." \
  --language en \
  --output-dir artifacts/sota_clone
```

For teacher-only output:

```bash
python scripts/run_sota_clone.py \
  --provider voxcpm2 \
  --ref-audio me_voice.mp3 \
  --ref-text "EXACT TRANSCRIPT OF me_voice.mp3" \
  --text "This is VoxCPM2 teacher-only cloning." \
  --language en \
  --output-dir artifacts/sota_clone
```

## Profiles

- `smoke`: tiny cache and 2 training steps, used to verify installs and tensor
  shapes.
- `core`: 1024 train examples, 64 eval examples, 20000 steps. This is the
  single-GPU default.
- `full`: all staged train examples, 256 eval examples, 100000 steps. This is
  much slower and should only be used after `core` sounds sane.

## Expected Behavior

The first quality gate is that VoxCPM2 teacher audio sounds human before Aoede
training. If teacher audio is bad, stop and fix provider/model/reference text
first. The Aoede refiner is only useful after the teacher cache is good.

If `auto` chooses teacher output, that is a safety behavior, not a failure. It
means Aoede did not pass the current local gate strongly enough to overwrite the
proven teacher audio.
