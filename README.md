# Aoede

Aoede is a teacher-anchored voice-cloning research scaffold.

The active research path is `sota_residualflow`: a permissive hybrid system
that uses VoxCPM2 as the default human-quality teacher, DAC as the codec target,
SpeechBrain ECAPA-TDNN as the speaker encoder, and a trainable Aoede
residual-flow refiner over teacher DAC latents. The paper frames this as a safe
adaptation problem: the teacher is the fallback, and Aoede only wins if the
refiner passes a non-regression gate.

## RunPod Quick Start

```bash
cd /workspace/new_research
bash scripts/runpod/bootstrap_workspace.sh
bash scripts/bootstrap_sota_models.sh
bash scripts/stage_sota_distill.sh core
bash scripts/train_sota_residualflow.sh core
```

Generate from the trained hybrid model:

```bash
python scripts/run_sota_clone.py \
  --provider auto \
  --model artifacts/experiments/sota_residualflow_core/artifacts/checkpoints/checkpoint-last.pt \
  --ref-audio me_voice.mp3 \
  --ref-text "EXACT TRANSCRIPT OF me_voice.mp3" \
  --text "This is the new Aoede SOTA hybrid voice cloning test." \
  --language en \
  --output-dir artifacts/sota_clone
```

`--provider auto` keeps the teacher output if the Aoede refiner fails or appears
to damage speaker similarity. Use `--provider voxcpm2` for teacher-only output,
or `--provider aoede-refiner` when you explicitly want the raw trained residual
refiner result.

Older MosaicFlow, AtlasFlow, F5, OmniVoice, and dual-RunPod pieces are legacy
scaffolding from earlier experiments. They are not the current paper direction
and should not be used for the main Aoede claims.

See `docs/sota_hybrid.md` for the full bootstrap, staging, training, and
generation workflow.

See `docs/repo_scope.md` for what is currently in scope versus legacy.
