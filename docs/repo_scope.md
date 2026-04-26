# Aoede Repo Scope

This repository is currently organized around the paper question in
`paper/main.tex`:

> Can a small residual flow model safely improve a strong permissive
> voice-cloning teacher in neural codec latent space under a formal
> non-regression constraint?

## Active Path

Use these files for the current Aoede research direction:

- `paper/main.tex`, `paper/evidence_matrix.md`, and `paper/scripts/`
- `aoede/providers.py`
- `aoede/audio/codec.py`, `aoede/audio/latent_stats.py`, and
  `aoede/audio/speaker.py`
- `aoede/data/sota_distill.py`
- `aoede/model/residualflow.py`
- `aoede/training/stage_sota_distill.py`
- `aoede/training/train_sota_residualflow.py`
- `scripts/bootstrap_sota_models.sh`
- `scripts/stage_sota_distill.sh`
- `scripts/train_sota_residualflow.sh`
- `scripts/run_sota_clone.py`
- `docs/sota_hybrid.md`
- `tests/test_sota_hybrid.py`, `tests/test_audio_codec.py`, and
  `tests/test_audio_io.py`

## Evidence To Keep

The current paper figures and tables are generated from preserved artifacts in
`results/aoede_sota_core/`. Do not delete that directory unless the paper is
regenerated from a newer run.

## Legacy / Not Main Claim

These areas are old scaffolding or comparison-only utilities. They are not the
current paper's primary claim path:

- MosaicFlow and AtlasFlow model variants
- old Aoede API/service demos
- F5 local demo wrapper
- OmniVoice comparison scripts and patches
- dual-RunPod orchestration
- old from-scratch training entry points

The legacy code is left in place for now because some tests and shared modules
still import it. Treat it as historical support code, not as the model described
by the paper.
