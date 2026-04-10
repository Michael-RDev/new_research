# MnemosVoice Runpod Execution Guide

This guide turns the research plan into concrete commands for a Runpod pod with
`/workspace` storage.

## What was added

- `omnivoice.scripts.build_hf_manifests`
- `omnivoice.scripts.orchestrate_tokenization`
- `omnivoice.runpod.launcher`
- `examples/runpod/bootstrap_omnivoice.sh`
- `examples/runpod/run_mnemosvoice_pipeline.sh`
- committed smoke / core / ablation configs in `examples/config/`

## Why the launcher needs your repo URL

The official `k2-fsa/OmniVoice` repository does not contain the local
MnemosVoice changes. The Runpod launcher therefore accepts `--repo_url` and
`--repo_branch` explicitly so the pod can clone the branch that actually
contains the MnemosVoice implementation and benchmark bridge.

## Stage-1 datasets

Automated in `build_hf_manifests`:

- `amphion/Emilia-Dataset`
- `MLCommons/peoples_speech`
- `google/WaxalNLP`
- `facebook/multilingual_librispeech`

Policy:

- filter empty transcripts
- keep clips between `1.5s` and `20s`
- deduplicate by `(dataset_name, source_id)`
- People’s Speech uses only `clean` and `clean_sa`
- Waxal keeps only supervised subsets with deterministic language mapping
- Emilia creates a small validation slice via deterministic hash split plus a
  per-language minimum floor

## Stage-2 datasets

Partially automated:

- Common Voice 17.0 recipes are scaffolded
- LibriTTS-R, VCTK, MUSAN, and RIRS_NOISES are represented by local-manifest
  placeholders under `stage2_local_templates`

That keeps the config format stable while allowing stage-2 local corpora to be
added without changing the training scripts.

## Suggested sequence on Runpod

1. Provision the pod:

```bash
python -m omnivoice.runpod.launcher create \
  --env_file /workspace/.env \
  --repo_url https://github.com/<you>/<your-omnivoice-fork>.git \
  --repo_branch <your-branch> \
  --name mnemosvoice-stage1 \
  --gpu_type_id "NVIDIA A100 80GB PCIe" \
  --data_center_id US-KS-2
```

2. On the pod, bootstrap completes automatically via `dockerStartCmd`. If you
   want to rerun it manually:

```bash
cd /workspace/OmniVoice
bash examples/runpod/bootstrap_omnivoice.sh
```

3. Run the staged pipeline:

```bash
cd /workspace/OmniVoice
bash examples/runpod/run_mnemosvoice_pipeline.sh 0 4
```

That covers:

- stage-1 smoke manifest build
- smoke tokenization
- baseline CloneEval smoke
- MnemosVoice smoke continue-train
- MnemosVoice smoke CloneEval comparison

4. Continue to the core run:

```bash
bash examples/runpod/run_mnemosvoice_pipeline.sh 5 8
```

## Direct commands

Build smoke manifests:

```bash
python -m omnivoice.scripts.build_hf_manifests \
  --stage stage1 \
  --profile smoke \
  --manifest_root /workspace/data/raw_manifests/stage1_smoke \
  --audio_root /workspace/data/raw_audio/stage1_smoke \
  --env_file /workspace/.env
```

Tokenize and write a data config:

```bash
python -m omnivoice.scripts.orchestrate_tokenization \
  --manifest_root /workspace/data/raw_manifests/stage1_smoke \
  --token_root /workspace/data/tokens/stage1_smoke \
  --data_config_out /workspace/OmniVoice/examples/config/data_config_mnemosvoice_smoke.json \
  --profile smoke
```

Train the smoke checkpoint:

```bash
accelerate launch -m omnivoice.cli.train \
  --train_config /workspace/OmniVoice/examples/config/train_config_mnemosvoice_smoke.json \
  --data_config /workspace/OmniVoice/examples/config/data_config_mnemosvoice_smoke.json \
  --output_dir /workspace/exp/mnemosvoice_smoke
```

Run CloneEval:

```bash
python -m omnivoice.eval.cloneval_benchmark \
  --model /workspace/exp/mnemosvoice_smoke/checkpoint-last \
  --test_list /workspace/data/cloneval/cloneval_smoke.jsonl \
  --work_dir /workspace/exp/mnemosvoice_cloneeval_smoke \
  --cloneval_repo /workspace/cloneval \
  --device_map cuda \
  --dtype float16 \
  --limit 2
```

Render a comparison table:

```bash
python -m omnivoice.scripts.render_cloneval_report \
  --baseline_dir /workspace/exp/baseline_cloneeval_smoke/metrics \
  --mnemosvoice_dir /workspace/exp/mnemosvoice_cloneeval_smoke/metrics \
  --output /workspace/exp/reports/cloneval_smoke_report.md
```

