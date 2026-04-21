# Dual RunPod Setup for Aoede Training and OmniVoice Evaluation

This workspace now includes a root-level dual-pod launcher and task scripts so
you can:

- run **Aoede training** on one pod
- run **released OmniVoice evaluation** on a second pod
- optionally run **Aoede vs OmniVoice comparison** later against the shared
  network volume

## Why a root-level launcher exists

The existing `OmniVoice` RunPod launcher clones only the nested `OmniVoice`
repository. That is enough for baseline OmniVoice work, but not for Aoede vs
OmniVoice comparison because the comparison flow also imports the root-level
`aoede` package.

The new launcher provisions two pods attached to the same RunPod network volume
and bootstraps a shared workspace that contains:

- the root repo under `/workspace/new_research`
- a real `OmniVoice` checkout under `/workspace/new_research/OmniVoice`
- the `cloneval` repository under `/workspace/cloneval`

If the root repo includes `patches/omnivoice-base-commit.txt` and
`patches/omnivoice-local.patch`, the bootstrap script checks out that upstream
`OmniVoice` base commit and reapplies the local Aoede/benchmark patch
automatically.

## Create the pods

Prepare an env file with:

```bash
RUNPOD_API_KEY=...
HF_TOKEN=...
```

Then dry-run the pod creation:

```bash
PYTHONPATH=/Users/michael/Desktop/new_research \
/Users/michael/Desktop/new_research/.venv_arm64/bin/python \
-m aoede.runpod.dual_pod_launcher create \
  --dry_run \
  --env_file /path/to/.env \
  --root_repo_url https://github.com/<you>/new_research.git \
  --root_repo_branch <your-branch> \
  --omnivoice_repo_url https://github.com/k2-fsa/OmniVoice.git \
  --omnivoice_repo_branch master
```

Create the real pods by removing `--dry_run`.

Default pod names:

- `aoede-train`
- `omnivoice-eval`

For the widest scheduling search, add `--any_data_center`. For the lower-cost
Community Cloud pricing tier, add `--cloud_type COMMUNITY`.

If RunPod rejects network volume creation because the account balance is below
their minimum threshold, create two independent pods instead:

```bash
PYTHONPATH=/Users/michael/Desktop/new_research \
/Users/michael/Desktop/new_research/.venv_arm64/bin/python \
-m aoede.runpod.dual_pod_launcher create \
  --env_file /path/to/.env \
  --root_repo_url https://github.com/<you>/new_research.git \
  --root_repo_branch <your-branch> \
  --any_data_center \
  --cloud_type COMMUNITY \
  --skip_network_volume \
  --pod_volume_gb 200
```

This keeps the training and evaluation pods separate, so checkpoints and logs do
not automatically appear on the other pod.

If you use `--skip_network_volume`, prefer **Stop Pod** when you are done for
the day. Stopping preserves the pod volume mounted at `/workspace`. **Terminate
Pod** deletes that workspace volume entirely.

## Connect to the pods

Use RunPod’s **Connect** tab and SSH into each pod. Official docs:

- [Connect to a Pod](https://docs.runpod.io/pods/connect-to-a-pod)
- [Connect to a Pod with SSH](https://docs.runpod.io/pods/configuration/override-public-keys)
- [Network volumes](https://docs.runpod.io/storage/network-volumes)

## Run Aoede training

On the training pod:

```bash
cd /workspace/new_research
source .venv/bin/activate
bash scripts/runpod/run_aoede_pipeline.sh --profile smoke
```

The single-script `smoke` profile stages a small sample from every configured
Hugging Face source, trains a smoke Aoede checkpoint, runs the standard
benchmark comparison, optionally runs CloneEval when you pass a test list, and
prints the exact `core` command to run later.

Default smoke staging caps:

- up to `128` train examples per source
- up to `16` eval examples per source

You can override the smoke staging caps explicitly:

```bash
bash scripts/runpod/run_aoede_pipeline.sh \
  --profile smoke \
  --max-train-examples 256 \
  --max-eval-examples 32
```

To include CloneEval in the smoke evaluation pass:

```bash
bash scripts/runpod/run_aoede_pipeline.sh \
  --profile smoke \
  --cloneval-test-list /workspace/data/cloneval/cloneval_smoke.jsonl
```

For the later larger stage-1 core run:

```bash
bash scripts/runpod/run_aoede_pipeline.sh --profile core
```

The `core` profile restages with larger caps before training and skips evals by
default:

- up to `1024` train examples per source
- up to `64` eval examples per source

## Run OmniVoice evaluation

On the evaluation pod:

```bash
cd /workspace/new_research
source .venv/bin/activate
bash scripts/runpod/run_omnivoice_eval.sh 1 6
```

This downloads the standard evaluation assets if needed and writes results under
`/workspace/exp/omnivoice_eval`.

## Compare Aoede to OmniVoice

After the Aoede checkpoint exists on the shared volume, run:

```bash
cd /workspace/new_research
source .venv/bin/activate
bash scripts/runpod/run_compare_aoede_vs_omnivoice.sh
```

If you also have a CloneEval JSONL:

```bash
CLONEVAL_TEST_LIST=/workspace/data/cloneval/cloneval_smoke.jsonl \
bash scripts/runpod/run_compare_aoede_vs_omnivoice.sh
```
