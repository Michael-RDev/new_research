#!/usr/bin/env bash

set -euo pipefail

stage="${1:-0}"
stop_stage="${2:-2}"
profile="${3:-smoke}"

WORKSPACE="${WORKSPACE:-/workspace}"
ROOT_REPO_DIR_NAME="${ROOT_REPO_DIR_NAME:-new_research}"
ROOT_REPO_DIR="${ROOT_REPO_DIR:-${WORKSPACE}/${ROOT_REPO_DIR_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/.venv/bin/python}"
OMNIVOICE_INIT="${OMNIVOICE_INIT:-k2-fsa/OmniVoice}"

case "${profile}" in
  smoke)
    OUTPUT_DIR="${WORKSPACE}/exp/aoede_smoke"
    MAX_SAMPLES="${MAX_SAMPLES:-256}"
    MAX_STEPS="${MAX_STEPS:-500}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    HF_MAX_TRAIN_EXAMPLES="${HF_MAX_TRAIN_EXAMPLES:-128}"
    HF_MAX_EVAL_EXAMPLES="${HF_MAX_EVAL_EXAMPLES:-16}"
    ;;
  core)
    OUTPUT_DIR="${WORKSPACE}/exp/aoede_stage1_core"
    MAX_SAMPLES="${MAX_SAMPLES:-0}"
    MAX_STEPS="${MAX_STEPS:-20000}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    HF_MAX_TRAIN_EXAMPLES="${HF_MAX_TRAIN_EXAMPLES:-1024}"
    HF_MAX_EVAL_EXAMPLES="${HF_MAX_EVAL_EXAMPLES:-64}"
    ;;
  *)
    echo "Unsupported profile: ${profile}. Use smoke or core."
    exit 1
    ;;
esac

MANIFEST_PATH="${MANIFEST_PATH:-artifacts/manifests/train.jsonl}"

cd "${ROOT_REPO_DIR}"
if [ ! -x "${PYTHON_BIN}" ] || ! "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1; then
  echo "Workspace Python env is missing required dependencies; running bootstrap first."
  bash scripts/runpod/bootstrap_workspace.sh
fi

PYTHON_BIN="${ROOT_REPO_DIR}/.venv/bin/python"
source .venv/bin/activate
export PYTHONPATH="${ROOT_REPO_DIR}:${ROOT_REPO_DIR}/OmniVoice:${PYTHONPATH:-}"

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
  echo "Stage 0: prepare Aoede manifests and tokenizer"
  echo "Sampling up to ${HF_MAX_TRAIN_EXAMPLES} train and ${HF_MAX_EVAL_EXAMPLES} eval examples per configured source"
  "${PYTHON_BIN}" -m aoede.data.huggingface \
    --project-root "${ROOT_REPO_DIR}" \
    --max-train-examples "${HF_MAX_TRAIN_EXAMPLES}" \
    --max-eval-examples "${HF_MAX_EVAL_EXAMPLES}"
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
  echo "Stage 1: Aoede preprocessing is handled in train_aoede.py"
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
  echo "Stage 2: train unified Aoede (${profile})"
  "${PYTHON_BIN}" -m aoede.training.train_aoede \
    --source-manifest "${MANIFEST_PATH}" \
    --output-root "${OUTPUT_DIR}" \
    --architecture-variant atlasflow \
    --batch-size "${BATCH_SIZE}" \
    --max-steps "${MAX_STEPS}" \
    --max-samples "${MAX_SAMPLES}" \
    --init-from-omnivoice "${OMNIVOICE_INIT}"
fi
