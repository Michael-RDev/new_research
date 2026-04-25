#!/usr/bin/env bash

set -euo pipefail

ROOT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-core}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/${VENV_DIR}/bin/python}"

case "${PROFILE}" in
smoke)
  DISTILL_ROOT="${DISTILL_ROOT:-artifacts/sota_distill_smoke}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/experiments/sota_residualflow_smoke}"
  MAX_STEPS="${MAX_STEPS:-2}"
  BATCH_SIZE="${BATCH_SIZE:-2}"
  ;;
core)
  DISTILL_ROOT="${DISTILL_ROOT:-artifacts/sota_distill_core}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/experiments/sota_residualflow_core}"
  MAX_STEPS="${MAX_STEPS:-20000}"
  BATCH_SIZE="${BATCH_SIZE:-2}"
  ;;
full)
  DISTILL_ROOT="${DISTILL_ROOT:-artifacts/sota_distill_full}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/experiments/sota_residualflow_full}"
  MAX_STEPS="${MAX_STEPS:-100000}"
  BATCH_SIZE="${BATCH_SIZE:-4}"
  ;;
*)
  echo "Unsupported profile: ${PROFILE}. Use smoke, core, or full."
  exit 1
  ;;
esac

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python executable not found: ${PYTHON_BIN}"
  exit 1
fi

cd "${ROOT_REPO_DIR}"
export PYTHONPATH="${ROOT_REPO_DIR}:${ROOT_REPO_DIR}/OmniVoice:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}"

train_cmd=(
  "${PYTHON_BIN}" -m aoede.training.train_sota_residualflow
  --train-manifest "${DISTILL_ROOT}/train.sota.jsonl"
  --eval-manifest "${DISTILL_ROOT}/eval.sota.jsonl"
  --latent-stats "${DISTILL_ROOT}/latent_stats.json"
  --tokenizer-path "${TOKENIZER_PATH:-artifacts/tokenizer.json}"
  --output-root "${OUTPUT_ROOT}"
  --device "${DEVICE:-cuda}"
  --batch-size "${BATCH_SIZE}"
  --max-steps "${MAX_STEPS}"
  --learning-rate "${LEARNING_RATE:-1e-4}"
  --checkpoint-every "${CHECKPOINT_EVERY:-250}"
  --eval-every "${EVAL_EVERY:-500}"
  --codec-latent-dim "${CODEC_LATENT_DIM:-1024}"
)

if [ -n "${RESUME_FROM:-}" ]; then
  train_cmd+=(--resume-from "${RESUME_FROM}")
fi

"${train_cmd[@]}"
