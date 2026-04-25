#!/usr/bin/env bash

set -euo pipefail

ROOT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-full}"
ENV_FILE="${ENV_FILE:-.env}"
MANIFEST_PATH="${MANIFEST_PATH:-artifacts/manifests/train.jsonl}"
ARCHITECTURE_VARIANT="${ARCHITECTURE_VARIANT:-mosaicflow}"
RESUME_FROM="${RESUME_FROM:-}"
DEVICE="${DEVICE:-}"
CODEC_BACKEND="${CODEC_BACKEND:-dac}"
CODEC_MODEL_TYPE="${CODEC_MODEL_TYPE:-24khz}"
CODEC_MODEL_PATH="${CODEC_MODEL_PATH:-}"
CODEC_LATENT_DIM="${CODEC_LATENT_DIM:-}"
CODEC_HOP_LENGTH="${CODEC_HOP_LENGTH:-}"
CODEC_DEVICE="${CODEC_DEVICE:-}"
AOEDE_SPEAKER_ENCODER="${AOEDE_SPEAKER_ENCODER:-frozen}"
AOEDE_SPEAKER_MODEL_SOURCE="${AOEDE_SPEAKER_MODEL_SOURCE:-speechbrain/spkrec-ecapa-voxceleb}"

case "${PROFILE}" in
smoke)
  OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/experiments/mosaicflow_smoke}"
  HF_MAX_TRAIN_EXAMPLES="${HF_MAX_TRAIN_EXAMPLES:-128}"
  HF_MAX_EVAL_EXAMPLES="${HF_MAX_EVAL_EXAMPLES:-16}"
  MAX_SAMPLES="${MAX_SAMPLES:-256}"
  MAX_STEPS="${MAX_STEPS:-500}"
  BATCH_SIZE="${BATCH_SIZE:-2}"
  ;;
core)
  OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/experiments/mosaicflow_core}"
  HF_MAX_TRAIN_EXAMPLES="${HF_MAX_TRAIN_EXAMPLES:-1024}"
  HF_MAX_EVAL_EXAMPLES="${HF_MAX_EVAL_EXAMPLES:-64}"
  MAX_SAMPLES="${MAX_SAMPLES:-0}"
  MAX_STEPS="${MAX_STEPS:-20000}"
  BATCH_SIZE="${BATCH_SIZE:-4}"
  ;;
full)
  OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/experiments/mosaicflow_full}"
  HF_MAX_TRAIN_EXAMPLES="${HF_MAX_TRAIN_EXAMPLES:-0}"
  HF_MAX_EVAL_EXAMPLES="${HF_MAX_EVAL_EXAMPLES:-256}"
  MAX_SAMPLES="${MAX_SAMPLES:-0}"
  MAX_STEPS="${MAX_STEPS:-100000}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  ;;
*)
  echo "Unsupported profile: ${PROFILE}. Use smoke, core, or full."
  exit 1
  ;;
esac

if [ -n "${VENV_DIR:-}" ]; then
  PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/${VENV_DIR}/bin/python}"
elif [ -d "${ROOT_REPO_DIR}/.venv_arm64" ]; then
  PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/.venv_arm64/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/.venv/bin/python}"
fi

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python executable not found: ${PYTHON_BIN}"
  exit 1
fi

cd "${ROOT_REPO_DIR}"
export PYTHONPATH="${ROOT_REPO_DIR}:${ROOT_REPO_DIR}/OmniVoice:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}"

echo "Stage Hugging Face data for ${ARCHITECTURE_VARIANT} (${PROFILE})"
stage_cmd=(
  "${PYTHON_BIN}" -m aoede.data.huggingface
  --project-root "${ROOT_REPO_DIR}"
  --env-file "${ENV_FILE}"
  --max-train-examples "${HF_MAX_TRAIN_EXAMPLES}"
  --max-eval-examples "${HF_MAX_EVAL_EXAMPLES}"
)

set +e
"${stage_cmd[@]}"
stage_status=$?
set -e

if [ "${stage_status}" -ne 0 ]; then
  if [ -s "${MANIFEST_PATH}" ]; then
    echo "Staging exited with status ${stage_status}, but ${MANIFEST_PATH} exists; continuing to training."
  else
    echo "Staging failed and ${MANIFEST_PATH} was not created."
    exit "${stage_status}"
  fi
fi

train_cmd=(
  "${PYTHON_BIN}" -m aoede.training.train_aoede
  --source-manifest "${MANIFEST_PATH}"
  --output-root "${OUTPUT_ROOT}"
  --architecture-variant "${ARCHITECTURE_VARIANT}"
  --codec-backend "${CODEC_BACKEND}"
  --codec-model-type "${CODEC_MODEL_TYPE}"
  --speaker-encoder "${AOEDE_SPEAKER_ENCODER}"
  --speaker-model-source "${AOEDE_SPEAKER_MODEL_SOURCE}"
  --batch-size "${BATCH_SIZE}"
  --max-steps "${MAX_STEPS}"
  --max-samples "${MAX_SAMPLES}"
)

if [ -n "${CODEC_MODEL_PATH}" ]; then
  train_cmd+=(--codec-model-path "${CODEC_MODEL_PATH}")
fi

if [ -n "${CODEC_LATENT_DIM}" ]; then
  train_cmd+=(--codec-latent-dim "${CODEC_LATENT_DIM}")
fi

if [ -n "${CODEC_HOP_LENGTH}" ]; then
  train_cmd+=(--codec-hop-length "${CODEC_HOP_LENGTH}")
fi

if [ -n "${CODEC_DEVICE}" ]; then
  train_cmd+=(--codec-device "${CODEC_DEVICE}")
fi

if [ -n "${RESUME_FROM}" ]; then
  train_cmd+=(--resume-from "${RESUME_FROM}")
fi

if [ -n "${DEVICE}" ]; then
  train_cmd+=(--device "${DEVICE}")
fi

echo "Train ${ARCHITECTURE_VARIANT} -> ${OUTPUT_ROOT}"
"${train_cmd[@]}"
