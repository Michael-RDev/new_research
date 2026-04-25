#!/usr/bin/env bash

set -euo pipefail

ROOT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-core}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/${VENV_DIR}/bin/python}"

case "${PROFILE}" in
smoke)
  OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/sota_distill_smoke}"
  SOTA_MAX_TRAIN_EXAMPLES="${SOTA_MAX_TRAIN_EXAMPLES:-8}"
  SOTA_MAX_EVAL_EXAMPLES="${SOTA_MAX_EVAL_EXAMPLES:-2}"
  ;;
core)
  OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/sota_distill_core}"
  SOTA_MAX_TRAIN_EXAMPLES="${SOTA_MAX_TRAIN_EXAMPLES:-1024}"
  SOTA_MAX_EVAL_EXAMPLES="${SOTA_MAX_EVAL_EXAMPLES:-64}"
  ;;
full)
  OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/sota_distill_full}"
  SOTA_MAX_TRAIN_EXAMPLES="${SOTA_MAX_TRAIN_EXAMPLES:-0}"
  SOTA_MAX_EVAL_EXAMPLES="${SOTA_MAX_EVAL_EXAMPLES:-256}"
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

"${PYTHON_BIN}" -m aoede.training.stage_sota_distill \
  --train-manifest "${TRAIN_MANIFEST:-artifacts/manifests/train.jsonl}" \
  --eval-manifest "${EVAL_MANIFEST:-artifacts/manifests/eval.jsonl}" \
  --output-root "${OUTPUT_ROOT}" \
  --provider "${AOEDE_TEACHER:-voxcpm2}" \
  --provider-model-id "${AOEDE_TEACHER_MODEL_ID:-}" \
  --device "${DEVICE:-cuda}" \
  --codec-backend "${CODEC_BACKEND:-dac}" \
  --codec-model-type "${CODEC_MODEL_TYPE:-24khz}" \
  --codec-hop-length "${CODEC_HOP_LENGTH:-320}" \
  --codec-latent-dim "${CODEC_LATENT_DIM:-1024}" \
  --speaker-encoder "${AOEDE_SPEAKER_ENCODER:-ecapa}" \
  --max-train-examples "${SOTA_MAX_TRAIN_EXAMPLES}" \
  --max-eval-examples "${SOTA_MAX_EVAL_EXAMPLES}"
