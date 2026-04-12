#!/usr/bin/env bash

set -euo pipefail

stage="${1:-0}"
stop_stage="${2:-2}"
profile="${3:-smoke}"

WORKSPACE="${WORKSPACE:-/workspace}"
ROOT_REPO_DIR_NAME="${ROOT_REPO_DIR_NAME:-new_research}"
ROOT_REPO_DIR="${ROOT_REPO_DIR:-${WORKSPACE}/${ROOT_REPO_DIR_NAME}}"
OMNIVOICE_DIR="${OMNIVOICE_DIR:-${ROOT_REPO_DIR}/OmniVoice}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/.venv/bin/python}"

case "${profile}" in
  smoke)
    MANIFEST_ROOT="${WORKSPACE}/data/raw_manifests/stage1_smoke"
    AUDIO_ROOT="${WORKSPACE}/data/raw_audio/stage1_smoke"
    TOKEN_ROOT="${WORKSPACE}/data/tokens/stage1_smoke"
    DATA_CONFIG="${OMNIVOICE_DIR}/examples/config/data_config_mnemosvoice_smoke.json"
    TRAIN_CONFIG="${OMNIVOICE_DIR}/examples/config/train_config_aoede_smoke.json"
    OUTPUT_DIR="${WORKSPACE}/exp/aoede_smoke"
    ;;
  core)
    MANIFEST_ROOT="${WORKSPACE}/data/raw_manifests/stage1_core"
    AUDIO_ROOT="${WORKSPACE}/data/raw_audio/stage1_core"
    TOKEN_ROOT="${WORKSPACE}/data/tokens/stage1_core"
    DATA_CONFIG="${OMNIVOICE_DIR}/examples/config/data_config_mnemosvoice_stage1_core.json"
    TRAIN_CONFIG="${OMNIVOICE_DIR}/examples/config/train_config_aoede_core_continue.json"
    OUTPUT_DIR="${WORKSPACE}/exp/aoede_stage1_core"
    ;;
  *)
    echo "Unsupported profile: ${profile}. Use smoke or core."
    exit 1
    ;;
esac

cd "${ROOT_REPO_DIR}"
source .venv/bin/activate
export PYTHONPATH="${ROOT_REPO_DIR}:${OMNIVOICE_DIR}:${PYTHONPATH:-}"

source "${ROOT_REPO_DIR}/scripts/runpod/check_audio_stack.sh"

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
  echo "Stage 0: build ${profile} manifests"
  "${PYTHON_BIN}" -m omnivoice.scripts.build_hf_manifests \
    --stage stage1 \
    --profile "${profile}" \
    --manifest_root "${MANIFEST_ROOT}" \
    --audio_root "${AUDIO_ROOT}" \
    --env_file "${WORKSPACE}/.env" \
    --write_recipe_plan
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
  echo "Stage 1: tokenize ${profile} manifests"
  "${PYTHON_BIN}" -m omnivoice.scripts.orchestrate_tokenization \
    --manifest_root "${MANIFEST_ROOT}" \
    --token_root "${TOKEN_ROOT}" \
    --data_config_out "${DATA_CONFIG}" \
    --profile "${profile}"
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
  echo "Stage 2: train Aoede (${profile})"
  accelerate launch -m omnivoice.cli.train \
    --train_config "${TRAIN_CONFIG}" \
    --data_config "${DATA_CONFIG}" \
    --output_dir "${OUTPUT_DIR}"
fi
