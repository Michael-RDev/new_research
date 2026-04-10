#!/usr/bin/env bash

set -euo pipefail

stage="${1:-0}"
stop_stage="${2:-8}"

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORKSPACE}/OmniVoice}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
CLONEVAL_REPO="${CLONEVAL_REPO:-${WORKSPACE}/cloneval}"
BASELINE_MODEL="${BASELINE_MODEL:-k2-fsa/OmniVoice}"
MNEMOSVOICE_MODEL="${MNEMOSVOICE_MODEL:-${WORKSPACE}/exp/mnemosvoice_stage1_core/checkpoint-last}"
SMOKE_TEST_LIST="${SMOKE_TEST_LIST:-${WORKSPACE}/data/cloneval/cloneval_smoke.jsonl}"
FULL_TEST_LIST="${FULL_TEST_LIST:-${WORKSPACE}/data/cloneval/cloneval_full.jsonl}"

SMOKE_MANIFEST_ROOT="${WORKSPACE}/data/raw_manifests/stage1_smoke"
SMOKE_AUDIO_ROOT="${WORKSPACE}/data/raw_audio/stage1_smoke"
SMOKE_TOKEN_ROOT="${WORKSPACE}/data/tokens/stage1_smoke"

CORE_MANIFEST_ROOT="${WORKSPACE}/data/raw_manifests/stage1_core"
CORE_AUDIO_ROOT="${WORKSPACE}/data/raw_audio/stage1_core"
CORE_TOKEN_ROOT="${WORKSPACE}/data/tokens/stage1_core"

STAGE2_MANIFEST_ROOT="${WORKSPACE}/data/raw_manifests/stage2_expansion"
STAGE2_AUDIO_ROOT="${WORKSPACE}/data/raw_audio/stage2_expansion"
STAGE2_TOKEN_ROOT="${WORKSPACE}/data/tokens/stage2_expansion"

SMOKE_DATA_CONFIG="${REPO_DIR}/examples/config/data_config_mnemosvoice_smoke.json"
CORE_DATA_CONFIG="${REPO_DIR}/examples/config/data_config_mnemosvoice_stage1_core.json"

cd "${REPO_DIR}"
source .venv/bin/activate

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
  echo "Stage 0: build stage-1 smoke manifests"
  ${PYTHON_BIN} -m omnivoice.scripts.build_hf_manifests \
    --stage stage1 \
    --profile smoke \
    --manifest_root "${SMOKE_MANIFEST_ROOT}" \
    --audio_root "${SMOKE_AUDIO_ROOT}" \
    --env_file "${WORKSPACE}/.env" \
    --write_recipe_plan
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
  echo "Stage 1: tokenize stage-1 smoke manifests"
  ${PYTHON_BIN} -m omnivoice.scripts.orchestrate_tokenization \
    --manifest_root "${SMOKE_MANIFEST_ROOT}" \
    --token_root "${SMOKE_TOKEN_ROOT}" \
    --data_config_out "${SMOKE_DATA_CONFIG}" \
    --profile smoke
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
  echo "Stage 2: baseline CloneEval smoke"
  ${PYTHON_BIN} -m omnivoice.eval.cloneval_benchmark \
    --model "${BASELINE_MODEL}" \
    --test_list "${SMOKE_TEST_LIST}" \
    --work_dir "${WORKSPACE}/exp/baseline_cloneeval_smoke" \
    --cloneval_repo "${CLONEVAL_REPO}" \
    --device_map cuda \
    --dtype float16 \
    --limit 2
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
  echo "Stage 3: MnemosVoice smoke continue-train"
  accelerate launch -m omnivoice.cli.train \
    --train_config "${REPO_DIR}/examples/config/train_config_mnemosvoice_smoke.json" \
    --data_config "${SMOKE_DATA_CONFIG}" \
    --output_dir "${WORKSPACE}/exp/mnemosvoice_smoke"
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
  echo "Stage 4: MnemosVoice smoke CloneEval comparison"
  ${PYTHON_BIN} -m omnivoice.eval.cloneval_benchmark \
    --model "${WORKSPACE}/exp/mnemosvoice_smoke/checkpoint-last" \
    --test_list "${SMOKE_TEST_LIST}" \
    --work_dir "${WORKSPACE}/exp/mnemosvoice_cloneeval_smoke" \
    --cloneval_repo "${CLONEVAL_REPO}" \
    --device_map cuda \
    --dtype float16 \
    --limit 2
fi

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
  echo "Stage 5: build stage-1 core manifests"
  ${PYTHON_BIN} -m omnivoice.scripts.build_hf_manifests \
    --stage stage1 \
    --profile core \
    --manifest_root "${CORE_MANIFEST_ROOT}" \
    --audio_root "${CORE_AUDIO_ROOT}" \
    --env_file "${WORKSPACE}/.env" \
    --write_recipe_plan
fi

if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 6 ]; then
  echo "Stage 6: tokenize stage-1 core manifests"
  ${PYTHON_BIN} -m omnivoice.scripts.orchestrate_tokenization \
    --manifest_root "${CORE_MANIFEST_ROOT}" \
    --token_root "${CORE_TOKEN_ROOT}" \
    --data_config_out "${CORE_DATA_CONFIG}" \
    --profile core
fi

if [ "${stage}" -le 7 ] && [ "${stop_stage}" -ge 7 ]; then
  echo "Stage 7: MnemosVoice stage-1 core continue-train"
  accelerate launch -m omnivoice.cli.train \
    --train_config "${REPO_DIR}/examples/config/train_config_mnemosvoice_core_continue.json" \
    --data_config "${CORE_DATA_CONFIG}" \
    --output_dir "${WORKSPACE}/exp/mnemosvoice_stage1_core"
fi

if [ "${stage}" -le 8 ] && [ "${stop_stage}" -ge 8 ]; then
  echo "Stage 8: stage-2 manifest scaffolding"
  ${PYTHON_BIN} -m omnivoice.scripts.build_hf_manifests \
    --stage stage2 \
    --profile core \
    --manifest_root "${STAGE2_MANIFEST_ROOT}" \
    --audio_root "${STAGE2_AUDIO_ROOT}" \
    --env_file "${WORKSPACE}/.env" \
    --write_recipe_plan
fi

