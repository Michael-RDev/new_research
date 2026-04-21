#!/usr/bin/env bash

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ROOT_REPO_DIR_NAME="${ROOT_REPO_DIR_NAME:-new_research}"
ROOT_REPO_DIR="${ROOT_REPO_DIR:-${WORKSPACE}/${ROOT_REPO_DIR_NAME}}"
OMNIVOICE_DIR="${OMNIVOICE_DIR:-${ROOT_REPO_DIR}/OmniVoice}"
CLONEVAL_REPO="${CLONEVAL_REPO:-${WORKSPACE}/cloneval}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/.venv/bin/python}"

CANDIDATE_MODEL="${CANDIDATE_MODEL:-${WORKSPACE}/exp/aoede_stage1_core/artifacts/checkpoints/checkpoint-last.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/exp/benchmark_compare}"
CLONEVAL_TEST_LIST="${CLONEVAL_TEST_LIST:-}"

cd "${ROOT_REPO_DIR}"
source .venv/bin/activate
export PYTHONPATH="${ROOT_REPO_DIR}:${OMNIVOICE_DIR}:${PYTHONPATH:-}"

cmd=(
  "${PYTHON_BIN}" -m omnivoice.scripts.benchmark_compare
  --baseline-model k2-fsa/OmniVoice
  --candidate-model "${CANDIDATE_MODEL}"
  --candidate-label Aoede
  --candidate-infer-module aoede.eval.infer_batch
  --candidate-cloneval-module aoede.eval.cloneval_benchmark
  --output-dir "${OUTPUT_DIR}"
  --prepare-assets
  --cloneval-repo "${CLONEVAL_REPO}"
)

if [ -n "${CLONEVAL_TEST_LIST}" ]; then
  cmd+=(--cloneval-test-list "${CLONEVAL_TEST_LIST}")
fi

"${cmd[@]}"
