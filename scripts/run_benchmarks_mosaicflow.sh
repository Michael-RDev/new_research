#!/usr/bin/env bash

set -euo pipefail

ROOT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-${ROOT_REPO_DIR}}"
OMNIVOICE_DIR="${OMNIVOICE_DIR:-${ROOT_REPO_DIR}/OmniVoice}"
CLONEVAL_REPO="${CLONEVAL_REPO:-${WORKSPACE}/cloneval}"
CHECKPOINT="${CHECKPOINT:-${ROOT_REPO_DIR}/artifacts/experiments/mosaicflow_full/artifacts/checkpoints/checkpoint-last.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/exp/benchmark_compare_aoede_mosaicflow}"
CLONEVAL_TEST_LIST="${CLONEVAL_TEST_LIST:-}"

if [ -n "${VENV_DIR:-}" ]; then
  PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/${VENV_DIR}/bin/python}"
elif [ -d "${ROOT_REPO_DIR}/.venv_arm64" ]; then
  PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/.venv_arm64/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/.venv/bin/python}"
fi

cd "${ROOT_REPO_DIR}"
export PYTHONPATH="${ROOT_REPO_DIR}:${OMNIVOICE_DIR}:${PYTHONPATH:-}"

cmd=(
  "${PYTHON_BIN}" -m omnivoice.scripts.benchmark_compare
  --baseline-model k2-fsa/OmniVoice
  --candidate-model "${CHECKPOINT}"
  --candidate-label Aoede-MosaicFlow
  --candidate-infer-module aoede.eval.infer_batch
  --candidate-cloneval-module aoede.eval.cloneval_benchmark
  --output-dir "${OUTPUT_DIR}"
  --cloneval-repo "${CLONEVAL_REPO}"
  --prepare-assets
  --benchmarks librispeech_pc seedtts_en seedtts_zh minimax
)

if [ -n "${CLONEVAL_TEST_LIST}" ]; then
  cmd+=(--cloneval-test-list "${CLONEVAL_TEST_LIST}")
fi

"${cmd[@]}"
