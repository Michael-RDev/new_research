#!/usr/bin/env bash

set -euo pipefail

ROOT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-${ROOT_REPO_DIR}}"
OMNIVOICE_DIR="${OMNIVOICE_DIR:-${ROOT_REPO_DIR}/OmniVoice}"
CLONEVAL_REPO="${CLONEVAL_REPO:-${WORKSPACE}/cloneval}"
CHECKPOINT="${CHECKPOINT:-${ROOT_REPO_DIR}/artifacts/experiments/mosaicflow_full/artifacts/checkpoints/checkpoint-last.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/exp/benchmark_compare_aoede_mosaicflow}"
CLONEVAL_TEST_LIST="${CLONEVAL_TEST_LIST:-}"
BENCHMARKS="${BENCHMARKS:-librispeech_pc seedtts_en seedtts_zh minimax}"
NUM_STEP="${NUM_STEP:-32}"
INFER_NJ_PER_GPU="${INFER_NJ_PER_GPU:-1}"
METRIC_NJ_PER_GPU="${METRIC_NJ_PER_GPU:-1}"
BATCH_DURATION="${BATCH_DURATION:-600.0}"
BATCH_SIZE="${BATCH_SIZE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [ -n "${VENV_DIR:-}" ]; then
  PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/${VENV_DIR}/bin/python}"
elif [ -d "${ROOT_REPO_DIR}/.venv_arm64" ]; then
  PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/.venv_arm64/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/.venv/bin/python}"
fi

cd "${ROOT_REPO_DIR}"
export PYTHONPATH="${ROOT_REPO_DIR}:${OMNIVOICE_DIR}:${PYTHONPATH:-}"
read -r -a BENCHMARK_ARRAY <<< "${BENCHMARKS}"

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
  --num-step "${NUM_STEP}"
  --infer-nj-per-gpu "${INFER_NJ_PER_GPU}"
  --metric-nj-per-gpu "${METRIC_NJ_PER_GPU}"
  --batch-duration "${BATCH_DURATION}"
  --batch-size "${BATCH_SIZE}"
  --benchmarks "${BENCHMARK_ARRAY[@]}"
)

if [ "${SKIP_EXISTING}" = "1" ]; then
  cmd+=(--skip-existing)
fi

if [ -n "${CLONEVAL_TEST_LIST}" ]; then
  cmd+=(--cloneval-test-list "${CLONEVAL_TEST_LIST}")
fi

"${cmd[@]}"
