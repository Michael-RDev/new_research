#!/usr/bin/env bash
set -euo pipefail

# Example usage:
#   bash examples/run_cloneval.sh \
#     /path/to/checkpoint_or_hf_repo \
#     /path/to/cloneval_test_list.jsonl \
#     /path/to/work_dir \
#     /path/to/cloneval

MODEL_PATH="${1:-k2-fsa/OmniVoice}"
TEST_LIST="${2:-data/cloneval_test_list.jsonl}"
WORK_DIR="${3:-exp/cloneval_eval}"
CLONEVAL_REPO="${4:-../cloneval}"

python -m omnivoice.eval.cloneval_benchmark \
  --model "${MODEL_PATH}" \
  --test_list "${TEST_LIST}" \
  --work_dir "${WORK_DIR}" \
  --cloneval_repo "${CLONEVAL_REPO}"
