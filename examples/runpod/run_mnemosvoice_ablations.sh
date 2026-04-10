#!/usr/bin/env bash

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORKSPACE}/OmniVoice}"
CLONEVAL_REPO="${CLONEVAL_REPO:-${WORKSPACE}/cloneval}"
TEST_LIST="${TEST_LIST:-${WORKSPACE}/data/cloneval/cloneval_smoke.jsonl}"
DATA_CONFIG="${DATA_CONFIG:-${REPO_DIR}/examples/config/data_config_mnemosvoice_stage1_core.json}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${WORKSPACE}/exp/ablations}"

cd "${REPO_DIR}"
source .venv/bin/activate

declare -A CONFIGS=(
  ["full"]="examples/config/train_config_mnemosvoice_core_continue.json"
  ["no_speaker_memory"]="examples/config/train_config_mnemosvoice_ablate_no_speaker_memory.json"
  ["no_prosody_planner"]="examples/config/train_config_mnemosvoice_ablate_no_prosody_planner.json"
  ["no_speaker_loss"]="examples/config/train_config_mnemosvoice_ablate_no_speaker_loss.json"
)

for name in "${!CONFIGS[@]}"; do
  train_config="${CONFIGS[$name]}"
  output_dir="${BASE_OUTPUT_DIR}/${name}"

  accelerate launch -m omnivoice.cli.train \
    --train_config "${REPO_DIR}/${train_config}" \
    --data_config "${DATA_CONFIG}" \
    --output_dir "${output_dir}"

  python -m omnivoice.eval.cloneval_benchmark \
    --model "${output_dir}/checkpoint-last" \
    --test_list "${TEST_LIST}" \
    --work_dir "${output_dir}/cloneval" \
    --cloneval_repo "${CLONEVAL_REPO}" \
    --device_map cuda \
    --dtype float16 \
    --limit 2
done

python -m omnivoice.scripts.render_cloneval_report \
  --baseline_dir "${WORKSPACE}/exp/baseline_cloneeval_smoke/metrics" \
  --mnemosvoice_dir "${BASE_OUTPUT_DIR}/full/cloneval/metrics" \
  --ablation "No speaker memory=${BASE_OUTPUT_DIR}/no_speaker_memory/cloneval/metrics" \
  --ablation "No prosody planner=${BASE_OUTPUT_DIR}/no_prosody_planner/cloneval/metrics" \
  --ablation "No speaker loss=${BASE_OUTPUT_DIR}/no_speaker_loss/cloneval/metrics" \
  --output "${BASE_OUTPUT_DIR}/cloneval_ablation_report.md"

