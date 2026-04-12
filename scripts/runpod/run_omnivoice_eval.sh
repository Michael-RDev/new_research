#!/usr/bin/env bash

set -euo pipefail

stage="${1:-1}"
stop_stage="${2:-6}"

WORKSPACE="${WORKSPACE:-/workspace}"
ROOT_REPO_DIR_NAME="${ROOT_REPO_DIR_NAME:-new_research}"
ROOT_REPO_DIR="${ROOT_REPO_DIR:-${WORKSPACE}/${ROOT_REPO_DIR_NAME}}"
OMNIVOICE_DIR="${OMNIVOICE_DIR:-${ROOT_REPO_DIR}/OmniVoice}"

CHECKPOINT="${CHECKPOINT:-k2-fsa/OmniVoice}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

download_dir="${OMNIVOICE_DIR}/download"
results_root="${WORKSPACE}/exp/omnivoice_eval"
TTS_EVAL_MODEL_DIR="${download_dir}/tts_eval_models"
TTS_EVAL_DATA_DIR="${download_dir}/tts_eval_datasets"

cd "${OMNIVOICE_DIR}"
source "${ROOT_REPO_DIR}/.venv/bin/activate"
export PYTHONPATH="${ROOT_REPO_DIR}:${OMNIVOICE_DIR}:${PYTHONPATH:-}"

get_test_list() {
  case "$1" in
    librispeech_pc) echo "${TTS_EVAL_DATA_DIR}/librispeech_pc_test_clean.jsonl" ;;
    seedtts_en) echo "${TTS_EVAL_DATA_DIR}/seedtts_test_en.jsonl" ;;
    seedtts_zh) echo "${TTS_EVAL_DATA_DIR}/seedtts_test_zh.jsonl" ;;
    minimax) echo "${TTS_EVAL_DATA_DIR}/minimax_multilingual_24.jsonl" ;;
    fleurs) echo "${TTS_EVAL_DATA_DIR}/fleurs_multilingual_102.jsonl" ;;
    *) echo ""; return 1 ;;
  esac
}

infer_options=(
  --preprocess_prompt False
  --postprocess_output False
  --batch_duration 600
  --audio_chunk_threshold 1000
)

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
  echo "Stage 1: download evaluation datasets and models"
  mkdir -p "${TTS_EVAL_DATA_DIR}"
  for file in \
    librispeech_pc_test_clean.jsonl \
    librispeech_pc_test_clean_transcript.jsonl \
    seedtts_test_en.jsonl \
    seedtts_test_zh.jsonl \
    minimax_multilingual_24.jsonl \
    fleurs_multilingual_102.jsonl; do
    hf download \
      --repo-type dataset \
      --local-dir "${TTS_EVAL_DATA_DIR}" \
      k2-fsa/TTS_eval_datasets \
      "${file}"
  done

  for file in \
    librispeech_pc_testset.tar.gz \
    seedtts_testset.tar.gz \
    minimax_multilingual_24.tar.gz \
    fleurs_multilingual_102.tar.gz; do
    hf download \
      --repo-type dataset \
      --local-dir "${TTS_EVAL_DATA_DIR}" \
      k2-fsa/TTS_eval_datasets \
      "${file}"
    tar -xzf "${TTS_EVAL_DATA_DIR}/${file}" -C "${TTS_EVAL_DATA_DIR}"
  done

  mkdir -p "${TTS_EVAL_MODEL_DIR}"
  hf download \
    --local-dir "${TTS_EVAL_MODEL_DIR}" \
    k2-fsa/TTS_eval_models
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
  wav_path="${results_root}/librispeech_pc"
  test_jsonl="$(get_test_list librispeech_pc)"
  transcript_jsonl="${TTS_EVAL_DATA_DIR}/librispeech_pc_test_clean_transcript.jsonl"
  python -m omnivoice.cli.infer_batch --model "${CHECKPOINT}" --test_list "${test_jsonl}" --res_dir "${wav_path}" "${infer_options[@]}"
  python -m omnivoice.eval.speaker_similarity.sim --wav-path "${wav_path}" --test-list "${test_jsonl}" --decode-path "${wav_path}.sim.log" --model-dir "${TTS_EVAL_MODEL_DIR}"
  python -m omnivoice.eval.wer.hubert --wav-path "${wav_path}" --test-list "${transcript_jsonl}" --decode-path "${wav_path}.wer.log" --model-dir "${TTS_EVAL_MODEL_DIR}"
  python -m omnivoice.eval.mos.utmos --wav-path "${wav_path}" --test-list "${test_jsonl}" --decode-path "${wav_path}.mos.log" --model-dir "${TTS_EVAL_MODEL_DIR}"
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
  wav_path="${results_root}/seedtts_en"
  test_jsonl="$(get_test_list seedtts_en)"
  python -m omnivoice.cli.infer_batch --model "${CHECKPOINT}" --test_list "${test_jsonl}" --res_dir "${wav_path}" "${infer_options[@]}"
  python -m omnivoice.eval.speaker_similarity.sim --wav-path "${wav_path}" --test-list "${test_jsonl}" --decode-path "${wav_path}.sim.log" --model-dir "${TTS_EVAL_MODEL_DIR}"
  python -m omnivoice.eval.wer.seedtts --wav-path "${wav_path}" --test-list "${test_jsonl}" --decode-path "${wav_path}.wer.log" --model-dir "${TTS_EVAL_MODEL_DIR}" --lang en
  python -m omnivoice.eval.mos.utmos --wav-path "${wav_path}" --test-list "${test_jsonl}" --decode-path "${wav_path}.mos.log" --model-dir "${TTS_EVAL_MODEL_DIR}"
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
  wav_path="${results_root}/seedtts_zh"
  test_jsonl="$(get_test_list seedtts_zh)"
  python -m omnivoice.cli.infer_batch --model "${CHECKPOINT}" --test_list "${test_jsonl}" --res_dir "${wav_path}" "${infer_options[@]}"
  python -m omnivoice.eval.speaker_similarity.sim --wav-path "${wav_path}" --test-list "${test_jsonl}" --decode-path "${wav_path}.sim.log" --model-dir "${TTS_EVAL_MODEL_DIR}"
  python -m omnivoice.eval.wer.seedtts --wav-path "${wav_path}" --test-list "${test_jsonl}" --decode-path "${wav_path}.wer.log" --model-dir "${TTS_EVAL_MODEL_DIR}" --lang zh
  python -m omnivoice.eval.mos.utmos --wav-path "${wav_path}" --test-list "${test_jsonl}" --decode-path "${wav_path}.mos.log" --model-dir "${TTS_EVAL_MODEL_DIR}"
fi

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
  wav_path="${results_root}/minimax"
  test_jsonl="$(get_test_list minimax)"
  python -m omnivoice.cli.infer_batch --model "${CHECKPOINT}" --test_list "${test_jsonl}" --res_dir "${wav_path}" "${infer_options[@]}"
  python -m omnivoice.eval.speaker_similarity.sim --wav-path "${wav_path}" --test-list "${test_jsonl}" --decode-path "${wav_path}.sim.log" --model-dir "${TTS_EVAL_MODEL_DIR}"
  python -m omnivoice.eval.wer.minimax --wav-path "${wav_path}" --test-list "${test_jsonl}" --decode-path "${wav_path}.wer.log" --model-dir "${TTS_EVAL_MODEL_DIR}"
fi

if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 6 ]; then
  wav_path="${results_root}/fleurs"
  test_jsonl="$(get_test_list fleurs)"
  python -m omnivoice.cli.infer_batch --model "${CHECKPOINT}" --test_list "${test_jsonl}" --res_dir "${wav_path}" "${infer_options[@]}"
  python -m omnivoice.eval.speaker_similarity.sim --wav-path "${wav_path}" --test-list "${test_jsonl}" --decode-path "${wav_path}.sim.log" --model-dir "${TTS_EVAL_MODEL_DIR}"
  python -m omnivoice.eval.wer.fleurs --wav-path "${wav_path}" --test-list "${test_jsonl}" --decode-path "${wav_path}.wer.log"
fi
