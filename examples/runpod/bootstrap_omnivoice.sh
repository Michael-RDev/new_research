#!/usr/bin/env bash

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR_NAME="${REPO_DIR_NAME:-OmniVoice}"
REPO_DIR="${REPO_DIR:-${WORKSPACE}/${REPO_DIR_NAME}}"
REPO_URL="${REPO_URL:-}"
REPO_BRANCH="${REPO_BRANCH:-master}"
CLONEVAL_REPO_URL="${CLONEVAL_REPO_URL:-https://github.com/amu-cai/cloneval.git}"
CLONEVAL_REPO_BRANCH="${CLONEVAL_REPO_BRANCH:-main}"
CLONEVAL_DIR="${CLONEVAL_DIR:-${WORKSPACE}/cloneval}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${WORKSPACE}"

if [ ! -d "${REPO_DIR}/.git" ]; then
  if [ -z "${REPO_URL}" ]; then
    echo "REPO_URL must be set when ${REPO_DIR} does not exist."
    exit 1
  fi
  git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi

if [ ! -d "${CLONEVAL_DIR}/.git" ]; then
  git clone --branch "${CLONEVAL_REPO_BRANCH}" "${CLONEVAL_REPO_URL}" "${CLONEVAL_DIR}"
fi

cd "${REPO_DIR}"

${PYTHON_BIN} -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel
python -m pip install -e ".[eval,research]"

if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential
fi

mkdir -p \
  "${WORKSPACE}/data/raw_manifests/stage1_smoke" \
  "${WORKSPACE}/data/raw_manifests/stage1_core" \
  "${WORKSPACE}/data/raw_manifests/stage2_expansion" \
  "${WORKSPACE}/data/raw_audio/stage1_smoke" \
  "${WORKSPACE}/data/raw_audio/stage1_core" \
  "${WORKSPACE}/data/raw_audio/stage2_expansion" \
  "${WORKSPACE}/data/tokens/stage1_smoke" \
  "${WORKSPACE}/data/tokens/stage1_core" \
  "${WORKSPACE}/data/tokens/stage2_expansion" \
  "${WORKSPACE}/exp" \
  "${WORKSPACE}/logs"

echo "Bootstrap complete."
echo "Repo: ${REPO_DIR}"
echo "CloneEval: ${CLONEVAL_DIR}"
echo "Activate: source ${REPO_DIR}/.venv/bin/activate"

