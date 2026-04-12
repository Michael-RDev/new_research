#!/usr/bin/env bash

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ROOT_REPO_DIR_NAME="${ROOT_REPO_DIR_NAME:-new_research}"
ROOT_REPO_DIR="${ROOT_REPO_DIR:-${WORKSPACE}/${ROOT_REPO_DIR_NAME}}"
ROOT_REPO_URL="${ROOT_REPO_URL:-}"
ROOT_REPO_BRANCH="${ROOT_REPO_BRANCH:-main}"
OMNIVOICE_REPO_URL="${OMNIVOICE_REPO_URL:-https://github.com/k2-fsa/OmniVoice.git}"
OMNIVOICE_REPO_BRANCH="${OMNIVOICE_REPO_BRANCH:-master}"
OMNIVOICE_DIR="${OMNIVOICE_DIR:-${ROOT_REPO_DIR}/OmniVoice}"
OMNIVOICE_BASE_COMMIT_FILE="${OMNIVOICE_BASE_COMMIT_FILE:-${ROOT_REPO_DIR}/patches/omnivoice-base-commit.txt}"
OMNIVOICE_PATCH_FILE="${OMNIVOICE_PATCH_FILE:-${ROOT_REPO_DIR}/patches/omnivoice-local.patch}"
FORCE_OMNIVOICE_RECLONE="${FORCE_OMNIVOICE_RECLONE:-0}"
CLONEVAL_REPO_URL="${CLONEVAL_REPO_URL:-https://github.com/amu-cai/cloneval.git}"
CLONEVAL_REPO_BRANCH="${CLONEVAL_REPO_BRANCH:-main}"
CLONEVAL_DIR="${CLONEVAL_DIR:-${WORKSPACE}/cloneval}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNPOD_TORCH_VERSION="${RUNPOD_TORCH_VERSION:-2.5.1}"
RUNPOD_TORCHAUDIO_VERSION="${RUNPOD_TORCHAUDIO_VERSION:-2.5.1}"
RUNPOD_TORCHVISION_VERSION="${RUNPOD_TORCHVISION_VERSION:-0.20.1}"
RUNPOD_DATASETS_VERSION="${RUNPOD_DATASETS_VERSION:-3.6.0}"

mkdir -p "${WORKSPACE}"

if [ ! -d "${ROOT_REPO_DIR}/.git" ]; then
  if [ -z "${ROOT_REPO_URL}" ]; then
    echo "ROOT_REPO_URL must be set when ${ROOT_REPO_DIR} does not exist."
    exit 1
  fi
  git clone --branch "${ROOT_REPO_BRANCH}" "${ROOT_REPO_URL}" "${ROOT_REPO_DIR}"
fi

if [ "${FORCE_OMNIVOICE_RECLONE}" = "1" ] && [ -d "${OMNIVOICE_DIR}" ]; then
  rm -rf "${OMNIVOICE_DIR}"
fi

if [ ! -d "${OMNIVOICE_DIR}/.git" ]; then
  rm -rf "${OMNIVOICE_DIR}"
  git clone --branch "${OMNIVOICE_REPO_BRANCH}" "${OMNIVOICE_REPO_URL}" "${OMNIVOICE_DIR}"
fi

if [ -f "${OMNIVOICE_BASE_COMMIT_FILE}" ]; then
  OMNIVOICE_BASE_COMMIT="$(tr -d '[:space:]' < "${OMNIVOICE_BASE_COMMIT_FILE}")"
  if [ -n "${OMNIVOICE_BASE_COMMIT}" ]; then
    git -C "${OMNIVOICE_DIR}" checkout "${OMNIVOICE_BASE_COMMIT}"
  fi
fi

if [ -f "${OMNIVOICE_PATCH_FILE}" ]; then
  git -C "${OMNIVOICE_DIR}" apply --binary "${OMNIVOICE_PATCH_FILE}"
fi

if [ ! -d "${CLONEVAL_DIR}/.git" ]; then
  git clone --branch "${CLONEVAL_REPO_BRANCH}" "${CLONEVAL_REPO_URL}" "${CLONEVAL_DIR}"
fi

cd "${ROOT_REPO_DIR}"

${PYTHON_BIN} -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel
python -m pip install \
  "torch==${RUNPOD_TORCH_VERSION}" \
  "torchaudio==${RUNPOD_TORCHAUDIO_VERSION}" \
  "torchvision==${RUNPOD_TORCHVISION_VERSION}" \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[audio,training,dev]"
python -m pip install -e "${OMNIVOICE_DIR}[eval,research]"
python -m pip install -U "huggingface_hub[cli]"
python -m pip install "datasets==${RUNPOD_DATASETS_VERSION}"

if [ -n "${HF_TOKEN:-}" ]; then
  hf auth login --token "${HF_TOKEN}" --add-to-git-credential
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
  "${WORKSPACE}/data/cloneval" \
  "${WORKSPACE}/exp" \
  "${WORKSPACE}/logs"

echo "Bootstrap complete."
echo "Root repo: ${ROOT_REPO_DIR}"
echo "OmniVoice repo: ${OMNIVOICE_DIR}"
if [ -f "${OMNIVOICE_PATCH_FILE}" ]; then
  echo "Applied OmniVoice patch: ${OMNIVOICE_PATCH_FILE}"
fi
echo "CloneEval repo: ${CLONEVAL_DIR}"
echo "Activate: source ${ROOT_REPO_DIR}/.venv/bin/activate"
echo "PYTHONPATH: export PYTHONPATH=${ROOT_REPO_DIR}:${OMNIVOICE_DIR}:\${PYTHONPATH:-}"
