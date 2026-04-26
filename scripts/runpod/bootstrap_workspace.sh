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
OMNIVOICE_PATCH_MARKER="${OMNIVOICE_PATCH_MARKER:-${OMNIVOICE_DIR}/.aoede_patch_sha256}"
FORCE_OMNIVOICE_RECLONE="${FORCE_OMNIVOICE_RECLONE:-0}"
BOOTSTRAP_OMNIVOICE="${BOOTSTRAP_OMNIVOICE:-0}"
CLONEVAL_REPO_URL="${CLONEVAL_REPO_URL:-https://github.com/amu-cai/cloneval.git}"
CLONEVAL_REPO_BRANCH="${CLONEVAL_REPO_BRANCH:-main}"
CLONEVAL_DIR="${CLONEVAL_DIR:-${WORKSPACE}/cloneval}"
BOOTSTRAP_CLONEVAL="${BOOTSTRAP_CLONEVAL:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNPOD_TORCH_VERSION="${RUNPOD_TORCH_VERSION:-2.6.0}"
RUNPOD_TORCHAUDIO_VERSION="${RUNPOD_TORCHAUDIO_VERSION:-2.6.0}"
RUNPOD_TORCHVISION_VERSION="${RUNPOD_TORCHVISION_VERSION:-0.21.0}"
RUNPOD_TORCHCODEC_SPEC="${RUNPOD_TORCHCODEC_SPEC:-torchcodec==0.2.1}"
RUNPOD_DATASETS_VERSION="${RUNPOD_DATASETS_VERSION:-3.6.0}"

mkdir -p "${WORKSPACE}"

OMNIVOICE_BASE_COMMIT=""
if [ -f "${OMNIVOICE_BASE_COMMIT_FILE}" ]; then
  OMNIVOICE_BASE_COMMIT="$(tr -d '[:space:]' <"${OMNIVOICE_BASE_COMMIT_FILE}")"
fi

ensure_omnivoice_checkout() {
  if [ ! -d "${OMNIVOICE_DIR}/.git" ]; then
    rm -rf "${OMNIVOICE_DIR}"
    git clone --branch "${OMNIVOICE_REPO_BRANCH}" "${OMNIVOICE_REPO_URL}" "${OMNIVOICE_DIR}"
  fi

  if [ -n "${OMNIVOICE_BASE_COMMIT}" ]; then
    git -C "${OMNIVOICE_DIR}" checkout "${OMNIVOICE_BASE_COMMIT}"
  fi
}

apply_omnivoice_patch() {
  if [ ! -f "${OMNIVOICE_PATCH_FILE}" ]; then
    return 0
  fi

  local patch_sha
  patch_sha="$(sha256sum "${OMNIVOICE_PATCH_FILE}" | awk '{print $1}')"
  if [ -f "${OMNIVOICE_PATCH_MARKER}" ] && [ "$(cat "${OMNIVOICE_PATCH_MARKER}")" = "${patch_sha}" ]; then
    echo "OmniVoice patch already applied; reusing checkout."
    return 0
  fi

  if git -C "${OMNIVOICE_DIR}" apply --check --binary "${OMNIVOICE_PATCH_FILE}" >/dev/null 2>&1; then
    git -C "${OMNIVOICE_DIR}" apply --binary "${OMNIVOICE_PATCH_FILE}"
  elif git -C "${OMNIVOICE_DIR}" apply --check --reverse --binary "${OMNIVOICE_PATCH_FILE}" >/dev/null 2>&1; then
    echo "OmniVoice patch already present in checkout; recording marker."
  else
    echo "Existing OmniVoice checkout is not patchable cleanly; recloning."
    rm -rf "${OMNIVOICE_DIR}"
    ensure_omnivoice_checkout
    git -C "${OMNIVOICE_DIR}" apply --binary "${OMNIVOICE_PATCH_FILE}"
  fi

  printf '%s\n' "${patch_sha}" >"${OMNIVOICE_PATCH_MARKER}"
}

if [ ! -d "${ROOT_REPO_DIR}/.git" ]; then
  if [ -z "${ROOT_REPO_URL}" ]; then
    echo "ROOT_REPO_URL must be set when ${ROOT_REPO_DIR} does not exist."
    exit 1
  fi
  git clone --branch "${ROOT_REPO_BRANCH}" "${ROOT_REPO_URL}" "${ROOT_REPO_DIR}"
fi

if [ "${BOOTSTRAP_OMNIVOICE}" = "1" ] && [ "${FORCE_OMNIVOICE_RECLONE}" = "1" ] && [ -d "${OMNIVOICE_DIR}" ]; then
  rm -rf "${OMNIVOICE_DIR}"
fi

if [ "${BOOTSTRAP_OMNIVOICE}" = "1" ]; then
  ensure_omnivoice_checkout
  apply_omnivoice_patch
fi

if [ "${BOOTSTRAP_CLONEVAL}" = "1" ] && [ ! -d "${CLONEVAL_DIR}/.git" ]; then
  git clone --branch "${CLONEVAL_REPO_BRANCH}" "${CLONEVAL_REPO_URL}" "${CLONEVAL_DIR}"
fi

cd "${ROOT_REPO_DIR}"

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg
fi

if [ -d .venv ]; then
  rm -rf .venv || mv .venv ".venv.bad.$(date +%s)"
fi
${PYTHON_BIN} -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
python -m pip install \
  "torch==${RUNPOD_TORCH_VERSION}" \
  "torchaudio==${RUNPOD_TORCHAUDIO_VERSION}" \
  "torchvision==${RUNPOD_TORCHVISION_VERSION}" \
  --extra-index-url https://download.pytorch.org/whl/cu124
python -m pip install \
  "${RUNPOD_TORCHCODEC_SPEC}" \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[audio,training,dev,codec,sota]"
if [ "${BOOTSTRAP_OMNIVOICE}" = "1" ]; then
  python -m pip install -e "${OMNIVOICE_DIR}[eval,research]"
fi
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
  "${WORKSPACE}/exp" \
  "${WORKSPACE}/logs"

if [ "${BOOTSTRAP_CLONEVAL}" = "1" ]; then
  mkdir -p "${WORKSPACE}/data/cloneval"
fi

echo "Bootstrap complete."
echo "Root repo: ${ROOT_REPO_DIR}"
if [ "${BOOTSTRAP_OMNIVOICE}" = "1" ]; then
  echo "OmniVoice repo: ${OMNIVOICE_DIR}"
  if [ -f "${OMNIVOICE_PATCH_FILE}" ]; then
    echo "Applied OmniVoice patch: ${OMNIVOICE_PATCH_FILE}"
  fi
else
  echo "OmniVoice bootstrap skipped. Set BOOTSTRAP_OMNIVOICE=1 for legacy comparisons."
fi
if [ "${BOOTSTRAP_CLONEVAL}" = "1" ]; then
  echo "CloneEval repo: ${CLONEVAL_DIR}"
else
  echo "CloneEval bootstrap skipped. Set BOOTSTRAP_CLONEVAL=1 for CloneEval evaluation."
fi
echo "Activate: source ${ROOT_REPO_DIR}/.venv/bin/activate"
if [ "${BOOTSTRAP_OMNIVOICE}" = "1" ]; then
  echo "PYTHONPATH: export PYTHONPATH=${ROOT_REPO_DIR}:${OMNIVOICE_DIR}:\${PYTHONPATH:-}"
else
  echo "PYTHONPATH: export PYTHONPATH=${ROOT_REPO_DIR}:\${PYTHONPATH:-}"
fi
