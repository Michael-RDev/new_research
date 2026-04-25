#!/usr/bin/env bash

set -euo pipefail

ROOT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/${VENV_DIR}/bin/python}"
SOTA_DOWNLOAD_MODELS="${SOTA_DOWNLOAD_MODELS:-1}"
VOXCPM_MODEL_ID="${VOXCPM_MODEL_ID:-openbmb/VoxCPM2}"
ECAPA_MODEL_ID="${ECAPA_MODEL_ID:-speechbrain/spkrec-ecapa-voxceleb}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python executable not found: ${PYTHON_BIN}"
  exit 1
fi

cd "${ROOT_REPO_DIR}"
source "${ROOT_REPO_DIR}/${VENV_DIR}/bin/activate"

python -m pip install -e ".[audio,training,dev,codec,sota]"
python -m dac download --model_type "${CODEC_MODEL_TYPE:-24khz}" || true

if [ "${SOTA_DOWNLOAD_MODELS}" = "1" ]; then
  python - <<PY
from huggingface_hub import snapshot_download

snapshot_download("${VOXCPM_MODEL_ID}", local_dir="pretrained_models/VoxCPM2")
snapshot_download("${ECAPA_MODEL_ID}", local_dir="pretrained_models/spkrec-ecapa-voxceleb")
print("downloaded SOTA model assets")
PY
else
  echo "Skipping heavyweight model downloads because SOTA_DOWNLOAD_MODELS=${SOTA_DOWNLOAD_MODELS}."
fi
