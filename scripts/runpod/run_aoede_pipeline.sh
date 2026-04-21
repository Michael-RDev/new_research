#!/usr/bin/env bash

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ROOT_REPO_DIR_NAME="${ROOT_REPO_DIR_NAME:-new_research}"
ROOT_REPO_DIR="${ROOT_REPO_DIR:-${WORKSPACE}/${ROOT_REPO_DIR_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/.venv/bin/python}"

cd "${ROOT_REPO_DIR}"
if [ ! -x "${PYTHON_BIN}" ] || ! "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1; then
  echo "Workspace Python env is missing required dependencies; running bootstrap first."
  bash scripts/runpod/bootstrap_workspace.sh
fi

PYTHON_BIN="${ROOT_REPO_DIR}/.venv/bin/python"
source .venv/bin/activate
export PYTHONPATH="${ROOT_REPO_DIR}:${ROOT_REPO_DIR}/OmniVoice:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m aoede.runpod.pipeline \
  --workspace "${WORKSPACE}" \
  --root-repo-dir "${ROOT_REPO_DIR}" \
  --python-bin "${PYTHON_BIN}" \
  "$@"
