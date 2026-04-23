#!/usr/bin/env bash

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ROOT_REPO_DIR_NAME="${ROOT_REPO_DIR_NAME:-new_research}"
ROOT_REPO_DIR="${ROOT_REPO_DIR:-${WORKSPACE}/${ROOT_REPO_DIR_NAME}}"

cd "${ROOT_REPO_DIR}"
PROFILE="${3:-full}"
exec bash scripts/stage_and_train_mosaicflow.sh "${PROFILE}"
