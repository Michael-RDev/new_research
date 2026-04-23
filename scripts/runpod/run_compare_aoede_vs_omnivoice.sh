#!/usr/bin/env bash

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ROOT_REPO_DIR_NAME="${ROOT_REPO_DIR_NAME:-new_research}"
ROOT_REPO_DIR="${ROOT_REPO_DIR:-${WORKSPACE}/${ROOT_REPO_DIR_NAME}}"

cd "${ROOT_REPO_DIR}"
exec bash scripts/run_benchmarks_mosaicflow.sh
