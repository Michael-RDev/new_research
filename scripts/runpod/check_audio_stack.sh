#!/usr/bin/env bash

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ROOT_REPO_DIR_NAME="${ROOT_REPO_DIR_NAME:-new_research}"
ROOT_REPO_DIR="${ROOT_REPO_DIR:-${WORKSPACE}/${ROOT_REPO_DIR_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_REPO_DIR}/.venv/bin/python}"
REQUIRE_TORCHCODEC="${AUDIO_PREFLIGHT_REQUIRE_TORCHCODEC:-0}"

echo "Audio preflight: checking ffmpeg, librosa, and torchcodec"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is not installed or not on PATH."
  exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "ERROR: ffprobe is not installed or not on PATH."
  exit 1
fi

echo "ffmpeg: $(ffmpeg -version | head -n 1)"
echo "ffprobe: $(ffprobe -version | head -n 1)"

"${PYTHON_BIN}" - <<'PY'
import importlib
import os
import sys


def check_module(module_name: str):
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"{module_name}: OK ({version})")
        return True
    except Exception as exc:
        print(f"{module_name}: FAIL ({exc})")
        return False


def check_torchcodec():
    try:
        module = importlib.import_module("torchcodec")
        version = getattr(module, "__version__", "unknown")
        from torchcodec._internally_replaced_utils import (
            load_torchcodec_shared_libraries,
        )

        load_torchcodec_shared_libraries()
        print(f"torchcodec: OK ({version})")
        return True
    except Exception as exc:
        required = os.environ.get("AUDIO_PREFLIGHT_REQUIRE_TORCHCODEC", "0") == "1"
        label = "FAIL" if required else "WARN"
        print(f"torchcodec: {label} ({exc})")
        return not required


ok = True
ok = check_module("librosa") and ok
ok = check_torchcodec() and ok

if not ok:
    sys.exit(1)
PY

echo "Audio preflight complete."
