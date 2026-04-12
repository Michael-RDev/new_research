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

mapfile -t audio_lib_dirs < <(
  "${PYTHON_BIN}" - <<'PY'
import site
from pathlib import Path

roots = []
for candidate in site.getsitepackages():
    p = Path(candidate)
    if p.exists():
        roots.append(p)

seen = set()
for root in roots:
    for rel in ("torch/lib",):
        path = root / rel
        if path.exists():
            resolved = str(path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                print(resolved)
    nvidia_root = root / "nvidia"
    if nvidia_root.exists():
        for child in sorted(nvidia_root.iterdir()):
            lib_dir = child / "lib"
            if lib_dir.exists():
                resolved = str(lib_dir.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    print(resolved)
PY
)

if [ "${#audio_lib_dirs[@]}" -gt 0 ]; then
  audio_ld_path="$(
    IFS=:
    echo "${audio_lib_dirs[*]}"
  )"
  export LD_LIBRARY_PATH="${audio_ld_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "LD_LIBRARY_PATH augmented with Python CUDA/audio libs"
fi

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
        from torchcodec.decoders import AudioDecoder  # noqa: F401
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
