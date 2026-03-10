#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
TORCH_LIB_REPLACEMENT="/opt/rocm/lib/libhsa-runtime64.so.1"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This setup script must run inside Linux. Use WSL2 Ubuntu 24.04." >&2
    exit 1
fi

if [[ -z "${WSL_INTEROP:-}" && "$(uname -r | tr '[:upper:]' '[:lower:]')" != *microsoft* ]]; then
    echo "Warning: WSL2 was not detected. Continuing, but this repo is only documented for ROCm on WSL2." >&2
fi

cd "${REPO_ROOT}"

uv venv --python 3.12 .venv
uv sync --python "${VENV_PYTHON}"
uv pip install --python "${VENV_PYTHON}" -r requirements-rocm-wsl.txt

TORCH_LIB_DIR="$("${VENV_PYTHON}" - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent / "lib")
PY
)"

if [[ -f "${TORCH_LIB_REPLACEMENT}" && -d "${TORCH_LIB_DIR}" ]]; then
    cp "${TORCH_LIB_REPLACEMENT}" "${TORCH_LIB_DIR}/libhsa-runtime64.so"
fi

"${VENV_PYTHON}" - <<'PY'
import platform
import torch

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False after ROCm install.")
if not getattr(torch.version, "hip", None):
    raise SystemExit("torch.version.hip is empty after ROCm install.")

print("Platform:", platform.platform())
print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("cuda_api_available:", torch.cuda.is_available())
print("device_name:", torch.cuda.get_device_name())
PY
