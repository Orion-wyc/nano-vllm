#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="${NANO_VLLM_ENV:-/mnt/workspace/Envs/nano_vllm_a3}"
CANN_SET_ENV="${CANN_SET_ENV:-/home/developer/Ascend/cann-9.3.0/set_env.sh}"
PACKAGES=(
    torch==2.9.0
    torch_npu==2.9.0.post9
    transformers==5.5.4
    safetensors
    xxhash
    jinja2
    tokenizers
    huggingface_hub
    tqdm
)

if [[ ! -d "$ENV_DIR" ]]; then
    echo "Creating virtual environment: $ENV_DIR"
    python3 -m venv --system-site-packages "$ENV_DIR"
    "$ENV_DIR/bin/python" -m ensurepip --upgrade
fi

if [[ -n "${PIP_INDEX_URL:-}" ]]; then
    echo "Using PIP_INDEX_URL: $PIP_INDEX_URL"
fi

"$ENV_DIR/bin/python" -m pip install "${PACKAGES[@]}"
"$ENV_DIR/bin/python" -m pip install --ignore-installed --no-deps tqdm

echo "Loading CANN environment: $CANN_SET_ENV"
if [[ ! -f "$CANN_SET_ENV" ]]; then
    echo "ERROR: CANN environment script not found: $CANN_SET_ENV" >&2
    exit 1
fi

set +u
source "$CANN_SET_ENV"
set -u

cd "$REPO_ROOT"
echo "Environment: $ENV_DIR"
echo "Python: $("$ENV_DIR/bin/python" --version)"
echo "Running NPU environment check"
"$ENV_DIR/bin/python" scripts/check_npu_env.py
