#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="${NANO_VLLM_ENV:-/mnt/workspace/Envs/nano_vllm_a3}"
CANN_SET_ENV="${CANN_SET_ENV:-/home/developer/Ascend/cann-9.3.0/set_env.sh}"

echo "Loading CANN environment: $CANN_SET_ENV"
if [[ ! -f "$CANN_SET_ENV" ]]; then
    echo "ERROR: CANN environment script not found: $CANN_SET_ENV" >&2
    exit 1
fi

set +u
source "$CANN_SET_ENV"
set -u

if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
    echo "ERROR: virtual environment not found: $ENV_DIR" >&2
    exit 1
fi
source "$ENV_DIR/bin/activate"
cd "$REPO_ROOT"

if [[ -z "${NANO_VLLM_MODEL:-}" ]]; then
    for candidate in \
        /home/developer/models/Qwen3.5-0.8B \
        /home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master \
        /mnt/workspace/models/Qwen3.5-0.8B
    do
        if [[ -d "$candidate" ]]; then
            export NANO_VLLM_MODEL="$candidate"
            break
        fi
    done
fi

if [[ -z "${NANO_VLLM_MODEL:-}" ]]; then
    echo "ERROR: model directory not found. Set NANO_VLLM_MODEL to override." >&2
    exit 1
fi

echo "Environment: $ENV_DIR"
echo "Model: $NANO_VLLM_MODEL"
python example.py "$@"
