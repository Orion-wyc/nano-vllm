#!/usr/bin/env bash
# Install the Mooncake client/master used by the Mooncake KV backend.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="${NANO_VLLM_ENV:-/mnt/workspace/Envs/nano_vllm_a3}"
CANN_SET_ENV="${CANN_SET_ENV:-/home/developer/Ascend/cann-9.3.0/set_env.sh}"
MOONCAKE_VERSION="${MOONCAKE_VERSION:-0.3.14}"
MOONCAKE_WHEEL="${MOONCAKE_WHEEL:-}"

if [[ -f "$CANN_SET_ENV" ]]; then
    set +u
    source "$CANN_SET_ENV"
    set -u
fi

if [[ -f "$ENV_DIR/bin/activate" ]]; then
    source "$ENV_DIR/bin/activate"
    PYTHON="${PYTHON:-$ENV_DIR/bin/python}"
else
    PYTHON="${PYTHON:-python3}"
fi

cd "$REPO_ROOT"

if [[ -n "$MOONCAKE_WHEEL" ]]; then
    echo "Installing Mooncake wheel: $MOONCAKE_WHEEL"
    "$PYTHON" -m pip install --force-reinstall "$MOONCAKE_WHEEL"
else
    echo "Installing mooncake-transfer-engine-non-cuda==$MOONCAKE_VERSION"
    "$PYTHON" -m pip install --force-reinstall "mooncake-transfer-engine-non-cuda==$MOONCAKE_VERSION"
fi

echo "Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"
"$PYTHON" - <<'PY'
import shutil
from mooncake.store import MooncakeDistributedStore

master = shutil.which("mooncake_master")
print(f"MooncakeDistributedStore: {MooncakeDistributedStore}")
print(f"mooncake_master: {master}")
if master is None:
    raise SystemExit("mooncake_master is not on PATH; check the Mooncake installation")
PY

echo "Mooncake environment is ready."
echo "Start the master with: $ENV_DIR/bin/mooncake_master --rpc_port 50051"
