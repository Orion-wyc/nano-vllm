#!/usr/bin/env bash
# Two-card Prefill/Decode demo fronted by vllm-router.
#
# Card 0 runs a prefill-only nano-vllm worker, card 1 runs a decode-only
# worker. Both are registered with vllm-router as static PD workers:
#
#   client -> vllm-router -> prefill worker (card 0)
#                          -> decode worker  (card 1)
#
# The decode worker fetches the staged KV cache and Qwen3.5 recurrent state
# from the prefill worker over the internal /internal/pd/transfer endpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="${NANO_VLLM_ENV:-/mnt/workspace/Envs/nano_vllm_a3}"
CANN_SET_ENV="${CANN_SET_ENV:-/home/developer/Ascend/cann-9.3.0/set_env.sh}"
VLLM_ROUTER_DIR="${VLLM_ROUTER_DIR:-$REPO_ROOT/../vllm-router}"

PREFILL_DEVICE="${PREFILL_DEVICE:-0}"
DECODE_DEVICE="${DECODE_DEVICE:-1}"
PREFILL_PORT="${PREFILL_PORT:-8101}"
DECODE_PORT="${DECODE_PORT:-8102}"
ROUTER_PORT="${ROUTER_PORT:-8080}"
METRICS_PORT="${METRICS_PORT:-29000}"
ROUTER_POLICY="${ROUTER_POLICY:-round_robin}"
KEEP_RUNNING="${KEEP_RUNNING:-1}"
LOG_DIR="${LOG_DIR:-$(mktemp -d /tmp/nano-vllm-pd-router.XXXXXX)}"

if [[ -f "$CANN_SET_ENV" ]]; then
    echo "Loading CANN environment: $CANN_SET_ENV"
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
if [[ -z "${NANO_VLLM_MODEL:-}" || ! -d "$NANO_VLLM_MODEL" ]]; then
    echo "ERROR: model directory not found. Set NANO_VLLM_MODEL." >&2
    exit 1
fi

find_router() {
    if [[ -n "${VLLM_ROUTER_BIN:-}" && -x "$VLLM_ROUTER_BIN" ]]; then
        echo "$VLLM_ROUTER_BIN"
        return 0
    fi
    local found
    found="$(command -v vllm-router || true)"
    if [[ -n "$found" ]]; then
        echo "$found"
        return 0
    fi
    if [[ -x "$VLLM_ROUTER_DIR/target/release/vllm-router" ]]; then
        echo "$VLLM_ROUTER_DIR/target/release/vllm-router"
        return 0
    fi
    return 1
}

if ! ROUTER_BIN="$(find_router)"; then
    shopt -s nullglob
    WHEELS=("$VLLM_ROUTER_DIR"/dist/vllm_router-*.whl)
    shopt -u nullglob
    if [[ "${INSTALL_ROUTER:-1}" == "1" && ${#WHEELS[@]} -gt 0 ]]; then
        echo "Installing vllm-router wheel: ${WHEELS[-1]}"
        "$PYTHON" -m pip install --force-reinstall "${WHEELS[-1]}"
        hash -r
        ROUTER_BIN="$(find_router)"
    fi
fi
if [[ -z "${ROUTER_BIN:-}" ]]; then
    echo "ERROR: vllm-router not found." >&2
    echo "Install a wheel from $VLLM_ROUTER_DIR/dist or set VLLM_ROUTER_BIN." >&2
    exit 1
fi

echo "Model:          $NANO_VLLM_MODEL"
echo "Router binary:  $ROUTER_BIN"
echo "Prefill worker: device $PREFILL_DEVICE, http://127.0.0.1:$PREFILL_PORT"
echo "Decode worker:  device $DECODE_DEVICE, http://127.0.0.1:$DECODE_PORT"
echo "Router:         http://127.0.0.1:$ROUTER_PORT (policy=$ROUTER_POLICY)"
echo "Logs:           $LOG_DIR"

mkdir -p "$LOG_DIR"
PIDS=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

wait_http() {
    local url=$1
    local name=$2
    local attempts=${3:-600}
    for _ in $(seq 1 "$attempts"); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "timed out waiting for $name at $url" >&2
    return 1
}

COMMON_ARGS=(
    --model "$NANO_VLLM_MODEL"
    --host 127.0.0.1
    --enforce-eager
    --max-model-len "${NANO_VLLM_MAX_MODEL_LEN:-512}"
    --max-num-seqs "${NANO_VLLM_MAX_NUM_SEQS:-8}"
    --max-num-batched-tokens "${NANO_VLLM_MAX_NUM_BATCHED_TOKENS:-512}"
    --gpu-memory-utilization "${NANO_VLLM_GPU_UTIL:-0.3}"
)

echo "=== starting prefill worker ==="
"$PYTHON" -u -m nanovllm.serving.api_server \
    "${COMMON_ARGS[@]}" \
    --port "$PREFILL_PORT" \
    --device-id "$PREFILL_DEVICE" \
    --role prefill \
    --worker-id prefill-1 \
    --advertise-url "http://127.0.0.1:$PREFILL_PORT" \
    >"$LOG_DIR/prefill.log" 2>&1 &
PIDS+=("$!")

echo "=== starting decode worker ==="
"$PYTHON" -u -m nanovllm.serving.api_server \
    "${COMMON_ARGS[@]}" \
    --port "$DECODE_PORT" \
    --device-id "$DECODE_DEVICE" \
    --role decode \
    --worker-id decode-1 \
    >"$LOG_DIR/decode.log" 2>&1 &
PIDS+=("$!")

wait_http "http://127.0.0.1:$PREFILL_PORT/health" "prefill worker" 600
wait_http "http://127.0.0.1:$DECODE_PORT/health" "decode worker" 600
echo "=== workers are healthy ==="

echo "=== starting vllm-router ==="
"$ROUTER_BIN" \
    --host 127.0.0.1 \
    --port "$ROUTER_PORT" \
    --vllm-pd-disaggregation \
    --prefill "http://127.0.0.1:$PREFILL_PORT" none \
    --decode "http://127.0.0.1:$DECODE_PORT" \
    --policy "$ROUTER_POLICY" \
    --worker-startup-timeout-secs 600 \
    --worker-startup-check-interval 2 \
    --prometheus-host 127.0.0.1 \
    --prometheus-port "$METRICS_PORT" \
    >"$LOG_DIR/router.log" 2>&1 &
PIDS+=("$!")

wait_http "http://127.0.0.1:$ROUTER_PORT/health" "vllm-router" 600
echo "=== vllm-router is healthy ==="

echo
echo "Send a request through the router:"
echo "  curl -sS http://127.0.0.1:$ROUTER_PORT/v1/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"$NANO_VLLM_MODEL\",\"prompt\":\"What is 1+1?\",\"max_tokens\":8,\"temperature\":0}'"
echo
echo "Run the automated check without keeping processes alive:"
echo "  KEEP_RUNNING=0 $0"
echo

if [[ "${SMOKE_REQUEST:-1}" == "1" ]]; then
    echo "=== smoke request ==="
    curl -fsS "http://127.0.0.1:$ROUTER_PORT/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$NANO_VLLM_MODEL\",\"prompt\":\"What is 1+1?\",\"max_tokens\":8,\"temperature\":0}" \
        | "$PYTHON" -m json.tool
    echo
fi

if [[ "$KEEP_RUNNING" == "1" ]]; then
    echo "Router is running at http://127.0.0.1:$ROUTER_PORT"
    echo "Metrics: http://127.0.0.1:$METRICS_PORT/metrics"
    echo "Press Ctrl+C to stop."
    wait "${PIDS[-1]}" || true
else
    echo "KEEP_RUNNING=0, shutting down."
fi
