#!/usr/bin/env bash
# Two-card PD demo backed by Mooncake Store.
#
# Starts mooncake_master, a prefill worker on card 0, a decode worker on card 1,
# and vllm-router with --kv-connector mooncake. The prefill worker publishes KV
# and state to Mooncake; the decode worker pulls them by transfer_id.
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
MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:-50051}"
MOONCAKE_BOOTSTRAP_PORT="${MOONCAKE_BOOTSTRAP_PORT:-8998}"
MOONCAKE_SEGMENT_MB="${MOONCAKE_SEGMENT_MB:-1024}"
MOONCAKE_BUFFER_MB="${MOONCAKE_BUFFER_MB:-256}"
ROUTER_POLICY="${ROUTER_POLICY:-round_robin}"
KEEP_RUNNING="${KEEP_RUNNING:-1}"
LOG_DIR="${LOG_DIR:-$(mktemp -d /tmp/nano-vllm-mooncake-pd.XXXXXX)}"

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
ROUTER_BIN="$(find_router)" || {
    echo "ERROR: vllm-router not found; set VLLM_ROUTER_BIN" >&2
    exit 1
}

MOONCAKE_MASTER_BIN="${MOONCAKE_MASTER_BIN:-$(command -v mooncake_master || true)}"
if [[ -z "$MOONCAKE_MASTER_BIN" ]]; then
    MOONCAKE_MASTER_BIN="$ENV_DIR/bin/mooncake_master"
fi
if [[ ! -x "$MOONCAKE_MASTER_BIN" ]]; then
    echo "ERROR: mooncake_master not found; run scripts/setup_mooncake_env.sh" >&2
    exit 1
fi

echo "Model:          $NANO_VLLM_MODEL"
echo "Router:         $ROUTER_BIN"
echo "Mooncake master:$MOONCAKE_MASTER_BIN (port $MOONCAKE_MASTER_PORT)"
echo "Prefill worker: device $PREFILL_DEVICE, port $PREFILL_PORT, bootstrap $MOONCAKE_BOOTSTRAP_PORT"
echo "Decode worker:  device $DECODE_DEVICE, port $DECODE_PORT"
echo "Logs:           $LOG_DIR"

mkdir -p "$LOG_DIR"
PIDS=()
MASTER_PID=""
cleanup() {
    trap - EXIT INT TERM
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    if [[ -n "$MASTER_PID" ]]; then
        kill "$MASTER_PID" 2>/dev/null || true
        wait "$MASTER_PID" 2>/dev/null || true
    fi
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

if ! (exec 3<>"/dev/tcp/127.0.0.1/$MOONCAKE_MASTER_PORT") 2>/dev/null; then
    echo "=== starting mooncake_master ==="
    "$MOONCAKE_MASTER_BIN" \
        "--rpc_port=$MOONCAKE_MASTER_PORT" \
        "--metrics_port=$((MOONCAKE_MASTER_PORT + 4000))" \
        --enable_metric_reporting=false \
        >"$LOG_DIR/mooncake_master.log" 2>&1 &
    MASTER_PID=$!
    for _ in $(seq 1 120); do
        if (exec 3<>"/dev/tcp/127.0.0.1/$MOONCAKE_MASTER_PORT") 2>/dev/null; then
            break
        fi
        sleep 1
    done
else
    echo "=== reusing mooncake_master on port $MOONCAKE_MASTER_PORT ==="
fi

COMMON_ARGS=(
    --model "$NANO_VLLM_MODEL"
    --host 127.0.0.1
    --enforce-eager
    --max-model-len "${NANO_VLLM_MAX_MODEL_LEN:-512}"
    --max-num-seqs "${NANO_VLLM_MAX_NUM_SEQS:-8}"
    --max-num-batched-tokens "${NANO_VLLM_MAX_NUM_BATCHED_TOKENS:-512}"
    --gpu-memory-utilization "${NANO_VLLM_GPU_UTIL:-0.3}"
    --kv-store mooncake
    --mooncake-master "127.0.0.1:$MOONCAKE_MASTER_PORT"
    --mooncake-global-segment-size "$((MOONCAKE_SEGMENT_MB * 1024 * 1024))"
    --mooncake-local-buffer-size "$((MOONCAKE_BUFFER_MB * 1024 * 1024))"
    --mooncake-state-snapshots "${MOONCAKE_STATE_SNAPSHOTS:-block}"
)

echo "=== starting prefill worker ==="
"$PYTHON" -u -m nanovllm.serving.api_server \
    "${COMMON_ARGS[@]}" \
    --port "$PREFILL_PORT" \
    --device-id "$PREFILL_DEVICE" \
    --role prefill \
    --worker-id prefill-1 \
    --advertise-url "http://127.0.0.1:$PREFILL_PORT" \
    --mooncake-bootstrap-port "$MOONCAKE_BOOTSTRAP_PORT" \
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
wait_http "http://127.0.0.1:$MOONCAKE_BOOTSTRAP_PORT/health" "mooncake bootstrap" 120
echo "=== workers are healthy ==="

echo "=== starting vllm-router with Mooncake connector ==="
"$ROUTER_BIN" \
    --host 127.0.0.1 \
    --port "$ROUTER_PORT" \
    --vllm-pd-disaggregation \
    --kv-connector mooncake \
    --prefill "http://127.0.0.1:$PREFILL_PORT" "$MOONCAKE_BOOTSTRAP_PORT" \
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
echo "    -d '{\"prompt\":\"What is 1+1?\",\"max_tokens\":8,\"temperature\":0}'"
echo

if [[ "${SMOKE_REQUEST:-1}" == "1" ]]; then
    echo "=== smoke request ==="
    curl -fsS "http://127.0.0.1:$ROUTER_PORT/v1/completions" \
        -H 'Content-Type: application/json' \
        -d '{"prompt":"What is 1+1?","max_tokens":8,"temperature":0}' \
        | "$PYTHON" -m json.tool
    echo
fi

if [[ "$KEEP_RUNNING" == "1" ]]; then
    echo "Router is running at http://127.0.0.1:$ROUTER_PORT"
    echo "Logs: $LOG_DIR"
    echo "Press Ctrl+C to stop."
    wait "${PIDS[-1]}" || true
else
    echo "KEEP_RUNNING=0, shutting down."
fi
