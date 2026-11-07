# 两卡 PD 混部与 vllm-router

本文给出 nano-vllm 的 Prefill / Decode 分离部署示例：两张 NPU 分别运行 prefill worker 和 decode worker，两者注册到 `vllm-router`（外部仓库），由 router 完成请求接入、PD 路由、健康检查和负载均衡。复现以 NPU 环境为准，环境准备和离线验证见 [A3_NPU_ADAPTATION.md](A3_NPU_ADAPTATION.md)。

```
client
  |
  v
vllm-router  --vllm-pd-disaggregation
  |-- prefill worker (device 0, port 8101)
  `-- decode worker  (device 1, port 8102)
```

## 1. 角色划分

| 进程 | 启动参数 | 职责 |
| --- | --- | --- |
| prefill worker | `--role prefill` | 执行 prompt 的 prefill，采样首个 token，导出 KV cache 和 Qwen3.5 线性注意力 state |
| decode worker | `--role decode` | 从 prefill worker 拉取 KV/state，导入后继续 decode |
| vllm-router | `--vllm-pd-disaggregation --prefill ... --decode ...` | 先调用 prefill，再把 prefill 返回的 `kv_transfer_params` 转发给 decode |

两个 worker 都是标准 OpenAI 风格 HTTP 服务，提供 `/health`、`/v1/models`、`/v1/completions`、`/v1/chat/completions`，并补充 router 需要的 `/health_generate`、`/get_server_info`、`/get_load`、`/flush_cache`。

## 2. 准备

### 2.1 环境

```bash
source /home/developer/Ascend/cann-9.3.0/set_env.sh
cd /mnt/workspace/Works/nano-vllm
source /mnt/workspace/Envs/nano_vllm_a3/bin/activate
export NANO_VLLM_MODEL=/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master
```

### 2.2 vllm-router

仓库外的 `../vllm-router` 已提供两种产物：

- whl：`../vllm-router/dist/vllm_router-*.whl`
- Rust 二进制：`../vllm-router/target/release/vllm-router`

安装 whl：

```bash
pip install --force-reinstall ../vllm-router/dist/vllm_router-*.whl
```

whl 的 Python 启动器依赖 `aiohttp`、`orjson` 等包；离线环境可以直接使用 Rust 二进制，并通过 `VLLM_ROUTER_BIN` 指定：

```bash
export VLLM_ROUTER_BIN=/mnt/workspace/Works/vllm-router/target/release/vllm-router
```

## 3. 启动

### 3.1 一键示例

```bash
bash scripts/run_pd_router_demo.sh
```

脚本会：

1. 在 device 0 启动 prefill worker（8101）。
2. 在 device 1 启动 decode worker（8102）。
3. 启动 vllm-router（8080），注册一个 prefill 和一个 decode worker。
4. 等健康检查通过后发送一次 smoke 请求。
5. 默认保持运行，按 `Ctrl+C` 停止。

只做启动检查并退出：

```bash
KEEP_RUNNING=0 bash scripts/run_pd_router_demo.sh
```

常用环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PREFILL_DEVICE` | `0` | prefill 使用的 NPU |
| `DECODE_DEVICE` | `1` | decode 使用的 NPU |
| `PREFILL_PORT` / `DECODE_PORT` | `8101` / `8102` | worker 端口 |
| `ROUTER_PORT` | `8080` | router 端口 |
| `ROUTER_POLICY` | `round_robin` | router 路由策略 |
| `NANO_VLLM_GPU_UTIL` | `0.3` | 每个 worker 的 KV cache 显存占比 |
| `VLLM_ROUTER_BIN` | 自动查找 | router 可执行文件 |
| `INSTALL_ROUTER` | `1` | 找不到可执行文件时是否安装 `../vllm-router/dist` 下的 whl |
| `KEEP_RUNNING` | `1` | `0` 表示 smoke 请求后退出 |

### 3.2 手工启动

prefill worker：

```bash
python -m nanovllm.serving.api_server \
  --model "$NANO_VLLM_MODEL" \
  --host 127.0.0.1 --port 8101 \
  --device-id 0 --role prefill \
  --worker-id prefill-1 \
  --advertise-url http://127.0.0.1:8101 \
  --enforce-eager \
  --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.3
```

decode worker：

```bash
python -m nanovllm.serving.api_server \
  --model "$NANO_VLLM_MODEL" \
  --host 127.0.0.1 --port 8102 \
  --device-id 1 --role decode \
  --worker-id decode-1 \
  --enforce-eager \
  --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.3
```

router：

```bash
"${VLLM_ROUTER_BIN:-../vllm-router/target/release/vllm-router}" \
  --host 127.0.0.1 --port 8080 \
  --vllm-pd-disaggregation \
  --prefill http://127.0.0.1:8101 none \
  --decode http://127.0.0.1:8102 \
  --policy round_robin \
  --worker-startup-timeout-secs 600 \
  --worker-startup-check-interval 2
```

`--prefill URL none` 表示不使用 vLLM bootstrap 端口。nano-vllm 的 KV 传输走自己的 HTTP 端点，不依赖 NIXL / Mooncake。

## 4. 请求

```bash
curl -sS http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-0.8B",
    "prompt": "What is 1+1?",
    "max_tokens": 8,
    "temperature": 0,
    "stream": false
  }'
```

响应由 decode worker 生成，额外带有两个便于排查的字段：

```json
{
  "worker_role": "decode",
  "worker_id": "decode-1",
  "pd_transfer": true,
  "choices": [{"index": 0, "text": "\n\n1+1 = 2\n\n", "finish_reason": "length"}],
  "usage": {"prompt_tokens": 7, "completion_tokens": 8, "total_tokens": 15}
}
```

`pd_transfer=true` 表示 decode worker 确实使用了 prefill 阶段导出的 KV/state。若该字段为 `false`，说明请求没有携带 `kv_transfer_params`，decode worker 回退为本地完整生成。

流式请求使用同一端点，`stream=true`：

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 32,
    "temperature": 0,
    "stream": true
  }'
```

## 5. 内部协议

vllm-router 的 PD 流程与 vLLM 一致：

1. router 将原始请求复制为 prefill 请求，强制 `max_tokens=1`、`stream=false`，并注入 `kv_transfer_params`。
2. prefill worker 只执行 prompt 的 prefill，采样首个 token，把 KV 和线性注意力 state 序列化后存入本地 transfer store。
3. prefill 响应中包含：

```json
{
  "kv_transfer_params": {
    "nano_pd": {
      "version": 1,
      "transfer_id": "…",
      "prefill_url": "http://127.0.0.1:8101",
      "token_ids": [ … ],
      "num_prompt_tokens": 7,
      "num_blocks": 1,
      "first_token": 271,
      "eos_token_id": 248044,
      "has_state": true
    }
  }
}
```

4. router 把 `kv_transfer_params` 原样转发给 decode worker。
5. decode worker 调用 `GET {prefill_url}/internal/pd/transfer/{transfer_id}` 拉取序列化数据，导入 KV/state，然后继续 decode。

KV 和 state 使用 `torch.save` 序列化，transfer store 默认保留 300 秒（`--kv-transfer-ttl`）。该端点没有鉴权，仅用于同机或可信网络内的 demo。

## 6. 负载均衡

router 的请求级策略由 `--policy`、`--prefill-policy`、`--decode-policy` 指定，可选：

- `round_robin`
- `random`
- `cache_aware`
- `power_of_two`
- `consistent_hash`

单 P 单 D 时策略只影响阶段选择；增加 worker 后即可横向扩展。例如两张卡各跑一个 prefill、再增加两张卡跑 decode 时：

```bash
vllm-router \
  --vllm-pd-disaggregation \
  --prefill http://127.0.0.1:8101 none \
  --prefill http://127.0.0.1:8103 none \
  --decode http://127.0.0.1:8102 \
  --decode http://127.0.0.1:8104 \
  --policy round_robin
```

`power_of_two` 要求每个阶段至少有两个 worker。`cache_aware` 和 `kv_aware` 可能查询 worker 的 `/get_server_info` 或 `/metrics`；nano-vllm worker 提供 `/get_server_info`，不导出 vLLM 风格的 `/metrics`，router 会记录警告并继续使用负载信息。

如果只需要普通数据并行的请求级负载均衡，不启用 PD，可以把两个 worker 都按 `--role regular` 启动，然后使用 `--worker-urls`：

```bash
python -m nanovllm.serving.api_server --model "$NANO_VLLM_MODEL"   --host 127.0.0.1 --port 8101 --device-id 0 --role regular --enforce-eager   --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 512

python -m nanovllm.serving.api_server --model "$NANO_VLLM_MODEL"   --host 127.0.0.1 --port 8102 --device-id 1 --role regular --enforce-eager   --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 512

vllm-router --host 127.0.0.1 --port 8080   --worker-urls http://127.0.0.1:8101 http://127.0.0.1:8102   --policy round_robin
```

普通模式下两个 worker 各自完整生成，不共享 KV；需要 KV 复用和阶段分离时使用本节前面的 PD 模式。

## 7. 自动验证

```bash
python scripts/verify_pd_router.py
```

脚本会启动 prefill/decode worker 和 router，并发发送多个 `temperature=0` 请求，检查：

- 响应来自 decode worker；
- `pd_transfer=true`；
- 流式请求以 `[DONE]` 结束；
- 返回文本与单请求直接生成的结果完全一致。

常用环境变量：

| 变量 | 默认值 |
| --- | --- |
| `NANO_VLLM_MODEL` | 自动查找 |
| `NANO_VLLM_PREFILL_DEVICE` | `0` |
| `NANO_VLLM_DECODE_DEVICE` | `1` |
| `NANO_VLLM_MAX_TOKENS` | `8` |
| `NANO_VLLM_ROUTER_TIMEOUT` | `600` |
| `NANO_VLLM_ROUTER_POLICY` | `round_robin` |
| `VLLM_ROUTER_BIN` | 自动查找 |

## 8. 远程 NPU 验证

在 CPU 节点使用 `npu-dev` 同步并执行：

```bash
export NPU_DEV=/home/developer/.config/opencode/skills/npu-dev/scripts/npu_dev.py
python3 "$NPU_DEV" sync
python3 "$NPU_DEV" run --no-sync -- bash -lc '
  source /home/developer/Ascend/cann-9.3.0/set_env.sh
  cd /mnt/workspace/Works/nano-vllm
  source /mnt/workspace/Envs/nano_vllm_a3/bin/activate
  export NANO_VLLM_MODEL=/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master
  export VLLM_ROUTER_BIN=/mnt/workspace/Works/vllm-router/target/release/vllm-router
  python scripts/verify_pd_router.py
'
```

## 9. 限制

- 两个 worker 必须使用同一份模型和 tokenizer；跨模型 PD 不在当前范围内。
- prefill 阶段串行执行，长 prompt 会阻塞后续请求的 prefill。
- KV/state 通过 HTTP 从 prefill worker 拉取，未接入 NIXL、Mooncake 或 RDMA。
- `tensor_parallel_size > 1` 的 PD 仍未支持。
- 内部 transfer 端点无鉴权、无 TLS。
- Qwen3.5 在 NPU 上保持 eager，未接入图执行。
