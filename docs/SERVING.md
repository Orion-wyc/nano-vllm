# nano-vllm 在线服务

本文说明 nano-vllm 的在线服务组件、API 用法、启动方式、PD 分离模式和验证脚本。

## A3 NPU 快速启动与请求示例

以下命令在 NPU 节点的终端中执行。首次运行可先执行 `bash scripts/setup_a3_env.sh` 安装依赖；如果代码、虚拟环境或模型位于其他路径，请相应替换。

```bash
source /home/developer/Ascend/cann-9.3.0/set_env.sh
cd /mnt/workspace/Works/nano-vllm
source /mnt/workspace/Envs/nano_vllm_a3/bin/activate
export NANO_VLLM_MODEL=/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master

python -m nanovllm.serving.api_server \
  --model "$NANO_VLLM_MODEL" \
  --host 0.0.0.0 --port 8000 \
  --device-id 0 --tp 1 --enforce-eager \
  --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 512
```

保持服务终端运行，在另一个终端检查服务并发送对话请求：

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"1+10086等于多少？"}],"max_tokens":128,"temperature":0,"repetition_penalty":1.1}'
```

`temperature=0` 使用贪心采样。模型的回答不保证数学正确，应在业务侧校验计算结果。若从其他机器访问，将 `127.0.0.1` 替换为 NPU 节点可达的地址。

流式对话使用相同端点并设置 `stream: true`，`curl -N` 可逐条显示 SSE：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"请介绍一下你自己"}],"max_tokens":128,"temperature":0.7,"top_p":0.8,"top_k":20,"repetition_penalty":1.1,"stream":true}'
```

原始文本续写可使用下列端点；它不会自动添加对话模板：

```bash
curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time","max_tokens":64,"temperature":0.7,"top_p":0.8,"top_k":20,"repetition_penalty":1.1}'
```

如需使用两张 NPU 执行 PD 分离，推荐按 5.3 启动两个独立 worker 并注册到 vllm-router。单进程 `--pd` 模式适合 CPU 或单 device 调试，说明见 5.2。

## 1. 组件

| 组件 | 位置 | 作用 |
| --- | --- | --- |
| `AsyncLLMEngine` | `nanovllm/serving/async_engine.py` | 在 asyncio 事件循环内包装 `LLMEngine`，为每个请求提供异步事件流 |
| API Server | `nanovllm/serving/api_server.py` | 基于 FastAPI 的 OpenAI 风格 HTTP 服务 |
| Router | `nanovllm/serving/router.py` | `BaseRouter`、`SingleRouter`、`RoundRobinRouter`，负责选择引擎 |
| `PDRouter` | `nanovllm/serving/pd.py` | Prefill / Decode 分离，prefill 结果通过 CPU staging 交给 decode 引擎 |

`AsyncLLMEngine` 复用 `LLMEngine` 的调度和模型执行，对上层提供：

- `generate(prompt, sampling_params, request_id=None, stop=None)`：异步生成器，逐事件返回结果。
- `add_prefilled(prefill_result, request_id, stop=None)`：导入 prefill 结果并开始 decode。
- `stream(request_id)`：继续消费已注册请求的事件流。
- `abort(request_id)`：中止请求。
- `shutdown()`：关闭后台任务和引擎。

`AsyncLLMEngine` 必须在运行中的 asyncio 事件循环内构造，不能在循环外提前创建。

## 2. 依赖安装

只安装服务依赖：

```bash
pip install -e '.[serving]'
```

`serving` extra 包含 `fastapi`、`uvicorn`、`pydantic`、`httpx`。原有的 `cuda` 和 `npu` extra 保持不变。

A3 NPU 环境可以继续使用初始化脚本，脚本会额外安装 FastAPI、uvicorn 和 httpx：

```bash
bash scripts/setup_a3_env.sh
```

## 3. 异步模型与事件格式

`AsyncLLMEngine` 在构造时绑定当前 asyncio 事件循环，并启动一个名为 `AsyncLLMEngine` 的后台工作线程。所有引擎操作（`add_request`、`abort_request`、`step_updates`、`exit`）都在这个线程内串行执行；异步侧通过线程安全队列提交命令，再用 `asyncio.wrap_future` 等待结果。每个 step 产生的更新通过 `loop.call_soon_threadsafe` 投递到请求对应的 `asyncio.Queue`，`generate()` 和 `stream()` 从队列中逐事件产出。多个请求共享同一个模型实例和同一个 batch 调度循环，模型计算仍在后台线程内同步完成。

`Context` 使用线程本地存储，RoPE cache 按当前设备缓存。因此 PD 模式下 prefill 线程和 decode 工作线程不会互相覆盖上下文，同一进程内也可以在不同 device 上创建多个引擎。

每个事件是一个字典：

| 键 | 含义 |
| --- | --- |
| `request_id` | 请求标识 |
| `seq_id` | 引擎内部的序列 ID |
| `token_ids` | 截至当前已生成的 completion token ID |
| `text` | 截至当前已解码的完整文本 |
| `delta_text` | 相对上一个事件的增量文本 |
| `finished` | 是否已结束 |
| `num_prompt_tokens` | prompt token 数 |
| `num_generated_tokens` | 已生成 token 数 |
| `finish_reason` | 结束原因，未结束时为 `null`，结束时通常为 `stop` 或 `length` |

最后一个事件的 `finished` 为 `true`。调用方应以该事件作为完成标志。

## 4. API 端点

服务默认监听 `http://127.0.0.1:8000`。下面的 `MODEL` 需要替换为 `/v1/models` 返回的模型 ID。

### 4.1 健康检查

```bash
curl -s http://127.0.0.1:8000/health
```

返回 `{"status":"ok"}`。

### 4.2 模型列表

```bash
curl -s http://127.0.0.1:8000/v1/models
```

### 4.3 文本补全（非流式）

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MODEL",
    "prompt": "Hello, introduce yourself in one sentence.",
    "max_tokens": 32,
    "temperature": 0.6
  }'
```

生成文本位于 `choices[0].text`。

`temperature=0` 使用贪心采样；大于 0 时按温度随机采样。可选参数 `top_p`、`top_k` 和 `repetition_penalty` 分别控制核采样、候选 token 数和重复惩罚，默认值为 `1.0`、`-1` 和 `1.0`，默认不改变原始采样分布。

`/v1/completions` 将 `prompt` 原样作为续写文本，不会自动转换为问答格式。对 Qwen3.5 指令模型提问时建议使用 `/v1/chat/completions`：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"1+10086等于多少？"}],"max_tokens":128,"temperature":0,"repetition_penalty":1.1}'
```

采样设置可以降低重复，但无法保证小模型的计算结果正确。

### 4.4 文本补全（流式）

```bash
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MODEL",
    "prompt": "Hello, introduce yourself in one sentence.",
    "max_tokens": 32,
    "temperature": 0.6,
    "stream": true
  }'
```

流式响应的每一行以 `data: ` 开头，最后一个事件为 `data: [DONE]`。

### 4.5 对话补全

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MODEL",
    "messages": [
      {"role": "user", "content": "Hello, introduce yourself in one sentence."}
    ],
    "max_tokens": 32,
    "temperature": 0.6
  }'
```

文本位于 `choices[0].message.content`。同样支持 `stream: true`。

## 5. 启动服务

```bash
python -m nanovllm.serving.api_server \
  --model /path/to/Qwen3.5-0.8B \
  --host 0.0.0.0 \
  --port 8000
```

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--model` | 本地模型目录 |
| `--host` / `--port` | 监听地址和端口 |
| `--device-id` | 单 worker 使用的设备编号 |
| `--max-model-len` | 最大模型长度 |
| `--max-num-seqs` | 最大并发序列数 |
| `--max-num-batched-tokens` | 单 step 最大 token 数 |
| `--gpu-memory-utilization` | KV cache 显存占比 |
| `--enforce-eager` | 关闭 CUDA graph / 编译路径 |
| `--workers` | 多 worker 数量 |
| `--pd` | 启用 Prefill / Decode 分离 |
| `--prefill-device` | PD 模式 prefill 设备 |
| `--decode-device` | PD 模式 decode 设备 |

### 5.1 多 worker

```bash
python -m nanovllm.serving.api_server \
  --model /path/to/model \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

每个 worker 创建一个 `AsyncLLMEngine`，设备编号从 `--device-id` 开始依次递增，请求由 `RoundRobinRouter` 轮询分发。

### 5.2 单进程 PD 模式

单进程 PD 使用一个同步 `LLMEngine` 做 prefill，一个 `AsyncLLMEngine` 做 decode，适合 CPU 或单 device 调试。当前 A3 环境下同时加载两个引擎可能受 torch_npu/驱动状态影响，不作为两卡 NPU 复现路径；两卡 NPU 请使用 5.3 的独立 worker + vllm-router 模式。

单 device 启动（device 0）：

```bash
python -m nanovllm.serving.api_server \
  --model "$NANO_VLLM_MODEL" \
  --host 0.0.0.0 --port 8000 \
  --pd --prefill-device 0 --decode-device 0 \
  --tp 1 --enforce-eager \
  --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 512
```

PD 模式的行为：

- prefill 引擎执行 prompt 并导出 KV cache；Qwen3.5 还会导出线性注意力 state。
- 导出结果通过 CPU staging 传给 decode 引擎，decode 引擎导入后继续生成。
- 同一时刻只处理一个 prefill，prefill 与 decode 串行执行，没有 overlap。
- 数据在同一进程内传递，没有跨节点网络传输。
- 单进程 PD 暂不支持 `tensor_parallel_size > 1`。
- prefill 和 decode 使用同一个 device 时，`create_router` 会把 `--gpu-memory-utilization` 自动减半。

启动后直接发请求：

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models

curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is 1+1?","max_tokens":8,"temperature":0}'

curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is 1+1?","max_tokens":8,"temperature":0,"stream":true}'

curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 1+1?"}],"max_tokens":8,"temperature":0}'
```

如果运行环境支持两个引擎同时驻留，可以使用 device 0 做 prefill、device 1 做 decode：

```bash
python -m nanovllm.serving.api_server \
  --model "$NANO_VLLM_MODEL" \
  --port 8000 \
  --pd --prefill-device 0 --decode-device 1 \
  --tp 1 --enforce-eager \
  --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 512
```

### 5.3 两卡 PD 与 vllm-router

`--pd` 是单进程内的 PD 分离。若要让 prefill 和 decode 分别跑在两张卡上，并由外部 vllm-router 做接入、路由和负载均衡，使用 worker 角色启动。

三终端分别在 NPU 节点执行：

```bash
# 终端 1：card 0，prefill worker
python -m nanovllm.serving.api_server --model "$NANO_VLLM_MODEL" \
  --host 127.0.0.1 --port 8101 --device-id 0 --role prefill \
  --advertise-url http://127.0.0.1:8101 --enforce-eager \
  --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 512

# 终端 2：card 1，decode worker
python -m nanovllm.serving.api_server --model "$NANO_VLLM_MODEL" \
  --host 127.0.0.1 --port 8102 --device-id 1 --role decode \
  --enforce-eager --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 512

# 终端 3：vllm-router
export VLLM_ROUTER_BIN=${VLLM_ROUTER_BIN:-/mnt/workspace/Works/vllm-router/target/release/vllm-router}
"$VLLM_ROUTER_BIN" --host 127.0.0.1 --port 8080 \
  --vllm-pd-disaggregation \
  --prefill http://127.0.0.1:8101 none \
  --decode http://127.0.0.1:8102 \
  --policy round_robin
```

prefill worker 导出 KV cache 和 Qwen3.5 线性注意力 state，decode worker 通过 `/internal/pd/transfer/{id}` 拉取。请求统一发到 router：

```bash
curl -sS http://127.0.0.1:8101/health
curl -sS http://127.0.0.1:8102/health
curl -sS http://127.0.0.1:8080/health

curl -sS http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is 1+1?","max_tokens":8,"temperature":0}'

curl -N http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is 1+1?","max_tokens":8,"temperature":0,"stream":true}'

curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 1+1?"}],"max_tokens":8,"temperature":0}'
```

`/v1/completions` 和 `/v1/chat/completions` 的非流式响应会带有：

```json
{
  "worker_role": "decode",
  "worker_id": "decode-1",
  "pd_transfer": true
}
```

`pd_transfer=true` 表示 decode worker 使用了 prefill 阶段导出的 KV/state。一键启动脚本为 `scripts/run_pd_router_demo.sh`，自动验证为 `scripts/verify_pd_router.py`，更完整的协议说明见 [PD_ROUTER.md](PD_ROUTER.md)。

## 6. 验证脚本

### 6.1 AsyncLLMEngine

```bash
python scripts/verify_async_engine.py
```

验证内容：构造 `AsyncLLMEngine`，发起一次异步生成，检查收到非空 `token_ids`，且最后一个事件 `finished=true`。

可覆盖的环境变量：

| 变量 | 默认值 |
| --- | --- |
| `NANO_VLLM_MODEL` | 自动查找模型目录 |
| `NANO_VLLM_PROMPT` | `Hello, introduce yourself in one sentence.` |
| `NANO_VLLM_MAX_TOKENS` | `16` |
| `NANO_VLLM_TEMPERATURE` | `1e-5` |
| `NANO_VLLM_DEVICE_ID` | `0` |

### 6.2 API Server

```bash
python scripts/verify_api_server.py
```

脚本以子进程启动 `nanovllm.serving.api_server`，等待 `/health`，依次验证非流式补全、流式 SSE 和可选的对话补全，最后终止服务。

关键环境变量：

| 变量 | 默认值 |
| --- | --- |
| `NANO_VLLM_MODEL` | 自动查找模型目录 |
| `NANO_VLLM_PROMPT` | `Hello, introduce yourself in one sentence.` |
| `NANO_VLLM_DEVICE_ID` | `0` |
| `NANO_VLLM_API_TIMEOUT` | `600` 秒，等待服务就绪 |

### 6.3 单进程 PD 模式

以下脚本用于单进程 PD 的引擎对拍，建议在 CPU 或单 device 环境运行；A3 两卡 PD 使用 6.4 的 `verify_pd_router.py`。

```bash
python scripts/verify_pd.py
```

验证内容：用 `PDRouter` 完成一次 prefill + decode，再用独立 `LLMEngine` 直接生成相同 prompt，比较两者的 `token_ids`。脚本要求完全一致，不一致时打印两组结果并返回非零。

多请求并发版本：

```bash
python scripts/verify_pd_concurrent.py
```

它同时发起两个 PD 请求，其中一个 prompt 长度为 160 token，用于覆盖 decode batching 和 chunked prefill，并分别与直接生成结果比较。

| 变量 | 默认值 |
| --- | --- |
| `NANO_VLLM_MODEL` | 自动查找模型目录 |
| `NANO_VLLM_PROMPT` | `Hello, introduce yourself in one sentence.` |
| `NANO_VLLM_PREFILL_DEVICE` | `0` |
| `NANO_VLLM_DECODE_DEVICE` | `1` |
| `NANO_VLLM_MAX_TOKENS` | `8`，并发脚本为 `16` |
| `NANO_VLLM_TEMPERATURE` | `1e-5` |
| `NANO_VLLM_GPU_UTIL` | `0.3` |

单设备环境设置 `NANO_VLLM_DECODE_DEVICE=0` 即可。

### 6.4 两卡 PD + vllm-router

在 NPU 环境中执行：

```bash
source /home/developer/Ascend/cann-9.3.0/set_env.sh
cd /mnt/workspace/Works/nano-vllm
source /mnt/workspace/Envs/nano_vllm_a3/bin/activate
export NANO_VLLM_MODEL=/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master
export VLLM_ROUTER_BIN=/mnt/workspace/Works/vllm-router/target/release/vllm-router

python scripts/verify_pd_router.py
```

脚本启动 prefill/decode worker 和 vllm-router，验证 PD 路由、`pd_transfer` 标记、流式响应，并比较 router 返回文本与单请求直接生成结果。也可以按 5.3 的步骤手工启动后，用 curl 直接发请求，两种方式等价。

### 6.5 NPU 环境中运行验证脚本

服务化复现统一在 NPU 环境执行。每个新终端先加载环境：

```bash
source /home/developer/Ascend/cann-9.3.0/set_env.sh
cd /mnt/workspace/Works/nano-vllm
source /mnt/workspace/Envs/nano_vllm_a3/bin/activate
export NANO_VLLM_MODEL=/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master
```

依次运行：

```bash
python scripts/verify_async_engine.py
python scripts/verify_api_server.py
NANO_VLLM_PREFILL_DEVICE=0 NANO_VLLM_DECODE_DEVICE=1 python scripts/verify_pd_router.py
```

`verify_pd.py` 和 `verify_pd_concurrent.py` 是单进程 PD 的引擎对拍脚本，建议在 CPU 或单 device 环境运行。

CPU 只用于数值对拍，不作为服务化复现环境。CPU 上可运行：

```bash
python scripts/verify_async_engine.py
NANO_VLLM_DECODE_DEVICE=0 python scripts/verify_pd.py
```

## 7. NPU 环境服务化复现清单

### 7.1 环境

```bash
source /home/developer/Ascend/cann-9.3.0/set_env.sh
cd /mnt/workspace/Works/nano-vllm
source /mnt/workspace/Envs/nano_vllm_a3/bin/activate
export NANO_VLLM_MODEL=/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master
pip install -e '.[serving]'
```

`npu-smi info` 应能看到两张可用设备。本文命令使用 device 0 和 device 1。

### 7.2 单 worker 服务

```bash
python -m nanovllm.serving.api_server \
  --model "$NANO_VLLM_MODEL" \
  --host 0.0.0.0 --port 8000 \
  --device-id 0 --tp 1 --enforce-eager \
  --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 512
```

另开终端直接验证：

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models

curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello, introduce yourself in one sentence.","max_tokens":8,"temperature":0}'

curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello, introduce yourself in one sentence.","max_tokens":8,"temperature":0,"stream":true}'

curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"1+10086等于多少？"}],"max_tokens":8,"temperature":0}'
```

### 7.3 单进程 PD 服务（CPU/单 device 调试）

单进程 PD 适合 CPU 或单 device 调试。A3 两卡复现使用 7.4。

```bash
python -m nanovllm.serving.api_server \
  --model "$NANO_VLLM_MODEL" \
  --host 0.0.0.0 --port 8000 \
  --pd --prefill-device 0 --decode-device 0 \
  --tp 1 --enforce-eager \
  --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 512
```

curl 命令与 7.2 相同，直接发到 `http://127.0.0.1:8000`。

### 7.4 两卡 PD + vllm-router 服务

按 5.3 启动 prefill worker（8101）、decode worker（8102）和 router（8080），然后：

```bash
curl -sS http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is 1+1?","max_tokens":8,"temperature":0}'

curl -N http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is 1+1?","max_tokens":8,"temperature":0,"stream":true}'

curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 1+1?"}],"max_tokens":8,"temperature":0}'
```

### 7.5 自动化验证

```bash
python scripts/verify_async_engine.py
python scripts/verify_api_server.py
python scripts/verify_pd_router.py
```

`verify_pd.py` 和 `verify_pd_concurrent.py` 用于单进程 PD 的引擎对拍，建议在 CPU 或单 device 环境运行。

或一键启动两卡 PD 服务并发送 smoke 请求：

```bash
KEEP_RUNNING=0 SMOKE_REQUEST=1 bash scripts/run_pd_router_demo.sh
```

## 8. 验证结果

### 8.1 Ascend 910_93 (A3)

在 NPU 节点上已验证：

- `verify_async_engine.py`：device 0 上正常生成。
- `verify_api_server.py`：device 0 上 `/health`、`/v1/models`、非流式补全、SSE 流式补全、chat completions 均通过；curl 直发同样通过。
- 两卡 PD + vllm-router：device 0 运行 prefill worker、device 1 运行 decode worker，router 使用 `--vllm-pd-disaggregation --policy round_robin`。curl 非流式、流式和 chat 均通过，`pd_transfer=true`，返回文本与单请求直接生成完全一致。
- `scripts/run_pd_router_demo.sh`：`KEEP_RUNNING=0 SMOKE_REQUEST=1` 在 A3 上完成两卡启动和 smoke 请求。

当前 A3 环境未将单进程 `--pd` 的双 device 启动作为验证项；两卡 PD 以 5.3 的独立 worker 模式为准。单进程 PD 引擎对拍见 8.2 CPU。

关键命令：

```bash
source /home/developer/Ascend/cann-9.3.0/set_env.sh
cd /mnt/workspace/Works/nano-vllm
source /mnt/workspace/Envs/nano_vllm_a3/bin/activate
export NANO_VLLM_MODEL=/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master
export VLLM_ROUTER_BIN=/mnt/workspace/Works/vllm-router/target/release/vllm-router

python scripts/verify_async_engine.py
python scripts/verify_api_server.py
python scripts/verify_pd_router.py
```

### 8.2 CPU

CPU 仅用于数值对拍和快速调试：

- `verify_async_engine.py`：异步事件流完整，最后事件 `finished=true`。
- `verify_api_server.py`：`/health`、`/v1/models`、非流式补全、SSE 流式补全、chat completions 均通过。
- `verify_pd.py`：PD 与直接生成 token_ids 完全一致。
- `verify_pd_router.py`：两卡 PD 路由在 CPU 上同样通过，用于不占用 NPU 时排查代码问题。

## 9. 已知限制

- 每个请求的 prefill 串行执行，长 prompt 会阻塞其他请求的 prefill。
- 单进程 `--pd` 的 KV cache 和线性 state 只在本进程内通过 CPU staging 传递。
- 单进程 `--pd` 在两个 NPU device 上同时加载两个引擎，当前 A3 环境可能受 torch_npu/驱动状态影响；两卡 PD 请使用独立 worker + vllm-router（5.3）。
- 两卡 `--role prefill/decode` 通过 HTTP 拉取 KV/state，未接入 NIXL、Mooncake 或 RDMA。
- PD 模式暂不支持 `tensor_parallel_size > 1`。
- 内部 `/internal/pd/transfer/{id}` 端点没有鉴权，只能用于同机或可信网络。
- 服务没有认证、限流和指标导出。
- 多 worker 在同一进程内轮询请求，不是独立进程模型。
- Qwen3.5 在 NPU/CPU 上保持 eager，未接入图执行。
