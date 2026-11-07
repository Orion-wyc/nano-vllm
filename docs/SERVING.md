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

如需使用两张 NPU 执行同进程 PD 分离，在环境准备完成后，将上面的单设备启动命令替换为：

```bash
python -m nanovllm.serving.api_server \
  --model "$NANO_VLLM_MODEL" \
  --host 0.0.0.0 --port 8000 --pd \
  --prefill-device 0 --decode-device 1 --tp 1 --enforce-eager \
  --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 512
```

PD 模式使用上面相同的 HTTP 端点；KV 和 Qwen3.5 的线性注意力状态经 CPU 中转，当前不支持跨节点传输。

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

### 5.2 PD 模式

```bash
python -m nanovllm.serving.api_server \
  --model /path/to/model \
  --host 0.0.0.0 \
  --port 8000 \
  --pd \
  --prefill-device 0 \
  --decode-device 1
```

PD 模式使用一个同步 `LLMEngine` 做 prefill，一个 `AsyncLLMEngine` 做 decode：

- prefill 引擎执行 prompt 并导出 KV cache；Qwen3.5 还会导出线性注意力 state。
- 导出结果通过 CPU staging 传给 decode 引擎，decode 引擎导入后继续生成。
- 同一时刻只处理一个 prefill，prefill 与 decode 串行执行，没有 overlap。
- 数据在同一进程内传递，没有跨节点网络传输。
- PD 暂不支持 `tensor_parallel_size > 1`。
- prefill 和 decode 使用同一个 device 时，`create_router` 会把 `--gpu-memory-utilization` 自动减半，避免两个引擎同时按 90% 申请 KV cache。

单设备环境可以让两个引擎共用 device 0：

```bash
python -m nanovllm.serving.api_server \
  --model /path/to/model \
  --port 8000 \
  --pd \
  --prefill-device 0 \
  --decode-device 0
```

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

### 6.3 PD 模式

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

### 6.4 CPU 与 NPU

CPU 上直接运行，设备参数保持默认或设为 0：

```bash
python scripts/verify_async_engine.py
NANO_VLLM_DECODE_DEVICE=0 python scripts/verify_pd.py
```

NPU 上先加载 CANN 并激活虚拟环境：

```bash
source /home/developer/Ascend/cann-9.3.0/set_env.sh
cd /mnt/workspace/Works/nano-vllm
source /mnt/workspace/Envs/nano_vllm_a3/bin/activate
export NANO_VLLM_MODEL=/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master

python scripts/verify_async_engine.py
python scripts/verify_api_server.py
NANO_VLLM_PREFILL_DEVICE=0 NANO_VLLM_DECODE_DEVICE=1 python scripts/verify_pd.py
```

## 7. A3 NPU 运行示例

```bash
source /home/developer/Ascend/cann-9.3.0/set_env.sh
cd /mnt/workspace/Works/nano-vllm
source /mnt/workspace/Envs/nano_vllm_a3/bin/activate
export NANO_VLLM_MODEL=/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master

pip install -e '.[serving]'

python -m nanovllm.serving.api_server \
  --model "$NANO_VLLM_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --device-id 0 \
  --enforce-eager \
  --max-model-len 512 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 512
```

另开终端验证：

```bash
curl -s http://127.0.0.1:8000/health
python scripts/verify_api_server.py
```

## 8. 验证结果

### 8.1 CPU

已验证：

- `verify_async_engine.py`：异步事件流完整，最后事件 `finished=true`。
- `verify_api_server.py`：`/health`、`/v1/models`、非流式补全、SSE 流式补全、chat completions 均通过。
- `verify_pd.py`：PD 与直接生成 token_ids 完全一致。
- `verify_pd_concurrent.py`：两个并发 PD 请求，包含 160 token 长 prompt，均与直接生成完全一致。

### 8.2 Ascend 910_93 (A3)

远程 A3 上已验证：

- `verify_async_engine.py`：device 0 上正常生成。
- `verify_pd.py`：prefill device 0、decode device 1，PD 与直接生成 token_ids 完全一致。
- `verify_pd_concurrent.py`：device 0/1 上两个并发 PD 请求，token_ids 完全一致。
- `verify_api_server.py`：device 0 上 API Server 的非流式、SSE、chat 均通过。
- PD API Server：`--pd --prefill-device 0 --decode-device 1`，`/v1/completions` 非流式和流式均通过。

关键验证命令：

```bash
source /home/developer/Ascend/cann-9.3.0/set_env.sh
cd /mnt/workspace/Works/nano-vllm
source /mnt/workspace/Envs/nano_vllm_a3/bin/activate
export NANO_VLLM_MODEL=/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master

python scripts/verify_async_engine.py
python scripts/verify_api_server.py
python scripts/verify_pd.py
python scripts/verify_pd_concurrent.py
```

## 9. 已知限制

- 每个请求的 prefill 串行执行，长 prompt 会阻塞其他请求的 prefill。
- PD 模式的 KV cache 和线性 state 只在本进程内通过 CPU staging 传递，没有跨节点网络传输。
- PD 模式暂不支持 `tensor_parallel_size > 1`。
- 服务没有认证、限流和指标导出。
- 多 worker 在同一进程内轮询请求，不是独立进程模型。
- Qwen3.5 在 NPU/CPU 上保持 eager，未接入图执行。
