# nano-vllm 在 Ascend 910_93 (A3) 上的适配说明

本文说明 nano-vllm 的 A3 NPU 适配内容，并给出两种运行方式：

- 在 NPU 环境内直接运行，不依赖 `npu-dev` 插件。
- 从 CPU 节点通过 `npu-dev` 自动同步后运行（可选）。

## 1. 适配目标

把 nano-vllm 从 CUDA 专用实现扩展到 Ascend 910_93（A3 NPU）可执行，并在 A3 上完成端到端验证。

验证环境：

| 项目 | 值 |
| --- | --- |
| 设备 | Ascend910_9362（单卡 2 die，验证使用 device 0） |
| CANN | 9.3.0，安装于 `/home/developer/Ascend/cann-9.3.0` |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post9 |
| transformers | 5.5.4 |
| Python | 3.12.9 |
| 虚拟环境 | `/mnt/workspace/Envs/nano_vllm_a3` |
| 模型 | Qwen3.5-0.8B |

常用路径：

| 内容 | 路径 |
| --- | --- |
| 仓库 | `/mnt/workspace/Works/nano-vllm` |
| CANN 环境脚本 | `/home/developer/Ascend/cann-9.3.0/set_env.sh` |
| 模型（符号链接） | `/home/developer/models/Qwen3.5-0.8B` |
| 模型（真实目录） | `/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master` |

`example.py`、`bench.py` 和验证脚本会按以下顺序解析模型目录：

1. `NANO_VLLM_MODEL`
2. `/home/developer/models/Qwen3.5-0.8B`
3. `/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master`
4. `/mnt/workspace/models/Qwen3.5-0.8B`

如果模型放在其他位置，设置 `NANO_VLLM_MODEL` 即可。

## 2. 主要改动

### 2.1 设备抽象

新增 `nanovllm/utils/device.py`，统一封装：

- 设备类型探测：`npu` / `cuda` / `cpu`
- 设备设置与同步
- 显存查询、峰值统计、`empty_cache`
- 分布式 backend：`hccl` / `nccl` / `gloo`
- Host 到设备拷贝
- `torch.compile` 保护：只在 CUDA 上启用，NPU/CPU 保持 eager

`model_runner` 中的 `torch.cuda.*`、`nccl`、`pin_memory().cuda()` 已替换为上述接口。

单卡（`tensor_parallel_size=1`）使用 `gloo` 和自动选择的空闲端口，避免 HCCL 初始化失败和固定 2333 端口冲突。多卡才使用 `hccl` / `nccl`，可通过 `NANO_VLLM_DIST_PORT` 和 `NANO_VLLM_DIST_TIMEOUT` 调整。

### 2.2 Attention

`nanovllm/layers/attention.py` 改为双路径：

- CUDA 且 `flash-attn` / Triton 可用：保留原 flash-attn 与 Triton KV 写入路径。
- NPU / CPU / Qwen3.5：使用 `F.scaled_dot_product_attention` + paged KV。

NPU 路径包括：

- prefill 无 paged：按 `cu_seqlens_q` 逐序列做 causal SDPA。
- prefill 有 paged：按 `context_lens` 和 `block_tables` 从 KV cache gather，causal 偏移为 `L - T`。
- decode paged：逐序列 gather 完整 KV，再对单 token query 做 SDPA。
- KV 写入使用 `index_copy_`，过滤 `slot_mapping < 0`。
- GQA 通过 `repeat_interleave` 展开 KV head。

`triton` 与 `flash_attn` 都改为可选导入，NPU 上不会因缺少依赖启动失败。

### 2.3 Qwen3.5 混合模型

新增 `nanovllm/models/qwen3_5.py`，文本部分结构：

- 24 层，其中 18 层 `linear_attention`（Gated DeltaNet），6 层 `full_attention`。
- `full_attention`：
  - Q 投影输出包含 query 和 gate 两部分。
  - Q/K 使用 `1 + weight` 形式的 RMSNorm。
  - partial RoPE，`rotary_dim = head_dim * 0.25`。
  - 文本场景下 MRoPE 退化为普通 partial RoPE。
  - attention 输出乘 `sigmoid(gate)` 后再过 `o_proj`。
- `linear_attention`：
  - causal conv1d、`in_proj_qkv/z/b/a`、`A_log`、`dt_bias`。
  - prefill 使用 chunk gated delta rule，decode 使用 recurrent gated delta rule。
  - 每序列维护 `conv_state` 与 `recurrent_state`。
- 最终 norm 同样使用 `1 + weight` 形式。
- `tie_word_embeddings=True`，`lm_head` 与 embedding 共享权重。
- 权重名与 checkpoint 完全对齐；vision 与 mtp 权重在 loader 中跳过。

### 2.4 状态缓存与调度

- `Sequence` 新增 `state_slot`。
- `BlockManager` 维护线性注意力状态槽的空闲列表。
- `Context` 新增 `state_slots`、`reset_slots`。
- 新序列首次 prefill 时清零对应状态槽；chunked prefill 和 decode 复用已有状态。
- Qwen3.5 关闭 prefix caching，避免线性注意力状态无法从 KV 前缀恢复。

### 2.5 KV cache

- KV cache 只按实际 full-attention 层数分配，不再为 linear 层浪费显存。
- CPU 调试时无显存信息，使用固定 256 个 block，便于本地对拍。

## 3. 在 NPU 环境直接运行（不使用 npu-dev）

适用于已经进入 NPU 容器或通过 SSH / 终端登录 NPU 节点的场景。下文的 `python` 均指 venv 中的 Python。

### 3.1 首次准备

如果 NPU 上已经有代码、环境和模型，可直接跳到 3.3。没有时按下面步骤准备。

加载 CANN：

```bash
source /home/developer/Ascend/cann-9.3.0/set_env.sh
```

检查设备：

```bash
npu-smi info
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
```

获取代码：

```bash
cd /mnt/workspace/Works
git clone git@github.com:Orion-wyc/nano-vllm.git
cd nano-vllm
git checkout feat/a3_integration
```

如果 NPU 无法访问 Git 远程，也可以从 CPU 节点用 `scp` / `rsync` 把仓库目录拷贝到同路径。

创建环境：

```bash
cd /mnt/workspace/Works/nano-vllm
bash scripts/setup_a3_env.sh
```

脚本会创建 `/mnt/workspace/Envs/nano_vllm_a3`，安装 torch 2.9.0、torch_npu 2.9.0.post9、transformers 5.5.4 等依赖，并执行 NPU 环境检查。

模型准备：

```bash
ls /home/developer/models/Qwen3.5-0.8B
ls /home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master
```

如果模型不在上述路径，先拷贝或下载，并在运行时设置：

```bash
export NANO_VLLM_MODEL=/path/to/Qwen3.5-0.8B
```

### 3.2 每次新开终端的最小步骤

```bash
source /home/developer/Ascend/cann-9.3.0/set_env.sh
cd /mnt/workspace/Works/nano-vllm
source /mnt/workspace/Envs/nano_vllm_a3/bin/activate
export NANO_VLLM_MODEL=/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master
```

之后可以直接运行本文中的任意 Python 命令。

### 3.3 运行方式

检查 NPU 环境：

```bash
python scripts/check_npu_env.py
```

最小生成：

```bash
python scripts/smoke_generate.py
```

完整示例：

```bash
python example.py
```

基准测试：

```bash
python bench.py
```

封装脚本也可以直接使用，它会自动 source CANN、激活 venv 并解析模型路径：

```bash
bash scripts/run_a3_example.sh
```

可覆盖的环境变量：

| 变量 | 作用 |
| --- | --- |
| `NANO_VLLM_MODEL` | 覆盖模型目录 |
| `NANO_VLLM_ENV` | 覆盖 venv 目录 |
| `NANO_VLLM_TP` | 覆盖 tensor parallel size |
| `NANO_VLLM_EAGER` | bench 中控制是否 eager |
| `NANO_VLLM_TEMPERATURE` | smoke 采样温度 |
| `NANO_VLLM_MAX_TOKENS` | smoke 最大生成 token 数 |
| `NANO_VLLM_PROMPT` | smoke 提示词 |
| `NANO_VLLM_MAX_NUM_SEQS` | smoke 最大并发序列数，默认 8 |
| `NANO_VLLM_MAX_BATCHED_TOKENS` | smoke 最大批处理 token 数，默认 512 |
| `NANO_VLLM_DIST_PORT` | 多卡分布式端口，默认 2333 |
| `NANO_VLLM_DIST_TIMEOUT` | 分布式初始化超时秒数，默认 300 |

示例：

```bash
NANO_VLLM_TEMPERATURE=1e-5 NANO_VLLM_MAX_TOKENS=16 python scripts/smoke_generate.py
NANO_VLLM_MODEL=/data/models/Qwen3.5-0.8B python example.py
NANO_VLLM_EAGER=1 python bench.py
```

### 3.4 一条命令运行

不激活 venv 时，可以直接使用绝对路径：

```bash
source /home/developer/Ascend/cann-9.3.0/set_env.sh && \
cd /mnt/workspace/Works/nano-vllm && \
NANO_VLLM_MODEL=/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master \
/mnt/workspace/Envs/nano_vllm_a3/bin/python example.py
```

或：

```bash
bash -lc '
source /home/developer/Ascend/cann-9.3.0/set_env.sh
cd /mnt/workspace/Works/nano-vllm
source /mnt/workspace/Envs/nano_vllm_a3/bin/activate
export NANO_VLLM_MODEL=/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master
python scripts/smoke_generate.py
'
```

### 3.5 运行验证脚本

Paged attention 单元验证，自带参考实现，不需要 CPU 参考文件：

```bash
python scripts/verify_npu_attention.py
```

多请求 + chunked prefill 端到端验证：

```bash
python scripts/verify_npu_generate.py
```

模型 logits 对拍：

```bash
# CPU 节点生成参考
python scripts/verify_npu_logits.py --save .ref_logits.pt

# 把 .ref_logits.pt 放到 NPU 仓库目录后执行
python scripts/verify_npu_logits.py --compare .ref_logits.pt
```

### 3.6 NPU 环境注意事项

- 每个新终端都需要重新 `source set_env.sh`。
- 不需要设置 `ASCEND_HOME_PATH`、`LD_LIBRARY_PATH`，`set_env.sh` 会处理。
- `torch_npu` 与 CANN 版本不匹配时，`torch.npu.is_available()` 可能为 `True`，但算子执行失败；优先使用本文表格中的版本组合。
- 首次运行会创建线性注意力状态缓存，默认 `max_num_seqs=512` 时约 9 GB，属于预期行为。

## 4. 环境准备与可选远程同步

### 4.1 CPU 节点

CPU 节点只用于数值对拍，不执行 NPU 算子：

```bash
cd /mnt/workspace/Works/nano-vllm
bash scripts/setup_a3_env.sh
```

CPU 没有 NPU，`check_npu_env.py` 会显示 `npu available False`，属正常。

### 4.2 使用 npu-dev 从 CPU 同步到 NPU（可选）

如果不在 NPU 环境内直接操作，可以通过 `npu-dev` 自动同步代码、环境和模型。远程实例可能在会话之间切换 pod，建议始终让 `run` 自动同步。

初始化同步配置：

```bash
NPU_DEV=/home/developer/.config/opencode/skills/npu-dev/scripts/npu_dev.py
python3 "$NPU_DEV" init
```

同步目录建议包含：

```text
/mnt/workspace/Works/nano-vllm
/mnt/workspace/Envs/nano_vllm_a3
/home/developer/models/models/Qwen--Qwen3.5-0.8B
exclude /mnt/workspace/Works/nano-vllm/.git
exclude /mnt/workspace/Works/nano-vllm/.npu-dev
exclude /mnt/workspace/Works/nano-vllm/**/__pycache__
```

远程环境检查：

```bash
python3 "$NPU_DEV" run \
  --workdir /mnt/workspace/Works/nano-vllm \
  --env-file /home/developer/Ascend/cann-9.3.0/set_env.sh \
  -- /mnt/workspace/Envs/nano_vllm_a3/bin/python scripts/check_npu_env.py
```

远程示例：

```bash
python3 "$NPU_DEV" run \
  --workdir /mnt/workspace/Works/nano-vllm \
  --env-file /home/developer/Ascend/cann-9.3.0/set_env.sh \
  -- /mnt/workspace/Envs/nano_vllm_a3/bin/python example.py
```

## 5. 验证结果

### 5.1 模型数学对拍

`scripts/verify_npu_logits.py` 直接构建模型并执行 prefill-only forward，不经过 paged KV。

CPU 与 transformers 原实现对拍：

```text
top5 一致
cosine = 0.999892
max_abs_diff = 0.21875
```

NPU 与 CPU 对拍：

```text
top5 一致
cosine = 0.999903
max_abs_diff = 0.234375
mean_abs_diff = 0.027975
```

误差来自 bf16 舍入，top-5 完全一致。

### 5.2 Paged attention 对拍

`scripts/verify_npu_attention.py` 用随机 Q/K/V 覆盖三种路径，并与手写 softmax 参考实现对比：

| 路径 | cosine | max_abs_diff |
| --- | --- | --- |
| prefill paged | 0.999999 | 0.006322 |
| chunked prefill paged | 0.999999 | 0.001951 |
| decode paged | 0.999998 | 0.003423 |

### 5.3 端到端确定性对拍

`scripts/verify_npu_generate.py` 使用 `temperature=1e-5` 近似贪心，覆盖两个请求，其中第二个是 160 token 长 prompt，用于触发 chunked prefill。

CPU 与 NPU 输出 token 完全一致：

```text
request=0 token_ids=[271, 9419, 11, 353, 1044, 264, 795, 26578, 13, 271, 248068, 271, 248069, 271, 9419, 11]
request=1 token_ids=[23066, 1814, 23066, 1814, 23066, 1814, 23066, 1814, 23066, 1814, 23066, 1814, 23066, 1814, 23066, 1814]
```

### 5.4 example.py

A3 上 `example.py` 正常生成。实测输出示例：

```text
Prompt: introduce yourself
Completion: Hello! I'm Qwen3.5, a large language model developed by Tongyi Lab ...

Prompt: list all prime numbers within 100
Completion: Here are all the prime numbers within the range of 100: 2, 3, 5, ...
```

单卡实测吞吐（Qwen3.5-0.8B，eager）：

```text
Prefill ≈ 58 tok/s
Decode  ≈ 9-19 tok/s
```

## 6. 复现验证清单

### 6.1 NPU 直接运行

1. 加载环境：

   ```bash
   source /home/developer/Ascend/cann-9.3.0/set_env.sh
   cd /mnt/workspace/Works/nano-vllm
   source /mnt/workspace/Envs/nano_vllm_a3/bin/activate
   ```

2. 检查设备和环境：

   ```bash
   npu-smi info
   python scripts/check_npu_env.py
   ```

3. 验证 paged attention：

   ```bash
   python scripts/verify_npu_attention.py
   ```

4. 验证 batching 与 chunked prefill：

   ```bash
   python scripts/verify_npu_generate.py
   ```

5. 运行 smoke 和 example：

   ```bash
   python scripts/smoke_generate.py
   python example.py
   ```

6. logits 对拍：

   ```bash
   # CPU 节点
   python scripts/verify_npu_logits.py --save .ref_logits.pt

   # NPU 节点，确保 .ref_logits.pt 已在仓库目录
   python scripts/verify_npu_logits.py --compare .ref_logits.pt

   rm -f .ref_logits.pt
   ```

### 6.2 npu-dev 模式

1. 同步并检查环境：

   ```bash
   python3 "$NPU_DEV" run \
     --workdir /mnt/workspace/Works/nano-vllm \
     --env-file /home/developer/Ascend/cann-9.3.0/set_env.sh \
     -- /mnt/workspace/Envs/nano_vllm_a3/bin/python scripts/check_npu_env.py
   ```

2. 依次运行验证脚本：

   ```bash
   for script in verify_npu_attention.py verify_npu_generate.py smoke_generate.py example.py; do
     python3 "$NPU_DEV" run \
       --workdir /mnt/workspace/Works/nano-vllm \
       --env-file /home/developer/Ascend/cann-9.3.0/set_env.sh \
       -- /mnt/workspace/Envs/nano_vllm_a3/bin/python "$script"
   done
   ```

## 7. 已知限制

- Qwen3.5 当前只支持 `tensor_parallel_size=1`。
- NPU 上强制 eager，未接入 `torch.npu` graph capture。
- Attention 使用逐序列 SDPA，decode batch 较大时 Python 循环和 D2H 同步开销明显。
- Gated DeltaNet 使用纯 torch 回退，未接入 NPU 融合算子。
- Qwen3.5 关闭 prefix caching。
- 线性注意力 recurrent state 使用 float32，默认 `max_num_seqs=512` 时约占 9 GB 显存。
- `torch.compile` 在 NPU/CPU 上禁用。

后续可优化方向：NPU Graph、Fused Infer Attention、GDN 融合算子、full-attention 层 prefix caching、多卡 HCCL tensor parallel。

## 8. 常见问题

### CANN 未加载

现象：`torch_npu` 导入失败或算子找不到。

处理：

```bash
source /home/developer/Ascend/cann-9.3.0/set_env.sh
```

`npu-dev run` 使用 `--env-file` 自动加载；直接运行时需要在每个新终端手动 source。

### 远程实例切换 pod

现象：刚安装的 venv 或 tqdm 在下次执行时消失。

原因是分机可能在会话之间切到不同 pod。使用 `npu-dev` 时始终让 `run` 自动同步；直接在 NPU 环境运行时，需要确认当前 pod 上代码、环境和模型仍然存在。

### 偶发 AI Core 异常

验证中遇到过一次 `L0B read/write conflict` 的偶发 AI Core 错误，重新执行后恢复正常。先检查 `npu-smi info` 的设备健康状态，再重试。

### smoke_generate.py 卡住

先看阶段输出：

```text
[1/5] model ...
[2/5] loading engine ...
[3/5] generating
[4/5] ...
[5/5] exiting
```

卡在 `[2/5]` 通常是设备初始化、模型加载或残留进程占用 device；卡在 `[3/5]` 通常是 prefill/decode 执行。

排查步骤：

```bash
npu-smi info
ps -eo pid,ppid,stat,etime,wchan:30,cmd | grep -E "[s]moke_generate|[e]xample.py"
ss -ltnp | grep -E ":2333|:2345|:2346|:2347"
```

如果进程状态是 `D`（不可中断睡眠），`kill -9` 也无法结束，说明卡在驱动层，需要重启 NPU 实例或容器：

```bash
bash /usr/local/bin/remote_tool.sh shutdown
# 等待 NPU_PHASE=STOPPED 后重新运行，npu-dev 会自动开机并同步
python3 "$NPU_DEV" run ... -- python scripts/smoke_generate.py
```

如果只是状态缓存分配或 warmup 较慢，先用小配置验证：

```bash
NANO_VLLM_MAX_NUM_SEQS=8 \
NANO_VLLM_MAX_BATCHED_TOKENS=512 \
timeout 600 python -u scripts/smoke_generate.py
```

单卡已经改用 `gloo` + 空闲端口，正常情况下不会再受 2333 端口冲突影响。多卡如果遇到端口冲突，设置：

```bash
export NANO_VLLM_DIST_PORT=24567
export NANO_VLLM_DIST_TIMEOUT=300
```

### CPU 上无法分配 KV cache

CPU 没有 `mem_get_info`，`allocate_kv_cache` 已固定回退到 256 个 block。CPU 仅用于数值对拍，不用于性能测试。

### venv 中缺少 tqdm

在 `--system-site-packages` 的 venv 中，如果全局 tqdm 不可导入，执行：

```bash
/mnt/workspace/Envs/nano_vllm_a3/bin/python -m pip install --ignore-installed --no-deps tqdm
```

`scripts/setup_a3_env.sh` 已包含该处理。

## 9. 变更文件

新增：

- `nanovllm/utils/device.py`
- `nanovllm/models/qwen3_5.py`
- `scripts/setup_a3_env.sh`
- `scripts/run_a3_example.sh`
- `scripts/check_npu_env.py`
- `scripts/smoke_generate.py`
- `scripts/verify_npu_logits.py`
- `scripts/verify_npu_attention.py`
- `scripts/verify_npu_generate.py`

修改：

- `nanovllm/config.py`
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/llm_engine.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/sequence.py`
- `nanovllm/layers/activation.py`
- `nanovllm/layers/attention.py`
- `nanovllm/layers/layernorm.py`
- `nanovllm/layers/rotary_embedding.py`
- `nanovllm/layers/sampler.py`
- `nanovllm/utils/context.py`
- `nanovllm/utils/loader.py`
- `example.py`
- `bench.py`
- `pyproject.toml`
- `.gitignore`
