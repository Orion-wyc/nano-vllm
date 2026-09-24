# 参考资料

本页汇总 nano-vllm 学习过程中值得参考的架构解析、源码精讲与教程。建议先通读源码，再结合下列资料对照理解。

## 基础知识

- [LLM 推理并行优化的必备知识](https://zhuanlan.zhihu.com/p/1937449564509545940)
  知乎专栏文章，梳理 LLM 推理中并行优化的基础，包括矩阵按列/行切分的计算原理，以及 DP、TP、SP、EP 等策略的组合方式，并讨论 PD 分离、MoE 模型并行等推理场景下的优化思路。理解张量并行的列切/行切是读懂 nano-vllm `linear.py` 中各类 ParallelLinear 的前提。

- [大模型训练并行技术理解：DP/TP/PP/SP/EP](https://zhuanlan.zhihu.com/p/1904506837543420662)
  对比数据并行、张量并行、流水线并行、序列并行与专家并行的切分对象、通信模式与适用场景，适合建立并行策略的整体概念。nano-vllm 使用 TP（`tensor_parallel_size`），其中 `ColumnParallelLinear`/`RowParallelLinear` 的 all-reduce 通信正对应文中 TP 的通信模式。

- [深度学习的分布式训练与集合通信（三）](https://www.hiascend.com/developer/techArticles/20250207-1)
  昇腾社区官方技术文章，系统讲解序列并行（SP）、上下文并行（CP）、混合序列并行 Ulysses、ZeRO 系列与 FSDP，逐一分析各策略涉及的 AllGather / ReduceScatter / AlltoAll 等集合通信操作，并在文末汇总并行方案与通信模式的对应关系。文中提到这些通信模式在 HCCL API 中均已支持，便于对照昇腾平台上的实现。

## 架构与源码解析

- [Structure of Nano-vLLM](https://github.com/CalvinXKY/nano-vllm/blob/main/docs/structures.md)
  英文架构说明，梳理整体架构与四大核心组件：LLM Engine、Block Manager、Scheduler、Model Runner，并给出各模块的执行流程与代码组织方式。适合先建立全局视图。

- [TuNaiChao/learn-nano-vllm](https://github.com/TuNaiChao/learn-nano-vllm)
  上游项目的 fork，代码保持原样，新增一套面向零基础的中文学习文档（约 10 篇）。涵盖导论、全局架构、AI Infra 理论（RoPE、KV Cache、PagedAttention、Flash Attention、Tensor Parallelism、CUDA Graph 等）、Python 语法清单和逐文件代码精讲，并提供「系统精读 / 快速了解 / 面试速查」三条学习路径。

## 中文教程

- [Nano-vLLM 学习教程：基于 Qwen3-0.6B 模型的深度剖析](https://docs.d.run/blogs/2026/nano-vllm)
  以 Qwen3-0.6B 为样例，从模型结构、`config.json`、GQA 讲起，逐类解析 `LLMEngine`、`Sequence`、`BlockManager`、`Scheduler`、`ModelRunner`、`load_model` 及各层算子实现（activation、embed_head、linear、attention、Qwen3ForCausalLM）。内容与源码章节一一对应，适合作为逐行阅读的对照材料。

- [知乎：nano-vllm 源码解析](https://zhuanlan.zhihu.com/p/2008285806222132143)
  知乎专栏文章，围绕 nano-vllm 的推理流程与关键机制展开分析，可作为上述教程的补充阅读。

## 昇腾 NPU 版本

当前开发环境为 Ascend NPU，以下为 nano-vllm 的昇腾移植版本与相关记录。

- [linzm1007/nano-vllm-ascend](https://github.com/linzm1007/nano-vllm-ascend)
  功能较完整的昇腾 NPU 移植版，将 GPU 侧的 FlashAttention / Triton 实现替换为 NPU 原生算子（如 `npu_fused_infer_attention_score_v2`、`_npu_reshape_and_cache`），引入 TorchAir Ascend IR 图编译与图缓存。支持 Qwen3、Qwen2、Llama、Qwen3-MoE、Qwen3-VL、MiniCPM4 等多个模型，含 PageAttention、自定义算子、在线推理和大量 Mermaid 流程图文档（engine / layers / models 三个模块）。仓库另提供 CPU 版 [nano-vllm-cpu](https://github.com/linzm1007/nano-vllm-cpu)。适合作为在昇腾上复现与优化的主要参考。

- [TobyMint/nano-vllm-ascend](https://github.com/TobyMint/nano-vllm-ascend)
  另一份昇腾适配的 fork，改动相对精简，保留约 1200 行核心实现，使用前缀缓存、张量并行与 NPU graph。仓库内 `install-ascend.md` 给出昇腾开发环境的部署步骤，适合快速跑通并对照上游代码理解移植点。

- [博客园：Nano-vLLM-Ascend](https://www.cnblogs.com/linzm14/p/19318861)
  作者对 linzm1007/nano-vllm-ascend 的环境搭建记录，包含镜像拉取、容器运行参数、SSH 配置、依赖安装、模型下载与 benchmark 数据，可作为昇腾容器化部署的操作参考。

## 推荐阅读顺序

1. 本仓库 `README.md` 与 `example.py`，跑通一次离线推理。
2. Structure of Nano-vLLM，建立整体架构认知。
3. 结合 d.run 教程逐模块阅读 `nanovllm/` 源码。
4. 需要补理论基础或准备面试时，查 TuNaiChao/learn-nano-vllm 的理论篇与速查篇。
5. 迁移到昇腾 NPU 时，对照 linzm1007/nano-vllm-ascend 的算子替换与图编译改动，部署细节参考其博客记录与 TobyMint 版的 `install-ascend.md`。
