<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Ascend NPU (A3)

nano-vllm can run on Ascend 910_93 (A3) with `torch_npu`. The port keeps the CUDA
flash-attn path and adds an NPU paged-attention path plus native Qwen3.5 hybrid
attention support.

```bash
bash scripts/setup_a3_env.sh
bash scripts/run_a3_example.sh
```

See [docs/A3_NPU_ADAPTATION.md](docs/A3_NPU_ADAPTATION.md) for environment setup,
remote A3 verification with `npu-dev`, limitations, and reproduction steps.

## Online Serving

Install the serving extras and start the OpenAI-style API server:

```bash
pip install -e '.[serving]'
python -m nanovllm.serving.api_server \
  --model /path/to/model \
  --host 0.0.0.0 \
  --port 8000
```

Verify the async engine, API server, and PD mode:

```bash
python scripts/verify_async_engine.py
python scripts/verify_api_server.py
python scripts/verify_pd.py
python scripts/verify_pd_concurrent.py
```

On a single-device machine, set `NANO_VLLM_DECODE_DEVICE=0` for the PD scripts.
See [docs/SERVING.md](docs/SERVING.md) for endpoints, streaming, multi-worker,
PD mode, and A3 NPU examples.

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)