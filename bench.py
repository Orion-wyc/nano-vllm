import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def resolve_model_path():
    env_path = os.environ.get("NANO_VLLM_MODEL")
    if env_path:
        path = os.path.expanduser(env_path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"NANO_VLLM_MODEL is not a directory: {path}")
        return path
    candidates = [
        "/home/developer/models/Qwen3.5-0.8B",
        "/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master",
        "/mnt/workspace/models/Qwen3.5-0.8B",
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "model directory not found, checked: " + ", ".join(candidates)
    )


def env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = resolve_model_path()
    enforce_eager = env_flag("NANO_VLLM_EAGER", True)
    tensor_parallel_size = int(os.environ.get("NANO_VLLM_TP", "1"))
    llm = LLM(
        path,
        enforce_eager=enforce_eager,
        max_model_len=4096,
        tensor_parallel_size=tensor_parallel_size,
    )

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
