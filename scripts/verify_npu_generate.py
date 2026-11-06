import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm import LLM, SamplingParams


def resolve_model_path():
    env_path = os.environ.get("NANO_VLLM_MODEL")
    candidates = [env_path] if env_path else []
    candidates += [
        "/home/developer/models/Qwen3.5-0.8B",
        "/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master",
        "/mnt/workspace/models/Qwen3.5-0.8B",
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(os.path.expanduser(candidate)):
            return os.path.expanduser(candidate)
    raise FileNotFoundError("model directory not found; set NANO_VLLM_MODEL")


def main():
    path = resolve_model_path()
    prompts = [
        "Hello, introduce yourself in one sentence.",
        " ".join(["hello world"] * 80),
    ]
    temperature = float(os.environ.get("NANO_VLLM_TEMPERATURE", "1e-5"))
    max_tokens = int(os.environ.get("NANO_VLLM_MAX_TOKENS", "16"))
    llm = LLM(
        path,
        enforce_eager=True,
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        tensor_parallel_size=1,
    )
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    for index, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"request={index} prompt_tokens={len(llm.tokenizer.encode(prompt))}")
        print(f"request={index} token_ids={output['token_ids']}")
    llm.exit()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
