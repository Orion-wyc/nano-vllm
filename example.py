import os

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


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


def main():
    path = resolve_model_path()
    tokenizer = AutoTokenizer.from_pretrained(path)
    tensor_parallel_size = int(os.environ.get("NANO_VLLM_TP", "1"))
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=tensor_parallel_size)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
