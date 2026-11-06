import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm import LLM, SamplingParams


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
    print(f"[1/5] model: {path}", flush=True)
    max_num_seqs = int(os.environ.get("NANO_VLLM_MAX_NUM_SEQS", "8"))
    max_num_batched_tokens = int(os.environ.get("NANO_VLLM_MAX_BATCHED_TOKENS", "512"))
    print(f"[2/5] loading engine max_num_seqs={max_num_seqs} max_num_batched_tokens={max_num_batched_tokens}", flush=True)
    llm = LLM(
        path,
        enforce_eager=True,
        max_model_len=512,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        tensor_parallel_size=1,
    )
    print("[3/5] generating", flush=True)
    prompt = os.environ.get("NANO_VLLM_PROMPT", "Hello, introduce yourself in one sentence.")
    temperature = float(os.environ.get("NANO_VLLM_TEMPERATURE", "0.6"))
    max_tokens = int(os.environ.get("NANO_VLLM_MAX_TOKENS", "32"))
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    output = outputs[0]
    print(f"[4/5] prompt: {prompt!r}", flush=True)
    print(f"[4/5] completion: {output['text']!r}", flush=True)
    print(f"[4/5] token_ids: {output['token_ids']}", flush=True)
    print("[5/5] exiting", flush=True)
    llm.exit()
    print("[5/5] done", flush=True)
    if not output["text"]:
        print("ERROR: empty completion", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
