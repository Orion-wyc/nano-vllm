import asyncio
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm import SamplingParams
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.serving.async_engine import AsyncLLMEngine
from nanovllm.serving.pd import PDRouter
from nanovllm.utils.device import empty_cache


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


async def collect(router, prompt, sampling_params, request_id):
    token_ids = []
    text = ""
    async for event in router.generate(prompt, sampling_params, request_id=request_id):
        token_ids = event.get("token_ids") or token_ids
        text = event.get("text", text)
    return token_ids, text


async def main():
    path = resolve_model_path()
    prompts = [
        os.environ.get("NANO_VLLM_PROMPT", "Hello, introduce yourself in one sentence."),
        " ".join(["hello world"] * 80),
    ]
    prefill_device = int(os.environ.get("NANO_VLLM_PREFILL_DEVICE", "0"))
    decode_device = int(os.environ.get("NANO_VLLM_DECODE_DEVICE", "1"))
    max_tokens = int(os.environ.get("NANO_VLLM_MAX_TOKENS", "16"))
    gpu_memory_utilization = float(os.environ.get("NANO_VLLM_GPU_UTIL", "0.3"))
    engine_kwargs = {
        "enforce_eager": True,
        "max_model_len": 512,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 128,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    params = [SamplingParams(temperature=1e-5, max_tokens=max_tokens) for _ in prompts]

    router = None
    direct = None
    try:
        direct = LLMEngine(path, device_id=prefill_device, **engine_kwargs)
        direct_outputs = direct.generate(prompts, params, use_tqdm=False)
        direct.exit()
        direct = None
        empty_cache()

        prefill_engine = LLMEngine(path, device_id=prefill_device, **engine_kwargs)
        decode_engine = AsyncLLMEngine(path, device_id=decode_device, **engine_kwargs)
        router = PDRouter(prefill_engine, decode_engine)
        results = await asyncio.gather(
            *[
                collect(router, prompt, params[index], f"pd-{index}")
                for index, prompt in enumerate(prompts)
            ]
        )

        ok = True
        for index, ((pd_token_ids, pd_text), direct_output) in enumerate(zip(results, direct_outputs)):
            match = pd_token_ids == direct_output["token_ids"]
            ok = ok and match
            print(f"request={index} prompt_tokens={len(prefill_engine.tokenizer.encode(prompts[index]))}")
            print(f"request={index} match={match}")
            print(f"request={index} pd_token_ids={pd_token_ids}")
            print(f"request={index} direct_token_ids={direct_output['token_ids']}")
            print(f"request={index} pd_text={pd_text!r}")
        print(f"prefill_device={prefill_device} decode_device={decode_device}")
        print(f"ALL_MATCH={ok}")
        return 0 if ok else 1
    finally:
        if router is not None:
            await router.shutdown()
        if direct is not None:
            direct.exit()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
