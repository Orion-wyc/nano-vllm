import asyncio
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm import SamplingParams
from nanovllm.serving.async_engine import AsyncLLMEngine


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


def pick_token_ids(events):
    for event in reversed(events):
        token_ids = event.get("token_ids") or []
        if token_ids:
            return token_ids
    return []


async def main():
    path = resolve_model_path()
    prompt = os.environ.get("NANO_VLLM_PROMPT", "Hello, introduce yourself in one sentence.")
    temperature = float(os.environ.get("NANO_VLLM_TEMPERATURE", "1e-5"))
    max_tokens = int(os.environ.get("NANO_VLLM_MAX_TOKENS", "16"))
    device_id = int(os.environ.get("NANO_VLLM_DEVICE_ID", "0"))
    engine = None
    try:
        engine = AsyncLLMEngine(
            path,
            enforce_eager=True,
            max_model_len=512,
            max_num_seqs=8,
            max_num_batched_tokens=512,
            device_id=device_id,
        )
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        events = []
        async for event in engine.generate(prompt, sampling_params):
            events.append(event)
        token_ids = pick_token_ids(events)
        last = events[-1] if events else {}
        text = last.get("text")
        if text is None:
            text = last.get("delta_text", "")
        print(f"model={path}")
        print(f"prompt={prompt!r}")
        print(f"events={len(events)}")
        print(f"token_ids={token_ids}")
        print(f"text={text!r}")
        print(f"finished={last.get('finished')}")
        if not events:
            print("ERROR: no events received", file=sys.stderr)
            return 1
        if not token_ids:
            print("ERROR: empty token_ids", file=sys.stderr)
            return 1
        if not last.get("finished"):
            print("ERROR: last event is not finished", file=sys.stderr)
            return 1
        return 0
    finally:
        if engine is not None:
            await engine.shutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
