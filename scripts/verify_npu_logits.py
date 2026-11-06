import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

from nanovllm.utils.device import device_type, set_device, dist_backend


def resolve_model_path(value):
    candidates = [value] if value else []
    candidates += [
        os.environ.get("NANO_VLLM_MODEL"),
        "/home/developer/models/Qwen3.5-0.8B",
        "/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master",
        "/mnt/workspace/models/Qwen3.5-0.8B",
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(os.path.expanduser(candidate)):
            return os.path.expanduser(candidate)
    raise FileNotFoundError("model directory not found; set --model or NANO_VLLM_MODEL")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default="Hello, introduce yourself in one sentence.")
    parser.add_argument("--save", default=None)
    parser.add_argument("--compare", default=None)
    parser.add_argument("--port", type=int, default=23470)
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    kind = device_type()
    if kind == "npu":
        import torch_npu  # noqa: F401

    set_device(0)
    dist.init_process_group(
        dist_backend(),
        init_method=f"tcp://127.0.0.1:{args.port}",
        world_size=1,
        rank=0,
    )

    from transformers import AutoConfig, AutoTokenizer
    from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
    from nanovllm.utils.loader import load_model
    from nanovllm.utils.context import set_context, reset_context

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(kind if kind != "cpu" else "cpu")

    config = AutoConfig.from_pretrained(model_path).text_config
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    input_ids = torch.tensor(token_ids, dtype=torch.int64)
    positions = torch.arange(len(token_ids), dtype=torch.int64)
    seqlen = len(token_ids)

    model = Qwen3_5ForCausalLM(config)
    load_model(model, model_path)
    model.allocate_state_cache(2)
    model.eval()

    set_context(
        True,
        cu_seqlens_q=torch.tensor([0, seqlen], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, seqlen], dtype=torch.int32),
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        slot_mapping=None,
        context_lens=torch.tensor([seqlen], dtype=torch.int32),
        block_tables=None,
        state_slots=[0],
        reset_slots=[0],
    )
    start = time.time()
    with torch.inference_mode():
        logits = model.compute_logits(model(input_ids, positions))[0].float().cpu()
    reset_context()
    elapsed = time.time() - start

    print(f"device={kind} model={model_path}")
    print(f"tokens={seqlen} time={elapsed:.3f}s")
    print(f"top5={logits.topk(5).indices.tolist()}")
    print(f"sum={logits.sum().item():.6f}")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"logits": logits, "token_ids": token_ids, "prompt": args.prompt}, args.save)
        print(f"saved={args.save}")

    if args.compare:
        ref = torch.load(args.compare, map_location="cpu")
        ref_logits = ref["logits"].float()
        diff = (logits - ref_logits).abs()
        cosine = torch.nn.functional.cosine_similarity(logits, ref_logits, dim=0).item()
        print(f"compare={args.compare}")
        print(f"max_abs_diff={diff.max().item():.6f}")
        print(f"mean_abs_diff={diff.mean().item():.6f}")
        print(f"cosine={cosine:.6f}")
        print(f"ref_top5={ref_logits.topk(5).indices.tolist()}")
        print(f"token_ids_match={token_ids == ref['token_ids']}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
