import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from nanovllm.layers.attention import Attention
from nanovllm.utils.context import set_context, reset_context
from nanovllm.utils.device import device_type


def manual_attention(q, k, v, scale, q_offset=0):
    kv_len = k.shape[0]
    q_len = q.shape[0]
    k = k.float()
    v = v.float()
    q = q.float()
    repeat = q.shape[1] // k.shape[1]
    if repeat > 1:
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    scores = torch.einsum("thd,shd->hts", q, k) * scale
    rows = torch.arange(q_len, device=q.device)
    cols = torch.arange(kv_len, device=q.device)
    allowed = cols[None, :] <= rows[:, None] + (kv_len - q_len)
    scores = scores.masked_fill(~allowed[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hts,shd->thd", probs, v)
    return out.to(q.dtype)


def make_inputs(device, tokens, num_heads=8, kv_heads=2, head_dim=64):
    torch.manual_seed(0)
    q = torch.randn(tokens, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    return q, k, v


def report(name, out, ref):
    diff = (out.float() - ref.float()).abs()
    cosine = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    print(f"{name}: max={diff.max().item():.6f} mean={diff.mean().item():.6f} cosine={cosine:.6f}")


def main():
    kind = device_type()
    if kind == "npu":
        import torch_npu  # noqa: F401
    dev = torch.device(kind if kind != "cpu" else "cpu")
    torch.set_default_device(dev)
    torch.set_default_dtype(torch.bfloat16)
    print("device", kind)

    num_heads, kv_heads, head_dim = 8, 2, 64
    scale = head_dim ** -0.5
    block_size = 256
    attn = Attention(num_heads, head_dim, scale, kv_heads)
    attn.k_cache = torch.zeros(4, block_size, kv_heads, head_dim, device=dev)
    attn.v_cache = torch.zeros(4, block_size, kv_heads, head_dim, device=dev)

    full_q, full_k, full_v = make_inputs(dev, 64, num_heads, kv_heads, head_dim)

    tokens = 64
    block_tables = torch.tensor([[0, -1, -1, -1]], dtype=torch.int32, device=dev)
    slots = torch.arange(tokens, dtype=torch.int32, device=dev)

    set_context(
        True,
        cu_seqlens_q=torch.tensor([0, tokens], dtype=torch.int32, device=dev),
        cu_seqlens_k=torch.tensor([0, tokens], dtype=torch.int32, device=dev),
        max_seqlen_q=tokens,
        max_seqlen_k=tokens,
        slot_mapping=slots,
        context_lens=torch.tensor([tokens], dtype=torch.int32, device=dev),
        block_tables=block_tables,
        state_slots=[0],
        reset_slots=[0],
    )
    out = attn(full_q, full_k, full_v)
    ref = manual_attention(full_q, full_k, full_v, scale)
    report("prefill_paged", out, ref)
    reset_context()

    chunk = 16
    q = full_q[tokens - chunk:]
    k = full_k[tokens - chunk:]
    v = full_v[tokens - chunk:]
    set_context(
        True,
        cu_seqlens_q=torch.tensor([0, chunk], dtype=torch.int32, device=dev),
        cu_seqlens_k=torch.tensor([0, tokens], dtype=torch.int32, device=dev),
        max_seqlen_q=chunk,
        max_seqlen_k=tokens,
        slot_mapping=torch.arange(tokens - chunk, tokens, dtype=torch.int32, device=dev),
        context_lens=torch.tensor([tokens], dtype=torch.int32, device=dev),
        block_tables=block_tables,
        state_slots=[0],
        reset_slots=[0],
    )
    out = attn(q, k, v)
    ref = manual_attention(q, full_k, full_v, scale)
    report("chunk_prefill_paged", out, ref)
    reset_context()

    new_q, new_k, new_v = make_inputs(dev, 1, num_heads, kv_heads, head_dim)
    all_k = torch.cat([full_k, new_k], dim=0)
    all_v = torch.cat([full_v, new_v], dim=0)
    set_context(
        False,
        slot_mapping=torch.tensor([tokens], dtype=torch.int32, device=dev),
        context_lens=torch.tensor([tokens + 1], dtype=torch.int32, device=dev),
        block_tables=block_tables,
        state_slots=[0],
        reset_slots=None,
    )
    out = attn(new_q, new_k, new_v)
    ref = manual_attention(new_q, all_k, all_v, scale)
    report("decode_paged", out, ref)
    reset_context()


if __name__ == "__main__":
    main()
