import torch
from torch import nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    _HAS_FLASH_ATTN = True
except Exception:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
    _HAS_FLASH_ATTN = False

from nanovllm.utils.context import get_context


if _HAS_TRITON:

    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def _store_kvcache_torch(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    num_slots = k_cache.shape[0] * k_cache.shape[1]
    D = k_cache.shape[2] * k_cache.shape[3]
    k_flat = k_cache.reshape(num_slots, D)
    v_flat = v_cache.reshape(num_slots, D)
    N = key.shape[0]
    slots = slot_mapping.reshape(-1)
    if slots.numel() != N:
        N = min(slots.numel(), N)
        slots = slots[:N]
        key = key[:N]
        value = value[:N]
    valid = slots >= 0
    valid_slots = slots[valid].long()
    k_flat.index_copy_(0, valid_slots, key.reshape(N, -1)[valid])
    v_flat.index_copy_(0, valid_slots, value.reshape(N, -1)[valid])


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    if _HAS_TRITON and key.is_cuda:
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N
        store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
    else:
        _store_kvcache_torch(key, value, k_cache, v_cache, slot_mapping)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.force_paged = False
        self.k_cache = self.v_cache = torch.tensor([])

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None):
        q = q.transpose(0, 1).unsqueeze(0).contiguous()
        k = k.transpose(0, 1).unsqueeze(0).contiguous()
        v = v.transpose(0, 1).unsqueeze(0).contiguous()
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        o = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )
        return o.squeeze(0).transpose(0, 1).contiguous()

    def _make_mask(self, q_len: int, kv_len: int, dtype: torch.dtype, device: torch.device):
        if q_len == 1:
            return None
        row = torch.arange(q_len, device=device)
        col = torch.arange(kv_len, device=device)
        allowed = col[None, :] <= row[:, None] + (kv_len - q_len)
        mask = torch.zeros((q_len, kv_len), dtype=dtype, device=device)
        return mask.masked_fill(~allowed, float("-inf"))

    def _gather_kv(self, k_cache: torch.Tensor, v_cache: torch.Tensor, block_table: torch.Tensor, length: int):
        block_size = k_cache.shape[1]
        num_blocks = (length + block_size - 1) // block_size
        blocks = block_table[:num_blocks].long()
        k = k_cache.index_select(0, blocks).reshape(-1, self.num_kv_heads, self.head_dim)[:length]
        v = v_cache.index_select(0, blocks).reshape(-1, self.num_kv_heads, self.head_dim)[:length]
        return k, v

    def _forward_sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context):
        outputs = []
        if context.is_prefill:
            cu_seqlens_q = context.cu_seqlens_q.tolist()
            if context.block_tables is None:
                for i in range(len(cu_seqlens_q) - 1):
                    start, end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
                    q_i, k_i, v_i = q[start:end], k[start:end], v[start:end]
                    mask = self._make_mask(q_i.shape[0], k_i.shape[0], q.dtype, q.device)
                    outputs.append(self._sdpa(q_i, k_i, v_i, mask))
            else:
                context_lens = context.context_lens.tolist()
                k_cache, v_cache = self.k_cache, self.v_cache
                for i in range(len(cu_seqlens_q) - 1):
                    start, end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
                    length = context_lens[i]
                    q_i = q[start:end]
                    k_i, v_i = self._gather_kv(k_cache, v_cache, context.block_tables[i], length)
                    mask = self._make_mask(q_i.shape[0], length, q.dtype, q.device)
                    outputs.append(self._sdpa(q_i, k_i, v_i, mask))
        else:
            context_lens = context.context_lens.tolist()
            k_cache, v_cache = self.k_cache, self.v_cache
            for i in range(q.shape[0]):
                length = context_lens[i]
                q_i = q[i:i + 1]
                k_i, v_i = self._gather_kv(k_cache, v_cache, context.block_tables[i], length)
                outputs.append(self._sdpa(q_i, k_i, v_i, None))
        if not outputs:
            return q.new_empty((0, self.num_heads, self.head_dim))
        return torch.cat(outputs, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        force_paged = self.force_paged or bool(getattr(context, "force_paged", False)) or bool(getattr(context, "use_paged_attention", False))
        if _HAS_FLASH_ATTN and q.is_cuda and not force_paged:
            if context.is_prefill:
                if context.block_tables is not None:
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
            return o
        return self._forward_sdpa(q, k, v, context)
