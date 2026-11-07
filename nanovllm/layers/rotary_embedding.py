from functools import lru_cache
import torch
from torch import nn

from nanovllm.utils.device import maybe_compile, device_type


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        assert rotary_dim <= head_size
        assert rotary_dim % 2 == 0
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float, device=device) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float, device=device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @maybe_compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert query.shape[-1] == self.rotary_dim
        assert key.shape[-1] == self.rotary_dim
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


def _current_device_str() -> str:
    kind = device_type()
    if kind == "npu":
        return f"npu:{torch.npu.current_device()}"
    if kind == "cuda":
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


@lru_cache(8)
def _get_rope_cached(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    device_str: str,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base, torch.device(device_str))
    return rotary_emb


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    return _get_rope_cached(head_size, rotary_dim, max_position, base, _current_device_str())
