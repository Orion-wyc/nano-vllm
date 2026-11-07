from dataclasses import dataclass
import threading

import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    state_slots: list[int] | None = None
    reset_slots: list[int] | None = None


_LOCAL = threading.local()


def get_context():
    context = getattr(_LOCAL, "context", None)
    if context is None:
        context = Context()
        _LOCAL.context = context
    return context


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, state_slots=None, reset_slots=None):
    _LOCAL.context = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, state_slots, reset_slots)


def reset_context():
    _LOCAL.context = Context()
