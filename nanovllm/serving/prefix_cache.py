from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

_TOKEN_SIZE = 4
_TOKEN_MAX = (1 << (8 * _TOKEN_SIZE)) - 1
_SCOPE_SEPARATOR = b"\x00"
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _encode_tokens(token_ids: list[int]) -> bytes:
    buffer = bytearray(len(token_ids) * _TOKEN_SIZE)
    for index, token in enumerate(token_ids):
        if not isinstance(token, int):
            raise TypeError(f"token id must be an int, got {type(token).__name__}")
        if token < 0 or token > _TOKEN_MAX:
            raise ValueError(f"token id {token} does not fit in uint32")
        buffer[index * _TOKEN_SIZE : (index + 1) * _TOKEN_SIZE] = token.to_bytes(
            _TOKEN_SIZE, "little"
        )
    return bytes(buffer)


class PrefixCacheKeyBuilder:
    """Builds deterministic Mooncake object keys for prefix KV caching.

    Prefix hashes are chained: a hash covers every token from position 0 up to
    the end of its block, so a hit at block *i* proves that the whole prefix
    matches. Only complete blocks participate; a trailing partial block is
    ignored.
    """

    def __init__(self, scope: str, block_size: int):
        if not scope:
            raise ValueError("scope must be a non-empty string")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.scope = scope
        self.block_size = block_size

    def block_hashes(self, token_ids: list[int]) -> list[str]:
        num_blocks = len(token_ids) // self.block_size
        if num_blocks == 0:
            return []

        scope_bytes = self.scope.encode("utf-8")
        hashes: list[str] = []
        previous: bytes | None = None
        for block_index in range(num_blocks):
            start = block_index * self.block_size
            block = token_ids[start : start + self.block_size]
            block_bytes = _encode_tokens(block)
            if previous is None:
                digest = hashlib.sha256(
                    scope_bytes + _SCOPE_SEPARATOR + block_bytes
                ).digest()
            else:
                digest = hashlib.sha256(previous + block_bytes).digest()
            hashes.append(digest.hex())
            previous = digest
        return hashes

    def kv_key(self, prefix_hash: str) -> str:
        return f"nano:kv:v1:{self.scope}:{prefix_hash}"

    def state_key(self, prefix_hash: str) -> str:
        return f"nano:state:v1:{self.scope}:{prefix_hash}"

    def manifest_key(self, transfer_id: str) -> str:
        return f"nano:manifest:v1:{transfer_id}"


@dataclass
class PrefixMatch:
    num_blocks: int = 0
    num_tokens: int = 0
    hashes: list[str] = field(default_factory=list)
    kv_keys: list[str] = field(default_factory=list)
    state_key: str | None = None


def _normalize_flags(flags, expected: int) -> list[int]:
    try:
        values = list(flags or [])
    except TypeError:
        return [0] * expected
    result = [0] * expected
    for index, value in enumerate(values[:expected]):
        try:
            result[index] = 1 if int(value) > 0 else 0
        except (TypeError, ValueError):
            result[index] = 0
    return result


def find_prefix(
    store,
    builder: PrefixCacheKeyBuilder,
    token_ids: list[int],
    require_state: bool = True,
) -> PrefixMatch:
    """Return the longest fully-contained prefix available in ``store``.

    ``store`` only needs a ``batch_is_exist(keys) -> list[int]`` method.
    KV blocks must exist contiguously from block 0. When ``require_state`` is
    true, a matching state snapshot is also required at every block boundary,
    and the match stops at the first boundary whose state is missing.
    """
    hashes = builder.block_hashes(token_ids)
    if not hashes:
        return PrefixMatch()

    kv_keys = [builder.kv_key(prefix_hash) for prefix_hash in hashes]
    kv_flags = _normalize_flags(store.batch_is_exist(kv_keys), len(hashes))

    kv_blocks = 0
    while kv_blocks < len(kv_flags) and kv_flags[kv_blocks] > 0:
        kv_blocks += 1

    if kv_blocks == 0:
        return PrefixMatch()

    state_blocks = kv_blocks
    if require_state:
        state_keys = [builder.state_key(prefix_hash) for prefix_hash in hashes[:kv_blocks]]
        state_flags = _normalize_flags(store.batch_is_exist(state_keys), kv_blocks)
        state_blocks = 0
        while state_blocks < kv_blocks and state_flags[state_blocks] > 0:
            state_blocks += 1

    if state_blocks == 0:
        return PrefixMatch()

    matched_hashes = hashes[:state_blocks]
    state_key = builder.state_key(matched_hashes[-1]) if require_state else None
    return PrefixMatch(
        num_blocks=state_blocks,
        num_tokens=state_blocks * builder.block_size,
        hashes=matched_hashes,
        kv_keys=kv_keys[:state_blocks],
        state_key=state_key,
    )


def _slugify(value: str) -> str:
    slug = _SLUG_PATTERN.sub("-", value.lower()).strip("-")
    return slug[:32] or "model"


def _normalize_dtype(dtype) -> str:
    value = str(dtype).lower()
    if "." in value:
        value = value.rsplit(".", 1)[-1]
    return value.replace(" ", "")


def compute_scope(
    model_path: str,
    dtype: str,
    block_size: int,
    num_kv_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> str:
    """Return a stable, readable namespace for model/layout-specific objects."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_kv_layers <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        raise ValueError("KV layout dimensions must be positive")

    path = os.path.normpath(os.path.expanduser(str(model_path)))
    dtype_name = _normalize_dtype(dtype)
    fingerprint = "\x1f".join(
        [
            path,
            dtype_name,
            str(block_size),
            str(num_kv_layers),
            str(num_kv_heads),
            str(head_dim),
        ]
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:8]
    model_slug = _slugify(os.path.basename(path))
    return (
        f"v1-{model_slug}-{dtype_name}-{block_size}"
        f"-l{num_kv_layers}-h{num_kv_heads}-d{head_dim}-{digest}"
    )
