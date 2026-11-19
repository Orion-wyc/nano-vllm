from __future__ import annotations

import io
import json
import time
from dataclasses import asdict, dataclass, field

import torch

from .prefix_cache import PrefixCacheKeyBuilder


MANIFEST_VERSION = 1


def request_kv_key(transfer_id: str, block_index: int) -> str:
    return f"nano:kv:req:{transfer_id}:{block_index}"


def request_state_key(transfer_id: str) -> str:
    return f"nano:state:req:{transfer_id}"


def encode_kv_block(block: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save({"version": 1, "block": block.detach().cpu()}, buffer)
    return buffer.getvalue()


def decode_kv_block(data: bytes) -> torch.Tensor:
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    block = payload["block"]
    if block.dim() == 5:
        block = block.unsqueeze(2)
    if block.dim() != 6:
        raise ValueError(f"unexpected KV block shape: {tuple(block.shape)}")
    return block


def encode_state(state) -> bytes:
    buffer = io.BytesIO()
    torch.save({"version": 1, "state": state}, buffer)
    return buffer.getvalue()


def decode_state(data: bytes):
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    return payload.get("state")


@dataclass(slots=True)
class MooncakeManifest:
    transfer_id: str
    engine_id: str
    model_scope: str
    token_ids: list[int]
    num_prompt_tokens: int
    block_keys: list[str]
    prefix_hashes: list[str]
    state_key: str | None
    first_token: int
    eos_token_id: int
    block_size: int
    kv_shape: list[int]
    prefix_hit_blocks: int = 0
    version: int = MANIFEST_VERSION
    created_at: float = field(default_factory=time.time)

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), ensure_ascii=False).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "MooncakeManifest":
        payload = json.loads(data.decode("utf-8"))
        if payload.get("version") != MANIFEST_VERSION:
            raise ValueError("unsupported Mooncake manifest version")
        return cls(**payload)


def store_manifest(store, builder: PrefixCacheKeyBuilder, manifest: MooncakeManifest) -> None:
    store.put(builder.manifest_key(manifest.transfer_id), manifest.to_bytes())


def load_manifest(store, builder: PrefixCacheKeyBuilder, transfer_id: str):
    data = store.get(builder.manifest_key(transfer_id))
    if not data:
        return None
    return MooncakeManifest.from_bytes(data)


def block_keys_for(
    builder: PrefixCacheKeyBuilder,
    transfer_id: str,
    token_ids: list[int],
    num_prompt_tokens: int,
    block_size: int,
    num_blocks: int,
):
    full_hashes = builder.block_hashes(token_ids[:num_prompt_tokens])
    keys = []
    for index in range(num_blocks):
        if index < len(full_hashes):
            keys.append(builder.kv_key(full_hashes[index]))
        else:
            keys.append(request_kv_key(transfer_id, index))
    return keys, full_hashes


def publish_prefill(
    store,
    builder: PrefixCacheKeyBuilder,
    transfer_id: str,
    engine_id: str,
    token_ids: list[int],
    num_prompt_tokens: int,
    block_tensors: list[torch.Tensor],
    state,
    first_token: int,
    eos_token_id: int,
    block_size: int,
    kv_shape: list[int],
    prefix_hit_blocks: int = 0,
    state_snapshots: str = "block",
    prefix_blocks: int | None = None,
) -> MooncakeManifest:
    block_keys, full_hashes = block_keys_for(
        builder,
        transfer_id,
        token_ids,
        num_prompt_tokens,
        block_size,
        len(block_tensors),
    )
    for key, block in zip(block_keys, block_tensors):
        store.put(key, encode_kv_block(block))
    state_key = None
    if state is not None:
        state_key = request_state_key(transfer_id)
        encoded_state = encode_state(state)
        store.put(state_key, encoded_state)
        if state_snapshots == "block" and num_prompt_tokens % block_size == 0:
            full_hashes = builder.block_hashes(token_ids[:num_prompt_tokens])
            if full_hashes:
                store.put(builder.state_key(full_hashes[-1]), encoded_state)
    manifest = MooncakeManifest(
        transfer_id=transfer_id,
        engine_id=engine_id,
        model_scope=builder.scope,
        token_ids=list(token_ids),
        num_prompt_tokens=int(num_prompt_tokens),
        block_keys=block_keys,
        prefix_hashes=full_hashes,
        state_key=state_key,
        first_token=int(first_token),
        eos_token_id=int(eos_token_id),
        block_size=int(block_size),
        kv_shape=list(kv_shape),
        prefix_hit_blocks=int(prefix_hit_blocks),
    )
    store_manifest(store, builder, manifest)
    return manifest


def publish_block_snapshot(
    store,
    builder: PrefixCacheKeyBuilder,
    prefix_hash: str,
    block: torch.Tensor,
    state,
) -> None:
    store.put(builder.kv_key(prefix_hash), encode_kv_block(block))
    if state is not None:
        store.put(builder.state_key(prefix_hash), encode_state(state))


def load_prefix(store, match):
    blocks = store.get_batch(match.kv_keys)
    if len(blocks) != len(match.kv_keys):
        raise ValueError("Mooncake returned an unexpected prefix block count")
    decoded = []
    for index, data in enumerate(blocks):
        if not data:
            raise ValueError(f"Mooncake prefix block missing: {match.kv_keys[index]}")
        decoded.append(decode_kv_block(data))
    state = None
    if match.state_key:
        data = store.get(match.state_key)
        if not data:
            raise ValueError(f"Mooncake prefix state missing: {match.state_key}")
        state = decode_state(data)
    return decoded, state


def load_prefill(store, manifest: MooncakeManifest):
    blocks = store.get_batch(manifest.block_keys)
    if len(blocks) != len(manifest.block_keys):
        raise ValueError("Mooncake returned an unexpected block count")
    decoded = []
    for index, data in enumerate(blocks):
        if not data:
            raise ValueError(f"Mooncake KV block missing: {manifest.block_keys[index]}")
        decoded.append(decode_kv_block(data))
    kv = torch.cat(decoded, dim=2)
    state = None
    if manifest.state_key:
        data = store.get(manifest.state_key)
        if not data:
            raise ValueError(f"Mooncake state missing: {manifest.state_key}")
        state = decode_state(data)
    return kv, state
