from __future__ import annotations

import io
import threading
import time
import uuid

import torch


TRANSFER_VERSION = 1
TRANSFER_KEY = "nano_pd"


def encode_payload(kv: torch.Tensor | None, state) -> bytes:
    buffer = io.BytesIO()
    torch.save({"version": TRANSFER_VERSION, "kv": kv, "state": state}, buffer)
    return buffer.getvalue()


def decode_payload(data: bytes) -> dict:
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("version") != TRANSFER_VERSION:
        raise ValueError("unsupported PD transfer payload")
    return payload


class PDTransferStore:

    def __init__(self, ttl_seconds: float = 300.0):
        self.ttl_seconds = ttl_seconds
        self._items: dict[str, tuple[float, bytes]] = {}
        self._lock = threading.Lock()

    def put(self, payload: bytes) -> str:
        transfer_id = uuid.uuid4().hex
        deadline = time.monotonic() + self.ttl_seconds
        with self._lock:
            self._items[transfer_id] = (deadline, payload)
            self._cleanup_locked()
        return transfer_id

    def get(self, transfer_id: str) -> bytes | None:
        now = time.monotonic()
        with self._lock:
            self._cleanup_locked(now)
            item = self._items.get(transfer_id)
            if item is None:
                return None
            _, payload = item
            return payload

    def remove(self, transfer_id: str) -> None:
        with self._lock:
            self._items.pop(transfer_id, None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def _cleanup_locked(self, now: float | None = None) -> None:
        if now is None:
            now = time.monotonic()
        expired = [key for key, (deadline, _) in self._items.items() if deadline <= now]
        for key in expired:
            self._items.pop(key, None)


def transfer_metadata(
    transfer_id: str,
    prefill_url: str,
    token_ids: list[int],
    num_prompt_tokens: int,
    num_blocks: int,
    first_token: int,
    eos_token_id: int,
    has_state: bool,
) -> dict:
    return {
        "version": TRANSFER_VERSION,
        "transfer_id": transfer_id,
        "prefill_url": prefill_url.rstrip("/"),
        "token_ids": list(token_ids),
        "num_prompt_tokens": num_prompt_tokens,
        "num_blocks": num_blocks,
        "first_token": first_token,
        "eos_token_id": eos_token_id,
        "has_state": has_state,
    }


def extract_metadata(kv_transfer_params) -> dict | None:
    if not isinstance(kv_transfer_params, dict):
        return None
    metadata = kv_transfer_params.get(TRANSFER_KEY)
    if not isinstance(metadata, dict):
        return None
    if metadata.get("version") != TRANSFER_VERSION:
        return None
    return metadata
