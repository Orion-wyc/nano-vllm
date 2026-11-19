from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)

LOCAL_HOSTNAME = "127.0.0.1"


@dataclass
class MooncakeStoreConfig:
    enabled: bool = False
    master: str = "127.0.0.1:50051"
    metadata: str = "P2PHANDSHAKE"
    protocol: str = "tcp"
    global_segment_size: int = 1024**3
    local_buffer_size: int = 256 * 1024**2
    tenant: str = "default"
    bootstrap_port: int = 8998
    state_snapshots: str = "block"
    missing_policy: str = "error"


def _load_store_class():
    """Import the Mooncake engine lazily so this module stays import-safe."""
    from mooncake.store import MooncakeDistributedStore

    return MooncakeDistributedStore


class MooncakeStoreClient:
    """Thin synchronous wrapper around ``MooncakeDistributedStore``.

    The underlying engine only returns integer status codes, and a missing
    object is reported as an empty byte string. This wrapper turns non-zero
    codes into ``RuntimeError`` and normalizes missing values so callers never
    mistake an empty payload for valid data.
    """

    def __init__(self, config: MooncakeStoreConfig):
        self._config = config
        self._engine_id = f"nano-{uuid.uuid4().hex[:8]}"
        self._store: Any | None = None
        self._setup_lock = threading.Lock()

    @property
    def config(self) -> MooncakeStoreConfig:
        return self._config

    @property
    def engine_id(self) -> str:
        return self._engine_id

    def setup(self) -> None:
        with self._setup_lock:
            if self._store is not None:
                return

            config = self._config
            try:
                store_cls = _load_store_class()
            except ImportError as exc:
                raise RuntimeError(
                    "Mooncake Store is not available: failed to import "
                    "'mooncake.store.MooncakeDistributedStore'. Install a "
                    "Mooncake wheel with Store support (for example "
                    "mooncake-transfer-engine) before enabling the mooncake "
                    "KV backend."
                ) from exc

            try:
                store = store_cls()
            except Exception as exc:
                raise RuntimeError("Failed to create a Mooncake store client") from exc

            try:
                ret = store.setup(
                    LOCAL_HOSTNAME,
                    config.metadata,
                    config.global_segment_size,
                    config.local_buffer_size,
                    config.protocol,
                    "",
                    config.master,
                    tenant_id=config.tenant,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Mooncake store setup failed (master={config.master}, "
                    f"metadata={config.metadata}, protocol={config.protocol}). "
                    "Check that mooncake_master is running and reachable."
                ) from exc

            if ret != 0:
                raise RuntimeError(
                    f"Mooncake store setup returned error code {ret} "
                    f"(master={config.master}, metadata={config.metadata}, "
                    f"protocol={config.protocol}). Check that mooncake_master "
                    "is running and reachable."
                )

            self._store = store

    def _engine(self) -> Any:
        if self._store is None:
            raise RuntimeError(
                "MooncakeStoreClient.setup() must be called before using the store"
            )
        return self._store

    def put(self, key: str, data: bytes) -> None:
        store = self._engine()
        try:
            ret = store.put(key, data)
        except Exception as exc:
            raise RuntimeError(f"Mooncake put failed for key {key!r}") from exc
        if ret != 0:
            raise RuntimeError(
                f"Mooncake put failed for key {key!r} with error code {ret}"
            )

    def get(self, key: str) -> bytes | None:
        store = self._engine()
        try:
            data = store.get(key)
        except Exception as exc:
            raise RuntimeError(f"Mooncake get failed for key {key!r}") from exc
        if not data:
            return None
        return bytes(data)

    def put_batch(self, keys: list[str], values: list[bytes]) -> None:
        if len(keys) != len(values):
            raise ValueError(
                f"put_batch requires matching lengths, got {len(keys)} keys "
                f"and {len(values)} values"
            )
        if not keys:
            return
        store = self._engine()
        try:
            ret = store.put_batch(list(keys), list(values))
        except Exception as exc:
            raise RuntimeError(
                f"Mooncake put_batch failed for {len(keys)} objects"
            ) from exc
        if ret != 0:
            raise RuntimeError(
                f"Mooncake put_batch failed for {len(keys)} objects with "
                f"error code {ret}"
            )

    def get_batch(self, keys: list[str]) -> list[bytes]:
        if not keys:
            return []
        store = self._engine()
        try:
            values = store.get_batch(list(keys))
        except Exception as exc:
            raise RuntimeError(
                f"Mooncake get_batch failed for {len(keys)} objects"
            ) from exc

        values = list(values or [])
        result = [b""] * len(keys)
        for index, value in enumerate(values[: len(keys)]):
            result[index] = bytes(value) if value else b""
        return result

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        if not keys:
            return []
        store = self._engine()
        try:
            flags = store.batch_is_exist(list(keys))
        except Exception as exc:
            raise RuntimeError(
                f"Mooncake batch_is_exist failed for {len(keys)} objects"
            ) from exc

        flags = list(flags or [])
        result = [0] * len(keys)
        for index, flag in enumerate(flags[: len(keys)]):
            try:
                result[index] = 1 if int(flag) > 0 else 0
            except (TypeError, ValueError):
                result[index] = 0
        return result

    def remove(self, key: str) -> None:
        store = self._engine()
        try:
            ret = store.remove(key)
        except Exception as exc:
            raise RuntimeError(f"Mooncake remove failed for key {key!r}") from exc
        if ret != 0:
            raise RuntimeError(
                f"Mooncake remove failed for key {key!r} with error code {ret}"
            )

    def close(self) -> None:
        store = self._store
        self._store = None
        if store is None:
            return
        try:
            store.close()
        except Exception:
            LOGGER.warning("Mooncake store close failed", exc_info=True)


def create_store(config: MooncakeStoreConfig) -> MooncakeStoreClient | None:
    if not config.enabled:
        return None
    return MooncakeStoreClient(config)
